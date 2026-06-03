"""
VITA model integration for drugrec_benchmark.

The original VITA core is embedded in this module so the benchmark can build
and run the model without runtime imports from the original repository.
"""
from __future__ import annotations

import math
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.core.io import Prediction
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from drugrec_benchmark.models.cognet import _COGNetBeam
from drugrec_benchmark.utils.metrics import sequence_output_process


class GraphConvolution(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_features))
		else:
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		stdv = 1.0 / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		output = torch.mm(adj, torch.mm(inputs, self.weight))
		if self.bias is not None:
			return output + self.bias
		return output


class VITAGCN(nn.Module):
	def __init__(self, voc_size: int, emb_dim: int, ehr_adj: np.ndarray, ddi_adj: np.ndarray, device: torch.device) -> None:
		super().__init__()
		self.voc_size = voc_size
		self.emb_dim = emb_dim
		self.device = device
		ehr_adj = self._normalize(np.asarray(ehr_adj, dtype=np.float32) + np.eye(ehr_adj.shape[0], dtype=np.float32))
		ddi_adj = self._normalize(np.asarray(ddi_adj, dtype=np.float32) + np.eye(ddi_adj.shape[0], dtype=np.float32))
		self.register_buffer("ehr_adj", torch.FloatTensor(ehr_adj))
		self.register_buffer("ddi_adj", torch.FloatTensor(ddi_adj))
		self.register_buffer("x", torch.eye(voc_size))
		self.gcn1 = GraphConvolution(voc_size, emb_dim)
		self.dropout = nn.Dropout(p=0.3)
		self.gcn2 = GraphConvolution(emb_dim, emb_dim)
		self.gcn3 = GraphConvolution(emb_dim, emb_dim)

	@staticmethod
	def _normalize(mx: np.ndarray) -> np.ndarray:
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.0
		return np.diagflat(r_inv).dot(mx)

	def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
		ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
		ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)
		ehr_node_embedding = F.relu(ehr_node_embedding)
		ddi_node_embedding = F.relu(ddi_node_embedding)
		ehr_node_embedding = self.dropout(ehr_node_embedding)
		ddi_node_embedding = self.dropout(ddi_node_embedding)
		ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
		ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
		return ehr_node_embedding, ddi_node_embedding


class MedTransformerDecoder(nn.Module):
	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5) -> None:
		super().__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)
		self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)
		self.activation = nn.ReLU()
		self.nhead = nhead

	def forward(
		self,
		input_medication_embedding: torch.Tensor,
		input_medication_memory: torch.Tensor,
		input_disease_embdding: torch.Tensor,
		input_medication_self_mask: torch.Tensor,
		d_mask: torch.Tensor,
	) -> torch.Tensor:
		input_len = input_medication_embedding.size(0)
		tgt_len = input_medication_embedding.size(1)
		subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len * self.nhead, input_disease_embdding.device)
		self_attn_mask = subsequent_mask + input_medication_self_mask
		x = input_medication_embedding + input_medication_memory
		x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, attn_mask=self_attn_mask, need_weights=False)[0]))
		x = self.norm2(x + self.dropout2(self.m2d_multihead_attn(x, input_disease_embdding, input_disease_embdding, attn_mask=d_mask, need_weights=False)[0]))
		return self.norm3(x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x))))))

	@staticmethod
	def generate_square_subsequent_mask(sz: int, batch_size: int, device: torch.device) -> torch.Tensor:
		mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, 0.0)
		return mask.unsqueeze(0).repeat(batch_size, 1, 1)


class _OriginalVITA(nn.Module):
	"""Self-contained VITA core adapted from the original implementation."""

	def __init__(self, voc_size: Tuple[int, int, int], ehr_adj: np.ndarray, ddi_adj: np.ndarray, ddi_mask_H: np.ndarray, emb_dim: int = 64, device: torch.device = torch.device("cpu")) -> None:
		super().__init__()
		self.voc_size = voc_size
		self.emb_dim = emb_dim
		self.device = device
		self.nhead = 2
		self.SOS_TOKEN = voc_size[2]
		self.END_TOKEN = voc_size[2] + 1
		self.MED_PAD_TOKEN = voc_size[2] + 2
		self.DIAG_PAD_TOKEN = voc_size[0] + 2
		self.PROC_PAD_TOKEN = voc_size[1] + 2
		self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
		self.concat_embedding = nn.Sequential(
			nn.Embedding(voc_size[0] + 3 + voc_size[1] + 3, emb_dim, self.DIAG_PAD_TOKEN + self.PROC_PAD_TOKEN),
			nn.Dropout(0.3),
		)
		self.linear_layer = nn.Linear(emb_dim, emb_dim)
		self.mlp_layer = nn.Linear(71, 1)
		self.med_embedding = nn.Sequential(
			nn.Embedding(voc_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
			nn.Dropout(0.3),
		)
		self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
		self.gcn = VITAGCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
		self.inter = nn.Parameter(torch.FloatTensor(1))
		self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim * 2, dropout=0.2, layer_norm_eps=1e-5)
		self.Wo = nn.Linear(emb_dim, voc_size[2] + 2)
		self.Wc = nn.Linear(emb_dim, emb_dim)
		self.W_z = nn.Linear(emb_dim, 1)
		self.MLP_layer2 = nn.Linear(71, 1)
		self.MLP_layer3 = nn.Linear(emb_dim, 1)
		self.MLP_layer4 = nn.Linear(2, 1)
		self.gumbel_tau = 0.6
		self.att_tau = 20
		self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)
		self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
		self.bipartite_output = nn.Linear(ddi_mask_H.shape[1], voc_size[2], bias=False)

	@staticmethod
	def _fit_code_width(tensor: torch.Tensor, target_width: int) -> torch.Tensor:
		width = tensor.size(-1)
		if width == target_width:
			return tensor
		if width == 0:
			return tensor.new_zeros(*tensor.shape[:-1], target_width)
		if width < target_width:
			return F.pad(tensor, (0, target_width - width))
		flat = tensor.reshape(-1, 1, width)
		pooled = F.adaptive_avg_pool1d(flat, target_width)
		return pooled.reshape(*tensor.shape[:-1], target_width)

	def decode(
		self,
		input_medications: torch.Tensor,
		input_disease_embedding: torch.Tensor,
		last_medication_embedding: torch.Tensor,
		last_medications: torch.Tensor,
		cross_visit_scores: torch.Tensor,
		d_mask_matrix: torch.Tensor,
		p_mask_matrix: torch.Tensor,
		m_mask_matrix: torch.Tensor,
		last_m_mask: torch.Tensor,
		drug_memory: torch.Tensor,
	) -> torch.Tensor:
		batch_size = input_medications.size(0)
		max_visit_num = input_medications.size(1)
		max_med_num = input_medications.size(2)
		max_diag_num = input_disease_embedding.size(2)
		input_medication_embs = self.med_embedding(input_medications).view(batch_size * max_visit_num, max_med_num, -1)
		input_medication_memory = drug_memory[input_medications].view(batch_size * max_visit_num, max_med_num, -1)
		d_p_mask_matrix = torch.cat([d_mask_matrix, p_mask_matrix], dim=-1)
		d_p_mask_2d = d_p_mask_matrix.view(batch_size * max_visit_num, -1)
		no_code_rows = d_p_mask_2d.eq(0.0).sum(dim=1) == 0
		if torch.any(no_code_rows):
			d_p_mask_2d = d_p_mask_2d.clone()
			d_p_mask_2d[no_code_rows, 0] = 0.0
			d_p_mask_matrix = d_p_mask_2d.view(batch_size, max_visit_num, -1)
		last_m_enc_mask = m_mask_matrix.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
		medication_self_mask = last_m_enc_mask.view(batch_size * max_visit_num * self.nhead, max_med_num, max_med_num)
		m2d_mask_matrix = d_p_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1)
		m2d_mask_matrix = m2d_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num)
		dec_hidden = self.decoder(
			input_medication_embedding=input_medication_embs,
			input_medication_memory=input_medication_memory,
			input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num, max_diag_num, -1),
			input_medication_self_mask=medication_self_mask,
			d_mask=m2d_mask_matrix,
		)
		score_g = self.Wo(dec_hidden).view(batch_size, max_visit_num, max_med_num, -1)
		prob_g = F.softmax(score_g, dim=-1)
		score_c = self.medication_level(dec_hidden.view(batch_size, max_visit_num, max_med_num, -1), last_medication_embedding, last_m_mask, cross_visit_scores)
		prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(batch_size, max_visit_num * max_med_num, -1)
		copy_source = last_medications.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)
		prob_c_to_g.scatter_add_(2, copy_source, score_c)
		prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)
		generate_prob = torch.sigmoid(self.W_z(dec_hidden)).view(batch_size, max_visit_num, max_med_num, 1)
		prob = prob_g * generate_prob + prob_c_to_g * (1.0 - generate_prob)
		prob[:, 0, :, :] = prob_g[:, 0, :, :]
		return torch.log(prob.clamp_min(1e-12))

	def forward(self, diseases: torch.Tensor, procedures: torch.Tensor, medications: torch.Tensor, d_mask_matrix: torch.Tensor, p_mask_matrix: torch.Tensor, m_mask_matrix: torch.Tensor, seq_length: torch.Tensor, dec_disease: torch.Tensor, stay_disease: torch.Tensor, dec_disease_mask: torch.Tensor, stay_disease_mask: torch.Tensor, dec_proc: torch.Tensor, stay_proc: torch.Tensor, dec_proc_mask: torch.Tensor, stay_proc_mask: torch.Tensor, max_len: int = 20):
		batch_size, max_seq_length, _ = medications.size()
		input_disease_embdding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, count, gumbel_pick_index = self.encode(
			diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask, stay_proc_mask, max_len=max_len
		)
		input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN, device=self.device, dtype=medications.dtype)
		input_medication = torch.cat([input_medication, medications], dim=2)
		m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float()
		m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)
		output_logits = self.decode(input_medication, input_disease_embdding, encoded_medication, last_seq_medication, cross_visit_scores, d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory)
		return output_logits, count, gumbel_pick_index, cross_visit_scores.detach().cpu().numpy()

	def calc_cross_visit_scores(self, embedding: torch.Tensor, gumbel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		diag_keys = embedding[:, :, :]
		diag_query = embedding[:, -1:, :]
		diag_scores = torch.bmm(self.linear_layer(diag_query), diag_keys.transpose(-2, -1)) / math.sqrt(diag_query.size(-1))
		diag_scores_encoder = diag_scores.squeeze(0).squeeze(0)
		diag_scores = diag_scores.squeeze(0).squeeze(0).masked_fill(gumbel == 0, -1e9)
		scores = F.softmax(diag_scores / self.att_tau, dim=-1)
		scores_encoder = F.softmax(diag_scores_encoder / self.att_tau, dim=-1)
		return scores, scores_encoder

	def medication_level(self, decode_input_hiddens: torch.Tensor, last_medications: torch.Tensor, last_m_mask: torch.Tensor, cross_visit_scores: torch.Tensor) -> torch.Tensor:
		max_visit_num = decode_input_hiddens.size(1)
		input_med_num = decode_input_hiddens.size(2)
		max_med_num = last_medications.size(2)
		copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num * input_med_num, self.emb_dim)
		attn_scores = torch.matmul(copy_query, last_medications.view(-1, max_visit_num * max_med_num, self.emb_dim).transpose(-2, -1)) / math.sqrt(self.emb_dim)
		med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
		attn_scores = F.softmax(attn_scores + med_mask, dim=-1)
		visit_scores = cross_visit_scores.unsqueeze(0).unsqueeze(-1).repeat(1, 1, max_med_num).view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
		scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
		row_scores = scores.sum(dim=-1, keepdim=True)
		return scores / row_scores



class VITA(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		di_mask_h: np.ndarray,
		emb_dim: int = 64,
		max_len: int = 45,
		beam_size: int = 4,
		max_diag_num: int = 39,
		max_proc_num: int = 32,
		max_med_num: int = 56,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "sequence"
		self.vocab_size = vocab_size
		self.max_len = int(max_len)
		self.beam_size = int(beam_size)
		self.max_diag_num = int(max_diag_num)
		self.max_proc_num = int(max_proc_num)
		self.max_med_num = int(max_med_num)
		self.use_beam_search = False
		self.ddi_adj = ddi_adj

		self.sos_token = vocab_size[2]
		self.end_token = vocab_size[2] + 1
		self.med_pad_token = vocab_size[2] + 2
		self.diag_pad_token = vocab_size[0] + 2
		self.proc_pad_token = vocab_size[1] + 2

		self.core = _OriginalVITA(
			vocab_size,
			ehr_adj,
			ddi_adj=ddi_adj,
			ddi_mask_H=di_mask_h,
			emb_dim=emb_dim,
			device=self.device,
		)
		self._patch_original_core()

	def _patch_original_core(self) -> None:
		core = self.core
		core.proc_offset = self.vocab_size[0]

		def make_query(core_self: Any, input_disease_embdding: torch.Tensor, medications: torch.Tensor):
			batch_size = 1
			max_visit_num = input_disease_embdding.size(0)
			emb_dim = core_self.emb_dim
			code_width = input_disease_embdding.size(1)
			input1 = core_self.MLP_layer3(input_disease_embdding).squeeze(-1)
			input1 = core_self._fit_code_width(input1, core_self.MLP_layer2.in_features)
			input2 = core_self.MLP_layer2(input1)
			current = input2[-1:, :]
			current2 = current.repeat(input2.size(0), 1)
			concat = torch.cat([input2, current2], dim=-1)
			concat2 = torch.sigmoid(core_self.MLP_layer4(concat))
			gumbel_input = torch.cat([concat2, 1 - concat2], dim=-1)
			pre_gumbel = F.gumbel_softmax(gumbel_input, tau=core_self.gumbel_tau, hard=True)[:, 0]
			gumbel = torch.cat([pre_gumbel[:-1], torch.ones(1, device=core_self.device)])
			picked = input_disease_embdding.mul(
				gumbel.unsqueeze(-1).unsqueeze(-1).expand(-1, code_width, emb_dim)
			)
			visit_input = core_self._fit_code_width(
				picked.transpose(-2, -1), core_self.mlp_layer.in_features
			)
			visit_diag_embedding = core_self.mlp_layer(visit_input).view(
				batch_size, max_visit_num, emb_dim
			)
			cross_visit_scores, scores_encoder = core_self.calc_cross_visit_scores(
				visit_diag_embedding, gumbel
			)
			score_emb = input_disease_embdding.mul(
				cross_visit_scores.unsqueeze(-1).unsqueeze(-1).expand(-1, code_width, emb_dim)
			)
			q_t = torch.sum(score_emb, dim=0, keepdim=True)
			gumbel_numpy = pre_gumbel.detach().cpu().numpy()
			gumbel_pick_index = [
				i + 1
				for i in list(filter(lambda x: gumbel_numpy[x] == 1, range(len(gumbel_numpy))))
			]
			if gumbel_pick_index == []:
				gumbel_pick_index = [0]
			return q_t, gumbel_pick_index, scores_encoder

		def encode(
			core_self: Any,
			diseases: torch.Tensor,
			procedures: torch.Tensor,
			medications: torch.Tensor,
			d_mask_matrix: torch.Tensor,
			p_mask_matrix: torch.Tensor,
			m_mask_matrix: torch.Tensor,
			seq_length: torch.Tensor,
			dec_disease: torch.Tensor,
			stay_disease: torch.Tensor,
			dec_disease_mask: torch.Tensor,
			stay_disease_mask: torch.Tensor,
			dec_proc: torch.Tensor,
			stay_proc: torch.Tensor,
			dec_proc_mask: torch.Tensor,
			stay_proc_mask: torch.Tensor,
			max_len: int = 20,
		):
			batch_size, max_visit_num, max_med_num = medications.size()
			max_diag_num = diseases.size(2)
			max_proc_num = procedures.size(2)
			p_change = procedures + int(core_self.proc_offset)
			adm_1_2 = torch.cat([diseases, p_change], dim=-1)
			input_disease_embdding = core_self.concat_embedding(adm_1_2).view(
				batch_size * max_visit_num,
				max_diag_num + max_proc_num,
				core_self.emb_dim,
			)

			queries = []
			gumbel_pick_index = [0]
			cross_visit_scores = torch.ones(max_visit_num, device=core_self.device)
			for i in range(1, input_disease_embdding.size(0)):
				q_t, gumbel_pick_index, cross_visit_scores = core_self.make_query(
					input_disease_embdding[: i + 1, :, :], medications
				)
				queries.append(q_t)
			if queries:
				pre_queries = torch.cat(queries)
				input_disease_embdding = torch.cat([input_disease_embdding[:1, :, :], pre_queries])
			input_disease_embdding = input_disease_embdding.unsqueeze(dim=0)

			last_seq_medication = torch.full(
				(batch_size, 1, max_med_num), 0, device=core_self.device, dtype=medications.dtype
			)
			last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
			last_m_mask = torch.full(
				(batch_size, 1, max_med_num), -1e9, device=core_self.device
			)
			last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
			last_m_mask_2d = last_m_mask.view(batch_size * max_visit_num, max_med_num)
			no_med_rows = last_m_mask_2d.eq(0.0).sum(dim=1) == 0
			if torch.any(no_med_rows):
				last_seq_flat = last_seq_medication.view(batch_size * max_visit_num, max_med_num).clone()
				last_seq_flat[no_med_rows, 0] = 0
				last_m_mask_2d = last_m_mask_2d.clone()
				last_m_mask_2d[no_med_rows, 0] = 0.0
				last_seq_medication = last_seq_flat.view(batch_size, max_visit_num, max_med_num)
				last_m_mask = last_m_mask_2d.view(batch_size, max_visit_num, max_med_num)

			last_seq_medication_emb = core_self.med_embedding(last_seq_medication)
			last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num)
			last_m_enc_mask = last_m_enc_mask.unsqueeze(dim=1).unsqueeze(dim=1).repeat(
				1, core_self.nhead, max_med_num, 1
			)
			last_m_enc_mask = last_m_enc_mask.view(
				batch_size * max_visit_num * core_self.nhead, max_med_num, max_med_num
			)
			encoded_medication = core_self.medication_encoder(
				last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, core_self.emb_dim),
				src_mask=last_m_enc_mask,
			)
			encoded_medication = encoded_medication.view(
				batch_size, max_visit_num, max_med_num, core_self.emb_dim
			)

			ehr_embedding, ddi_embedding = core_self.gcn()
			drug_memory = ehr_embedding - ddi_embedding * core_self.inter
			drug_memory_padding = torch.zeros((3, core_self.emb_dim), device=core_self.device).float()
			drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)
			return (
				input_disease_embdding,
				encoded_medication,
				cross_visit_scores,
				last_seq_medication,
				last_m_mask,
				drug_memory,
				0,
				gumbel_pick_index,
			)

		core.make_query = types.MethodType(make_query, core)
		core.encode = types.MethodType(encode, core)

	def set_beam_search(self, enabled: bool = True) -> None:
		self.use_beam_search = enabled

	def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		patient = self.get_patient(batch)
		if len(patient) < 2:
			return {
				"labels_flatten": torch.empty((0,), device=self.device),
				"logits": torch.empty((0, self.vocab_size[2] + 2), device=self.device),
			}
		inputs = self._build_model_inputs([patient])
		if self.training:
			output_logits, _, _, _ = self.core(
				inputs["diseases"],
				inputs["procedures"],
				inputs["medications"],
				inputs["d_mask_matrix"],
				inputs["p_mask_matrix"],
				inputs["m_mask_matrix"],
				inputs["seq_length"],
				inputs["dec_disease"],
				inputs["stay_disease"],
				inputs["dec_disease_mask"],
				inputs["stay_disease_mask"],
				inputs["dec_proc"],
				inputs["stay_proc"],
				inputs["dec_proc_mask"],
				inputs["stay_proc_mask"],
				max_len=self.max_len,
			)
		else:
			if self.use_beam_search:
				beam_hypotheses, beam_probs = self._beam_search_decode(inputs)
				return {"beam_hypotheses": beam_hypotheses, "beam_probs": beam_probs}
			output_logits = self._autoregressive_decode(inputs)

		labels_flatten, logits_flatten = self._output_flatten(
			inputs["medications"],
			output_logits,
			inputs["seq_length"],
			inputs["m_length_matrix"],
			self.vocab_size[2] + 2,
			self.end_token,
		)
		return {"labels_flatten": labels_flatten, "logits": logits_flatten}

	def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		labels = outputs.get("labels_flatten")
		if logits.numel() == 0 or labels is None or len(labels) == 0:
			return torch.tensor(0.0, device=self.device, requires_grad=True)
		return F.nll_loss(logits, labels.long())

	def decode(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Prediction:
		med_size = self.vocab_size[2]
		target = self.get_target_indices(batch)
		if "beam_hypotheses" in outputs:
			return self._decode_beam(outputs, target, med_size)
		logits = outputs.get("logits")
		logits_array = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
		if logits_array.size == 0:
			return Prediction(
				med_indices=[],
				med_scores=np.zeros((med_size,), dtype=float),
				target=target,
				task="sequence",
			)
		if logits_array.ndim == 1:
			logits_array = logits_array.reshape(1, -1)
		out_list, ranked = sequence_output_process(logits_array, [self.sos_token, self.end_token])
		scores = np.mean(logits_array[:, :med_size], axis=0)
		return Prediction(
			med_indices=list(out_list),
			med_scores=np.asarray(scores, dtype=float),
			target=target,
			task="sequence",
			ranked_med_indices=list(ranked),
		)

	def _decode_beam(self, outputs: Dict[str, Any], target: List[int], med_size: int) -> Prediction:
		beam_hypotheses = outputs.get("beam_hypotheses") or []
		beam_probs = outputs.get("beam_probs") or []
		if not beam_hypotheses:
			return Prediction(
				med_indices=[],
				med_scores=np.zeros((med_size,), dtype=float),
				target=target,
				task="sequence",
			)
		hypothesis = beam_hypotheses[-1]
		prob_steps = beam_probs[-1] if beam_probs else []
		out_list = []
		out_prob_list = []
		for med, prob in zip(hypothesis, prob_steps):
			if med in [self.sos_token, self.end_token]:
				break
			if 0 <= med < med_size:
				out_list.append(med)
				out_prob_list.append(np.asarray(prob[:med_size], dtype=float))
		if out_prob_list:
			scores = np.max(np.stack(out_prob_list, axis=0), axis=0)
			for med in out_list:
				scores[med] = out_prob_list[out_list.index(med)][med]
		else:
			scores = np.zeros((med_size,), dtype=float)
		return Prediction(
			med_indices=list(out_list),
			med_scores=np.asarray(scores, dtype=float),
			target=target,
			task="sequence",
			ranked_med_indices=list(out_list),
		)

	def _autoregressive_decode(self, inputs: Dict[str, Any]) -> torch.Tensor:
		(
			input_disease_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
			_,
			_,
		) = self.core.encode(
			inputs["diseases"],
			inputs["procedures"],
			inputs["medications"],
			inputs["d_mask_matrix"],
			inputs["p_mask_matrix"],
			inputs["m_mask_matrix"],
			inputs["seq_length"],
			inputs["dec_disease"],
			inputs["stay_disease"],
			inputs["dec_disease_mask"],
			inputs["stay_disease_mask"],
			inputs["dec_proc"],
			inputs["stay_proc"],
			inputs["dec_proc_mask"],
			inputs["stay_proc_mask"],
			max_len=self.max_len,
		)
		batch_size = inputs["medications"].size(0)
		max_visit_num = inputs["medications"].size(1)
		partial_input_medication = torch.full(
			(batch_size, max_visit_num, 1),
			self.sos_token,
			device=self.device,
			dtype=inputs["medications"].dtype,
		)
		partial_logits = None
		for _ in range(self.max_len):
			partial_m_mask_matrix = torch.zeros(
				(batch_size, max_visit_num, partial_input_medication.size(2)), device=self.device
			).float()
			partial_logits = self.core.decode(
				partial_input_medication,
				input_disease_embedding,
				encoded_medication,
				last_seq_medication,
				cross_visit_scores,
				inputs["d_mask_matrix"],
				inputs["p_mask_matrix"],
				partial_m_mask_matrix,
				last_m_mask,
				drug_memory,
			)
			_, next_medication = torch.topk(partial_logits[:, :, -1, :], 1, dim=-1)
			partial_input_medication = torch.cat([partial_input_medication, next_medication], dim=-1)
		return partial_logits if partial_logits is not None else torch.empty((0,), device=self.device)

	def _beam_search_decode(self, inputs: Dict[str, Any]) -> Tuple[List[List[int]], List[List[List[float]]]]:
		(
			input_disease_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
			_,
			_,
		) = self.core.encode(
			inputs["diseases"],
			inputs["procedures"],
			inputs["medications"],
			inputs["d_mask_matrix"],
			inputs["p_mask_matrix"],
			inputs["m_mask_matrix"],
			inputs["seq_length"],
			inputs["dec_disease"],
			inputs["stay_disease"],
			inputs["dec_disease_mask"],
			inputs["stay_disease_mask"],
			inputs["dec_proc"],
			inputs["stay_proc"],
			inputs["dec_proc_mask"],
			inputs["stay_proc_mask"],
			max_len=self.max_len,
		)
		batch_size = inputs["medications"].size(0)
		visit_num = inputs["medications"].size(1)
		if batch_size != 1:
			raise ValueError("VITA beam search expects one patient per batch.")
		beam_size = max(int(self.beam_size), 1)
		beams = [
			_COGNetBeam(beam_size, self.med_pad_token, self.sos_token, self.end_token, self.device)
			for _ in range(visit_num)
		]
		input_disease_embedding = input_disease_embedding.repeat_interleave(beam_size, dim=0)
		encoded_medication = encoded_medication.repeat_interleave(beam_size, dim=0)
		last_seq_medication = last_seq_medication.repeat_interleave(beam_size, dim=0)
		cross_visit_scores = cross_visit_scores.repeat_interleave(beam_size, dim=0)
		d_mask_matrix = inputs["d_mask_matrix"].repeat_interleave(beam_size, dim=0)
		p_mask_matrix = inputs["p_mask_matrix"].repeat_interleave(beam_size, dim=0)
		last_m_mask = last_m_mask.repeat_interleave(beam_size, dim=0)

		for i in range(self.max_len):
			len_dec_seq = i + 1
			dec_partial_inputs = torch.cat([b.get_current_state().unsqueeze(dim=1) for b in beams], dim=1)
			partial_m_mask_matrix = torch.zeros((beam_size, visit_num, len_dec_seq), device=self.device).float()
			partial_logits = self.core.decode(
				dec_partial_inputs,
				input_disease_embedding,
				encoded_medication,
				last_seq_medication,
				cross_visit_scores,
				d_mask_matrix,
				p_mask_matrix,
				partial_m_mask_matrix,
				last_m_mask,
				drug_memory,
			)
			word_lk = partial_logits[:, :, -1, :]
			active = []
			for beam_idx in range(visit_num):
				if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
					active.append(beam_idx)
			if not active:
				break

		all_hyp = []
		all_prob = []
		for beam in beams:
			_, tail_idxs = beam.sort_scores()
			all_hyp.append(beam.get_hypothesis(tail_idxs[0]))
			all_prob.append(beam.get_prob_list(tail_idxs[0]))
		return all_hyp, all_prob

	def _build_model_inputs(self, visits: Sequence[Sequence[Sequence[int]]]) -> Dict[str, Any]:
		(
			diseases,
			procedures,
			medications,
			seq_length,
			d_length_matrix,
			p_length_matrix,
			m_length_matrix,
			d_mask_matrix,
			p_mask_matrix,
			m_mask_matrix,
			dec_disease,
			stay_disease,
			dec_disease_mask,
			stay_disease_mask,
			dec_proc,
			stay_proc,
			dec_proc_mask,
			stay_proc_mask,
		) = self._pad_batch_vita(visits)
		return {
			"diseases": self._pad_num_replace(diseases, -1, self.diag_pad_token),
			"procedures": self._pad_num_replace(procedures, -1, self.proc_pad_token),
			"medications": medications,
			"seq_length": seq_length,
			"d_length_matrix": d_length_matrix,
			"p_length_matrix": p_length_matrix,
			"m_length_matrix": m_length_matrix,
			"d_mask_matrix": d_mask_matrix,
			"p_mask_matrix": p_mask_matrix,
			"m_mask_matrix": m_mask_matrix,
			"dec_disease": self._pad_num_replace(dec_disease, -1, self.diag_pad_token),
			"stay_disease": self._pad_num_replace(stay_disease, -1, self.diag_pad_token),
			"dec_disease_mask": dec_disease_mask,
			"stay_disease_mask": stay_disease_mask,
			"dec_proc": self._pad_num_replace(dec_proc, -1, self.proc_pad_token),
			"stay_proc": self._pad_num_replace(stay_proc, -1, self.proc_pad_token),
			"dec_proc_mask": dec_proc_mask,
			"stay_proc_mask": stay_proc_mask,
		}

	def _pad_num_replace(self, tensor: torch.Tensor, src_num: int, target_num: int) -> torch.Tensor:
		return torch.where(tensor == src_num, torch.tensor(target_num, device=tensor.device), tensor)

	def _pad_batch_vita(self, visits: Sequence[Sequence[Sequence[int]]]):
		seq_length = torch.tensor([len(data) for data in visits], device=self.device)
		batch_size = len(visits)
		max_seq = int(seq_length.max().item()) if batch_size else 0
		d_max_num = 0
		p_max_num = 0
		m_max_num = 0
		d_length_matrix = []
		p_length_matrix = []
		m_length_matrix = []
		d_dec_list = []
		d_stay_list = []
		p_dec_list = []
		p_stay_list = []
		for data in visits:
			d_buf, p_buf, m_buf = [], [], []
			d_dec_buf, d_stay_buf = [], []
			p_dec_buf, p_stay_buf = [], []
			for idx, seq in enumerate(data):
				d_buf.append(len(seq[0]))
				p_buf.append(len(seq[1]))
				m_buf.append(len(seq[2]))
				d_max_num = max(d_max_num, len(seq[0]))
				p_max_num = max(p_max_num, len(seq[1]))
				m_max_num = max(m_max_num, len(seq[2]))
				if idx == 0:
					d_dec_buf.append([])
					d_stay_buf.append([])
					p_dec_buf.append([])
					p_stay_buf.append([])
				else:
					cur_d, last_d = set(seq[0]), set(data[idx - 1][0])
					d_stay_buf.append(list(cur_d & last_d))
					d_dec_buf.append(list(last_d - cur_d))
					cur_p, last_p = set(seq[1]), set(data[idx - 1][1])
					p_stay_buf.append(list(cur_p & last_p))
					p_dec_buf.append(list(last_p - cur_p))
			d_length_matrix.append(d_buf)
			p_length_matrix.append(p_buf)
			m_length_matrix.append(m_buf)
			d_dec_list.append(d_dec_buf)
			d_stay_list.append(d_stay_buf)
			p_dec_list.append(p_dec_buf)
			p_stay_list.append(p_stay_buf)

		d_max_num = max(d_max_num, 1)
		p_max_num = max(p_max_num, 1)
		m_max_num = max(m_max_num, 1)
		disease_tensor = torch.full((batch_size, max_seq, d_max_num), -1, device=self.device, dtype=torch.long)
		procedure_tensor = torch.full((batch_size, max_seq, p_max_num), -1, device=self.device, dtype=torch.long)
		medication_tensor = torch.full((batch_size, max_seq, m_max_num), 0, device=self.device, dtype=torch.long)
		d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9, device=self.device)
		p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9, device=self.device)
		m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9, device=self.device)

		dec_disease = torch.full((batch_size, max_seq, d_max_num), -1, device=self.device, dtype=torch.long)
		stay_disease = torch.full((batch_size, max_seq, d_max_num), -1, device=self.device, dtype=torch.long)
		dec_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9, device=self.device)
		stay_disease_mask = torch.full((batch_size, max_seq, d_max_num), -1e9, device=self.device)
		dec_proc = torch.full((batch_size, max_seq, p_max_num), -1, device=self.device, dtype=torch.long)
		stay_proc = torch.full((batch_size, max_seq, p_max_num), -1, device=self.device, dtype=torch.long)
		dec_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9, device=self.device)
		stay_proc_mask = torch.full((batch_size, max_seq, p_max_num), -1e9, device=self.device)

		for b_id, data in enumerate(visits):
			for s_id, adm in enumerate(data):
				if adm[0]:
					disease_tensor[b_id, s_id, : len(adm[0])] = torch.tensor(adm[0], device=self.device)
					d_mask_matrix[b_id, s_id, : len(adm[0])] = 0.0
				if adm[1]:
					procedure_tensor[b_id, s_id, : len(adm[1])] = torch.tensor(adm[1], device=self.device)
					p_mask_matrix[b_id, s_id, : len(adm[1])] = 0.0
				if adm[2]:
					medication_tensor[b_id, s_id, : len(adm[2])] = torch.tensor(adm[2], device=self.device)
					m_mask_matrix[b_id, s_id, : len(adm[2])] = 0.0
			for s_id, dec_adm in enumerate(d_dec_list[b_id]):
				if dec_adm:
					dec_disease[b_id, s_id, : len(dec_adm)] = torch.tensor(dec_adm, device=self.device)
					dec_disease_mask[b_id, s_id, : len(dec_adm)] = 0.0
			for s_id, stay_adm in enumerate(d_stay_list[b_id]):
				if stay_adm:
					stay_disease[b_id, s_id, : len(stay_adm)] = torch.tensor(stay_adm, device=self.device)
					stay_disease_mask[b_id, s_id, : len(stay_adm)] = 0.0
			for s_id, dec_adm in enumerate(p_dec_list[b_id]):
				if dec_adm:
					dec_proc[b_id, s_id, : len(dec_adm)] = torch.tensor(dec_adm, device=self.device)
					dec_proc_mask[b_id, s_id, : len(dec_adm)] = 0.0
			for s_id, stay_adm in enumerate(p_stay_list[b_id]):
				if stay_adm:
					stay_proc[b_id, s_id, : len(stay_adm)] = torch.tensor(stay_adm, device=self.device)
					stay_proc_mask[b_id, s_id, : len(stay_adm)] = 0.0

		return (
			disease_tensor,
			procedure_tensor,
			medication_tensor,
			seq_length,
			d_length_matrix,
			p_length_matrix,
			m_length_matrix,
			d_mask_matrix,
			p_mask_matrix,
			m_mask_matrix,
			dec_disease,
			stay_disease,
			dec_disease_mask,
			stay_disease_mask,
			dec_proc,
			stay_proc,
			dec_proc_mask,
			stay_proc_mask,
		)

	def _output_flatten(
		self,
		labels: torch.Tensor,
		logits: torch.Tensor,
		seq_length: torch.Tensor,
		m_length_matrix: List[List[int]],
		med_num: int,
		end_token: int,
	):
		batch_size, max_seq_length = labels.size()[:2]
		if max_seq_length == 0:
			return torch.empty((0,), device=self.device), torch.empty((0, med_num), device=self.device)
		if self.training:
			whole_seqs_num = int(seq_length.sum().item())
			whole_med_sum = sum(sum(buf) for buf in m_length_matrix) + whole_seqs_num
			labels_flatten = torch.empty(whole_med_sum, device=self.device)
			logits_flatten = torch.empty((whole_med_sum, med_num), device=self.device)
			start_idx = 0
			for i in range(batch_size):
				for j in range(int(seq_length[i].item())):
					for k in range(m_length_matrix[i][j] + 1):
						labels_flatten[start_idx] = end_token if k == m_length_matrix[i][j] else labels[i, j, k]
						logits_flatten[start_idx, :] = logits[i, j, k, :]
						start_idx += 1
			return labels_flatten, logits_flatten
		labels_flatten = []
		logits_flatten = []
		for i in range(batch_size):
			for j in range(int(seq_length[i].item())):
				labels_flatten.append(labels[i, j, : m_length_matrix[i][j]].detach().cpu().numpy())
				logits_flatten.append(logits[i, j, : self.max_len, :].detach().cpu().numpy())
		return labels_flatten[-1], logits_flatten[-1]
