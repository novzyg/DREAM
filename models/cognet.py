"""
COGNet model implementation integrated with the benchmark base model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.core.io import Prediction
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from drugrec_benchmark.utils.metrics import sequence_output_process

class SelfAttend(nn.Module):
	def __init__(self, embedding_size: int) -> None:
		super().__init__()

		self.h1 = nn.Sequential(
			nn.Linear(embedding_size, 32),
			nn.Tanh(),
		)
		self.gate_layer = nn.Linear(32, 1)

	def forward(self, seqs: torch.Tensor, seq_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
		gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
		if seq_masks is not None:
			gates = gates + seq_masks
		p_attn = F.softmax(gates, dim=-1)
		p_attn = p_attn.unsqueeze(-1)
		h = seqs * p_attn
		output = torch.sum(h, dim=1)
		return output


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
		support = torch.mm(inputs, self.weight)
		output = torch.mm(adj, support)
		if self.bias is not None:
			return output + self.bias
		return output

	def __repr__(self) -> str:
		return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class MaskLinear(nn.Module):
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

	def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		weight = torch.mul(self.weight, mask)
		output = torch.mm(inputs, weight)
		if self.bias is not None:
			return output + self.bias
		return output

	def __repr__(self) -> str:
		return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class GCN(nn.Module):
	def __init__(
		self,
		voc_size: int,
		emb_dim: int,
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.voc_size = voc_size
		self.emb_dim = emb_dim
		self.device = device

		ehr_adj = self._normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
		ddi_adj = self._normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

		self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
		self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
		self.x = torch.eye(voc_size).to(device)

		self.gcn1 = GraphConvolution(voc_size, emb_dim)
		self.dropout = nn.Dropout(p=0.3)
		self.gcn2 = GraphConvolution(emb_dim, emb_dim)
		self.gcn3 = GraphConvolution(emb_dim, emb_dim)

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

	def _normalize(self, mx: np.ndarray) -> np.ndarray:
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.0
		r_mat_inv = np.diagflat(r_inv)
		return r_mat_inv.dot(mx)


class MedTransformerDecoder(nn.Module):
	def __init__(
		self,
		d_model: int,
		nhead: int,
		dim_feedforward: int = 2048,
		dropout: float = 0.1,
		layer_norm_eps: float = 1e-5,
	) -> None:
		super().__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
		self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
		input_proc_embedding: torch.Tensor,
		input_medication_self_mask: torch.Tensor,
		d_mask: torch.Tensor,
		p_mask: torch.Tensor,
	) -> torch.Tensor:
		input_len = input_medication_embedding.size(0)
		tgt_len = input_medication_embedding.size(1)

		subsequent_mask = self.generate_square_subsequent_mask(
			tgt_len,
			input_len * self.nhead,
			input_disease_embdding.device,
		)
		self_attn_mask = subsequent_mask + input_medication_self_mask

		x = input_medication_embedding + input_medication_memory

		x = self.norm1(x + self._sa_block(x, self_attn_mask))
		x = self.norm2(
			x
			+ self._m2d_mha_block(x, input_disease_embdding, d_mask)
			+ self._m2p_mha_block(x, input_proc_embedding, p_mask)
		)
		x = self.norm3(x + self._ff_block(x))

		return x

	def _sa_block(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
		x = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
		return self.dropout1(x)

	def _m2d_mha_block(self, x: torch.Tensor, mem: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
		x = self.m2d_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
		return self.dropout2(x)

	def _m2p_mha_block(self, x: torch.Tensor, mem: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
		x = self.m2p_multihead_attn(x, mem, mem, attn_mask=attn_mask, need_weights=False)[0]
		return self.dropout2(x)

	def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout3(x)

	def generate_square_subsequent_mask(
		self,
		sz: int,
		batch_size: int,
		device: torch.device,
	) -> torch.Tensor:
		mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, 0.0)
		mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
		return mask


class COGNetCore(nn.Module):
	def __init__(
		self,
		voc_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		device: torch.device = torch.device("cpu"),
	) -> None:
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


		self.diag_embedding = nn.Sequential(
			nn.Embedding(voc_size[0] + 3, emb_dim, self.DIAG_PAD_TOKEN),
			nn.Dropout(0.3),
		)

		self.proc_embedding = nn.Sequential(
			nn.Embedding(voc_size[1] + 3, emb_dim, self.PROC_PAD_TOKEN),
			nn.Dropout(0.3),
		)

		self.med_embedding = nn.Sequential(
			nn.Embedding(voc_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
			nn.Dropout(0.3),
		)

		self.medication_encoder = nn.TransformerEncoderLayer(
			emb_dim,
			self.nhead,
			batch_first=True,
			dropout=0.2,
		)
		self.diagnoses_encoder = nn.TransformerEncoderLayer(
			emb_dim,
			self.nhead,
			batch_first=True,
			dropout=0.2,
		)
		self.procedure_encoder = nn.TransformerEncoderLayer(
			emb_dim,
			self.nhead,
			batch_first=True,
			dropout=0.2,
		)

		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

		self.gcn = GCN(
			voc_size=voc_size[2],
			emb_dim=emb_dim,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			device=device,
		)
		self.inter = nn.Parameter(torch.ones(1))

		self.diag_self_attend = SelfAttend(emb_dim)
		self.proc_self_attend = SelfAttend(emb_dim)

		self.decoder = MedTransformerDecoder(
			emb_dim,
			self.nhead,
			dim_feedforward=emb_dim * 2,
			dropout=0.2,
			layer_norm_eps=1e-5,
		)

		self.dec_gru = nn.GRU(emb_dim * 3, emb_dim, batch_first=True)

		self.diag_attn = nn.Linear(emb_dim * 2, 1)
		self.proc_attn = nn.Linear(emb_dim * 2, 1)
		self.W_diag_attn = nn.Linear(emb_dim, emb_dim)
		self.W_proc_attn = nn.Linear(emb_dim, emb_dim)
		self.W_diff_attn = nn.Linear(emb_dim, emb_dim)
		self.W_diff_proc_attn = nn.Linear(emb_dim, emb_dim)

		self.Ws = nn.Linear(emb_dim * 2, emb_dim)
		self.Wo = nn.Linear(emb_dim, voc_size[2] + 2)
		self.Wc = nn.Linear(emb_dim, emb_dim)

		self.W_dec = nn.Linear(emb_dim, emb_dim)
		self.W_stay = nn.Linear(emb_dim, emb_dim)
		self.W_proc_dec = nn.Linear(emb_dim, emb_dim)
		self.W_proc_stay = nn.Linear(emb_dim, emb_dim)

		self.W_z = nn.Linear(emb_dim, 1)

		self.weight = nn.Parameter(torch.tensor([0.3]), requires_grad=True)


	def encode(
		self,
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
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		batch_size, max_visit_num, max_med_num = medications.size()
		max_diag_num = diseases.size()[2]
		max_proc_num = procedures.size()[2]

		if max_diag_num == 0:
			dummy_diags = torch.full(
				(batch_size, max_visit_num, 1),
				self.DIAG_PAD_TOKEN,
				device=self.device,
				dtype=diseases.dtype,
			)
			input_disease_embdding = self.diag_embedding(dummy_diags).view(
				batch_size * max_visit_num, 1, self.emb_dim
			)
			d_enc_mask_matrix = torch.zeros(
				(batch_size * max_visit_num * self.nhead, 1, 1),
				device=self.device,
			)
			input_disease_embdding = self.diagnoses_encoder(
				input_disease_embdding, src_mask=d_enc_mask_matrix
			).view(batch_size, max_visit_num, 1, self.emb_dim)
			diag_mask_2d = torch.zeros((batch_size * max_visit_num, 1), device=self.device)
			max_diag_num = 1
		else:
			diag_mask_2d = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num)
			no_diag_rows = diag_mask_2d.eq(0.0).sum(dim=1) == 0
			if torch.any(no_diag_rows):
				diseases_flat = diseases.view(batch_size * max_visit_num, max_diag_num).clone()
				diseases_flat[no_diag_rows, 0] = self.DIAG_PAD_TOKEN
				diag_mask_2d = diag_mask_2d.clone()
				diag_mask_2d[no_diag_rows, 0] = 0.0
				diseases = diseases_flat.view(batch_size, max_visit_num, max_diag_num)
				d_mask_matrix = diag_mask_2d.view(batch_size, max_visit_num, max_diag_num)

			input_disease_embdding = self.diag_embedding(diseases).view(
				batch_size * max_visit_num, max_diag_num, self.emb_dim
			)
			d_enc_mask_matrix = (
				diag_mask_2d
				.unsqueeze(dim=1)
				.unsqueeze(dim=1)
				.repeat(1, self.nhead, max_diag_num, 1)
			)
			d_enc_mask_matrix = d_enc_mask_matrix.view(
				batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num
			)
			input_disease_embdding = self.diagnoses_encoder(
				input_disease_embdding, src_mask=d_enc_mask_matrix
			).view(batch_size, max_visit_num, max_diag_num, self.emb_dim)
		if max_proc_num == 0:
			dummy_procs = torch.full(
				(batch_size, max_visit_num, 1),
				self.PROC_PAD_TOKEN,
				device=self.device,
				dtype=procedures.dtype,
			)
			input_proc_embedding = self.proc_embedding(dummy_procs).view(
				batch_size * max_visit_num, 1, self.emb_dim
			)
			p_enc_mask_matrix = torch.zeros(
				(batch_size * max_visit_num * self.nhead, 1, 1),
				device=self.device,
			)
			input_proc_embedding = self.procedure_encoder(
				input_proc_embedding, src_mask=p_enc_mask_matrix
			).view(batch_size, max_visit_num, 1, self.emb_dim)
		else:
			proc_mask_2d = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num)
			no_proc_rows = proc_mask_2d.eq(0.0).sum(dim=1) == 0
			if torch.any(no_proc_rows):
				procedures_flat = procedures.view(batch_size * max_visit_num, max_proc_num).clone()
				procedures_flat[no_proc_rows, 0] = self.PROC_PAD_TOKEN
				proc_mask_2d = proc_mask_2d.clone()
				proc_mask_2d[no_proc_rows, 0] = 0.0
				procedures = procedures_flat.view(batch_size, max_visit_num, max_proc_num)
				p_mask_matrix = proc_mask_2d.view(batch_size, max_visit_num, max_proc_num)

			input_proc_embedding = self.proc_embedding(procedures).view(
				batch_size * max_visit_num, max_proc_num, self.emb_dim
			)
			p_enc_mask_matrix = (
				proc_mask_2d
				.unsqueeze(dim=1)
				.unsqueeze(dim=1)
				.repeat(1, self.nhead, max_proc_num, 1)
			)
			p_enc_mask_matrix = p_enc_mask_matrix.view(
				batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num
			)
			input_proc_embedding = self.procedure_encoder(
				input_proc_embedding, src_mask=p_enc_mask_matrix
			).view(batch_size, max_visit_num, max_proc_num, self.emb_dim)
		visit_diag_embedding = self.diag_self_attend(
			input_disease_embdding.view(batch_size * max_visit_num, max_diag_num, -1),
			diag_mask_2d,
		)
		if max_proc_num == 0:
			visit_proc_embedding = torch.zeros(
				(batch_size * max_visit_num, self.emb_dim), device=self.device
			)
		else:
			visit_proc_embedding = self.proc_self_attend(
				input_proc_embedding.view(batch_size * max_visit_num, max_proc_num, -1),
				p_mask_matrix.view(batch_size * max_visit_num, -1),
			)
		visit_diag_embedding = visit_diag_embedding.view(batch_size, max_visit_num, -1)
		visit_proc_embedding = visit_proc_embedding.view(batch_size, max_visit_num, -1)

		cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding, visit_proc_embedding)
		last_seq_medication = torch.full(
			(batch_size, 1, max_med_num),
			0,
			device=self.device,
			dtype=medications.dtype,
		)
		last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
		last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(self.device)
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

		last_seq_medication_emb = self.med_embedding(last_seq_medication)
		last_m_enc_mask = (
			last_m_mask.view(batch_size * max_visit_num, max_med_num)
			.unsqueeze(dim=1)
			.unsqueeze(dim=1)
			.repeat(1, self.nhead, max_med_num, 1)
		)
		last_m_enc_mask = last_m_enc_mask.view(
			batch_size * max_visit_num * self.nhead, max_med_num, max_med_num
		)
		encoded_medication = self.medication_encoder(
			last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim),
			src_mask=last_m_enc_mask,
		)
		encoded_medication = encoded_medication.view(
			batch_size, max_visit_num, max_med_num, self.emb_dim
		)
		ehr_embedding, ddi_embedding = self.gcn()
		drug_memory = ehr_embedding - ddi_embedding * self.inter
		drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
		drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)

		return (
			input_disease_embdding,
			input_proc_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
		)

	def decode(
		self,
		input_medications: torch.Tensor,
		input_disease_embedding: torch.Tensor,
		input_proc_embedding: torch.Tensor,
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
		max_proc_num = input_proc_embedding.size(2)

		if p_mask_matrix.size(-1) == 0:
			p_mask_matrix = torch.zeros((batch_size, max_visit_num, 1), device=self.device)
			if max_proc_num == 0:
				input_proc_embedding = torch.zeros(
					(batch_size, max_visit_num, 1, self.emb_dim), device=self.device
				)
				max_proc_num = 1

		input_medication_embs = self.med_embedding(input_medications).view(
			batch_size * max_visit_num, max_med_num, -1
		)
		input_medication_memory = drug_memory[input_medications].view(
			batch_size * max_visit_num, max_med_num, -1
		)

		m_self_mask = m_mask_matrix

		last_m_enc_mask = (
			m_self_mask.view(batch_size * max_visit_num, max_med_num)
			.unsqueeze(dim=1)
			.unsqueeze(dim=1)
			.repeat(1, self.nhead, max_med_num, 1)
		)
		medication_self_mask = last_m_enc_mask.view(
			batch_size * max_visit_num * self.nhead, max_med_num, max_med_num
		)
		m2d_mask_matrix = (
			d_mask_matrix.view(batch_size * max_visit_num, max_diag_num)
			.unsqueeze(dim=1)
			.unsqueeze(dim=1)
			.repeat(1, self.nhead, max_med_num, 1)
		)
		m2d_mask_matrix = m2d_mask_matrix.view(
			batch_size * max_visit_num * self.nhead, max_med_num, max_diag_num
		)
		m2p_mask_matrix = (
			p_mask_matrix.view(batch_size * max_visit_num, max_proc_num)
			.unsqueeze(dim=1)
			.unsqueeze(dim=1)
			.repeat(1, self.nhead, max_med_num, 1)
		)
		m2p_mask_matrix = m2p_mask_matrix.view(
			batch_size * max_visit_num * self.nhead, max_med_num, max_proc_num
		)

		dec_hidden = self.decoder(
			input_medication_embedding=input_medication_embs,
			input_medication_memory=input_medication_memory,
			input_disease_embdding=input_disease_embedding.view(
				batch_size * max_visit_num, max_diag_num, -1
			),
			input_proc_embedding=input_proc_embedding.view(
				batch_size * max_visit_num, max_proc_num, -1
			),
			input_medication_self_mask=medication_self_mask,
			d_mask=m2d_mask_matrix,
			p_mask=m2p_mask_matrix,
		)

		score_g = self.Wo(dec_hidden)
		score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
		prob_g = F.softmax(score_g, dim=-1)
		score_c = self.copy_med(
			dec_hidden.view(batch_size, max_visit_num, max_med_num, -1),
			last_medication_embedding,
			last_m_mask,
			cross_visit_scores,
		)

		prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(
			batch_size, max_visit_num * max_med_num, -1
		)
		copy_source = last_medications.view(batch_size, 1, -1).repeat(
			1, max_visit_num * max_med_num, 1
		)
		copy_source = copy_source.clamp(min=0, max=self.voc_size[2] + 1)
		prob_c_to_g.scatter_add_(2, copy_source, score_c)
		prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)

		generate_prob = torch.sigmoid(self.W_z(dec_hidden)).view(
			batch_size, max_visit_num, max_med_num, 1
		)
		prob = prob_g * generate_prob + prob_c_to_g * (1.0 - generate_prob)
		prob[:, 0, :, :] = prob_g[:, 0, :, :]

		return torch.log(prob.clamp(min=1e-12))

	def forward(
		self,
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
	) -> torch.Tensor:
		batch_size, max_seq_length, _ = medications.size()

		(
			input_disease_embdding,
			input_proc_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
		) = self.encode(
			diseases,
			procedures,
			medications,
			d_mask_matrix,
			p_mask_matrix,
			m_mask_matrix,
			seq_length,
			dec_disease,
			stay_disease,
			dec_disease_mask,
			stay_disease_mask,
			dec_proc,
			stay_proc,
			dec_proc_mask,
			stay_proc_mask,
			max_len=max_len,
		)

		input_medication = torch.full(
			(batch_size, max_seq_length, 1), self.SOS_TOKEN, device=self.device
		)
		input_medication = torch.cat([input_medication, medications], dim=2)

		m_sos_mask = torch.zeros((batch_size, max_seq_length, 1), device=self.device).float()
		m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)

		output_logits = self.decode(
			input_medication,
			input_disease_embdding,
			input_proc_embedding,
			encoded_medication,
			last_seq_medication,
			cross_visit_scores,
			d_mask_matrix,
			p_mask_matrix,
			m_mask_matrix,
			last_m_mask,
			drug_memory,
		)
		return output_logits

	def calc_cross_visit_scores(
		self, visit_diag_embedding: torch.Tensor, visit_proc_embedding: torch.Tensor
	) -> torch.Tensor:
		max_visit_num = visit_diag_embedding.size(1)
		batch_size = visit_diag_embedding.size(0)

		mask = (
			torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1
		).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, 0.0)
		mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)

		padding = torch.zeros((batch_size, 1, self.emb_dim), device=self.device).float()
		diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]], dim=1)
		proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

		diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) / math.sqrt(
			visit_diag_embedding.size(-1)
		)
		proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) / math.sqrt(
			visit_proc_embedding.size(-1)
		)
		scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)
		return scores

	def copy_med(
		self,
		decode_input_hiddens: torch.Tensor,
		last_medications: torch.Tensor,
		last_m_mask: torch.Tensor,
		cross_visit_scores: torch.Tensor,
	) -> torch.Tensor:
		max_visit_num = decode_input_hiddens.size(1)
		input_med_num = decode_input_hiddens.size(2)
		max_med_num = last_medications.size(2)
		copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num * input_med_num, self.emb_dim)
		attn_scores = torch.matmul(
			copy_query,
			last_medications.view(-1, max_visit_num * max_med_num, self.emb_dim).transpose(-2, -1),
		) / math.sqrt(self.emb_dim)
		med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(
			1, max_visit_num * input_med_num, 1
		)
		attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

		visit_scores = cross_visit_scores.repeat(1, 1, input_med_num).view(
			-1, max_visit_num * input_med_num, max_visit_num
		)
		visit_scores = (
			visit_scores.unsqueeze(-1)
			.repeat(1, 1, 1, max_med_num)
			.view(-1, max_visit_num * input_med_num, max_visit_num * max_med_num)
		)

		scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)
		row_scores = scores.sum(dim=-1, keepdim=True)
		scores = scores / row_scores

		return scores


class PolicyNetwork(nn.Module):
	def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layers(x)


policy_network = PolicyNetwork



class _COGNetBeam:
	def __init__(
		self,
		size: int,
		pad_token: int,
		bos_token: int,
		eos_token: int,
		device: torch.device,
	) -> None:
		self.PAD = pad_token
		self.BOS = bos_token
		self.EOS = eos_token
		self.device = device
		self.size = size
		self.done = False
		self.beam_status = [False] * size
		self.scores = torch.zeros(size, device=device)
		self.all_scores: List[torch.Tensor] = []
		self.prev_ks: List[torch.Tensor] = []
		self.next_ys = [torch.full((size,), self.BOS, device=device, dtype=torch.long)]
		self.prob_list: List[List[List[float]]] = []

	def get_current_state(self, sort: bool = True) -> torch.Tensor:
		if sort:
			return self.get_tentative_hypothesis()
		return self.get_tentative_hypothesis_wo_sort()

	def advance(self, word_lk: torch.Tensor) -> bool:
		num_words = word_lk.size(1)
		if self.done:
			self.prev_ks.append(torch.arange(self.size, device=self.device))
			self.next_ys.append(torch.full((self.size,), self.EOS, device=self.device, dtype=torch.long))
			self.prob_list.append([[0.0] * num_words for _ in range(self.size)])
			return True

		active_beam_idx = torch.tensor(
			[idx for idx in range(self.size) if not self.beam_status[idx]],
			device=self.device,
			dtype=torch.long,
		)
		end_beam_idx = torch.tensor(
			[idx for idx in range(self.size) if self.beam_status[idx]],
			device=self.device,
			dtype=torch.long,
		)
		active_word_lk = word_lk[active_beam_idx]
		cur_output = self.get_current_state(sort=False)

		active_scores = self.scores[active_beam_idx]
		end_scores = self.scores[end_beam_idx]

		if len(self.prev_ks) > 0:
			beam_lk = active_word_lk + active_scores.unsqueeze(dim=1).expand_as(active_word_lk)
		else:
			beam_lk = active_word_lk[0]

		flat_beam_lk = beam_lk.reshape(-1)
		active_max_idx = len(flat_beam_lk)
		flat_beam_lk = torch.cat([flat_beam_lk, end_scores], dim=-1)

		self.all_scores.append(self.scores)
		sorted_scores, sorted_score_ids = torch.sort(flat_beam_lk, descending=True)

		select_num = 0
		cur_idx = 0
		selected_scores = []
		selected_words = []
		selected_beams = []
		new_active_status = []
		prob_buf = []

		while select_num < self.size and cur_idx < len(sorted_scores):
			cur_score = sorted_scores[cur_idx]
			cur_id = sorted_score_ids[cur_idx]
			if cur_id >= active_max_idx:
				which_beam = end_beam_idx[cur_id - active_max_idx]
				which_word = torch.tensor(self.EOS, device=self.device)
				new_active_status.append(True)
				selected_scores.append(cur_score)
				selected_beams.append(which_beam)
				selected_words.append(which_word)
				prob_buf.append([0.0] * num_words)
				select_num += 1
			else:
				which_beam_idx = cur_id // num_words
				which_beam = active_beam_idx[which_beam_idx]
				which_word = cur_id - which_beam_idx * num_words
				if which_word not in cur_output[which_beam]:
					new_active_status.append(bool(which_word in [self.EOS, self.BOS]))
					selected_scores.append(cur_score)
					selected_beams.append(which_beam)
					selected_words.append(which_word)
					prob_buf.append(active_word_lk[which_beam_idx].detach().cpu().numpy().tolist())
					select_num += 1
			cur_idx += 1

		self.prob_list.append(prob_buf)
		self.beam_status = new_active_status
		self.scores = torch.stack(selected_scores)
		self.prev_ks.append(torch.stack(selected_beams))
		self.next_ys.append(torch.stack(selected_words))
		self.done = all(self.beam_status)
		return self.done

	def sort_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
		return torch.sort(self.scores, 0, True)

	def get_tentative_hypothesis(self) -> torch.Tensor:
		if len(self.next_ys) == 1:
			return self.next_ys[0].unsqueeze(1)
		_, keys = self.sort_scores()
		hyps = [self.get_hypothesis(k) for k in keys]
		hyps = [[self.BOS] + h for h in hyps]
		return torch.as_tensor(hyps, device=self.device, dtype=torch.long)

	def get_tentative_hypothesis_wo_sort(self) -> torch.Tensor:
		if len(self.next_ys) == 1:
			return self.next_ys[0].unsqueeze(1)
		hyps = [self.get_hypothesis(k) for k in range(self.size)]
		hyps = [[self.BOS] + h for h in hyps]
		return torch.as_tensor(hyps, device=self.device, dtype=torch.long)

	def get_hypothesis(self, k: torch.Tensor) -> List[int]:
		hyp = []
		for j in range(len(self.prev_ks) - 1, -1, -1):
			hyp.append(self.next_ys[j + 1][k].item())
			k = self.prev_ks[j][k]
		return hyp[::-1]

	def get_prob_list(self, k: torch.Tensor) -> List[List[float]]:
		ret_prob_list = []
		for j in range(len(self.prev_ks) - 1, -1, -1):
			ret_prob_list.append(self.prob_list[j][k])
			k = self.prev_ks[j][k]
		return ret_prob_list[::-1]


class COGNet(BaseDrugRecommendationModel):
	"""
	COGNet wrapper integrated with the benchmark interfaces.

	Expected batch format:
	{
		"visits": List[List[Tuple[List[int], List[int], List[int]]]]
	}
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		max_len: int = 45,
		beam_size: int = 4,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.supports_batched_training = True
		self.model_type = "sequence"
		self.vocab_size = vocab_size
		self.max_len = max_len
		self.beam_size = beam_size
		self.use_beam_search = False
		self.ddi_adj = ddi_adj

		self.sos_token = vocab_size[2]
		self.end_token = vocab_size[2] + 1
		self.med_pad_token = vocab_size[2] + 2
		self.diag_pad_token = vocab_size[0] + 2
		self.proc_pad_token = vocab_size[1] + 2

		self.train_mode = self.training

		self.core = COGNetCore(
			voc_size=vocab_size,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		patients = self.get_patients(batch)
		if not patients:
			empty_logits = torch.empty((0, self.vocab_size[2]), device=self.device)
			return {"logits": empty_logits}
		inputs = self._build_model_inputs(patients)

		if self.training:
			output_logits = self.core(
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
				return {
					"beam_hypotheses": beam_hypotheses,
					"beam_probs": beam_probs,
				}
			output_logits = self._autoregressive_decode(inputs)
		labels_flatten, logits_flatten = self._output_flatten(
			inputs["medications"],
			output_logits,
			inputs["seq_length"],
			inputs["m_length_matrix"],
			self.vocab_size[2] + 2,
			self.end_token,
		)

		return {
			"labels_flatten": labels_flatten,
			"logits": logits_flatten,
		}

	def set_beam_search(self, enabled: bool = True) -> None:
		self.use_beam_search = enabled

	def decode(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> Prediction:
		if "beam_hypotheses" in outputs:
			med_size = self.vocab_size[2]
			target = self.get_target_indices(batch)
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
					first_idx = out_list.index(med)
					scores[med] = out_prob_list[first_idx][med]
			else:
				scores = np.zeros((med_size,), dtype=float)
			return Prediction(
				med_indices=list(out_list),
				med_scores=np.asarray(scores, dtype=float),
				target=target,
				task="sequence",
				ranked_med_indices=list(out_list),
			)

		logits = outputs.get("logits")
		logits_array = logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits)
		med_size = self.vocab_size[2]
		target = self.get_target_indices(batch)
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

	
	def compute_loss(self, outputs: Dict[str, Any], batch: Dict[str, Any]) -> torch.Tensor:
		labels_flatten = outputs["labels_flatten"]
		logits_flatten = outputs["logits"]
		return F.nll_loss(logits_flatten, labels_flatten.long())

	def _autoregressive_decode(self, inputs: Dict[str, Any]) -> torch.Tensor:
		(
			input_disease_embdding,
			input_proc_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
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
			partial_input_med_num = partial_input_medication.size(2)
			partial_m_mask_matrix = torch.zeros(
				(batch_size, max_visit_num, partial_input_med_num),
				device=self.device,
			).float()

			partial_logits = self.core.decode(
				partial_input_medication,
				input_disease_embdding,
				input_proc_embedding,
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
			partial_input_medication = torch.cat(
				[partial_input_medication, next_medication],
				dim=-1,
			)

		if partial_logits is None:
			return torch.empty(
				(batch_size, max_visit_num, 0, self.vocab_size[2] + 2),
				device=self.device,
			)
		return partial_logits

	def _beam_search_decode(self, inputs: Dict[str, Any]) -> Tuple[List[List[int]], List[List[List[float]]]]:
		(
			input_disease_embdding,
			input_proc_embedding,
			encoded_medication,
			cross_visit_scores,
			last_seq_medication,
			last_m_mask,
			drug_memory,
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
			raise ValueError("COGNet beam search expects one patient per batch.")

		beam_size = max(int(self.beam_size), 1)
		beams = [
			_COGNetBeam(
				beam_size,
				self.med_pad_token,
				self.sos_token,
				self.end_token,
				self.device,
			)
			for _ in range(visit_num)
		]

		input_disease_embdding = input_disease_embdding.repeat_interleave(beam_size, dim=0)
		input_proc_embedding = input_proc_embedding.repeat_interleave(beam_size, dim=0)
		encoded_medication = encoded_medication.repeat_interleave(beam_size, dim=0)
		last_seq_medication = last_seq_medication.repeat_interleave(beam_size, dim=0)
		cross_visit_scores = cross_visit_scores.repeat_interleave(beam_size, dim=0)
		d_mask_matrix = inputs["d_mask_matrix"].repeat_interleave(beam_size, dim=0)
		p_mask_matrix = inputs["p_mask_matrix"].repeat_interleave(beam_size, dim=0)
		last_m_mask = last_m_mask.repeat_interleave(beam_size, dim=0)

		for i in range(self.max_len):
			len_dec_seq = i + 1
			dec_partial_inputs = torch.cat(
				[b.get_current_state().unsqueeze(dim=1) for b in beams],
				dim=1,
			)
			partial_m_mask_matrix = torch.zeros(
				(beam_size, visit_num, len_dec_seq),
				device=self.device,
			).float()
			partial_logits = self.core.decode(
				dec_partial_inputs,
				input_disease_embdding,
				input_proc_embedding,
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

			active_beam_idx_list = []
			for beam_idx in range(visit_num):
				if not beams[beam_idx].advance(word_lk[:, beam_idx, :]):
					active_beam_idx_list.append(beam_idx)
			if not active_beam_idx_list:
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
		) = self._pad_batch_v2(visits)

		diseases = self._pad_num_replace(diseases, -1, self.diag_pad_token)
		procedures = self._pad_num_replace(procedures, -1, self.proc_pad_token)
		dec_disease = self._pad_num_replace(dec_disease, -1, self.diag_pad_token)
		stay_disease = self._pad_num_replace(stay_disease, -1, self.diag_pad_token)
		dec_proc = self._pad_num_replace(dec_proc, -1, self.proc_pad_token)
		stay_proc = self._pad_num_replace(stay_proc, -1, self.proc_pad_token)

		return {
			"diseases": diseases,
			"procedures": procedures,
			"medications": medications,
			"seq_length": seq_length,
			"d_length_matrix": d_length_matrix,
			"p_length_matrix": p_length_matrix,
			"m_length_matrix": m_length_matrix,
			"d_mask_matrix": d_mask_matrix,
			"p_mask_matrix": p_mask_matrix,
			"m_mask_matrix": m_mask_matrix,
			"dec_disease": dec_disease,
			"stay_disease": stay_disease,
			"dec_disease_mask": dec_disease_mask,
			"stay_disease_mask": stay_disease_mask,
			"dec_proc": dec_proc,
			"stay_proc": stay_proc,
			"dec_proc_mask": dec_proc_mask,
			"stay_proc_mask": stay_proc_mask,
		}


	def _pad_num_replace(self, tensor: torch.Tensor, src_num: int, target_num: int) -> torch.Tensor:
		return torch.where(tensor == src_num, torch.tensor(target_num, device=tensor.device), tensor)

	def _pad_batch_v2(
		self, visits: Sequence[Sequence[Sequence[int]]]
	) -> Tuple[
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		List[List[int]],
		List[List[int]],
		List[List[int]],
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
		torch.Tensor,
	]:
		seq_length = torch.tensor([len(data) for data in visits], device=self.device)
		batch_size = len(visits)
		max_seq = int(seq_length.max().item()) if batch_size > 0 else 0

		d_length_matrix: List[List[int]] = []
		p_length_matrix: List[List[int]] = []
		m_length_matrix: List[List[int]] = []
		d_max_num = 0
		p_max_num = 0
		m_max_num = 0
		d_dec_list: List[List[List[int]]] = []
		d_stay_list: List[List[List[int]]] = []
		p_dec_list: List[List[List[int]]] = []
		p_stay_list: List[List[List[int]]] = []

		for data in visits:
			d_buf: List[int] = []
			p_buf: List[int] = []
			m_buf: List[int] = []
			d_dec_list_buf: List[List[int]] = []
			d_stay_list_buf: List[List[int]] = []
			p_dec_list_buf: List[List[int]] = []
			p_stay_list_buf: List[List[int]] = []
			for idx, seq in enumerate(data):
				d_buf.append(len(seq[0]))
				p_buf.append(len(seq[1]))
				m_buf.append(len(seq[2]))
				d_max_num = max(d_max_num, len(seq[0]))
				p_max_num = max(p_max_num, len(seq[1]))
				m_max_num = max(m_max_num, len(seq[2]))
				if idx == 0:
					d_dec_list_buf.append([])
					d_stay_list_buf.append([])
					p_dec_list_buf.append([])
					p_stay_list_buf.append([])
				else:
					cur_d = set(seq[0])
					last_d = set(data[idx - 1][0])
					d_stay_list_buf.append(list(cur_d & last_d))
					d_dec_list_buf.append(list(last_d - cur_d))

					cur_p = set(seq[1])
					last_p = set(data[idx - 1][1])
					p_stay_list_buf.append(list(cur_p & last_p))
					p_dec_list_buf.append(list(last_p - cur_p))
			d_length_matrix.append(d_buf)
			p_length_matrix.append(p_buf)
			m_length_matrix.append(m_buf)
			d_dec_list.append(d_dec_list_buf)
			d_stay_list.append(d_stay_list_buf)
			p_dec_list.append(p_dec_list_buf)
			p_stay_list.append(p_stay_list_buf)

		d_mask_matrix = torch.full((batch_size, max_seq, d_max_num), -1e9, device=self.device)
		p_mask_matrix = torch.full((batch_size, max_seq, p_max_num), -1e9, device=self.device)
		m_mask_matrix = torch.full((batch_size, max_seq, m_max_num), -1e9, device=self.device)

		for i in range(batch_size):
			for j in range(len(d_length_matrix[i])):
				d_mask_matrix[i, j, : d_length_matrix[i][j]] = 0.0
			for j in range(len(p_length_matrix[i])):
				p_len = p_length_matrix[i][j]
				if p_len > 0:
					p_mask_matrix[i, j, :p_len] = 0.0
				elif p_max_num > 0:
					p_mask_matrix[i, j, 0] = 0.0
			for j in range(len(m_length_matrix[i])):
				m_mask_matrix[i, j, : m_length_matrix[i][j]] = 0.0

		dec_disease_tensor = torch.full(
			(batch_size, max_seq, d_max_num), -1, device=self.device
		)
		stay_disease_tensor = torch.full(
			(batch_size, max_seq, d_max_num), -1, device=self.device
		)
		dec_disease_mask = torch.full(
			(batch_size, max_seq, d_max_num), -1e9, device=self.device
		)
		stay_disease_mask = torch.full(
			(batch_size, max_seq, d_max_num), -1e9, device=self.device
		)
		for b_id, (dec_seqs, stay_seqs) in enumerate(zip(d_dec_list, d_stay_list)):
			for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
				if dec_adm:
					dec_disease_tensor[b_id, s_id, : len(dec_adm)] = torch.tensor(
						dec_adm, device=self.device
					)
					dec_disease_mask[b_id, s_id, : len(dec_adm)] = 0.0
				if stay_adm:
					stay_disease_tensor[b_id, s_id, : len(stay_adm)] = torch.tensor(
						stay_adm, device=self.device
					)
					stay_disease_mask[b_id, s_id, : len(stay_adm)] = 0.0

		dec_proc_tensor = torch.full(
			(batch_size, max_seq, p_max_num), -1, device=self.device
		)
		stay_proc_tensor = torch.full(
			(batch_size, max_seq, p_max_num), -1, device=self.device
		)
		dec_proc_mask = torch.full(
			(batch_size, max_seq, p_max_num), -1e9, device=self.device
		)
		stay_proc_mask = torch.full(
			(batch_size, max_seq, p_max_num), -1e9, device=self.device
		)
		for b_id, (dec_seqs, stay_seqs) in enumerate(zip(p_dec_list, p_stay_list)):
			for s_id, (dec_adm, stay_adm) in enumerate(zip(dec_seqs, stay_seqs)):
				if dec_adm:
					dec_proc_tensor[b_id, s_id, : len(dec_adm)] = torch.tensor(
						dec_adm, device=self.device
					)
					dec_proc_mask[b_id, s_id, : len(dec_adm)] = 0.0
				if stay_adm:
					stay_proc_tensor[b_id, s_id, : len(stay_adm)] = torch.tensor(
						stay_adm, device=self.device
					)
					stay_proc_mask[b_id, s_id, : len(stay_adm)] = 0.0

		disease_tensor = torch.full(
			(batch_size, max_seq, d_max_num), -1, device=self.device
		)
		procedure_tensor = torch.full(
			(batch_size, max_seq, p_max_num), -1, device=self.device
		)
		medication_tensor = torch.full(
			(batch_size, max_seq, m_max_num), 0, device=self.device
		)

		for b_id, data in enumerate(visits):
			for s_id, adm in enumerate(data):
				disease_tensor[b_id, s_id, : len(adm[0])] = torch.tensor(
					adm[0], device=self.device
				)
				procedure_tensor[b_id, s_id, : len(adm[1])] = torch.tensor(
					adm[1], device=self.device
				)
				medication_tensor[b_id, s_id, : len(adm[2])] = torch.tensor(
					adm[2], device=self.device
				)

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
			dec_disease_tensor,
			stay_disease_tensor,
			dec_disease_mask,
			stay_disease_mask,
			dec_proc_tensor,
			stay_proc_tensor,
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
	) -> Tuple[torch.Tensor, torch.Tensor]:
		batch_size, max_seq_length = labels.size()[:2]
		assert max_seq_length == int(seq_length.max().item())
		whole_seqs_num = int(seq_length.sum().item())

		if self.training:

			whole_med_sum = sum(sum(buf) for buf in m_length_matrix) + whole_seqs_num

			labels_flatten = torch.empty(whole_med_sum, device=self.device)
			logits_flatten = torch.empty((whole_med_sum, med_num), device=self.device)

			start_idx = 0
			for i in range(batch_size):
				for j in range(int(seq_length[i].item())):
					for k in range(m_length_matrix[i][j] + 1):
						if k == m_length_matrix[i][j]:
							labels_flatten[start_idx] = end_token
						else:
							labels_flatten[start_idx] = labels[i, j, k]
						logits_flatten[start_idx, :] = logits[i, j, k, :]
						start_idx += 1
		else:
			labels_flatten = []
			logits_flatten = []
			for i in range(batch_size):
				for j in range(int(seq_length[i].item())):
					# 提取当前访问的实际标签
					labels_flatten.append(
						labels[i, j, :m_length_matrix[i][j]].detach().cpu().numpy()
					)
					logits_flatten.append(logits[i, j, : self.max_len, :].detach().cpu().numpy())
			labels_flatten = labels_flatten[-1]
			logits_flatten = logits_flatten[-1]
		return labels_flatten, logits_flatten
