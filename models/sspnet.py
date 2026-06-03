"""
SSPNet model implementation integrated with drugrec_benchmark.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class GraphConvolution(nn.Module):
	def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
		super().__init__()
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


class MAB(nn.Module):
	def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, ln: bool = False) -> None:
		super().__init__()
		self.dim_v = dim_v
		self.num_heads = num_heads
		self.fc_q = nn.Linear(dim_q, dim_v)
		self.fc_k = nn.Linear(dim_k, dim_v)
		self.fc_v = nn.Linear(dim_k, dim_v)
		if ln:
			self.ln0 = nn.LayerNorm(dim_v)
			self.ln1 = nn.LayerNorm(dim_v)
		self.fc_o = nn.Linear(dim_v, dim_v)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		q = self.fc_q(q_input)
		k = self.fc_k(k_input)
		v = self.fc_v(k_input)
		dim_split = self.dim_v // self.num_heads
		q_split = torch.cat(q.split(dim_split, 2), 0)
		k_split = torch.cat(k.split(dim_split, 2), 0)
		v_split = torch.cat(v.split(dim_split, 2), 0)
		attn = q_split.bmm(k_split.transpose(1, 2)) / math.sqrt(self.dim_v)
		if src_mask is not None:
			attn = attn.masked_fill(src_mask < -1e8, -1e9)
		attn = self.softmax(attn)
		output = torch.cat((q_split + attn.bmm(v_split)).split(q.size(0), 0), 2)
		output = output if getattr(self, "ln0", None) is None else self.ln0(output)
		output = output + F.relu(self.fc_o(output))
		output = output if getattr(self, "ln1", None) is None else self.ln1(output)
		return output


class SAB(nn.Module):
	def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False) -> None:
		super().__init__()
		self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

	def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		return self.mab(x, x, src_mask)


class EncoderSAB(nn.Module):
	def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False) -> None:
		super().__init__()
		self.sab1 = SAB(dim_in, dim_out, num_heads, ln=ln)
		self.sab2 = SAB(dim_out, dim_out, num_heads, ln=ln)

	def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		return self.sab2(self.sab1(x, src_mask), src_mask)


class PMA(nn.Module):
	def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, ln: bool = False) -> None:
		super().__init__()
		self.seed = nn.Parameter(torch.Tensor(1, num_seeds, dim))
		nn.init.xavier_uniform_(self.seed)
		self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
		self.sab = MAB(dim, dim, dim, num_heads, ln=ln)

	def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		x = self.sab(x, x, src_mask)
		return self.mab(self.seed.repeat(x.size(0), 1, 1), x, src_mask)


class SSPNetGCN(nn.Module):
	def __init__(self, voc_size: int, emb_dim: int, ehr_adj: np.ndarray, ddi_adj: np.ndarray) -> None:
		super().__init__()
		ehr_norm = self._normalize(np.asarray(ehr_adj) + np.eye(ehr_adj.shape[0]))
		ddi_np = np.asarray(ddi_adj)
		ddi_norm = self._normalize(ddi_np + np.eye(ddi_np.shape[0]))
		self.register_buffer("ehr_adj", torch.FloatTensor(ehr_norm))
		self.register_buffer("ddi_adj", torch.FloatTensor(ddi_norm))
		self.register_buffer("x", torch.eye(voc_size))
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

	@staticmethod
	def _normalize(mx: np.ndarray) -> np.ndarray:
		rowsum = np.array(mx.sum(1))
		r_inv = np.power(rowsum, -1).flatten()
		r_inv[np.isinf(r_inv)] = 0.0
		return np.diagflat(r_inv).dot(mx)


class AdjAttenAgger(nn.Module):
	def __init__(self, qdim: int, kdim: int, mid_dim: int) -> None:
		super().__init__()
		self.model_dim = mid_dim
		self.q_dense = nn.Linear(qdim, mid_dim)
		self.k_dense = nn.Linear(kdim, mid_dim)

	def forward(self, main_feat: torch.Tensor, other_feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		q = self.q_dense(main_feat)
		k = self.k_dense(other_feat)
		attn = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(self.model_dim)
		if mask is not None:
			attn = torch.masked_fill(attn, mask, -(1 << 32))
		return attn


class MedTransformerDecoderAll(nn.Module):
	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5) -> None:
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

	def forward(self, input_med: torch.Tensor, input_disease_embedding: torch.Tensor, input_proc_embedding: torch.Tensor) -> torch.Tensor:
		x = input_med
		x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, need_weights=False)[0]))
		x = self.norm2(
			x
			+ self.dropout2(self.m2d_multihead_attn(x, input_disease_embedding, input_disease_embedding, need_weights=False)[0])
			+ self.dropout2(self.m2p_multihead_attn(x, input_proc_embedding, input_proc_embedding, need_weights=False)[0])
		)
		return self.norm3(x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x))))))


class SSPNetCore(nn.Module):
	def __init__(self, vocab_size: Tuple[int, int, int], ehr_adj: np.ndarray, ddi_adj: np.ndarray, emb_dim: int = 64, device: torch.device = torch.device("cpu"), dropout: float = 0.7) -> None:
		super().__init__()
		self.device = device
		self.emb_dim = emb_dim
		self.med_num = vocab_size[2]
		self.score_extractor = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Linear(emb_dim // 2, 1))
		self.gcn = SSPNetGCN(vocab_size[2], emb_dim, ehr_adj, ddi_adj)
		self.med_embedding = nn.Sequential(nn.Embedding(vocab_size[2], emb_dim), nn.Dropout(0.3))
		self.diag_embedding = nn.Sequential(nn.Embedding(vocab_size[0], emb_dim), nn.Dropout(0.3))
		self.proc_embedding = nn.Sequential(nn.Embedding(vocab_size[1], emb_dim), nn.Dropout(0.3))
		self.diag_encoder = EncoderSAB(emb_dim, emb_dim, 2)
		self.proc_encoder = EncoderSAB(emb_dim, emb_dim, 2)
		self.decoder = MedTransformerDecoderAll(emb_dim, 2, dropout=dropout)
		self.pma_d = PMA(emb_dim, 2)
		self.pma_p = PMA(emb_dim, 2)
		self.aggregator = AdjAttenAgger(emb_dim, emb_dim, emb_dim)
		self.W_z = nn.Sequential(nn.Linear(emb_dim, emb_dim // 2), nn.ReLU(), nn.Linear(emb_dim // 2, 1))
		self.inter = nn.Parameter(torch.ones(1))
		self.garm = nn.Parameter(torch.ones(1))
		self.W_visit = nn.Linear(emb_dim * 2, emb_dim)
		self.seq_encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True), nn.GRU(emb_dim, emb_dim, batch_first=True)])
		self.Out_visit = nn.Linear(emb_dim * 2, emb_dim)
		self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_adj))

	def _encode_codes(self, codes: Sequence[int], embedding: nn.Module, encoder: EncoderSAB) -> torch.Tensor:
		if len(codes) == 0:
			return torch.zeros((1, 1, self.emb_dim), device=self.device)
		idx = torch.LongTensor([codes]).to(self.device)
		return encoder(embedding(idx))

	def forward(self, med: torch.Tensor, patient_data: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
		med_emb = self.med_embedding(med)
		ehr_embedding, ddi_embedding = self.gcn()
		med_repr = med_emb + ehr_embedding - self.inter * ddi_embedding
		d_repr = self._encode_codes(patient_data[-1][0], self.diag_embedding, self.diag_encoder)
		p_repr = self._encode_codes(patient_data[-1][1], self.proc_embedding, self.proc_encoder)
		visit_len = len(patient_data)
		if visit_len > 1:
			d_history_pma = []
			p_history_pma = []
			for adm in patient_data:
				d_history_pma.append(self.pma_d(self._encode_codes(adm[0], self.diag_embedding, self.diag_encoder)).squeeze(0))
				p_history_pma.append(self.pma_p(self._encode_codes(adm[1], self.proc_embedding, self.proc_encoder)).squeeze(0))
			output1, _ = self.seq_encoders[0](torch.stack(d_history_pma, dim=0))
			output2, _ = self.seq_encoders[1](torch.stack(p_history_pma, dim=0))
			output_d_p = self.Out_visit(torch.cat([output1, output2], dim=-1)).squeeze(1)
			score_c = torch.softmax(self.aggregator(output_d_p[-1], output_d_p), dim=-1).squeeze(0)
			m_h = torch.zeros(self.med_num, device=self.device)
			for i in range(visit_len - 1):
				adm = patient_data[i]
				if len(adm[2]) == 0:
					continue
				m_emb_h = torch.zeros(self.med_num, device=self.device)
				m_emb_h[torch.LongTensor(adm[2]).to(self.device)] = 1.0
				m_h = m_h + m_emb_h * score_c[i]
			med_repr = med_repr * (torch.ones((1, self.med_num), device=self.device) + m_h.unsqueeze(0) * self.garm).t()
		hidden = self.decoder(med_repr.unsqueeze(0), d_repr, p_repr).squeeze(0)
		score = self.score_extractor(hidden).t()
		neg_pred_prob = torch.sigmoid(score)
		batch_neg = 0.0005 * torch.matmul(neg_pred_prob.t(), neg_pred_prob).mul(self.tensor_ddi_adj).sum()
		return score, batch_neg


class SSPNet(BaseDrugRecommendationModel):
	def __init__(self, vocab_size: Tuple[int, int, int], ehr_adj: np.ndarray, ddi_adj: np.ndarray, emb_dim: int = 64, target_ddi: float = 0.06, coef: float = 2.5, threshold: float = 0.5, dropout: float = 0.7, device: Optional[torch.device] = None) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.ddi_adj = ddi_adj
		self.target_ddi = float(target_ddi)
		self.coef = float(coef)
		self.threshold = float(threshold)
		self.register_buffer("med_indices", torch.arange(vocab_size[2], dtype=torch.long))
		self.core = SSPNetCore(vocab_size=vocab_size, ehr_adj=ehr_adj, ddi_adj=ddi_adj, emb_dim=emb_dim, device=self.device, dropout=dropout)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.vocab_size[2]), device=self.device), "ddi_loss": torch.empty((0,), device=self.device)}
		logits, ddi_penalty = self.core(self.med_indices.to(self.device), patient)
		return {"logits": logits, "ddi_loss": ddi_penalty.unsqueeze(0)}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		ddi_penalty = outputs["ddi_loss"]
		targets = self.build_target(batch)
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		target_multi = self.build_multilabel_target(targets)
		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), target_multi)
		current_ddi_rate = self._ddi_rate_from_labels(self._predict_labels(logits))
		if current_ddi_rate <= self.target_ddi:
			return 0.95 * loss_bce + 0.05 * loss_multi
		beta = self.coef * (1.0 - (current_ddi_rate / self.target_ddi))
		beta = min(math.exp(beta), 1.0)
		return beta * (0.95 * loss_bce + 0.05 * loss_multi) + (1.0 - beta) * ddi_penalty[0]

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()

	def _predict_labels(self, logits: torch.Tensor) -> List[int]:
		probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
		return np.where((probs >= self.threshold).astype(np.int32) == 1)[0].tolist()

	def _ddi_rate_from_labels(self, labels: List[int]) -> float:
		if len(labels) < 2:
			return 0.0
		ddi_count = 0
		total_count = 0
		for i, med_i in enumerate(labels):
			for med_j in labels[i + 1:]:
				total_count += 1
				if self.ddi_adj[med_i, med_j] == 1 or self.ddi_adj[med_j, med_i] == 1:
					ddi_count += 1
		return 0.0 if total_count == 0 else float(ddi_count) / float(total_count)
