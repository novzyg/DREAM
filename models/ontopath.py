"""
Ontopath model integration for drugrec_benchmark.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class Norm(nn.Module):
	def __init__(self, d_model: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.alpha = nn.Parameter(torch.ones(d_model))
		self.bias = nn.Parameter(torch.zeros(d_model))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias


def _attention(
	q: torch.Tensor,
	k: torch.Tensor,
	v: torch.Tensor,
	d_k: int,
	mask: Optional[torch.Tensor] = None,
	dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
	scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
	scores = F.softmax(scores, dim=-1)
	if dropout is not None:
		scores = dropout(scores)
	return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
	def __init__(self, heads: int, d_model: int, dropout: float = 0.1) -> None:
		super().__init__()
		if d_model % heads != 0:
			raise ValueError("d_model must be divisible by heads.")
		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads
		self.q_linear = nn.Linear(d_model, d_model)
		self.v_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model)
		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(d_model, d_model)

	def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		batch_size = q.size(0)
		q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
		k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
		v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
		scores = _attention(q, k, v, self.d_k, mask, self.dropout)
		concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
		return self.out(concat)


class FeedForward(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.linear_1 = nn.Linear(d_model, d_model * 4)
		self.dropout = nn.Dropout(dropout)
		self.linear_2 = nn.Linear(d_model * 4, d_model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class PositionalEncoder(nn.Module):
	def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
		super().__init__()
		self.d_model = d_model
		self.dropout = nn.Dropout(dropout)
		self.register_buffer("pe", self._build_pe(max_seq_len), persistent=False)

	def _build_pe(self, max_seq_len: int) -> torch.Tensor:
		pe = torch.zeros(max_seq_len, self.d_model)
		position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		if self.d_model > 1:
			pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
		return pe.unsqueeze(0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		seq_len = x.size(1)
		if seq_len > self.pe.size(1):
			self.pe = self._build_pe(seq_len).to(x.device)
		return x * math.sqrt(self.d_model) + self.pe[:, :seq_len].to(x.device)


class EncoderLayer(nn.Module):
	def __init__(self, d_model: int, heads: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
		self.ff = FeedForward(d_model, dropout=dropout)
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
		x2 = self.norm_2(x)
		return x + self.dropout_2(self.ff(x2))


class DecoderLayer(nn.Module):
	def __init__(self, d_model: int, heads: int, dropout: float = 0.1) -> None:
		super().__init__()
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
		self.ff = FeedForward(d_model, dropout=dropout)

	def forward(self, x: torch.Tensor, e_outputs: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
		x2 = self.norm_1(x)
		x = x + self.dropout_1(self.attn(x2, e_outputs, e_outputs, src_mask))
		x2 = self.norm_2(x)
		return x + self.dropout_2(self.ff(x2))


class Transformer(nn.Module):
	def __init__(self, d_model: int, layers: int, heads: int, dropout: float, order: bool = True) -> None:
		super().__init__()
		self.order = order
		self.pe = PositionalEncoder(d_model, dropout=dropout) if order else None
		self.encoder_layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, heads, dropout)) for _ in range(layers)])
		self.decoder_layers = nn.ModuleList([copy.deepcopy(DecoderLayer(d_model, heads, dropout)) for _ in range(layers)])
		self.encoder_norm = Norm(d_model)
		self.decoder_norm = Norm(d_model)

	def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
		encoded = self.pe(src) if self.pe is not None else src
		for layer in self.encoder_layers:
			encoded = layer(encoded, src_mask)
		encoded = self.encoder_norm(encoded)

		decoded = trg
		for layer in self.decoder_layers:
			decoded = layer(decoded, encoded, src_mask)
		return self.decoder_norm(decoded)


class OntopathCore(nn.Module):
	def __init__(
		self,
		atc_num: int,
		icd_num: int,
		patient_demo_dim: int,
		drug_se_dim: int,
		atc_path_length: int = 5,
		icd_path_length: int = 4,
		dim_emb: int = 64,
		dropout: float = 0.1,
		dim_network: int = 64,
		dim_output: int = 1,
		bidirectional: bool = False,
		initializer: str = "uniform",
		predictor: str = "dot",
		transformer_order: bool = True,
		transformer_layers: int = 1,
		transformer_heads: int = 2,
	) -> None:
		super().__init__()
		self.icd_path_length = icd_path_length
		self.atc_path_length = atc_path_length
		self.predictor = predictor
		self.icd_embedding = nn.Embedding(icd_num, dim_emb, padding_idx=0)
		self.atc_embedding = nn.Embedding(atc_num, dim_emb, padding_idx=0)
		self.transformer = Transformer(
			d_model=dim_emb,
			layers=transformer_layers,
			heads=transformer_heads,
			dropout=dropout,
			order=transformer_order,
		)

		hidden_size = int(dim_network / 2) if bidirectional else dim_network
		self.rnn_patient = nn.GRU(dim_emb, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
		self.rnn_drug = nn.GRU(dim_emb, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)
		dim_network = hidden_size * 2 if bidirectional else hidden_size
		self.C = nn.Parameter(torch.empty((dim_network, dim_network), requires_grad=True))
		self.att_activate = nn.Tanh()
		self.predict_activate = nn.ReLU()

		if predictor == "mlp":
			self.predict_layer = nn.Sequential(
				nn.Linear(dim_emb * 2 + dim_network * 2, dim_emb * 2),
				self.predict_activate,
				nn.Linear(dim_emb * 2, dim_emb),
				self.predict_activate,
				nn.Linear(dim_emb, dim_output),
			)
		else:
			self.predict_layer = nn.Sequential(nn.Linear(dim_emb * 2, dim_output))

		self.patient_demo_encoder = nn.Sequential(nn.Linear(patient_demo_dim, dim_emb))
		self.drug_se_encoder = nn.Sequential(nn.Linear(drug_se_dim, dim_emb))
		self._init_weight(initializer)

	def _init_weight(self, initializer: str) -> None:
		if initializer not in {"normal", "uniform"}:
			raise ValueError("initializer must be 'normal' or 'uniform'.")
		if initializer == "normal":
			init.normal_(self.atc_embedding.weight, std=0.01)
			init.normal_(self.icd_embedding.weight, std=0.01)
			init.normal_(self.C, std=0.01)
		else:
			init.normal_(self.atc_embedding.weight, std=0.01)
			init.normal_(self.icd_embedding.weight, std=0.01)
			init.xavier_uniform_(self.C)
		for module in [self.transformer, self.predict_layer, self.patient_demo_encoder, self.drug_se_encoder]:
			for param in module.parameters():
				if param.dim() > 1:
					init.normal_(param, std=0.01) if initializer == "normal" else init.xavier_uniform_(param)
				else:
					init.constant_(param, 0)

	def _gmf_prediction(self, health_rep_batch: torch.Tensor, path_rep_batch: torch.Tensor, se_batch: torch.Tensor) -> torch.Tensor:
		health_se = health_rep_batch * se_batch
		health_drug = health_rep_batch * path_rep_batch
		return torch.cat([health_drug, health_se], dim=1)

	def forward(
		self,
		demo_batch: torch.Tensor,
		se_batch: torch.Tensor,
		path_batch: torch.Tensor,
		ehr_batch: torch.Tensor,
		ehr_mask: torch.Tensor,
	) -> torch.Tensor:
		patient_emb = self.patient_demo_encoder(demo_batch)
		drug_emb = self.drug_se_encoder(se_batch)
		ehr_emb = self.icd_embedding(ehr_batch)
		patient_query = torch.repeat_interleave(patient_emb, self.icd_path_length, dim=0).unsqueeze(1)
		patient_health_emb = self.transformer(ehr_emb, patient_query, ehr_mask)
		drug_path_emb = self.atc_embedding(path_batch)

		g_health, _ = self.rnn_patient(patient_health_emb.view(patient_emb.size(0), -1, patient_emb.size(1)))
		g_path, _ = self.rnn_drug(drug_path_emb)
		att_mat = self.att_activate(torch.matmul(torch.matmul(g_health, self.C), g_path.transpose(1, 2)))
		health_att_logit = torch.max(att_mat, 2, keepdim=True)[0]
		path_att_logit = torch.max(att_mat, 1, keepdim=True)[0].transpose(1, 2)
		health_att = F.softmax(health_att_logit, dim=1)
		path_att = F.softmax(path_att_logit, dim=1)
		health_rep = torch.sum(g_health * health_att, dim=1)
		path_rep = torch.sum(g_path * path_att, dim=1)

		if self.predictor == "mlp":
			logits = self.predict_layer(torch.cat([health_rep, patient_emb, path_rep, drug_emb], dim=1))
		else:
			logits = self.predict_layer(self._gmf_prediction(health_rep, path_rep, drug_emb))
		return logits.squeeze(-1)


def _idx2word(voc: Any) -> Dict[int, Any]:
	return dict(getattr(voc, "idx2word"))


def _safe_icd_key(code: Any) -> Tuple[str, str, str, str]:
	text = str(code).strip()
	compact = text.replace(".", "") or "UNK"
	lv1 = compact[:1]
	lv2 = compact[:3] if len(compact) >= 3 else compact
	lv3 = compact[:4] if len(compact) >= 4 else compact
	return text, lv1, lv2, lv3


def build_icd_paths(diag_voc: Any) -> Tuple[np.ndarray, int]:
	diag_words = _idx2word(diag_voc)
	next_idx = len(diag_words) + 1
	node2idx: Dict[Tuple[str, str], int] = {}
	paths = np.zeros((len(diag_words), 4), dtype=np.int64)

	def node_id(level: str, key: str) -> int:
		nonlocal next_idx
		node_key = (level, key)
		if node_key not in node2idx:
			node2idx[node_key] = next_idx
			next_idx += 1
		return node2idx[node_key]

	for old_idx in range(len(diag_words)):
		_, lv1, lv2, lv3 = _safe_icd_key(diag_words[old_idx])
		icd_idx = old_idx + 1
		paths[old_idx] = [node_id("lv1", lv1), node_id("lv2", lv2), node_id("lv3", lv3), icd_idx]
	return paths, next_idx


def build_drug_paths(med_voc: Any) -> Tuple[np.ndarray, int]:
	med_words = _idx2word(med_voc)
	node2idx: Dict[str, int] = {"ROOT": 1}
	paths = np.zeros((len(med_words), 5), dtype=np.int64)

	def add_node(key: str) -> int:
		if key not in node2idx:
			node2idx[key] = len(node2idx) + 1
		return node2idx[key]

	for old_idx in range(len(med_words)):
		name = str(med_words[old_idx]).strip() or "UNK"
		compact = "".join(ch.lower() for ch in name if ch.isalnum()) or "unk"
		paths[old_idx] = [
			node2idx["ROOT"],
			add_node("initial:" + compact[:1]),
			add_node("prefix2:" + compact[:2]),
			add_node("prefix3:" + compact[:3]),
			add_node("drug:" + str(old_idx)),
		]
	return paths, len(node2idx) + 1


class Ontopath(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		drug_paths: np.ndarray,
		icd_paths: np.ndarray,
		atc_num: int,
		icd_num: int,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		dropout: float = 0.1,
		bidirectional: bool = False,
		initializer: str = "uniform",
		predictor: str = "dot",
		transformer_order: bool = True,
		transformer_layers: int = 1,
		transformer_heads: int = 2,
		threshold: float = 0.5,
		ddi_weight: float = 0.0,
		ddi_scale: float = 1e-5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.num_drugs = vocab_size[2]
		self.threshold = threshold
		self.ddi_weight = ddi_weight
		self.ddi_scale = ddi_scale
		self.ddi_adj = np.asarray(ddi_adj, dtype=np.float32)
		self.register_buffer("drug_paths", torch.as_tensor(drug_paths, dtype=torch.long, device=self.device))
		self.register_buffer("icd_paths", torch.as_tensor(icd_paths, dtype=torch.long, device=self.device))
		self.register_buffer("drug_se", torch.as_tensor(self.ddi_adj, dtype=torch.float32, device=self.device))
		self.register_buffer("tensor_ddi_adj", torch.as_tensor(self.ddi_adj, dtype=torch.float32, device=self.device))
		self.register_buffer("demo", torch.ones((1, 1), dtype=torch.float32, device=self.device))
		self.core = OntopathCore(
			atc_num=atc_num,
			icd_num=icd_num,
			patient_demo_dim=1,
			drug_se_dim=self.num_drugs,
			dim_emb=emb_dim,
			dim_network=emb_dim,
			dropout=dropout,
			bidirectional=bidirectional,
			initializer=initializer,
			predictor=predictor,
			transformer_order=transformer_order,
			transformer_layers=transformer_layers,
			transformer_heads=transformer_heads,
		)

	def _build_ehr_batch(self, diag_codes: Sequence[int]) -> torch.Tensor:
		valid = [int(code) for code in diag_codes if 0 <= int(code) < self.icd_paths.size(0)]
		if not valid:
			return torch.zeros((self.core.icd_path_length, 1), dtype=torch.long, device=self.device)
		paths = self.icd_paths[torch.as_tensor(valid, dtype=torch.long, device=self.device)]
		return paths.transpose(0, 1).contiguous()

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.num_drugs), device=self.device)}
		last_visit = patient[-1]
		diag_codes = last_visit[0] if len(last_visit) > 0 else []
		ehr_one = self._build_ehr_batch(diag_codes)
		ehr_batch = ehr_one.repeat(self.num_drugs, 1)
		ehr_mask = (ehr_batch != 0).unsqueeze(-2)
		demo_batch = self.demo.repeat(self.num_drugs, 1)
		logits = self.core(
			demo_batch=demo_batch,
			se_batch=self.drug_se,
			path_batch=self.drug_paths,
			ehr_batch=ehr_batch,
			ehr_mask=ehr_mask,
		)
		return {"logits": logits.unsqueeze(0)}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		targets = self.build_target(batch)
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		if self.ddi_weight > 0:
			probs = torch.sigmoid(logits)
			pair_prob = torch.matmul(probs.t(), probs)
			ddi_loss = self.ddi_scale * pair_prob.mul(self.tensor_ddi_adj).sum()
			loss = loss + self.ddi_weight * ddi_loss
		return loss

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()
