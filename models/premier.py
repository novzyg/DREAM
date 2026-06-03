"""
PREMIER model integration for drugrec_benchmark.

This is a benchmark-compatible implementation of the closed-source PREMIER
model described in "Personalizing Medication Recommendation with a Graph-Based
Approach". It implements the components supported by mimic-iii_all: diagnosis,
procedure, past medication history, EHR drug co-occurrence graph, and binary DDI
graph.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def _normalize_edges(adj: np.ndarray) -> np.ndarray:
	adj = np.asarray(adj, dtype=np.float32).copy()
	if adj.size == 0:
		return adj
	max_value = float(adj.max())
	if max_value > 0.0:
		adj = adj / max_value
	return adj


class WeightedGATLayer(nn.Module):
	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_heads: int = 1,
		dropout: float = 0.2,
		concat: bool = True,
	) -> None:
		super().__init__()
		self.out_dim = out_dim
		self.num_heads = num_heads
		self.concat = concat
		self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
		self.attn = nn.ParameterList([nn.Parameter(torch.empty(2 * out_dim)) for _ in range(num_heads)])
		self.dropout = nn.Dropout(dropout)
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		for linear in self.linears:
			nn.init.xavier_uniform_(linear.weight)
		for attn in self.attn:
			nn.init.xavier_uniform_(attn.view(1, -1))

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		outputs: List[torch.Tensor] = []
		mask = adj > 0
		log_weight = torch.log(adj.clamp_min(1e-12))
		for linear, attn in zip(self.linears, self.attn):
			h = linear(x)
			left = torch.matmul(h, attn[: self.out_dim])
			right = torch.matmul(h, attn[self.out_dim :])
			scores = self.leaky_relu(left.unsqueeze(1) + right.unsqueeze(0)) + log_weight
			scores = scores.masked_fill(~mask, -1e9)
			alpha = F.softmax(scores, dim=-1)
			alpha = self.dropout(alpha)
			outputs.append(torch.matmul(alpha, h))
		if self.concat:
			return torch.cat(outputs, dim=-1)
		return torch.stack(outputs, dim=0).mean(dim=0)


class DrugGraphGAT(nn.Module):
	def __init__(self, med_size: int, emb_dim: int, adj: np.ndarray, heads: int = 2, dropout: float = 0.2) -> None:
		super().__init__()
		adj = _normalize_edges(adj)
		adj = adj + np.eye(med_size, dtype=np.float32)
		self.register_buffer("adj", torch.FloatTensor(adj))
		self.node_features = nn.Parameter(torch.eye(med_size))
		self.gat1 = WeightedGATLayer(med_size, emb_dim, num_heads=heads, dropout=dropout, concat=True)
		self.gat2 = WeightedGATLayer(emb_dim * heads, emb_dim, num_heads=1, dropout=dropout, concat=False)

	def forward(self) -> torch.Tensor:
		x = F.elu(self.gat1(self.node_features, self.adj))
		return self.gat2(x, self.adj)


class CodeAttention(nn.Module):
	def __init__(self, emb_dim: int) -> None:
		super().__init__()
		self.scorer = nn.Linear(emb_dim, 1)

	def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		if embeddings.numel() == 0:
			empty = embeddings.new_zeros((embeddings.size(-1),))
			return empty, embeddings.new_zeros((0,))
		scores = self.scorer(torch.tanh(embeddings)).squeeze(-1)
		weights = F.softmax(scores, dim=0)
		return torch.sum(weights.unsqueeze(-1) * embeddings, dim=0), weights


class VisitAttention(nn.Module):
	def __init__(self, emb_dim: int) -> None:
		super().__init__()
		self.gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.scorer = nn.Linear(emb_dim, 1)

	def forward(self, visit_reprs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		if visit_reprs.size(0) == 0:
			empty = visit_reprs.new_zeros((visit_reprs.size(-1),))
			return empty, visit_reprs, visit_reprs.new_zeros((0,))
		encoded, _ = self.gru(visit_reprs.unsqueeze(0))
		encoded = encoded.squeeze(0)
		scores = self.scorer(torch.tanh(encoded)).squeeze(-1)
		weights = F.softmax(scores, dim=0)
		response = torch.sum(weights.unsqueeze(-1) * encoded, dim=0)
		return response, encoded, weights


class PREMIERCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		gat_heads: int = 2,
		dropout: float = 0.2,
		di_weight: float = 0.5,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.device = device
		self.emb_dim = emb_dim
		self.med_size = vocab_size[2]
		self.di_weight = di_weight

		self.diag_embedding = nn.Embedding(vocab_size[0], emb_dim)
		self.proc_embedding = nn.Embedding(vocab_size[1], emb_dim)
		self.med_embedding = nn.Embedding(vocab_size[2], emb_dim)
		self.diag_code_attention = CodeAttention(emb_dim)
		self.proc_code_attention = CodeAttention(emb_dim)
		self.diag_visit_attention = VisitAttention(emb_dim)
		self.proc_visit_attention = VisitAttention(emb_dim)
		self.context = nn.Sequential(nn.Linear(3 * emb_dim, emb_dim), nn.Tanh())
		self.history_key = nn.Linear(emb_dim, emb_dim, bias=False)
		self.history_query = nn.Linear(emb_dim, emb_dim, bias=False)
		self.query_layer = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)

		self.ehr_gat = DrugGraphGAT(vocab_size[2], emb_dim, ehr_adj, heads=gat_heads, dropout=dropout)
		self.ddi_gat = DrugGraphGAT(vocab_size[2], emb_dim, ddi_adj, heads=gat_heads, dropout=dropout)
		self.graph_fusion = nn.Sequential(
			nn.Linear(3 * emb_dim, emb_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.output = nn.Linear(emb_dim, vocab_size[2])
		self.dropout = nn.Dropout(dropout)
		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		for embedding in (self.diag_embedding, self.proc_embedding, self.med_embedding):
			embedding.weight.data.uniform_(-initrange, initrange)

	def _embed_codes(self, embedding: nn.Embedding, codes: Sequence[int], vocab_size: int) -> torch.Tensor:
		valid = [int(code) for code in codes if 0 <= int(code) < vocab_size]
		if not valid:
			return torch.empty((0, self.emb_dim), device=self.device)
		index = torch.LongTensor(valid).to(self.device)
		return embedding(index)

	def _visit_med_embedding(self, meds: Sequence[int]) -> torch.Tensor:
		valid = [int(code) for code in meds if 0 <= int(code) < self.med_size]
		if not valid:
			return torch.zeros((self.emb_dim,), device=self.device)
		index = torch.LongTensor(valid).to(self.device)
		return self.med_embedding(index).mean(dim=0)

	def _encode_modalities(
		self,
		visits: Sequence[Sequence[Sequence[int]]],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		diag_visits: List[torch.Tensor] = []
		proc_visits: List[torch.Tensor] = []
		for visit in visits:
			diag_emb = self._embed_codes(self.diag_embedding, visit[0], self.vocab_size[0])
			proc_emb = self._embed_codes(self.proc_embedding, visit[1], self.vocab_size[1])
			diag_rep, _ = self.diag_code_attention(diag_emb)
			proc_rep, _ = self.proc_code_attention(proc_emb)
			diag_visits.append(diag_rep)
			proc_visits.append(proc_rep)

		diag_stack = torch.stack(diag_visits, dim=0)
		proc_stack = torch.stack(proc_visits, dim=0)
		diag_response, diag_encoded, _ = self.diag_visit_attention(self.dropout(diag_stack))
		proc_response, proc_encoded, _ = self.proc_visit_attention(self.dropout(proc_stack))
		current_diag = diag_encoded[-1] if diag_encoded.size(0) > 0 else diag_response
		current_proc = proc_encoded[-1] if proc_encoded.size(0) > 0 else proc_response
		return diag_response, proc_response, current_diag, current_proc, torch.stack([diag_stack[-1], proc_stack[-1]], dim=0)

	def _history_vector(self, visits: Sequence[Sequence[Sequence[int]]], current_context: torch.Tensor) -> torch.Tensor:
		if len(visits) <= 1:
			return torch.zeros((self.emb_dim,), device=self.device)
		keys: List[torch.Tensor] = []
		values: List[torch.Tensor] = []
		for visit in visits[:-1]:
			diag_emb = self._embed_codes(self.diag_embedding, visit[0], self.vocab_size[0])
			proc_emb = self._embed_codes(self.proc_embedding, visit[1], self.vocab_size[1])
			diag_rep, _ = self.diag_code_attention(diag_emb)
			proc_rep, _ = self.proc_code_attention(proc_emb)
			key = self.context(torch.cat([diag_rep, proc_rep, 0.5 * (diag_rep + proc_rep)], dim=-1))
			keys.append(key)
			values.append(self._visit_med_embedding(visit[2]))
		key_tensor = torch.stack(keys, dim=0)
		value_tensor = torch.stack(values, dim=0)
		scores = torch.matmul(self.history_key(key_tensor), self.history_query(current_context)) / math.sqrt(self.emb_dim)
		weights = F.softmax(scores, dim=0)
		return torch.sum(weights.unsqueeze(-1) * value_tensor, dim=0)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		if not visits:
			empty = torch.empty((0, self.med_size), device=self.device)
			return empty, {}

		diag_response, proc_response, current_diag, current_proc, current_modalities = self._encode_modalities(visits)
		current_context = self.context(torch.cat([diag_response, proc_response, 0.5 * (current_diag + current_proc)], dim=-1))
		history = self._history_vector(visits, current_context)
		query = self.query_layer(torch.cat([current_context, history], dim=-1)).unsqueeze(0)

		ehr_repr = self.ehr_gat()
		ddi_repr = self.ddi_gat()
		ehr_weights = F.softmax(torch.matmul(query, ehr_repr.t()), dim=-1)
		di_weights = F.softmax(torch.matmul(query, ddi_repr.t()), dim=-1)
		ehr_fact = torch.matmul(ehr_weights, ehr_repr)
		di_fact = torch.matmul(di_weights, ddi_repr)
		fused = self.graph_fusion(torch.cat([query, ehr_fact, di_fact], dim=-1))
		logits = self.output(query + fused)
		logits = logits + torch.matmul(query, ehr_repr.t()) - self.di_weight * torch.matmul(query, ddi_repr.t())
		state = {
			"query": query,
			"ehr_weights": ehr_weights,
			"ddi_weights": di_weights,
			"current_modalities": current_modalities,
		}
		return logits, state


class PREMIER(BaseDrugRecommendationModel):
	"""SafeDrug-style multilabel wrapper for PREMIER."""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		gat_heads: int = 2,
		dropout: float = 0.2,
		threshold: float = 0.5,
		multilabel_weight: float = 0.05,
		ddi_weight: float = 0.0005,
		di_graph_weight: float = 0.5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.multilabel_weight = multilabel_weight
		self.ddi_weight = ddi_weight
		self.ddi_adj = np.asarray(ddi_adj, dtype=np.float32)
		self.register_buffer("tensor_ddi_adj", torch.FloatTensor(self.ddi_adj).to(self.device))
		self.core = PREMIERCore(
			vocab_size=vocab_size,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			gat_heads=gat_heads,
			dropout=dropout,
			di_weight=di_graph_weight,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.vocab_size[2]), device=self.device)}
		logits, state = self.core(patient)
		return {"logits": logits, "state": state}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		targets = self.build_target(batch)
		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), self.build_multilabel_target(targets))
		probs = torch.sigmoid(logits)
		pair_prob = torch.matmul(probs.t(), probs)
		loss_ddi = pair_prob.mul(self.tensor_ddi_adj).mean()
		return loss_bce + self.multilabel_weight * loss_multi + self.ddi_weight * loss_ddi

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()
