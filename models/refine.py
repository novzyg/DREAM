"""
REFINE-style medication recommendation model for drugrec_benchmark.

This implementation keeps the parts of REFINE that are supported by the
benchmark MIMIC-III data: longitudinal diagnoses/procedures, medication-history
trends, EHR drug co-occurrence, and binary DDI graphs.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def _normalize_weights(adj: np.ndarray) -> np.ndarray:
	adj = np.asarray(adj, dtype=np.float32).copy()
	if adj.size == 0:
		return adj
	max_value = float(adj.max())
	if max_value > 0.0:
		adj = adj / max_value
	return adj


class WeightedGATLayer(nn.Module):
	"""Dense GATv2-like layer with edge weights folded into attention."""

	def __init__(
		self,
		in_dim: int,
		out_dim: int,
		num_heads: int = 1,
		dropout: float = 0.1,
		concat: bool = True,
	) -> None:
		super().__init__()
		self.out_dim = out_dim
		self.num_heads = num_heads
		self.concat = concat
		self.linear = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_heads)])
		self.attn = nn.ParameterList([nn.Parameter(torch.empty(2 * out_dim)) for _ in range(num_heads)])
		self.dropout = nn.Dropout(dropout)
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		for layer in self.linear:
			nn.init.xavier_uniform_(layer.weight)
		for attn in self.attn:
			nn.init.xavier_uniform_(attn.view(1, -1))

	def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
		outputs: List[torch.Tensor] = []
		mask = adj > 0
		log_weight = torch.log(adj.clamp_min(1e-12))
		for linear, attn in zip(self.linear, self.attn):
			h = linear(x)
			left = torch.matmul(h, attn[: self.out_dim])
			right = torch.matmul(h, attn[self.out_dim :])
			score = self.leaky_relu(left.unsqueeze(1) + right.unsqueeze(0))
			score = score + log_weight
			score = score.masked_fill(~mask, -1e9)
			alpha = F.softmax(score, dim=-1)
			alpha = self.dropout(alpha)
			outputs.append(torch.matmul(alpha, h))
		if self.concat:
			return torch.cat(outputs, dim=-1)
		return torch.stack(outputs, dim=0).mean(dim=0)


class DrugGraphEncoder(nn.Module):
	def __init__(
		self,
		num_drugs: int,
		emb_dim: int,
		adj: np.ndarray,
		num_heads: int = 4,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		adj = _normalize_weights(adj)
		adj = adj + np.eye(num_drugs, dtype=np.float32)
		self.register_buffer("adj", torch.FloatTensor(adj))
		self.node_embedding = nn.Parameter(torch.empty(num_drugs, emb_dim))
		self.gat1 = WeightedGATLayer(emb_dim, emb_dim, num_heads=num_heads, dropout=dropout, concat=True)
		self.gat2 = WeightedGATLayer(emb_dim * num_heads, emb_dim, num_heads=1, dropout=dropout, concat=False)
		nn.init.xavier_uniform_(self.node_embedding)

	def forward(self) -> torch.Tensor:
		x = F.elu(self.gat1(self.node_embedding, self.adj))
		return torch.sigmoid(self.gat2(x, self.adj))


class REFINECore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 128,
		num_heads: int = 4,
		transformer_layers: int = 2,
		dropout: float = 0.2,
		ddi_weight: float = 1.0,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.device = device
		self.emb_dim = emb_dim
		self.num_drugs = vocab_size[2]

		self.diag_embedding = nn.Embedding(vocab_size[0], emb_dim)
		self.proc_embedding = nn.Embedding(vocab_size[1], emb_dim)
		self.visit_norm = nn.LayerNorm(emb_dim)
		self.dropout = nn.Dropout(dropout)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=emb_dim,
			nhead=num_heads,
			dim_feedforward=emb_dim * 4,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.visit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
		self.med_trend_projection = nn.Sequential(
			nn.Linear(self.num_drugs * 3, emb_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(emb_dim, emb_dim),
		)

		self.ehr_graph = DrugGraphEncoder(self.num_drugs, emb_dim, ehr_adj, num_heads=num_heads, dropout=dropout)
		self.ddi_graph = DrugGraphEncoder(self.num_drugs, emb_dim, ddi_adj, num_heads=num_heads, dropout=dropout)
		self.graph_gate = nn.Parameter(torch.tensor(float(ddi_weight)))
		self.output = nn.Linear(emb_dim, self.num_drugs)
		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		self.diag_embedding.weight.data.uniform_(-initrange, initrange)
		self.proc_embedding.weight.data.uniform_(-initrange, initrange)

	def _mean_codes(self, embedding: nn.Embedding, codes: Sequence[int]) -> torch.Tensor:
		if not codes:
			return torch.zeros((self.emb_dim,), device=self.device)
		index = torch.LongTensor(list(codes)).to(self.device)
		return embedding(index).mean(dim=0)

	def _build_visit_sequence(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		visit_reprs: List[torch.Tensor] = []
		for visit in visits:
			diag_repr = self._mean_codes(self.diag_embedding, visit[0])
			proc_repr = self._mean_codes(self.proc_embedding, visit[1])
			visit_reprs.append(self.visit_norm(diag_repr + proc_repr))
		return torch.stack(visit_reprs, dim=0).unsqueeze(0)

	def _medication_trend(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		if not visits:
			return torch.zeros((self.num_drugs * 3,), device=self.device)

		history = torch.zeros((len(visits), self.num_drugs), device=self.device)
		for row, visit in enumerate(visits):
			meds = [idx for idx in visit[2] if 0 <= idx < self.num_drugs]
			if meds:
				history[row, torch.LongTensor(meds).to(self.device)] = 1.0

		latest = history[-1]
		if history.size(0) == 1:
			variance = torch.zeros_like(latest)
			slope = torch.zeros_like(latest)
		else:
			variance = torch.var(history, dim=0, unbiased=False)
			time = torch.arange(history.size(0), dtype=torch.float32, device=self.device)
			centered_time = time - time.mean()
			centered_history = history - history.mean(dim=0, keepdim=True)
			denominator = torch.sum(centered_time * centered_time).clamp_min(1e-6)
			slope = torch.sum(centered_time.unsqueeze(1) * centered_history, dim=0) / denominator
		return torch.cat([latest, variance, slope], dim=0)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		if not visits:
			return torch.empty((0, self.num_drugs), device=self.device)

		visit_sequence = self._build_visit_sequence(visits)
		encoded_visits = self.visit_encoder(self.dropout(visit_sequence)).squeeze(0)
		health_repr = encoded_visits[-1]
		trend_repr = self.med_trend_projection(self._medication_trend(visits[:-1]))
		query = health_repr + trend_repr

		drug_repr = self.ehr_graph() + self.graph_gate * self.ddi_graph()
		attention = F.softmax(torch.matmul(drug_repr, query), dim=0)
		context = torch.matmul(attention.unsqueeze(0), drug_repr).squeeze(0)
		return self.output(query + context).unsqueeze(0)


class REFINE(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 128,
		num_heads: int = 4,
		transformer_layers: int = 2,
		dropout: float = 0.2,
		threshold: float = 0.5,
		gamma_bce: float = 0.90,
		gamma_hinge: float = 0.05,
		beta: float = 0.50,
		bdi_scale: float = 1e-4,
		ddi_weight: float = 1.0,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.gamma_bce = gamma_bce
		self.gamma_hinge = gamma_hinge
		self.beta = beta
		self.bdi_scale = bdi_scale
		self.ddi_adj = np.asarray(ddi_adj, dtype=np.float32)

		ehr_norm = _normalize_weights(ehr_adj)
		ddi_norm = _normalize_weights(ddi_adj)
		np.fill_diagonal(ehr_norm, 0.0)
		np.fill_diagonal(ddi_norm, 0.0)
		self.register_buffer("ehr_weight", torch.FloatTensor(ehr_norm).to(self.device))
		self.register_buffer("ddi_weight", torch.FloatTensor(ddi_norm).to(self.device))

		self.core = REFINECore(
			vocab_size=vocab_size,
			ehr_adj=ehr_norm,
			ddi_adj=ddi_norm,
			emb_dim=emb_dim,
			num_heads=num_heads,
			transformer_layers=transformer_layers,
			dropout=dropout,
			ddi_weight=ddi_weight,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.vocab_size[2]), device=self.device)}
		return {"logits": self.core(patient)}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)

		targets = self.build_target(batch)
		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_hinge = F.multilabel_margin_loss(torch.sigmoid(logits), self.build_multilabel_target(targets))
		loss_bdi = self._balanced_ddi_loss(logits)

		graph_weight = max(0.0, 1.0 - self.gamma_bce - self.gamma_hinge)
		return self.gamma_bce * loss_bce + self.gamma_hinge * loss_hinge + graph_weight * loss_bdi

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()

	def _balanced_ddi_loss(self, logits: torch.Tensor) -> torch.Tensor:
		probs = torch.sigmoid(logits)
		pair_prob = torch.matmul(probs.t(), probs)
		balance = self.beta * self.ddi_weight - (1.0 - self.beta) * self.ehr_weight
		return self.bdi_scale * torch.sum(pair_prob * balance) / max(self.vocab_size[2] * self.vocab_size[2], 1)
