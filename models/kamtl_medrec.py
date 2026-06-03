"""
KAMTL/MedRec-style medicine recommendation model for drugrec_benchmark.

The original paper models recommendation together with graph representation
learning over a medical knowledge graph and a medicine attribute graph. This
benchmark implementation keeps the components supported by mimic-iii_all:
diagnosis/procedure-to-medicine co-occurrence, EHR medicine co-occurrence,
SMILES-derived medicine attributes, and multi-task auxiliary graph losses.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(matrix, dtype=np.float32).copy()
	rowsum = matrix.sum(axis=1, keepdims=True)
	rowsum[rowsum == 0.0] = 1.0
	return matrix / rowsum


def _sym_normalize(adj: np.ndarray) -> np.ndarray:
	adj = np.asarray(adj, dtype=np.float32).copy()
	adj = adj + np.eye(adj.shape[0], dtype=np.float32)
	degree = adj.sum(axis=1)
	degree[degree == 0.0] = 1.0
	d_inv_sqrt = np.power(degree, -0.5)
	return d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]


class SimpleGCN(nn.Module):
	def __init__(self, size: int, emb_dim: int, adj: np.ndarray, dropout: float = 0.2) -> None:
		super().__init__()
		self.register_buffer("adj", torch.FloatTensor(_sym_normalize(adj)))
		self.input_embedding = nn.Parameter(torch.empty(size, emb_dim))
		self.linear1 = nn.Linear(emb_dim, emb_dim)
		self.linear2 = nn.Linear(emb_dim, emb_dim)
		self.dropout = nn.Dropout(dropout)
		nn.init.xavier_uniform_(self.input_embedding)

	def forward(self) -> torch.Tensor:
		x = torch.matmul(self.adj, self.input_embedding)
		x = F.relu(self.linear1(x))
		x = self.dropout(x)
		x = torch.matmul(self.adj, x)
		return self.linear2(x)


class KAMTLCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		diag_med_adj: np.ndarray,
		proc_med_adj: np.ndarray,
		ehr_adj: np.ndarray,
		attr_adj: np.ndarray,
		emb_dim: int = 128,
		dropout: float = 0.3,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.device = device
		self.emb_dim = emb_dim
		self.med_size = vocab_size[2]

		self.diag_embedding = nn.Embedding(vocab_size[0], emb_dim)
		self.proc_embedding = nn.Embedding(vocab_size[1], emb_dim)
		self.med_embedding = nn.Embedding(vocab_size[2], emb_dim)
		self.dropout = nn.Dropout(dropout)
		self.diag_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.proc_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.query = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)

		self.register_buffer("diag_med_adj", torch.FloatTensor(_row_normalize(diag_med_adj)))
		self.register_buffer("proc_med_adj", torch.FloatTensor(_row_normalize(proc_med_adj)))
		self.register_buffer("diag_med_target", torch.FloatTensor((diag_med_adj > 0).astype(np.float32)))
		self.register_buffer("proc_med_target", torch.FloatTensor((proc_med_adj > 0).astype(np.float32)))
		self.register_buffer("attr_target", torch.FloatTensor((attr_adj > 0).astype(np.float32)))
		self.register_buffer("ehr_target", torch.FloatTensor((ehr_adj > 0).astype(np.float32)))

		self.ehr_gcn = SimpleGCN(vocab_size[2], emb_dim, ehr_adj, dropout=dropout)
		self.attr_gcn = SimpleGCN(vocab_size[2], emb_dim, attr_adj, dropout=dropout)
		self.med_fusion = nn.Sequential(
			nn.Linear(3 * emb_dim, emb_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
		)
		self.output_bias = nn.Linear(emb_dim, vocab_size[2])
		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		for embedding in (self.diag_embedding, self.proc_embedding, self.med_embedding):
			embedding.weight.data.uniform_(-initrange, initrange)

	def _enhanced_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		base_diag = self.diag_embedding.weight
		base_proc = self.proc_embedding.weight
		base_med = self.med_embedding.weight
		kg_diag = base_diag + torch.matmul(self.diag_med_adj, base_med)
		kg_proc = base_proc + torch.matmul(self.proc_med_adj, base_med)
		med_from_diag = torch.matmul(self.diag_med_adj.t(), base_diag)
		med_from_proc = torch.matmul(self.proc_med_adj.t(), base_proc)
		med_graph = 0.5 * (self.ehr_gcn() + self.attr_gcn())
		kg_med = self.med_fusion(torch.cat([base_med, med_from_diag + med_from_proc, med_graph], dim=-1))
		return kg_diag, kg_proc, kg_med

	def _mean_codes(self, table: torch.Tensor, codes: Sequence[int]) -> torch.Tensor:
		valid = [int(code) for code in codes if 0 <= int(code) < table.size(0)]
		if not valid:
			return torch.zeros((1, 1, self.emb_dim), device=self.device)
		index = torch.LongTensor(valid).to(self.device)
		return table.index_select(0, index).mean(dim=0, keepdim=True).unsqueeze(0)

	def _encode_patient(
		self,
		visits: Sequence[Sequence[Sequence[int]]],
		kg_diag: torch.Tensor,
		kg_proc: torch.Tensor,
	) -> torch.Tensor:
		diag_seq: List[torch.Tensor] = []
		proc_seq: List[torch.Tensor] = []
		for visit in visits:
			diag_seq.append(self._mean_codes(kg_diag, visit[0]))
			proc_seq.append(self._mean_codes(kg_proc, visit[1]))
		diag_input = self.dropout(torch.cat(diag_seq, dim=1))
		proc_input = self.dropout(torch.cat(proc_seq, dim=1))
		diag_out, _ = self.diag_encoder(diag_input)
		proc_out, _ = self.proc_encoder(proc_input)
		return self.query(torch.cat([diag_out[:, -1, :], proc_out[:, -1, :]], dim=-1))

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		if not visits:
			empty = torch.empty((0, self.med_size), device=self.device)
			return empty, {"query": empty, "med_repr": empty}
		kg_diag, kg_proc, kg_med = self._enhanced_embeddings()
		query = self._encode_patient(visits, kg_diag, kg_proc)
		logits = torch.matmul(query, kg_med.t()) + self.output_bias(query)
		return logits, {"query": query, "kg_diag": kg_diag, "kg_proc": kg_proc, "med_repr": kg_med}

	def auxiliary_graph_loss(
		self,
		state: Dict[str, torch.Tensor],
		batch: Dict[str, Any],
		max_codes: int = 64,
	) -> torch.Tensor:
		if not state or "med_repr" not in state:
			return torch.tensor(0.0, device=self.device)
		patient = batch.get("visit", [])
		if not patient:
			return torch.tensor(0.0, device=self.device)

		last_visit = patient[-1]
		med_repr = state["med_repr"]
		losses: List[torch.Tensor] = []

		diag_codes = [int(c) for c in last_visit[0] if 0 <= int(c) < self.vocab_size[0]][:max_codes]
		if diag_codes:
			diag_idx = torch.LongTensor(diag_codes).to(self.device)
			diag_logits = torch.matmul(state["kg_diag"].index_select(0, diag_idx), med_repr.t())
			diag_targets = self.diag_med_target.index_select(0, diag_idx)
			losses.append(F.binary_cross_entropy_with_logits(diag_logits, diag_targets))

		proc_codes = [int(c) for c in last_visit[1] if 0 <= int(c) < self.vocab_size[1]][:max_codes]
		if proc_codes:
			proc_idx = torch.LongTensor(proc_codes).to(self.device)
			proc_logits = torch.matmul(state["kg_proc"].index_select(0, proc_idx), med_repr.t())
			proc_targets = self.proc_med_target.index_select(0, proc_idx)
			losses.append(F.binary_cross_entropy_with_logits(proc_logits, proc_targets))

		med_codes = [int(c) for c in last_visit[2] if 0 <= int(c) < self.med_size][:max_codes]
		if med_codes:
			med_idx = torch.LongTensor(med_codes).to(self.device)
			selected = med_repr.index_select(0, med_idx)
			pair_logits = torch.matmul(selected, med_repr.t())
			graph_targets = torch.clamp(
				self.attr_target.index_select(0, med_idx) + self.ehr_target.index_select(0, med_idx),
				max=1.0,
			)
			losses.append(F.binary_cross_entropy_with_logits(pair_logits, graph_targets))

		if not losses:
			return torch.tensor(0.0, device=self.device)
		return torch.stack(losses).mean()


class KAMTLMedRec(BaseDrugRecommendationModel):
	"""Benchmark wrapper with SafeDrug-style multilabel outputs."""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		diag_med_adj: np.ndarray,
		proc_med_adj: np.ndarray,
		ehr_adj: np.ndarray,
		attr_adj: np.ndarray,
		emb_dim: int = 128,
		dropout: float = 0.3,
		threshold: float = 0.5,
		aux_weight: float = 0.05,
		ddi_weight: float = 0.0,
		ddi_adj: Optional[np.ndarray] = None,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.aux_weight = aux_weight
		self.ddi_weight = ddi_weight
		self.ddi_adj = ddi_adj
		if ddi_adj is None:
			self.register_buffer("tensor_ddi_adj", torch.zeros((vocab_size[2], vocab_size[2]), device=self.device))
		else:
			self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_adj).to(self.device))
		self.core = KAMTLCore(
			vocab_size=vocab_size,
			diag_med_adj=diag_med_adj,
			proc_med_adj=proc_med_adj,
			ehr_adj=ehr_adj,
			attr_adj=attr_adj,
			emb_dim=emb_dim,
			dropout=dropout,
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
		loss = F.binary_cross_entropy_with_logits(logits, targets)
		if self.aux_weight > 0.0:
			loss = loss + self.aux_weight * self.core.auxiliary_graph_loss(outputs.get("state", {}), batch)
		if self.ddi_weight > 0.0:
			probs = torch.sigmoid(logits)
			pair_prob = torch.matmul(probs.t(), probs)
			loss = loss + self.ddi_weight * pair_prob.mul(self.tensor_ddi_adj).mean()
		return loss

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()


def build_medical_cooccurrence(
	records: Sequence[Sequence[Sequence[int]]],
	vocab_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
	diag_med = np.zeros((vocab_size[0], vocab_size[2]), dtype=np.float32)
	proc_med = np.zeros((vocab_size[1], vocab_size[2]), dtype=np.float32)
	for patient in records:
		for visit in patient:
			diags, procs, meds = visit[:3]
			valid_meds = [int(m) for m in meds if 0 <= int(m) < vocab_size[2]]
			if not valid_meds:
				continue
			for diag in diags:
				if 0 <= int(diag) < vocab_size[0]:
					diag_med[int(diag), valid_meds] += 1.0
			for proc in procs:
				if 0 <= int(proc) < vocab_size[1]:
					proc_med[int(proc), valid_meds] += 1.0
	return diag_med, proc_med


def _smiles_tokens(smiles_values: Iterable[str]) -> set[str]:
	tokens: set[str] = set()
	for smiles in smiles_values:
		for token in re.findall(r"Cl|Br|[A-Z][a-z]?|[cnops]|\[[^\]]+\]|=|#|\(|\)|\d", str(smiles)):
			tokens.add(token)
	return tokens


def build_smiles_attribute_graph(molecule: Dict[str, Sequence[str]], med_idx2word: Dict[int, str], top_k: int = 20) -> np.ndarray:
	med_count = len(med_idx2word)
	token_sets: List[set[str]] = []
	for idx in range(med_count):
		name = med_idx2word[idx]
		smiles_values = molecule.get(name, []) if isinstance(molecule, dict) else []
		token_sets.append(_smiles_tokens(smiles_values))

	adj = np.zeros((med_count, med_count), dtype=np.float32)
	for i in range(med_count):
		if not token_sets[i]:
			continue
		scores: List[Tuple[float, int]] = []
		for j in range(med_count):
			if i == j or not token_sets[j]:
				continue
			union = token_sets[i] | token_sets[j]
			if not union:
				continue
			score = len(token_sets[i] & token_sets[j]) / float(len(union))
			if score > 0.0:
				scores.append((score, j))
		for score, j in sorted(scores, reverse=True)[:top_k]:
			adj[i, j] = score
			adj[j, i] = max(adj[j, i], score)
	return adj
