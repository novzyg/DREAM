"""
Carmen model integration for drugrec_benchmark.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
	matrix = np.asarray(matrix, dtype=np.float32)
	row_sum = matrix.sum(axis=1, keepdims=True)
	row_sum[row_sum == 0] = 1.0
	return matrix / row_sum


def build_carmen_matrices(
	records: Sequence[Any],
	vocab_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Build Carmen's medication-to-EHR-context matrices from benchmark records."""
	n_diag, n_proc, n_med = vocab_size
	med_count = np.zeros((n_med,), dtype=np.float32)
	med2diag = np.zeros((n_med, n_diag), dtype=np.float32)
	med2proc = np.zeros((n_med, n_proc), dtype=np.float32)
	ehr_adj = np.zeros((n_med, n_med), dtype=np.float32)

	for patient in records:
		for visit in patient:
			if len(visit) < 3:
				continue
			diag_codes = [int(code) for code in visit[0] if 0 <= int(code) < n_diag]
			proc_codes = [int(code) for code in visit[1] if 0 <= int(code) < n_proc]
			med_codes = sorted({int(code) for code in visit[2] if 0 <= int(code) < n_med})
			for med in med_codes:
				med_count[med] += 1.0
				med2diag[med, diag_codes] += 1.0
				med2proc[med, proc_codes] += 1.0
			for i, med_i in enumerate(med_codes):
				for med_j in med_codes[i + 1:]:
					ehr_adj[med_i, med_j] += 1.0
					ehr_adj[med_j, med_i] += 1.0

	med_count[med_count == 0] = 1.0
	med2diag = med2diag / med_count.reshape(-1, 1)
	med2proc = med2proc / med_count.reshape(-1, 1)
	return _row_normalize(med2diag), _row_normalize(med2proc), _row_normalize(ehr_adj)


class CarmenCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		med2diag: np.ndarray,
		med2proc: np.ndarray,
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		dropout: float = 0.5,
		max_visits: int = 2,
		use_ehr_aug: bool = True,
		use_ddi_encoding: bool = True,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.num_drugs = vocab_size[2]
		self.emb_dim = emb_dim
		self.max_visits = max_visits
		self.use_ehr_aug = use_ehr_aug
		self.use_ddi_encoding = use_ddi_encoding
		self.device = device

		self.diag_embedding = nn.Embedding(vocab_size[0] + 1, emb_dim, padding_idx=vocab_size[0])
		self.proc_embedding = nn.Embedding(vocab_size[1] + 1, emb_dim, padding_idx=vocab_size[1])
		self.diag_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.proc_encoder = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.dropout = nn.Dropout(dropout)
		self.query = nn.Sequential(
			nn.ReLU(),
			nn.Linear(2 * emb_dim, emb_dim),
		)

		self.med_base = nn.Embedding(self.num_drugs, emb_dim)
		self.viewcat = nn.Linear(2 * emb_dim, emb_dim)
		self.fc_selector = nn.Linear(emb_dim, emb_dim)
		self.med_layernorm = nn.LayerNorm(self.num_drugs)

		self.register_buffer("med2diag", torch.as_tensor(med2diag, dtype=torch.float32, device=device))
		self.register_buffer("med2proc", torch.as_tensor(med2proc, dtype=torch.float32, device=device))
		self.register_buffer("ehr_adj", torch.as_tensor(ehr_adj, dtype=torch.float32, device=device))
		self.register_buffer("ddi_adj_norm", torch.as_tensor(_row_normalize(ddi_adj), dtype=torch.float32, device=device))
		self.ddi_proj = nn.Linear(emb_dim, emb_dim)
		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		self.diag_embedding.weight.data.uniform_(-initrange, initrange)
		self.proc_embedding.weight.data.uniform_(-initrange, initrange)
		self.med_base.weight.data.uniform_(-initrange, initrange)

	def _pad_visit_codes(self, visits: Sequence[Sequence[Sequence[int]]], field: int, vocab_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
		selected = list(visits[-self.max_visits :])
		visit_len = len(selected)
		max_width = max([len(visit[field]) if len(visit) > field else 0 for visit in selected] + [1])
		codes = torch.full((1, self.max_visits, max_width), vocab_dim, dtype=torch.long, device=self.device)
		offset = self.max_visits - visit_len
		for row, visit in enumerate(selected):
			values = [int(code) for code in visit[field] if 0 <= int(code) < vocab_dim] if len(visit) > field else []
			if values:
				codes[0, offset + row, : len(values)] = torch.as_tensor(values, dtype=torch.long, device=self.device)
		return codes, torch.as_tensor([visit_len], dtype=torch.long, device=self.device)

	def _encode_patient(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		diag, visit_len = self._pad_visit_codes(visits, 0, self.vocab_size[0])
		proc, _ = self._pad_visit_codes(visits, 1, self.vocab_size[1])
		diag_seq = self.dropout(self.diag_embedding(diag).sum(dim=-2))
		proc_seq = self.dropout(self.proc_embedding(proc).sum(dim=-2))
		diag_out, _ = self.diag_encoder(diag_seq)
		proc_out, _ = self.proc_encoder(proc_seq)
		diag_last = torch.stack([diag_out[i, visit_len[i] - 1, :] for i in range(visit_len.shape[0])])
		proc_last = torch.stack([proc_out[i, visit_len[i] - 1, :] for i in range(visit_len.shape[0])])
		query = self.query(torch.cat([diag_last, proc_last], dim=-1))
		norm = torch.norm(query, 2, 1, keepdim=True).clamp_min(1e-8)
		return (norm / (1 + norm)) * (query / norm)

	def _medication_embedding(self) -> torch.Tensor:
		diag_emb = self.diag_embedding(torch.arange(self.vocab_size[0], device=self.device))
		proc_emb = self.proc_embedding(torch.arange(self.vocab_size[1], device=self.device))
		med_diagview = torch.matmul(self.med2diag, diag_emb)
		med_procview = torch.matmul(self.med2proc, proc_emb)
		med_rec = self.viewcat(torch.cat([med_diagview, med_procview], dim=-1))
		if self.use_ehr_aug:
			aug_emb = torch.matmul(self.ehr_adj, med_rec)
			selector = torch.tanh(self.fc_selector(med_rec))
			med_rec = med_rec + selector * aug_emb
		med_emb = self.med_base.weight + med_rec
		if self.use_ddi_encoding:
			med_emb = med_emb + self.ddi_proj(torch.matmul(self.ddi_adj_norm, med_emb))
		return med_emb

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		if not visits:
			return torch.empty((0, self.num_drugs), device=self.device)
		query = self._encode_patient(visits)
		med_emb = self._medication_embedding()
		normed_med = med_emb / torch.norm(med_emb, 2, 1, keepdim=True).clamp_min(1e-8)
		logits = torch.matmul(query, normed_med.t())
		return self.med_layernorm(logits)


class Carmen(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		med2diag: np.ndarray,
		med2proc: np.ndarray,
		ehr_adj: np.ndarray,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		dropout: float = 0.5,
		max_visits: int = 2,
		threshold: float = 0.5,
		gamma_bce: float = 0.95,
		gamma_margin: float = 0.05,
		ddi_weight: float = 0.0,
		ddi_scale: float = 5e-4,
		use_ehr_aug: bool = True,
		use_ddi_encoding: bool = True,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.gamma_bce = gamma_bce
		self.gamma_margin = gamma_margin
		self.ddi_weight = ddi_weight
		self.ddi_scale = ddi_scale
		self.ddi_adj = np.asarray(ddi_adj, dtype=np.float32)
		self.register_buffer("tensor_ddi_adj", torch.as_tensor(self.ddi_adj, dtype=torch.float32, device=self.device))
		self.core = CarmenCore(
			vocab_size=vocab_size,
			med2diag=med2diag,
			med2proc=med2proc,
			ehr_adj=ehr_adj,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			dropout=dropout,
			max_visits=max_visits,
			use_ehr_aug=use_ehr_aug,
			use_ddi_encoding=use_ddi_encoding,
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
		loss_margin = F.multilabel_margin_loss(torch.sigmoid(logits), self.build_multilabel_target(targets))
		loss = self.gamma_bce * loss_bce + self.gamma_margin * loss_margin
		if self.ddi_weight > 0:
			probs = torch.sigmoid(logits)
			pair_prob = torch.matmul(probs.t(), probs)
			loss = loss + self.ddi_weight * self.ddi_scale * pair_prob.mul(self.tensor_ddi_adj).sum()
		return loss

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		return (torch.sigmoid(logits) >= self.threshold).float()
