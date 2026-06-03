"""
4SDrug model integration for drugrec_benchmark.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class Attention(nn.Module):
	def __init__(self, embed_dim: int = 64, output_dim: int = 1) -> None:
		super().__init__()
		self.aggregation = nn.Linear(embed_dim, output_dim)

	def _aggregate(self, x: torch.Tensor) -> torch.Tensor:
		weight = self.aggregation(x)
		return torch.tanh(weight)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weight = torch.softmax(self._aggregate(x), dim=-2)
		agg_embeds = torch.matmul(x.transpose(-1, -2).float(), weight).squeeze(-1)
		return agg_embeds


class FourSDrug(BaseDrugRecommendationModel):
	"""
	A v2-compatible 4SDrug wrapper.

	This integration keeps the core symptom-to-drug scoring and DDI regularization
	while adapting to the v2 per-patient batch format.
	"""

	def __init__(
		self,
		n_sym: int,
		n_drug: int,
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		dropout: float = 0.4,
		threshold: float = 0.5,
		entropy_weight: float = 0.5,
		ddi_weight: float = 1.0,
		ddi_scale: float = 1e-5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.n_sym = n_sym
		self.n_drug = n_drug
		self.threshold = threshold
		self.entropy_weight = entropy_weight
		self.ddi_weight = ddi_weight
		self.ddi_scale = ddi_scale
		self.dropout = dropout

		self.ddi_adj = ddi_adj
		self.tensor_ddi_adj = torch.as_tensor(ddi_adj, dtype=torch.float32, device=self.device)

		self.sym_embeddings = nn.Embedding(self.n_sym, emb_dim)
		self.drug_embeddings = nn.Embedding(self.n_drug, emb_dim)
		self.sym_agg = Attention(emb_dim)
		self.dropout_layer = nn.Dropout(p=dropout)
		self._init_parameters(emb_dim)

	def _init_parameters(self, emb_dim: int) -> None:
		stdv = 1.0 / np.sqrt(float(emb_dim))
		for weight in self.parameters():
			if weight.requires_grad:
				weight.data.uniform_(-stdv, stdv)

	def _extract_last_diag(self, patient: Sequence[Sequence[Sequence[int]]]) -> Sequence[int]:
		if not patient:
			return []
		adm = patient[-1]
		if not adm:
			return []
		return adm[0] if len(adm) > 0 else []

	def _build_target(self, patient: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		return self.build_target({"visit": patient})

	def _encode_symptoms(self, sym_codes: Sequence[int]) -> torch.Tensor:
		if not sym_codes:
			return torch.zeros((1, self.sym_embeddings.embedding_dim), device=self.device)
		sym_tensor = torch.tensor(sym_codes, dtype=torch.long, device=self.device).unsqueeze(0)
		sym_embeds = self.dropout_layer(self.sym_embeddings(sym_tensor))
		set_embed = self.sym_agg(sym_embeds)
		return set_embed

	def _compute_logits_from_set(self, set_embed: torch.Tensor) -> torch.Tensor:
		all_drug_idx = torch.arange(self.n_drug, dtype=torch.long, device=self.device)
		drug_embeds = self.drug_embeddings(all_drug_idx)
		logits = torch.mm(set_embed, drug_embeds.transpose(-1, -2))
		return logits

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		sym_codes = self._extract_last_diag(patient)
		set_embed = self._encode_symptoms(sym_codes)
		logits = self._compute_logits_from_set(set_embed)

		neg_pred_prob = torch.sigmoid(logits)
		pair_prob = torch.mm(neg_pred_prob.transpose(-1, -2), neg_pred_prob)
		ddi_loss = self.ddi_scale * pair_prob.mul(self.tensor_ddi_adj).sum()

		targets = self.build_target(batch)
		return {
			"logits": logits,
			"ddi_loss": ddi_loss.unsqueeze(0),
			"targets": targets,
		}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		targets = outputs["targets"]
		ddi_loss = outputs["ddi_loss"][0]

		bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
		sig_scores = torch.sigmoid(logits)
		safe_sig = torch.clamp(sig_scores, min=1e-8)
		entropy = -torch.mean(sig_scores * (torch.log(safe_sig) - 1.0))

		return bce_loss + self.entropy_weight * entropy + self.ddi_weight * ddi_loss

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()
