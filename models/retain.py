"""
RETAIN model implementation integrated with the benchmark base model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class RetainCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		emb_dim: int = 64,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.device = device
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.input_len = vocab_size[0] + vocab_size[1] + vocab_size[2]
		self.output_len = vocab_size[2]

		self.embedding = nn.Sequential(
			nn.Embedding(self.input_len + 1, self.emb_dim, padding_idx=self.input_len),
			nn.Dropout(0.5),
		)

		self.alpha_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)
		self.beta_gru = nn.GRU(emb_dim, emb_dim, batch_first=True)

		self.alpha_li = nn.Linear(emb_dim, 1)
		self.beta_li = nn.Linear(emb_dim, emb_dim)

		self.output = nn.Linear(emb_dim, self.output_len)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		if not visits:
			return torch.zeros((1, self.output_len), device=self.device)

		max_len = max(len(v[0]) + len(v[1]) + len(v[2]) for v in visits)
		input_np: List[List[int]] = []
		for visit in visits:
			input_tmp: List[int] = []
			input_tmp.extend(visit[0])
			input_tmp.extend(list(np.array(visit[1]) + self.vocab_size[0]))
			input_tmp.extend(
				list(np.array(visit[2]) + self.vocab_size[0] + self.vocab_size[1])
			)
			if len(input_tmp) < max_len:
				input_tmp.extend([self.input_len] * (max_len - len(input_tmp)))
			input_np.append(input_tmp)

		visit_emb = self.embedding(torch.LongTensor(input_np).to(self.device))
		visit_emb = torch.sum(visit_emb, dim=1)

		g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))
		h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))

		g = g.squeeze(dim=0)
		h = h.squeeze(dim=0)
		attn_g = F.softmax(self.alpha_li(g), dim=-1)
		attn_h = torch.tanh(self.beta_li(h))

		context = attn_g * attn_h * visit_emb
		context = torch.sum(context, dim=0).unsqueeze(dim=0)

		return self.output(context)


class Retain(BaseDrugRecommendationModel):
	"""
	RETAIN wrapper integrated with the benchmark interfaces.

	Expected batch format:
	{
		"visits": List[List[Tuple[List[int], List[int], List[int]]]],
		"labels": Tensor
	}
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		emb_dim: int = 64,
		ddi_adj: Optional[np.ndarray] = None,
		threshold: float = 0.5,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.ddi_adj = ddi_adj

		self.core = RetainCore(
			vocab_size=vocab_size,
			emb_dim=emb_dim,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			return {"logits": torch.empty((0, self.vocab_size[2]), device=self.device)}
		logits = self.core(patient[:-1])
		return {"logits": logits}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		targets = self.build_target(batch)
		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)

		return F.binary_cross_entropy_with_logits(logits, targets)

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

	def _build_target(self, patient: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		return self.build_target({"visit": patient})
