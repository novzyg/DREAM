"""
MICRON model implementation integrated with the benchmark base model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.core.io import Prediction
from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class MICRONCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		emb_dim: int = 256,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.device = device

		self.embeddings = nn.ModuleList(
			[nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
		)
		self.dropout = nn.Dropout(p=0.5)

		self.health_net = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim),
		)

		self.prescription_net = nn.Sequential(
			nn.Linear(emb_dim, emb_dim * 4),
			nn.ReLU(),
			nn.Linear(emb_dim * 4, vocab_size[2]),
		)

		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
		self.ddi_vocab_size = float(self.tensor_ddi_adj.shape[0])
		self._init_weights()

	def _sum_embedding(self, emb: torch.Tensor) -> torch.Tensor:
		return emb.sum(dim=1).unsqueeze(dim=0)

	def _embed_visit(self, visit: Sequence[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
		diag_codes = visit[0]
		proc_codes = visit[1]

		diag_emb = self._sum_embedding(
			self.dropout(
				self.embeddings[0](
					torch.LongTensor(diag_codes).unsqueeze(dim=0).to(self.device)
				)
			)
		)
		proc_emb = self._sum_embedding(
			self.dropout(
				self.embeddings[1](
					torch.LongTensor(proc_codes).unsqueeze(dim=0).to(self.device)
				)
			)
		)
		return diag_emb, proc_emb

	def forward(
		self,
		visits: Sequence[Sequence[Sequence[int]]],
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		if not visits:
			empty = torch.zeros((1, self.prescription_net[-1].out_features), device=self.device)
			zero = torch.tensor(0.0, device=self.device)
			return empty, empty, empty, zero, zero

		diag_emb, proc_emb = self._embed_visit(visits[-1])

		if len(visits) < 2:
			diag_emb_last = torch.zeros_like(diag_emb)
			proc_emb_last = torch.zeros_like(proc_emb)
		else:
			diag_emb_last, proc_emb_last = self._embed_visit(visits[-2])

		health_repr = torch.cat([diag_emb, proc_emb], dim=-1).squeeze(dim=0)
		health_repr_last = torch.cat([diag_emb_last, proc_emb_last], dim=-1).squeeze(dim=0)

		health_rep = self.health_net(health_repr)[-1:, :]
		health_rep_last = self.health_net(health_repr_last)[-1:, :]
		health_residual_rep = health_rep - health_rep_last

		drug_rep = self.prescription_net(health_rep)
		drug_rep_last = self.prescription_net(health_rep_last)
		drug_residual_rep = self.prescription_net(health_residual_rep)

		rec_loss = (
			1.0
			/ self.ddi_vocab_size
			* torch.sum(
				torch.pow(
					torch.sigmoid(drug_rep)
					- torch.sigmoid(drug_rep_last + drug_residual_rep),
					2,
				)
			)
		)

		neg_pred_prob = torch.sigmoid(drug_rep)
		neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
		batch_neg = (
			1.0 / self.ddi_vocab_size * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
		)

		return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

	def _init_weights(self) -> None:
		initrange = 0.1
		for item in self.embeddings:
			item.weight.data.uniform_(-initrange, initrange)


class MICRON(BaseDrugRecommendationModel):
	"""
	MICRON wrapper integrated with the benchmark interfaces.
	"""

	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		emb_dim: int = 64,
		threshold: float = 0.5,
		alpha_current: float = 0.75,
		multilabel_weight: float = 5e-2,
		lambda_bce: float = 0.25,
		lambda_multi: float = 0.25,
		lambda_ddi: float = 0.25,
		lambda_rec: float = 0.25,
		target_ddi: float = 0.08,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.threshold = threshold
		self.ddi_adj = ddi_adj
		self.alpha_current = alpha_current
		self.multilabel_weight = multilabel_weight
		self.lambda_bce = lambda_bce
		self.lambda_multi = lambda_multi
		self.lambda_ddi = lambda_ddi
		self.lambda_rec = lambda_rec
		self.target_ddi = target_ddi

		self.core = MICRONCore(
			vocab_size=vocab_size,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			device=self.device,
		)

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		if not patient:
			empty_logits = torch.empty((0, self.vocab_size[2]), device=self.device)
			empty_scalar = torch.empty((0,), device=self.device)
			empty_bool = torch.empty((0,), dtype=torch.bool, device=self.device)
			return {
				"logits": empty_logits,
				"prev_logits": empty_logits,
				"ddi_loss": empty_scalar,
				"rec_loss": empty_scalar,
				"has_prev": empty_bool,
				"target_cur": empty_logits,
				"target_prev": empty_logits,
			}

		if self.training:
			if len(patient) < 2:
				empty_logits = torch.empty((0, self.vocab_size[2]), device=self.device)
				empty_scalar = torch.empty((0,), device=self.device)
				empty_bool = torch.empty((0,), dtype=torch.bool, device=self.device)
				return {
					"logits": empty_logits,
					"prev_logits": empty_logits,
					"ddi_loss": empty_scalar,
					"rec_loss": empty_scalar,
					"has_prev": empty_bool,
					"target_cur": empty_logits,
					"target_prev": empty_logits,
				}

			logits_list: List[torch.Tensor] = []
			prev_logits_list: List[torch.Tensor] = []
			ddi_loss_list: List[torch.Tensor] = []
			rec_loss_list: List[torch.Tensor] = []
			target_cur_list: List[torch.Tensor] = []
			target_prev_list: List[torch.Tensor] = []

			for adm_idx in range(1, len(patient)):
				seq_input = patient[: adm_idx + 1]
				logits, prev_logits, _, ddi_loss, rec_loss = self.core(seq_input)

				target_cur = torch.zeros((1, self.vocab_size[2]), device=self.device)
				target_prev = torch.zeros((1, self.vocab_size[2]), device=self.device)
				target_cur[0, patient[adm_idx][2]] = 1
				target_prev[0, patient[adm_idx - 1][2]] = 1

				logits_list.append(logits)
				prev_logits_list.append(prev_logits)
				ddi_loss_list.append(ddi_loss.unsqueeze(0))
				rec_loss_list.append(rec_loss.unsqueeze(0))
				target_cur_list.append(target_cur)
				target_prev_list.append(target_prev)

			steps = len(logits_list)
			return {
				"logits": torch.cat(logits_list, dim=0),
				"prev_logits": torch.cat(prev_logits_list, dim=0),
				"ddi_loss": torch.cat(ddi_loss_list, dim=0),
				"rec_loss": torch.cat(rec_loss_list, dim=0),
				"has_prev": torch.ones((steps,), dtype=torch.bool, device=self.device),
				"target_cur": torch.cat(target_cur_list, dim=0),
				"target_prev": torch.cat(target_prev_list, dim=0),
			}

		logits, prev_logits, _, ddi_loss, rec_loss = self.core(patient)
		target_cur = torch.zeros((1, self.vocab_size[2]), device=self.device)
		target_prev = torch.zeros((1, self.vocab_size[2]), device=self.device)
		target_cur[0, patient[-1][2]] = 1
		prev_adm = patient[-2] if len(patient) >= 2 else patient[-1]
		target_prev[0, prev_adm[2]] = 1

		return {
			"logits": logits,
			"prev_logits": prev_logits,
			"ddi_loss": ddi_loss.unsqueeze(0),
			"rec_loss": rec_loss.unsqueeze(0),
			"has_prev": torch.tensor([len(patient) >= 2], dtype=torch.bool, device=self.device),
			"target_cur": target_cur,
			"target_prev": target_prev,
		}

	def decode(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Prediction:
		logits = outputs["logits"]
		target = self.get_target_indices(batch)
		med_size = self.vocab_size[2]
		if logits.numel() == 0:
			return Prediction(
				med_indices=[],
				med_scores=np.zeros((med_size,), dtype=float),
				target=target,
				task="multilabel",
			)
		current_logits = logits[-1:].detach()
		probs = torch.sigmoid(current_logits).cpu().numpy()[0]
		med_indices = np.where(probs >= self.threshold)[0].tolist()
		return Prediction(
			med_indices=list(med_indices),
			med_scores=np.asarray(probs, dtype=float),
			target=target,
			task="multilabel",
			ranked_med_indices=np.argsort(probs)[::-1].tolist(),
		)

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		prev_logits = outputs["prev_logits"]
		ddi_loss = outputs["ddi_loss"]
		rec_loss = outputs["rec_loss"]
		has_prev = outputs["has_prev"]
		targets_cur = outputs["target_cur"]
		targets_prev = outputs["target_prev"]

		if logits.shape[0] == 0:
			return torch.tensor(0.0, device=self.device)
		losses: List[torch.Tensor] = []

		for i in range(logits.shape[0]):
			target_cur = targets_cur[i : i + 1]
			target_prev = targets_prev[i : i + 1]

			loss_bce = self.alpha_current * F.binary_cross_entropy_with_logits(
				logits[i : i + 1],
				target_cur,
			)
			if bool(has_prev[i].item()):
				loss_bce = loss_bce + (1 - self.alpha_current) * F.binary_cross_entropy_with_logits(
					prev_logits[i : i + 1],
					target_prev,
				)

			target_multi_cur = self._build_multilabel_target(target_cur)
			loss_multi = self.alpha_current * F.multilabel_margin_loss(
				torch.sigmoid(logits[i : i + 1]),
				target_multi_cur,
			)
			if bool(has_prev[i].item()):
				target_multi_prev = self._build_multilabel_target(target_prev)
				loss_multi = loss_multi + (1 - self.alpha_current) * F.multilabel_margin_loss(
					torch.sigmoid(prev_logits[i : i + 1]),
					target_multi_prev,
				)

			loss_multi = self.multilabel_weight * loss_multi
			current_ddi_rate = self._ddi_rate_from_logits(logits[i : i + 1])

			if current_ddi_rate > self.target_ddi:
				loss = (
					self.lambda_bce * loss_bce
					+ self.lambda_multi * loss_multi
					+ self.lambda_ddi * ddi_loss[i]
					+ self.lambda_rec * rec_loss[i]
				)
			else:
				loss = (
					self.lambda_bce * loss_bce
					+ self.lambda_multi * loss_multi
					+ self.lambda_rec * rec_loss[i]
				)
			losses.append(loss)

		return torch.stack(losses).mean()

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.shape[0] == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()

	def _build_multilabel_target(self, target_bce: torch.Tensor) -> torch.Tensor:
		target = torch.full_like(target_bce, -1)
		indices = torch.nonzero(target_bce[0], as_tuple=False).squeeze(-1)
		for idx, med_idx in enumerate(indices.tolist()):
			target[0, idx] = med_idx
		return target.long()

	def _ddi_rate_from_logits(self, logits: torch.Tensor) -> float:
		probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
		preds = (probs >= self.threshold).astype(np.int32)
		labels = np.where(preds == 1)[0].tolist()
		if len(labels) < 2:
			return 0.0
		ddi_count = 0
		total_count = 0
		for i, med_i in enumerate(labels):
			for j in range(i + 1, len(labels)):
				med_j = labels[j]
				total_count += 1
				if self.ddi_adj[med_i, med_j] == 1 or self.ddi_adj[med_j, med_i] == 1:
					ddi_count += 1
		if total_count == 0:
			return 0.0
		return ddi_count / total_count
