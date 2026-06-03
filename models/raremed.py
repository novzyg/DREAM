"""
RAREMed model integration for drugrec_benchmark.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel


class LearnablePositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 1000) -> None:
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.embeddings = nn.Embedding(max_len, d_model)
		self.embeddings.weight.data.uniform_(-0.1, 0.1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		pos = torch.arange(0, x.size(0), device=x.device).long()
		pos_emb = self.embeddings(pos).unsqueeze(1).expand_as(x)
		return self.dropout(x + pos_emb)


class RareMedCore(nn.Module):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		emb_dim: int = 512,
		encoder_layers: int = 3,
		nhead: int = 4,
		dropout: float = 0.3,
		patient_separate: bool = False,
		ddi_scale: float = 5e-4,
		device: torch.device = torch.device("cpu"),
	) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.emb_dim = emb_dim
		self.patient_separate = patient_separate
		self.ddi_scale = ddi_scale
		self.device = device

		self.special_tokens = {
			"CLS": torch.LongTensor([0]).to(self.device),
			"SEP": torch.LongTensor([1]).to(self.device),
		}
		self.segment_embedding = nn.Embedding(2, emb_dim)

		if patient_separate:
			half_dim = emb_dim // 2
			if emb_dim % 2 != 0:
				raise ValueError("RAREMed patient_separate=True requires an even emb_dim.")
			self.embeddings = nn.ModuleList(
				[nn.Embedding(vocab_size[i], half_dim) for i in range(2)]
			)
			self.special_embeddings = nn.Embedding(2, half_dim)
			self.transformer_disease = nn.TransformerEncoder(
				nn.TransformerEncoderLayer(
					d_model=half_dim,
					nhead=nhead,
					dropout=dropout,
					batch_first=False,
				),
				num_layers=encoder_layers,
			)
			self.transformer_procedure = nn.TransformerEncoder(
				nn.TransformerEncoderLayer(
					d_model=half_dim,
					nhead=nhead,
					dropout=dropout,
					batch_first=False,
				),
				num_layers=encoder_layers,
			)
			self.positional_embedding_layer_disease = LearnablePositionalEncoding(
				d_model=half_dim
			)
			self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
				d_model=half_dim
			)
		else:
			self.embeddings = nn.ModuleList(
				[nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
			)
			self.special_embeddings = nn.Embedding(2, emb_dim)
			self.transformer_visit = nn.TransformerEncoder(
				nn.TransformerEncoderLayer(
					d_model=emb_dim,
					nhead=nhead,
					dropout=dropout,
					batch_first=False,
				),
				num_layers=encoder_layers,
			)
			self.positional_embedding_layer_disease = LearnablePositionalEncoding(
				d_model=emb_dim
			)
			self.positional_embedding_layer_procedure = LearnablePositionalEncoding(
				d_model=emb_dim
			)

		self.cls_final = nn.Linear(emb_dim, vocab_size[2])
		self.tensor_ddi_adj = torch.as_tensor(ddi_adj, dtype=torch.float32, device=device)
		self._init_weights()

	def _init_weights(self) -> None:
		initrange = 0.1
		for embedding in self.embeddings:
			embedding.weight.data.uniform_(-initrange, initrange)
		self.segment_embedding.weight.data.uniform_(-initrange, initrange)
		self.special_embeddings.weight.data.uniform_(-initrange, initrange)

	def _encode_visit_unified(self, visit: Sequence[Sequence[int]]) -> torch.Tensor:
		diseases = list(visit[0]) if len(visit) > 0 else []
		procedures = list(visit[1]) if len(visit) > 1 else []

		disease_tensor = torch.LongTensor(diseases).unsqueeze(1).to(self.device)
		procedure_tensor = torch.LongTensor(procedures).unsqueeze(1).to(self.device)
		disease_embedding = self.embeddings[0](disease_tensor)
		procedure_embedding = self.embeddings[1](procedure_tensor)

		cls_embedding = self.special_embeddings(self.special_tokens["CLS"]).unsqueeze(1)
		sep_embedding = self.special_embeddings(self.special_tokens["SEP"]).unsqueeze(1)
		disease_embedding = torch.cat((cls_embedding, disease_embedding), dim=0)
		procedure_embedding = torch.cat((sep_embedding, procedure_embedding), dim=0)

		disease_embedding = self.positional_embedding_layer_disease(disease_embedding)
		procedure_embedding = self.positional_embedding_layer_procedure(procedure_embedding)
		combined_embedding = torch.cat((disease_embedding, procedure_embedding), dim=0)

		segments = torch.tensor(
			[0] * (len(diseases) + 2) + [1] * len(procedures),
			dtype=torch.long,
			device=self.device,
		)
		input_embedding = combined_embedding + self.segment_embedding(segments).unsqueeze(1)
		visit_representation = self.transformer_visit(input_embedding)[0]
		return visit_representation.reshape(1, -1)

	def _encode_visit_separate(self, visit: Sequence[Sequence[int]]) -> torch.Tensor:
		diseases = list(visit[0]) if len(visit) > 0 else []
		procedures = list(visit[1]) if len(visit) > 1 else []

		disease_tensor = torch.LongTensor(diseases).unsqueeze(1).to(self.device)
		procedure_tensor = torch.LongTensor(procedures).unsqueeze(1).to(self.device)
		disease_embedding = self.embeddings[0](disease_tensor)
		procedure_embedding = self.embeddings[1](procedure_tensor)

		cls_embedding_dis = self.special_embeddings(self.special_tokens["CLS"]).unsqueeze(1)
		cls_embedding_pro = self.special_embeddings(self.special_tokens["SEP"]).unsqueeze(1)
		disease_embedding = torch.cat((cls_embedding_dis, disease_embedding), dim=0)
		procedure_embedding = torch.cat((cls_embedding_pro, procedure_embedding), dim=0)

		disease_embedding = self.positional_embedding_layer_disease(disease_embedding)
		procedure_embedding = self.positional_embedding_layer_procedure(procedure_embedding)
		disease_repr = self.transformer_disease(disease_embedding)[0].mean(dim=0)
		procedure_repr = self.transformer_procedure(procedure_embedding)[0].mean(dim=0)
		return torch.cat((disease_repr, procedure_repr), dim=-1)

	def encode_visits(self, visits: Sequence[Sequence[Sequence[int]]]) -> torch.Tensor:
		if not visits:
			return torch.zeros((0, self.emb_dim), device=self.device)
		encoded: List[torch.Tensor] = []
		for visit in visits:
			if self.patient_separate:
				encoded.append(self._encode_visit_separate(visit))
			else:
				encoded.append(self._encode_visit_unified(visit))
		return torch.cat(encoded, dim=0)

	def forward(self, visits: Sequence[Sequence[Sequence[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
		if not visits:
			empty = torch.empty((0, self.vocab_size[2]), device=self.device)
			return empty, torch.tensor(0.0, device=self.device)

		patient_repr = self.encode_visits(visits)
		logits = self.cls_final(patient_repr)
		pred_prob = torch.sigmoid(logits)
		pair_prob = torch.matmul(pred_prob.t(), pred_prob)
		ddi_loss = self.ddi_scale * pair_prob.mul(self.tensor_ddi_adj).sum()
		return logits, ddi_loss


class RareMed(BaseDrugRecommendationModel):
	def __init__(
		self,
		vocab_size: Tuple[int, int, int],
		ddi_adj: np.ndarray,
		emb_dim: int = 512,
		encoder_layers: int = 3,
		nhead: int = 4,
		dropout: float = 0.3,
		threshold: float = 0.5,
		weight_multi: float = 0.005,
		weight_ddi: float = 0.1,
		ddi_scale: float = 5e-4,
		patient_separate: bool = False,
		train_all_visits: bool = False,
		device: Optional[torch.device] = None,
	) -> None:
		super().__init__(device=device)
		self.model_type = "multilabel"
		self.vocab_size = vocab_size
		self.ddi_adj = ddi_adj
		self.threshold = threshold
		self.weight_multi = weight_multi
		self.weight_ddi = weight_ddi
		self.train_all_visits = train_all_visits

		self.core = RareMedCore(
			vocab_size=vocab_size,
			ddi_adj=ddi_adj,
			emb_dim=emb_dim,
			encoder_layers=encoder_layers,
			nhead=nhead,
			dropout=dropout,
			patient_separate=patient_separate,
			ddi_scale=ddi_scale,
			device=self.device,
		)

	def _selected_visits(self, patient: Sequence[Sequence[Sequence[int]]]) -> Sequence[Sequence[Sequence[int]]]:
		if not patient:
			return []
		if self.training and self.train_all_visits:
			return patient
		return [patient[-1]]

	def _build_targets_for_visits(
		self,
		visits: Sequence[Sequence[Sequence[int]]],
	) -> torch.Tensor:
		targets = torch.zeros((len(visits), self.vocab_size[2]), device=self.device)
		for row, visit in enumerate(visits):
			medications = list(visit[2]) if len(visit) > 2 else []
			valid = [idx for idx in medications if 0 <= idx < self.vocab_size[2]]
			if valid:
				targets[row, valid] = 1.0
		return targets

	def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
		patient = self.get_patient(batch)
		visits = self._selected_visits(patient)
		logits, ddi_loss = self.core(visits)
		targets = self._build_targets_for_visits(visits)
		return {
			"logits": logits,
			"ddi_loss": ddi_loss.unsqueeze(0),
			"targets": targets,
		}

	def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.numel() == 0:
			return torch.tensor(0.0, device=self.device)
		targets = outputs["targets"]
		loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
		loss_multi = F.multilabel_margin_loss(torch.sigmoid(logits), self.build_multilabel_target(targets))
		loss_ddi = outputs["ddi_loss"][0]
		return (1.0 - self.weight_multi) * loss_bce + self.weight_multi * loss_multi + self.weight_ddi * loss_ddi

	def predict(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
		logits = outputs["logits"]
		if logits.numel() == 0:
			return torch.empty_like(logits)
		probs = torch.sigmoid(logits)
		return (probs >= self.threshold).float()
