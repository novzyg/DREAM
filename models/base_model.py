"""
Model base class and interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn

from drugrec_benchmark.core.io import Prediction, as_model_output
from drugrec_benchmark.utils.metrics import sequence_output_process


class ModelOutput(TypedDict, total=False):
	logits: torch.Tensor
	seq_logits: List[torch.Tensor]
	ddi_loss: torch.Tensor


class BaseDrugRecommendationModel(nn.Module, ABC):
	"""
	Base class for drug recommendation models.

	Subclasses must implement forward, compute_loss, and predict to provide a
	unified interface for the trainer and evaluator.
	"""

	def __init__(self, device: Optional[torch.device] = None) -> None:
		super().__init__()
		self.device = device if device is not None else torch.device("cpu")
		self.model_type: Optional[str] = None

	@abstractmethod
	def forward(self, batch: Dict[str, Any]) -> ModelOutput:
		"""
		Run forward pass for a batch.

		Args:
			batch: A dict containing model inputs.

		Returns:
			Model outputs (logits or structured outputs).
		"""
		raise NotImplementedError

	@abstractmethod
	def compute_loss(self, outputs: ModelOutput, batch: Dict[str, Any]) -> torch.Tensor:
		"""
		Compute training loss from outputs and batch labels.
		"""
		raise NotImplementedError


	def predict(self, outputs: ModelOutput) -> torch.Tensor:
		"""
		Convert raw outputs into predictions for evaluation.
		"""
		raise NotImplementedError

	def predict_set(self, outputs: ModelOutput) -> List[int]:
		"""
		Return a list of predicted medication indices for a single instance.
		"""
		preds = self.predict(outputs)
		if preds.numel() == 0:
			return []
		return (preds[0].detach().cpu().numpy() == 1).nonzero()[0].tolist()

	def decode(self, outputs: ModelOutput, batch: Dict[str, Any]) -> Prediction:
		"""Decode model-specific outputs into a unified prediction object."""
		model_task = self.model_type or "multilabel"
		task = "sequence" if model_task == "sequence" else "multilabel"
		normalized = as_model_output(outputs, default_task=task)
		sample = batch.get("sample")
		target = list(getattr(sample, "target", batch.get("target", [])) or [])

		if normalized.task == "sequence" or task == "sequence":
			logits = normalized.logits
			if isinstance(logits, torch.Tensor):
				logits_array = logits.detach().cpu().numpy()
			else:
				logits_array = np.asarray(logits)
			med_size = int(getattr(self, "vocab_size", [0, 0, logits_array.shape[-1]])[2])
			if logits_array.size == 0 or med_size <= 0:
				return Prediction(med_indices=[], med_scores=np.zeros((max(med_size, 0),), dtype=float), target=target, task="sequence")
			out_list, ranked = sequence_output_process(logits_array, [med_size, med_size + 1])
			scores = np.mean(logits_array[:, :med_size], axis=0)
			return Prediction(
				med_indices=list(out_list),
				med_scores=np.asarray(scores, dtype=float),
				target=target,
				task="sequence",
				ranked_med_indices=list(ranked),
			)

		logits = normalized.logits
		if not isinstance(logits, torch.Tensor):
			logits = torch.as_tensor(logits, device=self.device)
		if logits.numel() == 0:
			return Prediction(med_indices=[], med_scores=np.zeros((0,), dtype=float), target=target, task="multilabel")
		probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
		med_indices = self.predict_set(outputs)
		ranked = np.argsort(probs)[::-1].tolist()
		return Prediction(
			med_indices=list(med_indices),
			med_scores=np.asarray(probs, dtype=float),
			target=target,
			task="multilabel",
			ranked_med_indices=ranked,
		)

	def get_patient(self, batch: Dict[str, Any]) -> Any:
		"""Return the raw patient visits kept for legacy model cores."""
		return batch.get("visit", [])

	def get_patients(self, batch: Dict[str, Any]) -> List[Any]:
		"""Return one or more raw patient visit sequences from a unified batch."""
		patients = batch.get("visits")
		if patients is not None:
			return list(patients)
		patient = self.get_patient(batch)
		return [patient] if patient else []

	def _coerce_target_rows(self, value: Any) -> List[List[int]]:
		if value is None:
			return []
		if isinstance(value, torch.Tensor):
			value = value.detach().cpu().tolist()
		if isinstance(value, np.ndarray):
			value = value.tolist()
		if not isinstance(value, (list, tuple)):
			return [[int(value)]]
		if len(value) == 0:
			return [[]]
		first = value[0]
		if isinstance(first, torch.Tensor):
			first = first.detach().cpu().tolist()
		if isinstance(first, np.ndarray):
			first = first.tolist()
		if isinstance(first, (list, tuple, set)):
			return [list(map(int, row)) for row in value]
		return [list(map(int, value))]

	def get_target_indices_list(self, batch: Dict[str, Any]) -> List[List[int]]:
		"""Return target medication indices for every sample in a batch."""
		samples = batch.get("samples")
		if isinstance(samples, list):
			return [list(getattr(sample, "target", []) or []) for sample in samples]
		sample = batch.get("sample")
		if isinstance(sample, list):
			return [list(getattr(item, "target", []) or []) for item in sample]
		if sample is not None and getattr(sample, "target", None) is not None:
			return [list(sample.target)]

		for key in ("target_indices", "target"):
			rows = self._coerce_target_rows(batch.get(key))
			if rows:
				return rows

		patients = self.get_patients(batch)
		if patients:
			return [list(patient[-1][2]) if patient else [] for patient in patients]
		return []

	def get_target_indices(self, batch: Dict[str, Any]) -> List[int]:
		"""Return target medication indices from the first sample in a batch."""
		rows = self.get_target_indices_list(batch)
		return rows[0] if rows else []

	def _med_vocab_size(self) -> int:
		vocab_size = getattr(self, "vocab_size", None)
		if vocab_size is not None and len(vocab_size) > 2:
			return int(vocab_size[2])
		return int(getattr(self, "n_drug", 0))

	def build_target(self, batch: Dict[str, Any], *, batch_dim: bool = True) -> torch.Tensor:
		"""Build multi-hot medication targets from the unified batch."""
		med_size = self._med_vocab_size()
		if med_size <= 0:
			shape = (0, 0) if batch_dim else (0,)
			return torch.empty(shape, device=self.device)
		target_rows = self.get_target_indices_list(batch)
		if not target_rows:
			shape = (0, med_size) if batch_dim else (med_size,)
			return torch.zeros(shape, dtype=torch.float32, device=self.device)
		target = torch.zeros((len(target_rows), med_size), dtype=torch.float32, device=self.device)
		for row, row_indices in enumerate(target_rows):
			indices = [idx for idx in row_indices if 0 <= idx < med_size]
			if indices:
				target[row, indices] = 1.0
		if batch_dim:
			return target
		return target[0] if target.shape[0] else torch.zeros((med_size,), dtype=torch.float32, device=self.device)

	def build_multilabel_target(self, target_bce: torch.Tensor) -> torch.Tensor:
		"""Convert multi-hot BCE targets to multilabel margin-loss targets."""
		target = torch.full_like(target_bce, -1)
		if target_bce.numel() == 0:
			return target.long()
		for row in range(target_bce.shape[0]):
			indices = torch.nonzero(target_bce[row], as_tuple=False).squeeze(-1)
			for idx, med_idx in enumerate(indices.tolist()):
				target[row, idx] = med_idx
		return target.long()

	def get_ddi_adj(self) -> Optional[Any]:
		return getattr(self, "ddi_adj", None)

	def move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Move tensors in the batch to the model device.
		"""
		moved: Dict[str, Any] = {}
		for key, value in batch.items():
			if isinstance(value, torch.Tensor):
				moved[key] = value.to(self.device)
			else:
				moved[key] = value
		return moved

	def save(self, path: str) -> None:
		"""
		Save model weights.
		"""
		torch.save(self.state_dict(), path)

	def load(self, path: str, map_location: Optional[str] = None) -> None:
		"""
		Load model weights.
		"""
		state = torch.load(path, map_location=map_location)
		self.load_state_dict(state)

