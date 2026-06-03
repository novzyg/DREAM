"""Unified model input/output structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class Visit:
	diagnoses: List[int]
	procedures: List[int]
	medications: List[int]


@dataclass
class PatientSample:
	visits: List[Visit]
	target: List[int]
	history: List[Visit]
	patient_id: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBatch:
	sample: PatientSample
	raw_visit: Any
	target_indices: List[int]
	target_multi_hot: Optional[torch.Tensor] = None


@dataclass
class DrugRecModelOutput:
	task: str
	logits: Optional[Any] = None
	sequence_logits: Optional[Any] = None
	loss_terms: Dict[str, torch.Tensor] = field(default_factory=dict)
	extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
	med_indices: List[int]
	med_scores: np.ndarray
	target: List[int]
	task: str = "multilabel"
	ranked_med_indices: List[int] = field(default_factory=list)
	extra: Dict[str, Any] = field(default_factory=dict)


def visit_from_raw(raw_visit: Sequence[Sequence[int]]) -> Visit:
	diagnoses = list(raw_visit[0]) if len(raw_visit) > 0 else []
	procedures = list(raw_visit[1]) if len(raw_visit) > 1 else []
	medications = list(raw_visit[2]) if len(raw_visit) > 2 else []
	return Visit(diagnoses=diagnoses, procedures=procedures, medications=medications)


def patient_sample_from_raw(patient: Sequence[Sequence[Sequence[int]]]) -> PatientSample:
	visits = [visit_from_raw(adm) for adm in patient]
	target = visits[-1].medications if visits else []
	history = visits[:-1] if visits else []
	return PatientSample(visits=visits, target=list(target), history=history)


def target_multi_hot(target: Sequence[int], med_vocab_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
	labels = torch.zeros((med_vocab_size,), dtype=torch.float32, device=device)
	if target:
		labels[list(target)] = 1.0
	return labels


def as_model_output(output: Any, default_task: str = "multilabel") -> DrugRecModelOutput:
	if isinstance(output, DrugRecModelOutput):
		return output
	if isinstance(output, dict):
		task = str(output.get("task") or default_task)
		loss_terms = output.get("loss_terms") or {}
		extra = {
			key: value
			for key, value in output.items()
			if key not in {"task", "logits", "seq_logits", "sequence_logits", "loss_terms"}
		}
		return DrugRecModelOutput(
			task=task,
			logits=output.get("logits"),
			sequence_logits=output.get("sequence_logits", output.get("seq_logits")),
			loss_terms=loss_terms,
			extra=extra,
		)
	return DrugRecModelOutput(task=default_task, logits=output)
