"""
Dataset helper utilities.
"""
from __future__ import annotations

from collections import defaultdict
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

from drugrec_benchmark.core.io import patient_sample_from_raw

import numpy as np
import torch


def load_pickle(path: str) -> Any:
	try:
		import dill
		with open(path, "rb") as handle:
			try:
				return dill.load(handle)
			except AttributeError as exc:
				error_text = str(exc)
				if "Voc" not in error_text:
						raise

				candidates = ["__main__", "__mp_main__"]
				for module_name in candidates:
					module = sys.modules.get(module_name)
					if module is None:
						continue
					if not hasattr(module, "Voc"):
						class Voc:  # type: ignore[unused-class]
							pass
						setattr(module, "Voc", Voc)

				handle.seek(0)
				return dill.load(handle)
	except ImportError:
		import pickle
		with open(path, "rb") as handle:
			return pickle.load(handle)


def split_records(
	records: List[Any],
	train_ratio: float,
	val_ratio: float,
	seed: int,
) -> Tuple[List[Any], List[Any], List[Any]]:
	rng = np.random.RandomState(seed)
	indices = np.arange(len(records))
	rng.shuffle(indices)

	train_end = int(len(records) * train_ratio)
	val_end = train_end + int(len(records) * val_ratio)

	train_idx = indices[:train_end]
	val_idx = indices[train_end:val_end]
	test_idx = indices[val_end:]

	train_records = [records[i] for i in train_idx]
	val_records = [records[i] for i in val_idx]
	test_records = [records[i] for i in test_idx]


	# debug

	# train_records = train_records[:100]
	# val_records = val_records[:100]

	return train_records, val_records, test_records

"""
Dataset loading and batching.
"""


# def build_labels(visits: Sequence[Sequence[Sequence[int]]], med_vocab_size: int) -> torch.Tensor:
# 	labels: List[torch.Tensor] = []
# 	for patient in visits:
# 		if not patient:
# 			continue
# 		last_adm = patient[-1]
# 		target = torch.zeros((med_vocab_size,), dtype=torch.float32)
# 		target[last_adm[2]] = 1
# 		labels.append(target)
# 	if not labels:
# 		return torch.empty((0, med_vocab_size), dtype=torch.float32)
# 	return torch.stack(labels, dim=0)


def build_batches(records: Sequence[Any], batch_size: int = 1) -> List[Dict[str, Any]]:
	batches: List[Dict[str, Any]] = []
	batch_size = max(1, int(batch_size or 1))
	if batch_size == 1:
		for visit in records:
			sample = patient_sample_from_raw(visit)
			batches.append(
				{
					"visit": visit,
					"sample": sample,
					"target": sample.target,
					"target_indices": sample.target,
					"batch_size": 1,
				}
			)
		return batches

	for start in range(0, len(records), batch_size):
		visits = list(records[start : start + batch_size])
		samples = [patient_sample_from_raw(visit) for visit in visits]
		target_indices = [sample.target for sample in samples]
		batches.append(
			{
				"visit": visits[0] if len(visits) == 1 else visits,
				"visits": visits,
				"sample": samples[0] if len(samples) == 1 else samples,
				"samples": samples,
				"target": target_indices[0] if len(target_indices) == 1 else target_indices,
				"target_indices": target_indices[0] if len(target_indices) == 1 else target_indices,
				"batch_size": len(visits),
			}
		)
	return batches
