"""
Evaluation flow and metric aggregation.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

import torch

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel
from drugrec_benchmark.utils.metrics import multi_label_metric, sequence_metric, sequence_output_process

class Evaluator:
	"""
	Evaluator with metric aggregation over unified model predictions.
	"""

	def __init__(self, metrics: Optional[Dict[str, Callable[[Any, Any], float]]] = None) -> None:
		self.metrics = metrics or {}

	def evaluate(
		self,
		model: BaseDrugRecommendationModel,
		data_loader: Iterable[Dict[str, Any]],
		log_interval: Optional[int] = None,
		prefix: str = "test step",
		model_type: Optional[str] = None,
	) -> Dict[str, float]:
		model.eval()
		ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
		smm_record = []
		med_cnt = 0
		visit_cnt = 0
		total_steps = len(data_loader) if hasattr(data_loader, "__len__") else None
		step_count = 0

		with torch.no_grad():
			for batch in data_loader:
				patient = batch.get("visit")
				if not patient:
					continue

				step_count += 1
				outputs = model(batch)
				prediction = model.decode(outputs, batch)
				med_size = int(prediction.med_scores.shape[-1]) if prediction.med_scores.size else int(getattr(model, "vocab_size", [0, 0, 0])[2])
				if med_size <= 0:
					continue

				y_gt_tmp = np.zeros((med_size,), dtype=float)
				valid_target = [idx for idx in prediction.target if 0 <= idx < med_size]
				y_gt_tmp[valid_target] = 1

				pred_set = [idx for idx in prediction.med_indices if 0 <= idx < med_size]
				y_pred_tmp = np.zeros((med_size,), dtype=float)
				y_pred_tmp[pred_set] = 1
				prob = np.asarray(prediction.med_scores, dtype=float)[:med_size]
				if prob.shape[0] < med_size:
					prob = np.pad(prob, (0, med_size - prob.shape[0]))

				if prediction.task == "sequence":
					ranked = prediction.ranked_med_indices or pred_set
					y_pred_label = [sorted([idx for idx in ranked if 0 <= idx < med_size])]
					adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(
						np.array([y_gt_tmp]),
						np.array([y_pred_tmp]),
						np.array([prob]),
						y_pred_label,
					)
					smm_record.append(y_pred_label)
				else:
					y_pred_label = [sorted(pred_set)]
					adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
						np.array([y_gt_tmp]),
						np.array([y_pred_tmp]),
						np.array([prob]),
					)
					smm_record.append(y_pred_label)

				visit_cnt += 1
				med_cnt += len(pred_set)
				ja.append(adm_ja)
				prauc.append(adm_prauc)
				avg_p.append(adm_avg_p)
				avg_r.append(adm_avg_r)
				avg_f1.append(adm_avg_f1)

				if log_interval and step_count % log_interval == 0:
					if total_steps is None:
						message = f"{prefix}: {step_count}"
					else:
						message = f"{prefix}: {step_count} / {total_steps}"
					print(f"\r{message}", end="", flush=True)

		if log_interval:
			print()

		ddi_rate = 0.0
		ddi_adj = model.get_ddi_adj()
		if ddi_adj is not None:
			all_cnt = 0
			ddi_cnt = 0
			for patient in smm_record:
				for adm in patient:
					for i, med_i in enumerate(adm):
						for j, med_j in enumerate(adm):
							if j <= i:
								continue
							all_cnt += 1
							if ddi_adj[med_i, med_j] == 1 or ddi_adj[med_j, med_i] == 1:
								ddi_cnt += 1
			ddi_rate = 0.0 if all_cnt == 0 else ddi_cnt / all_cnt

		return {
			"ddi_rate": float(ddi_rate),
			"jaccard": float(np.mean(ja) if ja else 0.0),
			"prauc": float(np.mean(prauc) if prauc else 0.0),
			"avg_prc": float(np.mean(avg_p) if avg_p else 0.0),
			"avg_recall": float(np.mean(avg_r) if avg_r else 0.0),
			"avg_f1": float(np.mean(avg_f1) if avg_f1 else 0.0),
			"avg_med": float(med_cnt / visit_cnt) if visit_cnt > 0 else 0.0,
		}
