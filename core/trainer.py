"""
Training loop and checkpointing.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional

import torch

from drugrec_benchmark.models.base_model import BaseDrugRecommendationModel

class Trainer:
	"""
	Minimal trainer skeleton for supervised learning.
	"""

	def __init__(
		self,
		model: BaseDrugRecommendationModel,
		optimizer: torch.optim.Optimizer,
		device: Optional[torch.device] = None,
		grad_clip: Optional[float] = None,
	) -> None:
		self.model = model
		self.optimizer = optimizer
		self.device = device if device is not None else model.device
		self.grad_clip = grad_clip
		self.model.to(self.device)

	def train_epoch(
		self,
		data_loader: Iterable[Dict[str, Any]],
		epoch: Optional[int] = None,
		log_interval: Optional[int] = None,
	) -> Dict[str, float]:
		self.model.train()
		total_loss = 0.0
		step_count = 0
		total_steps = len(data_loader) if hasattr(data_loader, "__len__") else None
		for batch in data_loader:
			step_count += 1
			outputs = self.model(batch)  
			loss = self.model.compute_loss(outputs, batch)
			self.optimizer.zero_grad(set_to_none=True)
			loss.backward()
			if self.grad_clip is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
			self.optimizer.step()

			total_loss += float(loss.detach().cpu().item())

			if log_interval and step_count % log_interval == 0:
				prefix = f"epoch {epoch + 1} " if epoch is not None else ""
				if total_steps is None:
					message = f"train {prefix}step {step_count} loss {loss.item():.4f}"
				else:
					message = (
						f"train {prefix}step {step_count}/{total_steps} loss {loss.item():.4f}"
					)
				print(f"\r{message}", end="", flush=True)

		avg_loss = total_loss / max(step_count, 1)
		return {"loss": avg_loss}

	def fit(
		self,
		train_loader: Iterable[Dict[str, Any]],
		model_type: Optional[str] = None,
		epochs: int=50,
		val_loader: Optional[Iterable[Dict[str, Any]]] = None,
		evaluator: Optional[Any] = None,
		checkpoint_dir: Optional[str] = None,
		metric_key: Optional[str] = None,
		early_stop_patience: Optional[int] = None,
		log_interval: Optional[int] = None,
	) -> Dict[str, Any]:
		history: Dict[str, list] = {"train": [], "val": []}
		best_metric = float("-inf")
		best_checkpoint: Optional[str] = None
		no_improve_epochs = 0

		if checkpoint_dir:
			os.makedirs(checkpoint_dir, exist_ok=True)
		for epoch in range(epochs):
			print(f"\nepoch {epoch + 1} --------------------------")
			train_start = time.time()
			train_metrics = self.train_epoch(
				train_loader,
				epoch=epoch,
				log_interval=log_interval,
			)
			train_time = time.time() - train_start
			history["train"].append(train_metrics)

			val_metrics = None
			if val_loader is not None and evaluator is not None:
				eval_start = time.time()
				val_metrics = evaluator.evaluate(
					self.model,
					val_loader,
					log_interval=log_interval,
					prefix="test step",
				)
				eval_time = time.time() - eval_start
				history["val"].append(val_metrics)

			if val_metrics:
				ddi_rate = val_metrics.get("ddi_rate", 0.0)
				ja = val_metrics.get("jaccard", 0.0)
				prauc = val_metrics.get("prauc", 0.0)
				avg_prc = val_metrics.get("avg_prc", 0.0)
				avg_recall = val_metrics.get("avg_recall", 0.0)
				avg_f1 = val_metrics.get("avg_f1", 0.0)
				avg_med = val_metrics.get("avg_med", 0.0)
				print(
					"DDI Rate: {:.5}, Jaccard: {:.4},  PRAUC: {:.4}, "
					"AVG_PRC: {:.4}, AVG_RECALL: {:.4}, AVG_F1: {:.4}, AVG_MED: {:.2f}".format(
						ddi_rate,
						ja,
						prauc,
						avg_prc,
						avg_recall,
						avg_f1,
						avg_med,
					)
				)
				print(
					"training time: {:.4f}, test time: {:.4f}".format(
						train_time,
						eval_time,
					)
				)

			if checkpoint_dir:
				metric_value = None
				if metric_key and val_metrics and metric_key in val_metrics:
					metric_value = val_metrics[metric_key]
				if metric_value is None:
					filename = f"epoch_{epoch + 1:03d}.pth"
				else:
					filename = f"epoch_{epoch + 1:03d}_{metric_key}_{metric_value:.4f}.pth"
				checkpoint_path = os.path.join(checkpoint_dir, filename)
				self.model.save(checkpoint_path)

				best_file = os.path.join(checkpoint_dir, "best_checkpoint.txt")
				if metric_value is not None and metric_value > best_metric:
					best_metric = metric_value
					best_checkpoint = checkpoint_path
					with open(best_file, "w", encoding="utf-8") as handle:
						handle.write(best_checkpoint)
					no_improve_epochs = 0
				else:
					if best_checkpoint is None:
						best_checkpoint = checkpoint_path
						with open(best_file, "w", encoding="utf-8") as handle:
							handle.write(best_checkpoint)
					no_improve_epochs += 1

				if early_stop_patience is not None and no_improve_epochs >= early_stop_patience:
					print(
						"Early stopping: no improvement in {} epochs.".format(
							early_stop_patience
						)
					)
					break

			if best_checkpoint:
				best_epoch = int(best_checkpoint.split("epoch_")[-1].split("_")[0]) - 1
				print(f"best_epoch: {best_epoch}")

			on_epoch_end = getattr(self.model, "on_epoch_end", None)
			if callable(on_epoch_end):
				on_epoch_end()

		return {
			"history": history,
			"best_checkpoint": best_checkpoint,
			"best_metric": best_metric if best_checkpoint else None,
		}
