"""
Train and evaluate pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from drugrec_benchmark.scripts.test import run_test
from drugrec_benchmark.scripts.train import run_train
from drugrec_benchmark.utils.load_config import load_config, resolve_seeds


def _extract_single_result(output: object) -> Dict[str, object]:
	if isinstance(output, dict) and "payload" in output:
		status = int(output.get("status", 1))
		if status != 0:
			error = str(output.get("error") or "sub-command failed")
			raise RuntimeError(error)
		payload = output.get("payload")
		if isinstance(payload, dict):
			result = payload.get("result")
			if isinstance(result, dict):
				return result
			results = payload.get("results")
			if isinstance(results, list) and results and isinstance(results[0], dict):
				return results[0]
		return {}

	if isinstance(output, list) and output and isinstance(output[0], dict):
		return output[0]
	if isinstance(output, dict):
		return output
	return {}



def _extract_results(output: object) -> List[Dict[str, object]]:
	if isinstance(output, dict) and "payload" in output:
		status = int(output.get("status", 1))
		if status != 0:
			error = str(output.get("error") or "sub-command failed")
			raise RuntimeError(error)
		payload = output.get("payload")
		if isinstance(payload, dict):
			results = payload.get("results")
			if isinstance(results, list):
				return [item for item in results if isinstance(item, dict)]
			result = payload.get("result")
			if isinstance(result, dict):
				return [result]
		return []

	if isinstance(output, list):
		return [item for item in output if isinstance(item, dict)]
	if isinstance(output, dict):
		return [output]
	return []

def _write_cross_seed_summary(
	model: str,
	dataset: str,
	seed_results: List[Dict[str, object]],
) -> str:
	reports_dir = os.path.join(
		"drugrec_benchmark",
		"results",
		model,
		dataset,
		"reports",
	)
	os.makedirs(reports_dir, exist_ok=True)

	metric_values: Dict[str, List[float]] = {}
	for item in seed_results:
		metrics = item.get("metrics", {})
		if not isinstance(metrics, dict):
			continue
		for key, value in metrics.items():
			if isinstance(value, (int, float)):
				metric_values.setdefault(key, []).append(float(value))

	summary = {}
	for key, values in metric_values.items():
		array = np.asarray(values, dtype=float)
		summary[key] = {
			"mean": float(np.mean(array)),
			"std": float(np.std(array)),
			"n": int(array.size),
		}

	report = {
		"model": model,
		"dataset": dataset,
		"seeds": [item.get("seed") for item in seed_results],
		"per_seed": seed_results,
		"summary": summary,
	}

	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
	report_path = os.path.join(reports_dir, f"cross_seed_summary_{timestamp}.json")
	with open(report_path, "w", encoding="utf-8") as handle:
		json.dump(report, handle, indent=2)
	return report_path


def _find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
	best_file = os.path.join(checkpoint_dir, "best_checkpoint.txt")
	if os.path.isfile(best_file):
		with open(best_file, "r", encoding="utf-8") as handle:
			path = handle.read().strip()
			return path or None

	if not os.path.isdir(checkpoint_dir):
		return None

	best_score = float("-inf")
	best_path = None
	candidates = []
	for name in os.listdir(checkpoint_dir):
		if not name.endswith(".pth"):
			continue
		path = os.path.join(checkpoint_dir, name)
		candidates.append(path)
		parts = name.split("_")
		try:
			score = float(parts[-1].replace(".pth", ""))
			if score > best_score:
				best_score = score
				best_path = path
		except Exception:
			continue
	if best_path is None and candidates:
		best_path = max(candidates, key=os.path.getmtime)
	return best_path


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
	try:
		config = load_config(args.model, override_config=None)
		seed_override = getattr(args, "seed", None)
		expected_seeds = [int(seed_override)] if seed_override is not None else resolve_seeds(config)
		seed_results: List[Dict[str, object]] = []

		train_args = argparse.Namespace(**vars(args))
		train_args.checkpoint = None
		train_args.run_dir = None
		train_output = run_train(train_args)
		train_results = _extract_results(train_output)
		results_by_seed = {
			int(item["seed"]): item
			for item in train_results
			if item.get("seed") is not None
		}

		for seed in expected_seeds:
			result = results_by_seed.get(seed)
			if result is None:
				raise RuntimeError(f"Training result not found for seed {seed}.")

			run_dir = None
			models_dir = None
			best_checkpoint = None
			if isinstance(result, dict):
				run_dir = result.get("run_dir")
				models_dir = result.get("models_dir")
				best_checkpoint = result.get("best_checkpoint")
			if not best_checkpoint:
				search_dir = models_dir or os.path.join(
					"drugrec_benchmark",
					"results",
					"models",
					args.model,
					args.dataset,
				)
				best_checkpoint = _find_best_checkpoint(search_dir)

			if not best_checkpoint:
				raise FileNotFoundError(f"Best checkpoint not found after training for seed {seed}.")

			seed_args = argparse.Namespace(**vars(args))
			seed_args.seed = seed
			seed_args.checkpoint = best_checkpoint
			if run_dir:
				seed_args.run_dir = run_dir
			test_output = run_test(seed_args)
			metrics = _extract_single_result(test_output)
			metadata_keys = {"seed", "checkpoint", "run_dir", "metrics_path", "log_path"}
			metrics_only = {
				key: value
				for key, value in metrics.items()
				if key not in metadata_keys and isinstance(value, (int, float))
			}
			seed_results.append(
				{
					"seed": seed,
					"run_dir": run_dir,
					"checkpoint": best_checkpoint,
					"metrics": metrics_only,
					"test_result": metrics,
				}
			)

		report_path = _write_cross_seed_summary(args.model, args.dataset, seed_results)
		print(f"cross-seed summary saved to: {report_path}")
		return {
			"status": 0,
			"error": None,
			"payload": {
				"report_path": report_path,
				"seed_results": seed_results,
			},
		}
	except Exception as exc:
		return {
			"status": 1,
			"error": str(exc),
			"payload": None,
		}


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--seed", type=int)
	parser.add_argument("--cuda", type=int, default=0)
	parser.add_argument("--batch-size", type=int, help="Override training batch size for models with batched training support")
	args = parser.parse_args()
	output = run_pipeline(args)
	status = int(output.get("status", 1))
	if status != 0 and output.get("error"):
		print(output["error"])
	return status


if __name__ == "__main__":
	raise SystemExit(main())
