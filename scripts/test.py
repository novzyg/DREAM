"""
Evaluate a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from drugrec_benchmark.utils.dataset_utils import load_pickle, split_records, build_batches
from drugrec_benchmark.core.evaluator import Evaluator
from drugrec_benchmark.utils.load_config import load_config, resolve_seeds, set_seed
from drugrec_benchmark.models.registry import build_model
from drugrec_benchmark.utils.logs import make_run_dirs


def _resolve_run_seeds(config: Dict[str, object], args: argparse.Namespace) -> List[int]:
	seed_override = getattr(args, "seed", None)
	if seed_override is None:
		return resolve_seeds(config)
	return [int(seed_override)]


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
	for name in os.listdir(checkpoint_dir):
		if not name.endswith(".pth"):
			continue
		parts = name.split("_")
		try:
			score = float(parts[-1].replace(".pth", ""))
			if score > best_score:
				best_score = score
				best_path = os.path.join(checkpoint_dir, name)
		except Exception:
			continue
	return best_path


def _find_latest_run_dir(seed_root: str) -> Optional[str]:
	if not os.path.isdir(seed_root):
		return None
	entries = [
		os.path.join(seed_root, name)
		for name in os.listdir(seed_root)
		if os.path.isdir(os.path.join(seed_root, name))
	]
	if not entries:
		return None
	entries.sort(key=lambda path: os.path.getmtime(path), reverse=True)
	return entries[0]


def _write_metrics_report(metrics: Dict[str, float], path: str) -> None:
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(metrics, handle, indent=2)


def _write_test_context(context: Dict[str, Any], path: str) -> None:
	with open(path, "w", encoding="utf-8") as handle:
		json.dump(context, handle, indent=2)


def _timestamp() -> str:
	return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _infer_run_dir_from_checkpoint(checkpoint: Optional[str]) -> Optional[str]:
	if not checkpoint:
		return None
	checkpoint_dir = os.path.dirname(checkpoint)
	if os.path.basename(checkpoint_dir) != "models":
		return None
	run_dir = os.path.dirname(checkpoint_dir)
	if not run_dir or run_dir == checkpoint_dir:
		return None
	return run_dir if os.path.isdir(run_dir) else None


def _extract_seed_from_path(path: Optional[str]) -> Optional[int]:
	if not path:
		return None
	for part in os.path.normpath(path).split(os.sep):
		if not part.startswith("seed_"):
			continue
		try:
			return int(part.split("seed_", 1)[1])
		except ValueError:
			return None
	return None


class _Tee:
	def __init__(self, console, file_stream) -> None:
		self._console = console
		self._file = file_stream

	def write(self, data: str) -> int:
		self._console.write(data)
		self._console.flush()
		if "\r" not in data:
			self._file.write(data)
			self._file.flush()
		return len(data)

	def flush(self) -> None:
		self._console.flush()
		self._file.flush()

	def isatty(self) -> bool:
		return getattr(self._console, "isatty", lambda: False)()

	def fileno(self) -> int:
		return self._console.fileno()


def run_test(args: argparse.Namespace) -> Dict[str, Any]:
	config = load_config(args.model, override_config=None)
	config["dataset"] = config.get("dataset", {})
	config["dataset"]["name"] = args.dataset

	data_root = config.get("paths", {}).get("data_root", "drugrec_benchmark/data")
	template = config.get("paths", {}).get("dataset_dir_template", "{data_root}/{dataset}")
	config["dataset"]["data_dir"] = template.format(
		data_root=data_root,
		dataset=args.dataset,
	)

	data_dir = config["dataset"]["data_dir"]
	records_path = os.path.join(data_dir, config["dataset"]["records_file"])
	records = load_pickle(records_path)

	train_ratio = config.get("split", {}).get("train_ratio", 0.66)
	val_ratio = config.get("split", {}).get("val_ratio", 0.17)
	seeds = _resolve_run_seeds(config, args)
	initial_checkpoint = getattr(args, "checkpoint", None)
	if initial_checkpoint is not None and len(seeds) > 1:
		raise ValueError(
			"A single --checkpoint cannot be used for multiple seeds. "
			"Run without --checkpoint for per-seed auto-discovery, or pass a single --seed."
		)

	use_cuda = config.get("device", {}).get("use_cuda", True)
	device = torch.device(
		f"cuda:{args.cuda}" if use_cuda and torch.cuda.is_available() else "cpu"
	)

	metrics = config.get("evaluation", {}).get("metrics", [])
	evaluator = Evaluator(metrics=metrics) if metrics else Evaluator(metrics={})
	log_interval = config.get("logging", {}).get("log_interval", 50)

	results_list: List[Dict[str, Any]] = []
	for seed in seeds:
		set_seed(seed)

		model, meta = build_model(config["model"]["name"], config, device)
		_ = meta["vocab_size"]

		checkpoint = getattr(args, "checkpoint", None)
		run_dir = getattr(args, "run_dir", None)
		seed_run_dir = None
		checkpoint_source = "argument" if checkpoint else "auto"

		if run_dir:
			if os.path.isdir(os.path.join(run_dir, "models")):
				seed_run_dir = run_dir
			elif os.path.isdir(os.path.join(run_dir, f"seed_{seed}")):
				seed_run_dir = _find_latest_run_dir(os.path.join(run_dir, f"seed_{seed}"))

		if checkpoint is not None and seed_run_dir is None:
			seed_run_dir = _infer_run_dir_from_checkpoint(checkpoint)

		checkpoint_seed = _extract_seed_from_path(seed_run_dir) or _extract_seed_from_path(checkpoint)
		if checkpoint_seed is not None and checkpoint_seed != seed:
			raise ValueError(
				f"Checkpoint seed {checkpoint_seed} does not match evaluation seed {seed}: {checkpoint}"
			)

		if checkpoint is None:
			if seed_run_dir is None:
				seed_root = os.path.join(
					"drugrec_benchmark",
					"results",
					args.model,
					args.dataset,
					f"seed_{seed}",
				)
				seed_run_dir = _find_latest_run_dir(seed_root)
			checkpoint_dir = None
			if seed_run_dir is not None:
				checkpoint_dir = os.path.join(seed_run_dir, "models")
			if checkpoint_dir:
				checkpoint = _find_best_checkpoint(checkpoint_dir)
				if checkpoint is not None:
					checkpoint_source = "auto"
			if checkpoint is None:
				raise FileNotFoundError(
					f"Best checkpoint not found for seed {seed}."
				)

		model.load(checkpoint, map_location=str(device))
		model.to(device)
		set_beam_search = getattr(model, "set_beam_search", None)
		if callable(set_beam_search):
			set_beam_search(True)

		_, _, test_records = split_records(records, train_ratio, val_ratio, seed)

		test_loader = build_batches(test_records)

		if seed_run_dir:
			reports_dir = os.path.join(seed_run_dir, "reports")
			logs_dir = os.path.join(seed_run_dir, "logs")
			os.makedirs(reports_dir, exist_ok=True)
			os.makedirs(logs_dir, exist_ok=True)
		else:
			run_dirs = make_run_dirs(args.model, args.dataset, seed)
			seed_run_dir = run_dirs["run_dir"]
			reports_dir = run_dirs["reports"]
			logs_dir = run_dirs["logs"]

		test_id = _timestamp()
		log_path = os.path.join(logs_dir, f"test_{test_id}.log")
		original_stdout = sys.stdout
		original_stderr = sys.stderr
		with open(log_path, "w", encoding="utf-8") as log_file:
			tee = _Tee(original_stdout, log_file)
			sys.stdout = tee
			sys.stderr = tee
			try:
				results = evaluator.evaluate(
					model,
					test_loader,
					log_interval=log_interval,
					prefix="test",
				)
			finally:
				sys.stdout = original_stdout
				sys.stderr = original_stderr
		metrics_path = os.path.join(reports_dir, f"metrics_test_{test_id}.json")
		context_path = os.path.join(reports_dir, f"test_context_{test_id}.json")
		_write_metrics_report(results, metrics_path)
		_write_test_context(
			{
				"model": args.model,
				"dataset": args.dataset,
				"seed": seed,
				"checkpoint": checkpoint,
				"checkpoint_source": checkpoint_source,
				"run_dir": seed_run_dir,
				"metrics_path": metrics_path,
				"log_path": log_path,
			},
			context_path,
		)

		for key, value in results.items():
			print(f"seed {seed} {key}: {value:.4f}")

		result_payload = dict(results)
		result_payload["seed"] = seed
		result_payload["checkpoint"] = checkpoint
		result_payload["run_dir"] = seed_run_dir
		result_payload["metrics_path"] = metrics_path
		result_payload["log_path"] = log_path
		results_list.append(result_payload)

	payload: Dict[str, Any] = {
		"results": results_list,
		"result": results_list[0] if len(results_list) == 1 else None,
	}
	return {
		"status": 0,
		"error": None,
		"payload": payload,
	}


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--config")
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--checkpoint")
	parser.add_argument("--seed", type=int)
	parser.add_argument("--cuda", type=int, default=0)
	parser.add_argument("--batch-size", type=int, help="Accepted for CLI compatibility; evaluation remains batch_size=1")
	args = parser.parse_args()
	output = run_test(args)
	status = int(output.get("status", 1))
	if status != 0 and output.get("error"):
		print(output["error"])
	return status


if __name__ == "__main__":
	raise SystemExit(main())
