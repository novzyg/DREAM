"""
MR-DTR custom training pipeline.
Handles graph data preprocessing for the MR-DTR model.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import torch

from drugrec_benchmark.utils.load_config import load_config, resolve_seeds, set_seed
from drugrec_benchmark.utils.dataset_utils import load_pickle, split_records
from drugrec_benchmark.models.registry import build_model
from drugrec_benchmark.core.evaluator import Evaluator
from drugrec_benchmark.core.trainer import Trainer
from drugrec_benchmark.utils.logs import make_run_dirs
from drugrec_benchmark.models.mrdtr import (
    preprocess_mrdtr_data,
    load_mrdtr_data,
    mrdtr_collate,
    _build_graph,
    _generate_graph_samples,
)


def _resolve_run_seeds(config: Dict[str, Any], args: argparse.Namespace) -> List[int]:
    seed_override = getattr(args, "seed", None)
    if seed_override is None:
        return resolve_seeds(config)
    return [int(seed_override)]


def _write_config_snapshot(config: Dict[str, Any], path: str) -> None:
    try:
        import yaml
    except ImportError:
        return
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


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


def _preprocess_and_build_batches(records, config, data_dir, seed):
    """Preprocess records into MR-DTR graph format and build batches."""
    print(f"Preprocessing MR-DTR graph data for {len(records)} records...", flush=True)
    graph = _build_graph(records)
    n = len(records)
    split_point = int(n * 2 / 3)
    eval_len = int((n - split_point) / 2)
    train_records = records[:split_point]
    val_records = records[split_point:split_point + eval_len]
    test_records = records[split_point + eval_len:]

    train_samples = _generate_graph_samples(graph, 0, split_point, records)
    val_samples = _generate_graph_samples(graph, split_point, split_point + eval_len, records)

    train_batches = [mrdtr_collate(s) for s in train_samples]
    val_batches = [mrdtr_collate(s) for s in val_samples]

    print(f"MR-DTR preprocessing done: {len(train_batches)} train, {len(val_batches)} val", flush=True)
    return train_batches, val_batches


def run_train(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.model, override_config=None)
    batch_size_override = getattr(args, "batch_size", None)
    if batch_size_override is not None:
        config.setdefault("training", {})["batch_size"] = max(1, int(batch_size_override))
    config["dataset"] = config.get("dataset", {})
    config["dataset"]["name"] = args.dataset
    data_root = config.get("paths", {}).get("data_root", "drugrec_benchmark/data")
    template = config.get("paths", {}).get("dataset_dir_template", "{data_root}/{dataset}")
    config["dataset"]["data_dir"] = template.format(
        data_root=data_root, dataset=args.dataset,
    )
    seeds = _resolve_run_seeds(config, args)

    use_cuda = config.get("device", {}).get("use_cuda", True)
    device = torch.device(
        f"cuda:{args.cuda}" if use_cuda and torch.cuda.is_available() else "cpu"
    )

    data_dir = config["dataset"]["data_dir"]
    records_path = os.path.join(data_dir, config["dataset"]["records_file"])
    results: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        model, meta = build_model(config["model"]["name"], config, device)
        voc_size = meta["vocab_size"]

        records = load_pickle(records_path)

        train_loader, val_loader = _preprocess_and_build_batches(records, config, data_dir, seed)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["optimizer"].get("lr", 5e-4),
            weight_decay=config["optimizer"].get("weight_decay", 0.0),
        )

        metric_names = config.get("evaluation", {}).get("metrics", [])
        if "jaccard" not in metric_names:
            metric_names = metric_names + ["jaccard"]

        evaluator = Evaluator(metrics=metric_names) if metric_names else None

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            device=device,
            grad_clip=config["training"].get("grad_clip"),
        )

        log_interval = config.get("logging", {}).get("log_interval", 50)
        run_dirs = make_run_dirs(args.model, args.dataset, seed)
        checkpoint_dir = run_dirs["models"]
        config_snapshot = os.path.join(run_dirs["reports"], "config.yaml")
        config_snapshot_data = dict(config)
        config_snapshot_data["seed"] = seed
        config_snapshot_data["seeds"] = seeds
        _write_config_snapshot(config_snapshot_data, config_snapshot)
        log_path = os.path.join(run_dirs["logs"], "train.log")
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with open(log_path, "w", encoding="utf-8") as log_file:
            tee = _Tee(original_stdout, log_file)
            sys.stdout = tee
            sys.stderr = tee
            try:
                result = trainer.fit(
                    train_loader=train_loader,
                    epochs=config["training"].get("epochs", 50),
                    val_loader=val_loader,
                    evaluator=evaluator,
                    checkpoint_dir=checkpoint_dir,
                    metric_key="jaccard",
                    early_stop_patience=config["training"].get("early_stop_patience"),
                    log_interval=log_interval,
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        result["run_dir"] = run_dirs["run_dir"]
        result["reports_dir"] = run_dirs["reports"]
        result["models_dir"] = run_dirs["models"]
        result["logs_dir"] = run_dirs["logs"]
        result["seed"] = seed
        results.append(result)

    payload: Dict[str, Any] = {
        "results": results,
        "result": results[0] if len(results) == 1 else None,
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    output = run_train(args)
    status = int(output.get("status", 1))
    if status != 0 and output.get("error"):
        print(output["error"])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
