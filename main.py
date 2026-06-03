"""
Drug recommendation benchmark unified CLI entry.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import deque
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

try:
	from rich.console import Group
	from rich.live import Live
	from rich.panel import Panel
	from rich.table import Table
except ImportError:
	Group = None
	Live = None
	Panel = None
	Table = None


def _parse_csv(value: Optional[str]) -> List[str]:
	if not value:
		return []
	items = [item.strip() for item in value.split(",")]
	return [item for item in items if item]


def _resolve_targets(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
	models = _parse_csv(getattr(args, "models", None))
	datasets = _parse_csv(getattr(args, "datasets", None))

	if not models:
		single_model = getattr(args, "model", None)
		if single_model:
			models = [single_model]

	if not datasets:
		single_dataset = getattr(args, "dataset", None)
		if single_dataset:
			datasets = [single_dataset]

	if not models or not datasets:
		raise ValueError("Either --model/--dataset or --models/--datasets must be provided.")

	return models, datasets


def _resolve_cuda_slots(args: argparse.Namespace, task_count: int) -> List[int]:
	cuda_values = _parse_csv(getattr(args, "cuda_list", None))
	if cuda_values:
		slots: List[int] = [int(value) for value in cuda_values]
		if not slots:
			return [int(getattr(args, "cuda", 0))] * task_count
		return [slots[index % len(slots)] for index in range(task_count)]
	return [int(getattr(args, "cuda", 0))] * task_count


def _resolve_gpu_pool(args: argparse.Namespace) -> List[int]:
	cuda_values = _parse_csv(getattr(args, "cuda_list", None))
	if cuda_values:
		return [int(value) for value in cuda_values]
	return [int(getattr(args, "cuda", 0))]


def _build_interleaved_tasks(models: List[str], datasets: List[str]) -> List[Tuple[str, str]]:
	queues: List[deque[Tuple[str, str]]] = [
		deque((model, dataset) for dataset in datasets)
		for model in models
	]
	ordered_tasks: List[Tuple[str, str]] = []
	while any(queue for queue in queues):
		for queue in queues:
			if queue:
				ordered_tasks.append(queue.popleft())
	return ordered_tasks


def _query_gpu_status() -> Dict[int, Dict[str, float]]:
	command = [
		"nvidia-smi",
		"--query-gpu=index,memory.used,memory.total,utilization.gpu",
		"--format=csv,noheader,nounits",
	]
	try:
		result = subprocess.run(
			command,
			check=True,
			capture_output=True,
			text=True,
		)
	except (subprocess.CalledProcessError, FileNotFoundError) as exc:
		raise RuntimeError(f"Failed to query GPU status via nvidia-smi: {exc}") from exc

	status_map: Dict[int, Dict[str, float]] = {}
	for raw_line in result.stdout.strip().splitlines():
		line = raw_line.strip()
		if not line:
			continue
		parts = [item.strip() for item in line.split(",")]
		if len(parts) != 4:
			continue
		try:
			gpu_index = int(parts[0])
			memory_used = float(parts[1])
			memory_total = float(parts[2])
			utilization = float(parts[3])
		except ValueError:
			continue
		memory_ratio = (memory_used / memory_total) if memory_total > 0 else 1.0
		status_map[gpu_index] = {
			"memory_ratio": memory_ratio,
			"utilization": utilization,
		}
	return status_map


def _find_available_gpu(
	gpu_pool: List[int],
	status_map: Dict[int, Dict[str, float]],
	gpu_running_counts: Dict[int, int],
	memory_threshold: float,
	util_threshold: float,
	max_procs_per_gpu: int,
	gpu_last_launch: Optional[Dict[int, float]] = None,
	launch_interval: float = 0.0,
) -> Optional[int]:
	now = time.time()
	candidates: List[Tuple[int, float, float, int]] = []
	for gpu in gpu_pool:
		running_count = int(gpu_running_counts.get(gpu, 0))
		if max_procs_per_gpu > 0 and running_count >= max_procs_per_gpu:
			continue
		if gpu_last_launch and running_count > 0 and launch_interval > 0:
			last_launch = float(gpu_last_launch.get(gpu, 0.0))
			if now - last_launch < launch_interval:
				continue
		status = status_map.get(gpu)
		if status is None:
			continue
		if status["memory_ratio"] <= memory_threshold and status["utilization"] <= util_threshold:
			candidates.append(
				(
					running_count,
					float(status["memory_ratio"]),
					float(status["utilization"]),
					gpu,
				)
			)
	if not candidates:
		return None
	candidates.sort()
	return candidates[0][3]


def _resolve_worker_count(
	raw_parallel: int,
	job_count: int,
	gpu_pool: List[int],
	max_procs_per_gpu: int,
) -> int:
	if job_count <= 0:
		return 0
	if raw_parallel > 0:
		return max(1, min(raw_parallel, job_count))
	per_gpu_capacity = max_procs_per_gpu if max_procs_per_gpu > 0 else 2
	auto_capacity = max(1, len(gpu_pool) * per_gpu_capacity)
	return min(auto_capacity, job_count)


def _configure_worker_threads(thread_count: int) -> None:
	if thread_count <= 0:
		return
	value = str(thread_count)
	for name in (
		"OMP_NUM_THREADS",
		"MKL_NUM_THREADS",
		"OPENBLAS_NUM_THREADS",
		"NUMEXPR_NUM_THREADS",
		"VECLIB_MAXIMUM_THREADS",
		"BLIS_NUM_THREADS",
	):
		os.environ[name] = value
	try:
		import torch

		torch.set_num_threads(thread_count)
		try:
			torch.set_num_interop_threads(max(1, min(thread_count, 4)))
		except RuntimeError:
			pass
	except Exception:
		pass


def _format_elapsed(started_at: Optional[float], ended_at: Optional[float]) -> str:
	if started_at is None:
		return "-"
	end_time = ended_at if ended_at is not None else time.time()
	seconds = max(0, int(end_time - started_at))
	minutes = seconds // 60
	remain = seconds % 60
	return f"{minutes:02d}:{remain:02d}"


def _normalize_command_result(result: Any) -> Tuple[int, Optional[str]]:
	if result is None:
		return 0, None

	if isinstance(result, bool):
		return (0, None) if result else (1, "command returned False")

	if isinstance(result, (int, float)):
		status = int(result)
		if status == 0:
			return 0, None
		return status, f"non-zero status: {status}"

	if isinstance(result, dict):
		if "status" in result:
			try:
				status = int(result.get("status", 0) or 0)
			except (TypeError, ValueError):
				status = 1
			error = result.get("error")
			return status, str(error) if error else (None if status == 0 else f"non-zero status: {status}")
		error = result.get("error")
		if error:
			return 1, str(error)
		return 0, None

	if isinstance(result, (list, tuple)):
		failures: List[str] = []
		for item in result:
			status, error = _normalize_command_result(item)
			if status != 0:
				failures.append(error or f"non-zero status: {status}")
		if failures:
			preview = "; ".join(failures[:3])
			return 1, preview
		return 0, None

	return 0, None


_TRAIN_STEP_RE = re.compile(r"train\s+epoch\s+(\d+)\s+step\s+(\d+)(?:/(\d+))?\s+loss\s+([0-9eE.+-]+)")
_EPOCH_RE = re.compile(r"^epoch\s+(\d+)\s+-+")
_EVAL_STEP_RE = re.compile(r"(?:test|test step):\s*(\d+)\s*/\s*(\d+)")
_METRIC_RE = re.compile(r"DDI Rate:|Jaccard:|seed\s+\d+\s+")


class _TaskDashboard:
    def __init__(
        self,
        enabled: bool,
        total_tasks: int,
        gpu_pool: List[int],
        memory_threshold: float,
        util_threshold: float,
        worker_count: int,
        max_starting_tasks: int,
    ) -> None:
        self.enabled = bool(enabled and Live is not None and Table is not None and Panel is not None and Group is not None)
        self.total_tasks = total_tasks
        self.gpu_pool = gpu_pool
        self.memory_threshold = memory_threshold
        self.util_threshold = util_threshold
        self.worker_count = worker_count
        self.max_starting_tasks = max_starting_tasks
        self._live = Live(refresh_per_second=2, transient=False) if self.enabled else None

    def __enter__(self):
        if self._live is not None:
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc_val, exc_tb)

    def _build_task_table(self, task_records: List[Dict[str, object]]) -> object:
        table = Table(title="Tasks", expand=True)
        table.add_column("ID", justify="right", no_wrap=True)
        table.add_column("Command", no_wrap=True)
        table.add_column("Model", no_wrap=True)
        table.add_column("Dataset", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("GPU", justify="right", no_wrap=True)
        table.add_column("Epoch", justify="right", no_wrap=True)
        table.add_column("Step", justify="right", no_wrap=True)
        table.add_column("Loss", justify="right", no_wrap=True)
        table.add_column("Elapsed", justify="right", no_wrap=True)
        table.add_column("Log")
        table.add_column("Error")

        for item in sorted(task_records, key=lambda rec: int(rec.get("task_id", 0))):
            state = str(item.get("state", "QUEUED"))
            gpu_value = item.get("gpu")
            gpu_text = "-" if gpu_value is None else str(gpu_value)
            epoch_text = str(item.get("epoch") or "-")
            step = item.get("step")
            total_steps = item.get("total_steps")
            if step and total_steps:
                step_text = f"{step}/{total_steps}"
            elif step:
                step_text = str(step)
            else:
                step_text = "-"
            loss = item.get("loss")
            loss_text = f"{float(loss):.4f}" if isinstance(loss, (int, float)) else "-"
            log_path = str(item.get("log_path") or "")
            log_text = os.path.basename(log_path) if log_path else "-"
            error_text = str(item.get("error") or "")
            if len(error_text) > 40:
                error_text = error_text[:37] + "..."
            table.add_row(
                str(item.get("task_id", "-")),
                str(item.get("command", "")),
                str(item.get("model", "")),
                str(item.get("dataset", "")),
                state,
                gpu_text,
                epoch_text,
                step_text,
                loss_text,
                _format_elapsed(
                    item.get("started_at") if isinstance(item.get("started_at"), (int, float)) else None,
                    item.get("ended_at") if isinstance(item.get("ended_at"), (int, float)) else None,
                ),
                log_text,
                error_text,
            )
        return table

    def _build_gpu_table(
        self,
        gpu_status: Dict[int, Dict[str, float]],
        gpu_running_counts: Dict[int, int],
        running_by_gpu: Dict[int, List[int]],
        max_procs_per_gpu: int,
    ) -> object:
        table = Table(title="GPUs", expand=True)
        table.add_column("GPU", justify="right", no_wrap=True)
        table.add_column("MemRatio", justify="right", no_wrap=True)
        table.add_column("Util%", justify="right", no_wrap=True)
        table.add_column("Slots", justify="right", no_wrap=True)
        table.add_column("Sched", no_wrap=True)
        table.add_column("TaskIDs")

        for gpu in self.gpu_pool:
            status = gpu_status.get(gpu)
            running_count = int(gpu_running_counts.get(gpu, 0))
            task_ids = running_by_gpu.get(gpu, [])
            task_ids_text = ",".join(str(task_id) for task_id in task_ids) if task_ids else "-"
            if status is None:
                mem_text = "n/a"
                util_text = "n/a"
                available = "NO"
            else:
                mem_ratio = float(status.get("memory_ratio", 1.0))
                util = float(status.get("utilization", 100.0))
                mem_text = f"{mem_ratio:.2f}"
                util_text = f"{util:.1f}"
                slot_ok = max_procs_per_gpu <= 0 or running_count < max_procs_per_gpu
                available = "YES" if mem_ratio <= self.memory_threshold and util <= self.util_threshold and slot_ok else "NO"
            slot_text = f"{running_count}/{max_procs_per_gpu}" if max_procs_per_gpu > 0 else str(running_count)
            table.add_row(str(gpu), mem_text, util_text, slot_text, available, task_ids_text)
        return table

    def update(
        self,
        task_records: List[Dict[str, object]],
        gpu_status: Dict[int, Dict[str, float]],
        gpu_running_counts: Dict[int, int],
        running_by_gpu: Dict[int, List[int]],
        max_procs_per_gpu: int,
        pending_count: int,
        wait_reason: Optional[str],
    ) -> None:
        if not self.enabled:
            return

        counts = _state_counts(task_records)
        summary = (
            f"Total: {self.total_tasks} | Workers: {self.worker_count} | MaxStarting: {self.max_starting_tasks} | "
            f"Queued: {counts.get('QUEUED', 0)} | Waiting: {counts.get('WAITING', 0)} | Starting: {counts.get('STARTING', 0)} | "
            f"Training: {counts.get('TRAINING', 0)} | Evaluating: {counts.get('EVALUATING', 0)} | "
            f"Done: {counts.get('DONE', 0)} | Failed: {counts.get('FAILED', 0)} | "
            f"Threshold(mem<={self.memory_threshold:.2f}, util<={self.util_threshold:.1f})"
        )
        if wait_reason:
            summary += f"\nWait: {wait_reason}"

        header_panel = Panel(summary, title="Scheduler", border_style="cyan")
        tasks_table = self._build_task_table(task_records)
        gpus_table = self._build_gpu_table(gpu_status, gpu_running_counts, running_by_gpu, max_procs_per_gpu)
        self._live.update(Group(header_panel, tasks_table, gpus_table), refresh=True)


def _runtime_states() -> Tuple[str, ...]:
    return ("STARTING", "TRAINING", "EVALUATING")


def _state_counts(task_records: List[Dict[str, object]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in task_records:
        state = str(item.get("state", "QUEUED"))
        counts[state] = counts.get(state, 0) + 1
    return counts


def _active_records(task_records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    active_states = set(_runtime_states())
    return [item for item in task_records if str(item.get("state")) in active_states]


def _running_by_gpu(task_records: List[Dict[str, object]]) -> Dict[int, List[int]]:
    by_gpu: Dict[int, List[int]] = {}
    for item in _active_records(task_records):
        gpu_value = item.get("gpu")
        if gpu_value is None:
            continue
        gpu = int(gpu_value)
        by_gpu.setdefault(gpu, []).append(int(item.get("task_id", 0)))
    for task_ids in by_gpu.values():
        task_ids.sort()
    return by_gpu


def _counts_as_starting(record: Dict[str, object], warmup_seconds: float) -> bool:
    if str(record.get("state")) != "STARTING":
        return False
    started_at = record.get("started_at")
    if not isinstance(started_at, (int, float)):
        return True
    return time.time() - float(started_at) < warmup_seconds


def _sanitize_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "task"


def _task_snapshot(record: Dict[str, object]) -> Dict[str, object]:
    skip = {"process", "log_handle"}
    snapshot: Dict[str, object] = {}
    for key, value in record.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value
        elif isinstance(value, list):
            snapshot[key] = value
        elif isinstance(value, dict):
            snapshot[key] = value
        else:
            snapshot[key] = str(value)
    return snapshot


def _write_scheduler_event(state_log_path: str, event: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(state_log_path), exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **payload,
    }
    with open(state_log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _tail_text(path: str, max_lines: int = 12, max_bytes: int = 8192) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read().decode("utf-8", errors="replace")
    lines = [line.strip() for line in data.replace("\r", "\n").splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])


def _update_task_from_log(record: Dict[str, object]) -> None:
    log_path = str(record.get("log_path") or "")
    if not log_path or not os.path.isfile(log_path):
        return
    offset = int(record.get("log_offset", 0) or 0)
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        data = handle.read()
        record["log_offset"] = handle.tell()
    if not data:
        return

    terminal_states = {"DONE", "FAILED"}
    for raw_line in data.replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record["last_log"] = line[-240:]

        train_match = _TRAIN_STEP_RE.search(line)
        if train_match:
            record["state"] = "TRAINING"
            record["epoch"] = int(train_match.group(1))
            record["step"] = int(train_match.group(2))
            record["total_steps"] = int(train_match.group(3)) if train_match.group(3) else None
            record["loss"] = float(train_match.group(4))
            continue

        epoch_match = _EPOCH_RE.search(line)
        if epoch_match and str(record.get("state")) not in terminal_states:
            record["state"] = "TRAINING"
            record["epoch"] = int(epoch_match.group(1))
            continue

        eval_match = _EVAL_STEP_RE.search(line)
        if eval_match and str(record.get("state")) not in terminal_states:
            record["state"] = "EVALUATING"
            record["step"] = int(eval_match.group(1))
            record["total_steps"] = int(eval_match.group(2))
            continue

        if _METRIC_RE.search(line) and str(record.get("state")) not in terminal_states:
            record["state"] = "EVALUATING"
            continue

        if "Traceback" in line or "RuntimeError" in line or "FileNotFoundError" in line:
            record["error"] = line[-240:]


def _build_worker_env(worker_threads: int) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if worker_threads > 0:
        value = str(worker_threads)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[name] = value
    return env


def _build_subprocess_command(task: Dict[str, object]) -> List[str]:
    command = str(task["command"])
    module = f"drugrec_benchmark.scripts.{command}"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        module,
        "--model",
        str(task["model"]),
        "--dataset",
        str(task["dataset"]),
        "--cuda",
        str(task["cuda"]),
    ]
    seed = task.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    batch_size = task.get("batch_size")
    if batch_size is not None and command in {"train", "run"}:
        cmd.extend(["--batch-size", str(batch_size)])
    checkpoint = task.get("checkpoint")
    if command == "test" and checkpoint:
        cmd.extend(["--checkpoint", str(checkpoint)])
    return cmd


def _task_log_path(scheduler_dir: str, task: Dict[str, object]) -> str:
    filename = "task_{:03d}_{}_{}_{}.log".format(
        int(task["task_id"]),
        _sanitize_name(task["command"]),
        _sanitize_name(task["model"]),
        _sanitize_name(task["dataset"]),
    )
    return os.path.join(scheduler_dir, "logs", filename)


def _launch_task(task: Dict[str, object], scheduler_dir: str, worker_threads: int) -> None:
    log_path = _task_log_path(scheduler_dir, task)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    task["log_path"] = log_path
    task["log_offset"] = 0
    task["state"] = "STARTING"
    task["started_at"] = time.time()
    task["ended_at"] = None
    task["error"] = None

    cmd = _build_subprocess_command(task)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    log_handle.write("$ " + " ".join(cmd) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=_build_worker_env(worker_threads),
        start_new_session=True,
    )
    task["process"] = process
    task["log_handle"] = log_handle
    task["pid"] = process.pid


def _finish_task(task: Dict[str, object], return_code: int) -> Dict[str, object]:
    _update_task_from_log(task)
    task["ended_at"] = time.time()
    task["return_code"] = return_code
    if return_code == 0:
        task["state"] = "DONE"
        task["error"] = None
    else:
        task["state"] = "FAILED"
        if not task.get("error"):
            task["error"] = _tail_text(str(task.get("log_path") or "")) or f"return code {return_code}"

    handle = task.get("log_handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    task.pop("process", None)
    task.pop("log_handle", None)
    return {
        "command": task["command"],
        "model": task["model"],
        "dataset": task["dataset"],
        "cuda": task.get("gpu"),
        "status": 0 if return_code == 0 else 1,
        "error": task.get("error"),
        "log_path": task.get("log_path"),
    }


def _terminate_running_tasks(task_records: List[Dict[str, object]]) -> None:
    for task in _active_records(task_records):
        process = task.get("process")
        if process is None:
            continue
        try:
            os.killpg(int(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
    time.sleep(2.0)
    for task in _active_records(task_records):
        process = task.get("process")
        if process is None:
            continue
        try:
            if process.poll() is None:
                os.killpg(int(process.pid), signal.SIGKILL)
        except Exception:
            pass


def _preflight_check(models: List[str], datasets: List[str], gpu_pool: List[int]) -> None:
    errors: List[str] = []
    try:
        gpu_status = _query_gpu_status()
    except RuntimeError as exc:
        gpu_status = {}
        errors.append(str(exc))
    for gpu in gpu_pool:
        if gpu not in gpu_status:
            errors.append(f"GPU {gpu} is not visible in nvidia-smi output")

    try:
        from drugrec_benchmark.models.registry import _MODEL_REGISTRY
        from drugrec_benchmark.utils.load_config import load_config, resolve_model_config
    except Exception as exc:
        raise ValueError(f"Preflight failed: cannot import registry/config loader: {exc}") from exc

    for model in models:
        config_path = resolve_model_config(model)
        if not os.path.isfile(config_path):
            errors.append(f"config not found for model={model}: {config_path}")
            continue
        try:
            config = load_config(model, override_config=None)
        except Exception as exc:
            errors.append(f"cannot load config for model={model}: {exc}")
            continue

        registered_name = str(config.get("model", {}).get("name") or model).lower()
        if registered_name not in _MODEL_REGISTRY and model.lower() not in _MODEL_REGISTRY:
            errors.append(f"model not registered: model={model}, config model.name={registered_name}")

        data_root = config.get("paths", {}).get("data_root", "drugrec_benchmark/data")
        template = config.get("paths", {}).get("dataset_dir_template", "{data_root}/{dataset}")
        dataset_config = config.get("dataset", {})
        required_files = [
            dataset_config.get("records_file", "records_final.pkl"),
            dataset_config.get("vocab_file", "voc_final.pkl"),
            dataset_config.get("ddi_adj_file", "ddi_A_final.pkl"),
        ]
        for dataset in datasets:
            data_dir = template.format(data_root=data_root, dataset=dataset)
            if not os.path.isdir(data_dir):
                errors.append(f"dataset dir not found: model={model} dataset={dataset} path={data_dir}")
                continue
            for filename in required_files:
                path = os.path.join(data_dir, str(filename))
                if not os.path.isfile(path):
                    errors.append(f"required data file missing: model={model} dataset={dataset} file={path}")

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:30])
        extra = "" if len(errors) <= 30 else f"\n  ... and {len(errors) - 30} more"
        raise ValueError(f"Preflight failed:\n{preview}{extra}")


def _print_heartbeat(
    task_records: List[Dict[str, object]],
    running_by_gpu: Dict[int, List[int]],
    gpu_pool: List[int],
    wait_reason: Optional[str],
) -> None:
    counts = _state_counts(task_records)
    gpu_text = " ".join(f"{gpu}:{len(running_by_gpu.get(gpu, []))}" for gpu in gpu_pool)
    print(
        "[scheduler] "
        f"queued={counts.get('QUEUED', 0)} waiting={counts.get('WAITING', 0)} "
        f"starting={counts.get('STARTING', 0)} training={counts.get('TRAINING', 0)} "
        f"evaluating={counts.get('EVALUATING', 0)} done={counts.get('DONE', 0)} "
        f"failed={counts.get('FAILED', 0)} gpu_running={gpu_text} wait={wait_reason or '-'}",
        flush=True,
    )


def _run_batch(args: argparse.Namespace) -> int:
    models, datasets = _resolve_targets(args)
    tasks = _build_interleaved_tasks(models, datasets)
    gpu_pool = _resolve_gpu_pool(args)
    memory_threshold = float(getattr(args, "gpu_mem_threshold", 0.1))
    util_threshold = float(getattr(args, "gpu_util_threshold", 10.0))
    poll_interval = max(1.0, float(getattr(args, "poll_interval", 5.0)))
    max_procs_per_gpu = max(0, int(getattr(args, "max_procs_per_gpu", 0)))
    gpu_launch_interval = max(0.0, float(getattr(args, "gpu_launch_interval", 3.0)))
    worker_threads = max(0, int(getattr(args, "worker_threads", 1)))
    scheduler_log_interval = max(0.0, float(getattr(args, "scheduler_log_interval", 30.0)))
    task_warmup_seconds = max(0.0, float(getattr(args, "task_warmup_seconds", 30.0)))
    raw_max_starting = int(getattr(args, "max_starting_tasks", 0))
    max_starting_tasks = max(1, raw_max_starting) if raw_max_starting > 0 else max(1, len(gpu_pool))

    if not bool(getattr(args, "skip_preflight", False)):
        _preflight_check(models, datasets, gpu_pool)

    job_specs: List[Dict[str, object]] = []
    for index, (model, dataset) in enumerate(tasks, start=1):
        job_specs.append(
            {
                "task_id": index,
                "command": args.command,
                "model": model,
                "dataset": dataset,
                "cuda": int(getattr(args, "cuda", 0)),
                "checkpoint": getattr(args, "checkpoint", None),
                "seed": getattr(args, "seed", None),
                "batch_size": getattr(args, "batch_size", None),
                "state": "QUEUED",
                "gpu": None,
                "pid": None,
                "error": None,
                "started_at": None,
                "ended_at": None,
                "epoch": None,
                "step": None,
                "total_steps": None,
                "loss": None,
                "log_path": None,
                "log_offset": 0,
            }
        )

    raw_parallel = int(getattr(args, "parallel", 0))
    worker_count = _resolve_worker_count(raw_parallel, len(job_specs), gpu_pool, max_procs_per_gpu)
    if worker_count <= 0:
        return 0

    scheduler_dir = getattr(args, "scheduler_dir", None)
    if not scheduler_dir:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scheduler_dir = os.path.join("drugrec_benchmark", "results", "_scheduler", f"run_{timestamp}")
    os.makedirs(os.path.join(scheduler_dir, "logs"), exist_ok=True)
    state_log_path = os.path.join(scheduler_dir, "scheduler_state.jsonl")

    dashboard_enabled = bool(getattr(args, "dashboard", True))
    results: List[Dict[str, object]] = []
    gpu_running_counts: Dict[int, int] = {gpu: 0 for gpu in gpu_pool}
    gpu_last_launch: Dict[int, float] = {gpu: 0.0 for gpu in gpu_pool}
    pending_jobs: deque[Dict[str, object]] = deque(job_specs)

    print(
        f"Scheduler: tasks={len(job_specs)} workers={worker_count} gpus={gpu_pool} "
        f"max_per_gpu={max_procs_per_gpu or 'unlimited'} max_starting={max_starting_tasks} "
        f"warmup={task_warmup_seconds:.1f}s worker_threads={worker_threads} "
        f"batch_size={getattr(args, 'batch_size', None) or 'config'} "
        f"launch_interval={gpu_launch_interval:.1f}s scheduler_dir={scheduler_dir}",
        flush=True,
    )
    _write_scheduler_event(
        state_log_path,
        "plan",
        {
            "tasks": len(job_specs),
            "workers": worker_count,
            "gpus": gpu_pool,
            "max_procs_per_gpu": max_procs_per_gpu,
            "max_starting_tasks": max_starting_tasks,
            "task_warmup_seconds": task_warmup_seconds,
            "batch_size": getattr(args, "batch_size", None),
        },
    )

    last_scheduler_log = 0.0
    last_snapshot_log = 0.0
    gpu_status: Dict[int, Dict[str, float]] = {}
    wait_reason: Optional[str] = None

    try:
        with _TaskDashboard(
            enabled=dashboard_enabled,
            total_tasks=len(job_specs),
            gpu_pool=gpu_pool,
            memory_threshold=memory_threshold,
            util_threshold=util_threshold,
            worker_count=worker_count,
            max_starting_tasks=max_starting_tasks,
        ) as dashboard:
            while pending_jobs or _active_records(job_specs):
                gpu_status = _query_gpu_status()

                for record in _active_records(job_specs):
                    _update_task_from_log(record)

                for record in list(_active_records(job_specs)):
                    process = record.get("process")
                    if process is None:
                        continue
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    result = _finish_task(record, int(return_code))
                    results.append(result)
                    assigned_gpu = record.get("gpu")
                    if assigned_gpu is not None:
                        gpu_running_counts[int(assigned_gpu)] = max(0, gpu_running_counts.get(int(assigned_gpu), 0) - 1)
                    _write_scheduler_event(state_log_path, "finish", {"task": _task_snapshot(record)})

                wait_reason = None
                launched_any = False
                while pending_jobs and len(_active_records(job_specs)) < worker_count:
                    starting_count = sum(1 for item in job_specs if _counts_as_starting(item, task_warmup_seconds))
                    if starting_count >= max_starting_tasks:
                        wait_reason = f"starting tasks at limit {starting_count}/{max_starting_tasks}"
                        break

                    gpu_status = _query_gpu_status()
                    available_gpu = _find_available_gpu(
                        gpu_pool=gpu_pool,
                        status_map=gpu_status,
                        gpu_running_counts=gpu_running_counts,
                        memory_threshold=memory_threshold,
                        util_threshold=util_threshold,
                        max_procs_per_gpu=max_procs_per_gpu,
                        gpu_last_launch=gpu_last_launch,
                        launch_interval=gpu_launch_interval,
                    )
                    if available_gpu is None:
                        wait_reason = f"no GPU slot available in {gpu_pool}"
                        break

                    record = pending_jobs.popleft()
                    record["gpu"] = available_gpu
                    record["cuda"] = available_gpu
                    _launch_task(record, str(scheduler_dir), worker_threads)
                    gpu_running_counts[available_gpu] = int(gpu_running_counts.get(available_gpu, 0)) + 1
                    gpu_last_launch[available_gpu] = time.time()
                    launched_any = True
                    _write_scheduler_event(state_log_path, "launch", {"task": _task_snapshot(record)})

                for record in pending_jobs:
                    record["state"] = "WAITING" if wait_reason else "QUEUED"

                running_by_gpu = _running_by_gpu(job_specs)
                dashboard.update(
                    task_records=job_specs,
                    gpu_status=gpu_status,
                    gpu_running_counts=gpu_running_counts,
                    running_by_gpu=running_by_gpu,
                    max_procs_per_gpu=max_procs_per_gpu,
                    pending_count=len(pending_jobs),
                    wait_reason=wait_reason,
                )

                now = time.time()
                if not dashboard_enabled and scheduler_log_interval > 0 and now - last_scheduler_log >= scheduler_log_interval:
                    _print_heartbeat(job_specs, running_by_gpu, gpu_pool, wait_reason)
                    last_scheduler_log = now

                if scheduler_log_interval > 0 and now - last_snapshot_log >= scheduler_log_interval:
                    _write_scheduler_event(
                        state_log_path,
                        "snapshot",
                        {
                            "wait_reason": wait_reason,
                            "gpu_status": gpu_status,
                            "tasks": [_task_snapshot(item) for item in job_specs],
                        },
                    )
                    last_snapshot_log = now

                if not pending_jobs and not _active_records(job_specs):
                    break
                if not launched_any:
                    time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nScheduler interrupted; terminating running tasks...", flush=True)
        _terminate_running_tasks(job_specs)
        _write_scheduler_event(state_log_path, "interrupted", {"tasks": [_task_snapshot(item) for item in job_specs]})
        return 130
    finally:
        for record in job_specs:
            handle = record.get("log_handle")
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

    results.sort(key=lambda item: (str(item["model"]), str(item["dataset"])))
    failures = [item for item in results if int(item.get("status", 1)) != 0]

    print("\nBatch execution summary:")
    for item in results:
        status = "OK" if int(item.get("status", 1)) == 0 else "FAILED"
        message = f"[{status}] {item['command']} model={item['model']} dataset={item['dataset']} cuda={item['cuda']} log={item.get('log_path')}"
        error = item.get("error")
        if error:
            message += f" error={error}"
        print(message)
    print(f"Scheduler state log: {state_log_path}")

    return 0 if not failures else 1


def _add_common_targets(parser: argparse.ArgumentParser, *, checkpoint_required: bool) -> None:
	parser.add_argument("--model", help="Single model name")
	parser.add_argument("--dataset", help="Single dataset name")
	parser.add_argument("--models", help="Comma-separated model names")
	parser.add_argument("--datasets", help="Comma-separated dataset names")
	parser.add_argument("--cuda", type=int, default=0, help="Default CUDA id")
	parser.add_argument("--seed", type=int, help="Optional single seed override")
	parser.add_argument("--cuda-list", help="Comma-separated GPU ids for dynamic scheduling")
	parser.add_argument("--parallel", type=int, default=0, help="Total max concurrent tasks (<=0 means auto: gpu_count * max_procs_per_gpu)")
	parser.set_defaults(dashboard=True)
	parser.add_argument("--dashboard", action="store_true", help="Enable live terminal dashboard")
	parser.add_argument("--no-dashboard", action="store_false", dest="dashboard", help="Disable live terminal dashboard")
	parser.add_argument("--gpu-mem-threshold", type=float, default=0.5, help="GPU memory usage ratio threshold [0, 1]")
	parser.add_argument("--gpu-util-threshold", type=float, default=50.0, help="GPU utilization threshold (percent)")
	parser.add_argument("--max-procs-per-gpu", type=int, default=3, help="Max concurrent tasks per GPU (<=0 means no fixed cap)")
	parser.add_argument("--gpu-launch-interval", type=float, default=3.0, help="Seconds to wait before launching another task on the same GPU")
	parser.add_argument("--worker-threads", type=int, default=1, help="CPU/BLAS/PyTorch threads per worker process (<=0 means unchanged)")
	parser.add_argument("--batch-size", type=int, help="Override training batch size for models with batched training support")
	parser.add_argument("--scheduler-log-interval", type=float, default=30.0, help="Plain scheduler heartbeat interval and state snapshot interval (<=0 disables)")
	parser.add_argument("--max-starting-tasks", type=int, default=0, help="Max tasks allowed in STARTING warmup at once (<=0 means GPU count)")
	parser.add_argument("--task-warmup-seconds", type=float, default=30.0, help="Seconds a STARTING task counts against max-starting-tasks")
	parser.add_argument("--scheduler-dir", help="Directory for scheduler logs/state (default: results/_scheduler/run_TIMESTAMP)")
	parser.add_argument("--skip-preflight", action="store_true", help="Skip model/config/data/GPU preflight checks")
	parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval seconds when no GPU is available")
	if checkpoint_required:
		parser.add_argument("--checkpoint", help="Checkpoint path (optional for auto-discovery)")


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Drug Recommendation Benchmark CLI",
	)

	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Train a model")
	_add_common_targets(train_parser, checkpoint_required=False)

	test_parser = subparsers.add_parser("test", help="Evaluate a model")
	_add_common_targets(test_parser, checkpoint_required=True)

	run_parser = subparsers.add_parser("run", help="Train then test best checkpoint")
	_add_common_targets(run_parser, checkpoint_required=False)


	return parser


def main() -> int:
	parser = _build_parser()
	args = parser.parse_args()
	try:
		return _run_batch(args)
	except ValueError as exc:
		print(str(exc))
		parser.print_help()
		return 1

	parser.print_help()
	return 1


if __name__ == "__main__":
	sys.exit(main())
