import yaml
import os
import random
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np
import torch


def load_yaml(path: str) -> Dict[str, Any]:
	try:
		import yaml
	except ImportError as exc:
		raise ImportError("PyYAML is required to load configs.") from exc
	with open(path, "r", encoding="utf-8") as handle:
		return yaml.safe_load(handle) or {}

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key not in base:
            base[key] = value
        elif isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
    return base


def resolve_model_config(model: str) -> str:
	if os.path.isfile(model):
		return model
	model_name = model.lower()
	return os.path.join("drugrec_benchmark", "configs", f"{model_name}.yaml")


def load_config(model: str, override_config: str | None = None) -> Dict[str, Any]:
	base_config = load_yaml(os.path.join("drugrec_benchmark", "configs", "base_config.yaml"))
	model_config = load_yaml(resolve_model_config(model))
	config = deep_update(base_config, model_config)
	if override_config:
		config = deep_update(config, load_yaml(override_config))
	return config

def resolve_seeds(config: Dict[str, Any]) -> List[int]:
	seeds = config.get("seeds")
	if seeds is None:
		seed = config.get("seed", 1203)
		return [int(seed)]
	if isinstance(seeds, (list, tuple)):
		return [int(seed) for seed in seeds]
	return [int(seeds)]


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False