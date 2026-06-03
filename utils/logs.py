from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np
import torch

def make_run_dirs(model: str, dataset: str, seed: int) -> Dict[str, str]:
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	run_id = f"run_{timestamp}"
	base = os.path.join(
		"drugrec_benchmark",
		"results",
		model,
		dataset,
		f"seed_{seed}",
		run_id,
	)
	paths = {
		"run_dir": base,
		"logs": os.path.join(base, "logs"),
		"models": os.path.join(base, "models"),
		"reports": os.path.join(base, "reports"),
	}
	for path in paths.values():
		os.makedirs(path, exist_ok=True)
	return paths