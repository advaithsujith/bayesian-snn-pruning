"""
Cross-cutting helpers: reproducibility, logging, checkpointing, device
selection. Nothing here is specific to Bayesian pruning or SNNs.
"""

import dataclasses
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix every source of randomness we control.

    Deterministic cuDNN algorithms are enabled where practical; this can
    slow down convolutions somewhat but makes runs comparable across
    machines, which matters for a dissertation's reported numbers.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Select CUDA if available and requested, otherwise fall back to CPU."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, log_dir: str, filename: str) -> logging.Logger:
    """
    Create a logger that writes to both stdout and a log file.

    Re-invoking this with the same `name` returns a logger with duplicate
    handlers removed first, so run_all.py can safely call it once per
    experiment without accumulating duplicate log lines.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def print_banner(title: str) -> None:
    """Print the section-header style banner used throughout run_all.py."""
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}")


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save a model/optimizer state dict (or any picklable dict) to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint previously written by save_checkpoint."""
    return torch.load(path, map_location=map_location)


def save_config_json(config: Any, path: str) -> None:
    """Serialise a (possibly nested) dataclass config to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    with open(path, "w") as f:
        json.dump(_to_dict(config), f, indent=2)


def ensure_dirs(*dirs: str) -> None:
    """Create every directory in `dirs` if it does not already exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def count_trainable_parameters(model: torch.nn.Module, exclude_substr: Optional[str] = None) -> int:
    """
    Count trainable parameters, optionally excluding parameters whose name
    contains `exclude_substr` (used to exclude gate parameters such as
    'log_alpha' when reporting the "real" architectural parameter count).
    """
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_substr is not None and exclude_substr in name:
            continue
        total += p.numel()
    return total
