"""
Compression, resource, and timing metrics: parameter counts, FLOPs,
remaining channels/neurons, GPU memory, and simple CSV-writing helpers.
"""

import csv
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from bayesian_layers import BayesianConv2d, BayesianLinear, collect_bayesian_layers


def count_parameters(model: nn.Module, exclude_gates: bool = True) -> int:
    """
    Count the model's "real" (architectural) parameters.

    By default excludes `log_alpha` gate parameters, since those are an
    artifact of the Bayesian training procedure, not part of the deployed
    network's weights -- reporting them as part of "model size" would be
    misleading once the model has been physically pruned into a plain
    (non-Bayesian) network by pruning.py.
    """
    total = 0
    for name, p in model.named_parameters():
        if exclude_gates and "log_alpha" in name:
            continue
        total += p.numel()
    return total


def count_remaining_structures(model: nn.Module, threshold: float) -> Dict[str, Dict[str, int]]:
    """
    For every Bayesian layer in `model`, report (total, remaining) neuron
    or channel counts under the given pruning threshold. Does not modify
    the model -- this is a read-only diagnostic used both before and after
    physical pruning.
    """
    report: Dict[str, Dict[str, int]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            prunable = module.prunable_mask(threshold)
            total = prunable.numel()
            remaining = int((~prunable).sum().item())
            report[name] = {"total": total, "remaining": remaining, "pruned": total - remaining}
    return report


def compression_ratio(original_params: int, remaining_params: int) -> float:
    """Ratio of original to remaining parameters (>= 1.0 for a compressed model)."""
    if remaining_params == 0:
        return float("inf")
    return original_params / remaining_params


def pruning_percentage(original_params: int, remaining_params: int) -> float:
    """Percentage of parameters removed."""
    if original_params == 0:
        return 0.0
    return 100.0 * (1.0 - remaining_params / original_params)


class _FlopCounter:
    """Accumulates multiply-accumulate FLOPs via forward hooks on Conv2d/Linear."""

    def __init__(self) -> None:
        self.total_flops = 0
        self._handles: List[Any] = []

    def _conv_hook(self, module: nn.Conv2d, inputs: Any, output: torch.Tensor) -> None:
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels_per_group = module.in_channels // module.groups
        kernel_flops = (
            in_channels_per_group * module.kernel_size[0] * module.kernel_size[1]
        )
        flops_per_instance = kernel_flops * out_channels * out_h * out_w
        self.total_flops += flops_per_instance * batch_size

    def _linear_hook(self, module: nn.Linear, inputs: Any, output: torch.Tensor) -> None:
        batch_size = output.shape[0]
        self.total_flops += module.in_features * module.out_features * batch_size

    def register(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self._handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear_hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def estimate_flops(model: nn.Module, input_shape: "tuple[int, int, int, int]", device: torch.device) -> int:
    """
    Estimate total multiply-accumulate FLOPs for one full forward pass
    (i.e. across all `num_steps` simulated timesteps, since the model's
    own forward() internally loops over time).

    A lightweight hand-rolled hook-based counter is used (rather than an
    external profiling library) so it works transparently with the custom
    BayesianConv2d / BayesianLinear wrapper modules without needing custom
    op registrations.
    """
    was_training = model.training
    model.eval()
    counter = _FlopCounter()
    counter.register(model)

    dummy_input = torch.randn(*input_shape, device=device)
    model(dummy_input)

    flops = counter.total_flops
    counter.remove()
    model.train(was_training)
    return flops


def measure_gpu_memory_mb(device: torch.device) -> float:
    """Peak CUDA memory allocated, in megabytes. Returns 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    input_shape: "tuple[int, int, int, int]",
    device: torch.device,
    num_warmup: int,
    num_runs: int,
) -> float:
    """
    Average wall-clock inference latency in milliseconds, over `num_runs`
    forward passes following `num_warmup` untimed warmup passes.
    """
    was_training = model.training
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    for _ in range(num_warmup):
        model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.train(was_training)
    return (elapsed / num_runs) * 1000.0


def gather_log_alpha_values(model: nn.Module) -> Dict[str, "torch.Tensor"]:
    """Return a dict of {layer_name: log_alpha tensor} for every Bayesian layer.

    Used by run_all.py to build the uncertainty-histogram / posterior-
    variance-distribution plots requested for the dissertation figures.
    """
    values: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            values[name] = module.log_alpha.detach().cpu().clone()
    return values


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of flat dicts to a CSV file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(row: Dict[str, Any], path: str) -> None:
    """Append a single row to a CSV file, writing a header if the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
