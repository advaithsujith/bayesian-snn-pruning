"""
Full-model evaluation: accuracy/loss on a data split, plus the resource
metrics (parameter count, FLOPs, latency, GPU memory) needed for
final_results.csv. Thin orchestration layer over train.evaluate_loader
and metrics.py.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics import (
    count_parameters,
    estimate_flops,
    measure_gpu_memory_mb,
    measure_latency_ms,
)
from train import evaluate_loader


def full_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    input_shape: "tuple[int, int, int, int]",
    latency_num_warmup: int,
    latency_num_runs: int,
) -> Dict[str, Any]:
    """
    Run the complete evaluation suite used at every "Testing..." /
    "Final evaluation" pipeline stage: test accuracy, test loss,
    parameter count, estimated FLOPs, inference latency, and peak GPU
    memory.
    """
    model.to(device)

    test_stats = evaluate_loader(model, test_loader, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    flops = estimate_flops(model, input_shape, device)
    latency_ms = measure_latency_ms(model, input_shape, device, latency_num_warmup, latency_num_runs)
    gpu_memory_mb = measure_gpu_memory_mb(device)
    n_params = count_parameters(model, exclude_gates=True)

    return {
        "test_loss": test_stats["loss"],
        "test_accuracy": test_stats["accuracy"],
        "parameters": n_params,
        "flops": flops,
        "latency_ms": latency_ms,
        "gpu_memory_mb": gpu_memory_mb,
    }
