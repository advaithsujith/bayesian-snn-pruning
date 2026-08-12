"""Measure the SynOps of an unpruned pretrained model.

The SynOps budgets used throughout this project are expressed as a *fraction*
of the unpruned network's cost (see pruning.synops_budget_plan), but that
denominator was never recorded, which leaves every reported SynOps figure
uninterpretable in absolute terms. This script measures it directly.

No training, no gradients: it loads the saved baseline checkpoint, runs a
handful of test batches with forward hooks, and writes the result. Seconds of
GPU, and it can also be run on CPU.

Usage
-----
    python measure_baseline_synops.py                      # dpap_repl
    python measure_baseline_synops.py --model lenet
    python measure_baseline_synops.py --model dpap_repl --batches 16
    python measure_baseline_synops.py --all                # every model with a checkpoint

Writes `outputs/<model>/baseline_synops.json` and prints a summary table.
"""

import argparse
import json
import os

import torch

from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS
from datasets import get_cifar10_loaders
from metrics import measure_synops
from models import build_model
from utils import set_seed

# Use config.ALL_EXPERIMENTS rather than a second registry maintained here.
# This file used to keep its own dict of four configs, which then silently
# fell behind: `--model spear_repl` and `--model spear_repl_resnet18` failed
# argparse with exit code 2, so both SynOps measurement jobs died in two
# minutes and the baseline_synops.json they were supposed to produce never
# appeared. A duplicated registry is a registry that drifts.
CONFIGS = ALL_EXPERIMENTS


def measure_one(model_name: str, batches: int, device: torch.device) -> dict:
    cfg = CONFIGS[model_name]()
    set_seed(cfg.seed)

    checkpoint = os.path.join(cfg.output_dir, "trained_model.pt")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"No pretrained baseline at '{checkpoint}'. This script measures the "
            "unpruned network, so it needs the checkpoint the pruning runs forked from."
        )

    # Gates are built but left inert: the baseline is the ordinary SNN, exactly
    # as it was before any pruning stage touched it.
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    set_bayesian_mode(model, False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # The test loader is the right one here: SynOps depends on firing rates,
    # and the figure we want is the cost of an inference on held-out data.
    _, _, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)

    with torch.no_grad():
        result = measure_synops(model, test_loader, device, max_batches=batches)

    synops = float(result["synops_per_sample"])
    dense = float(result["dense_macs_per_sample"])

    record = {
        "model": model_name,
        "checkpoint": checkpoint,
        "batches_measured": batches,
        "unpruned_synops_per_sample": synops,
        "unpruned_dense_macs_per_sample": dense,
        "event_driven_fraction": synops / dense if dense else None,
        # The budgets actually used in the sparsity-curve runs, resolved into
        # the absolute SynOps ceilings they correspond to.
        "budget_ceilings": {
            f"{b:.2f}": synops * b for b in (0.5, 0.3)
        },
        "per_layer": result["per_layer"],
    }

    out_path = os.path.join(cfg.output_dir, "baseline_synops.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\n=== {model_name} ===")
    print(f"  checkpoint            : {checkpoint}")
    print(f"  unpruned SynOps/sample: {synops/1e6:12.2f} M")
    print(f"  dense MACs/sample     : {dense/1e6:12.2f} M")
    print(f"  event-driven fraction : {synops/dense:12.4f}" if dense else "")
    print(f"  budget 0.5 ceiling    : {synops*0.5/1e6:12.2f} M")
    print(f"  budget 0.3 ceiling    : {synops*0.3/1e6:12.2f} M")
    print(f"  written to            : {out_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpap_repl", choices=sorted(CONFIGS))
    parser.add_argument(
        "--batches",
        type=int,
        default=8,
        help="Test batches to average over. 8 matches metrics.measure_synops's default.",
    )
    parser.add_argument("--all", action="store_true", help="Every model that has a checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    names = sorted(CONFIGS) if args.all else [args.model]
    for name in names:
        try:
            measure_one(name, args.batches, device)
        except FileNotFoundError as exc:
            if args.all:
                print(f"\n=== {name} ===\n  skipped: {exc}")
            else:
                raise


if __name__ == "__main__":
    main()
