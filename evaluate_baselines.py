"""Re-measure the unpruned test accuracy of a saved baseline checkpoint.

Why this exists
---------------
The three unpruned baselines the dissertation reports (94.35% on `dpap_repl`,
90.62% on `spear_repl`, 93.05% on `spear_repl_resnet18`) have no committed file
behind them. They survive only as prose in `HANDOFF.md` and `docs/`. The
artefacts that *are* committed at `outputs/dpap_repl/summary.txt` and
`outputs/dpap_repl/metrics.csv` are stale: they record the first replication
attempt, which read 91.96% before pruning and 9.99% after with 183 parameters
left, and they have never been overwritten because every later run used
`reuse_pretrained` and wrote into a `sparsity_curve_<tag>/` subdirectory
instead.

The replication gap (94.35% against DPAP's published 94.54%) is objective O1 in
its entirety and the first row of the results section, so it needs a committed
measurement rather than a note.

No training and no gradients: this loads the checkpoint the pruning runs forked
from, evaluates it on the CIFAR-10 test set, and writes the result. Seconds of
GPU, and it runs on CPU too.

Usage
-----
    python evaluate_baselines.py --all            # every model with a checkpoint
    python evaluate_baselines.py                  # dpap_repl only
    python evaluate_baselines.py --model spear_repl

Writes `outputs/<model>/baseline_eval.json` and prints a summary table.
Commit the JSON files: they are the evidence for Table 7 of the dissertation.
"""

import argparse
import json
import os

import torch

from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS
from datasets import get_cifar10_loaders
from models import build_model
from train import evaluate_loader
from utils import set_seed

# Same convention as measure_baseline_synops.py: defer to the one registry in
# config rather than keeping a second copy here, which is how that script's
# `--model spear_repl` silently failed argparse for two whole jobs.
CONFIGS = ALL_EXPERIMENTS


def evaluate_one(model_name: str, device: torch.device) -> dict:
    cfg = CONFIGS[model_name]()
    set_seed(cfg.seed)

    checkpoint = os.path.join(cfg.output_dir, "trained_model.pt")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"No pretrained baseline at '{checkpoint}'. This script measures the "
            "unpruned network, so it needs the checkpoint the pruning runs forked "
            "from. On CSF3 those are backed up under ~/snn_checkpoints/."
        )

    # Gates are built but left inert, so this is the ordinary deterministic SNN
    # exactly as it stood before any pruning stage touched it. This is the same
    # construction measure_baseline_synops.py uses, so the accuracy reported
    # here and the SynOps reported there describe the same network.
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    set_bayesian_mode(model, False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    _, _, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)

    # loss_type must match the one the platform trained under or the loss is on
    # a different scale; accuracy ranks by summed spike count and is unaffected.
    with torch.no_grad():
        result = evaluate_loader(model, test_loader, device, loss_type=cfg.loss_type)

    total_params = sum(p.numel() for p in model.parameters())
    gate_params = sum(
        p.numel() for n, p in model.named_parameters() if "log_alpha" in n
    )

    record = {
        "model": model_name,
        "checkpoint": checkpoint,
        "test_accuracy": float(result["accuracy"]),
        "test_loss": float(result["loss"]),
        "loss_type": cfg.loss_type,
        # Weight parameters excludes one log_alpha per prunable unit. The
        # sparsity-curve CSVs report the weight-only figure, so both are
        # recorded here and the dissertation can quote them consistently.
        "weight_parameters": total_params - gate_params,
        "gate_parameters": gate_params,
        "num_steps": cfg.snn.num_steps,
        # val_fraction 0.0 means the test set was returned as the validation
        # loader during pretraining. Recorded so the caveat in the methodology
        # can be checked against the config rather than taken on trust.
        "pretrain_val_fraction": cfg.data.val_fraction,
    }

    out_path = os.path.join(cfg.output_dir, "baseline_eval.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\n=== {model_name} ===")
    print(f"  checkpoint         : {checkpoint}")
    print(f"  test accuracy      : {100 * record['test_accuracy']:8.2f} %")
    print(f"  test loss ({cfg.loss_type}) : {record['test_loss']:.4f}")
    print(f"  weight parameters  : {record['weight_parameters']:,}")
    print(f"  written to         : {out_path}")
    return record


def resolve_device(choice: str) -> torch.device:
    """Pick a device, falling back to CPU if CUDA is present but unusable.

    On a CSF3 login node `torch.cuda.is_available()` returns True while the
    device itself is busy or not allocatable, so the first `.to(device)` dies
    with "CUDA-capable device(s) is/are busy or unavailable". Availability is
    therefore not enough: the allocation has to be attempted. This eval is a
    few minutes on CPU, so silently falling back is better than failing.
    """
    if choice == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1).to("cuda")
        return torch.device("cuda")
    except RuntimeError as exc:
        if choice == "cuda":
            raise
        print(f"CUDA present but unusable ({str(exc).splitlines()[0]}); falling back to CPU.")
        return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpap_repl", choices=sorted(CONFIGS))
    parser.add_argument(
        "--all", action="store_true",
        help="evaluate every model that has a saved checkpoint, skipping those that do not",
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="auto (default) uses CUDA if it is actually usable, else CPU",
    )
    args = parser.parse_args()

    # Choosing CPU is not enough on a login node. get_cifar10_loaders builds its
    # DataLoaders with pin_memory=True, and the pin-memory thread targets CUDA
    # device 0 whenever torch.cuda.is_available() is True, regardless of where
    # the model lives. On a node where the GPU exists but is not allocatable
    # that thread dies with "CUDA-capable device(s) is/are busy or unavailable"
    # after the model has already been placed on CPU. Hiding the device makes
    # is_available() False, which is the flag DataLoader actually consults.
    # Must be set before the first CUDA call, hence before resolve_device.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = resolve_device(args.device)
    print(f"device: {device}")

    names = sorted(CONFIGS) if args.all else [args.model]
    records = []
    for name in names:
        try:
            records.append(evaluate_one(name, device))
        except FileNotFoundError as exc:
            if not args.all:
                raise
            print(f"\n=== {name} ===\n  skipped: {exc}")

    if len(records) > 1:
        print("\n" + "=" * 52)
        print(f"{'model':<24}{'test acc %':>12}{'weights':>16}")
        print("-" * 52)
        for r in records:
            print(f"{r['model']:<24}{100 * r['test_accuracy']:>12.2f}"
                  f"{r['weight_parameters']:>16,}")


if __name__ == "__main__":
    main()
