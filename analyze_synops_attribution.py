"""
Attribute the SynOps result between the KL term and the SynOps budget term.

The question this answers
--------------------------
The SynOps budget term pushes log_alpha in the same direction the KL does,
so "budget-trained gates beat KL-only gates" could in principle be nothing
more than extra pruning pressure wearing a costume. The claim worth making
is stronger: that the term helps *because it knows which units are
expensive*. Separating the two takes three arms, all evaluated with the
same cost-blind selection at the same measured SynOps budgets:

1. **KL only** -- gates trained with no budget term, budgeted at selection
   time (`--tag synops`).
2. **Uniform-cost budget** -- the identical Lagrangian machinery with every
   unit's cost set to 1, i.e. the same kind of pressure with the energy
   information removed (`--tag uniformcost0.5`, via
   `--synops-uniform-costs`).
3. **Real-cost budget** -- measured per-unit SynOps costs
   (`--tag lagrangian0.5`).

Then, per budget:

    effect of budget-in-loss overall   = (3) - (1)
    ... from generic pressure          = (2) - (1)
    ... from the SynOps information    = (3) - (2)

Honest caveat, printed with the table: with uniform costs the *training*
constraint is "at most B of the units expected to survive", which is not
the same tightness as "at most B of the SynOps", so arm 2 removes the cost
information rather than holding everything else numerically identical.
There is no ablation that does both; this is the standard one.

Usage (CSF3, venv active, after the ablation job finishes):

    python analyze_synops_attribution.py

Zero GPU required; the only compute is a few forward passes to measure the
dense baseline's SynOps (skippable with --skip-dense).
"""

import argparse
import csv
import os
from typing import Dict, List, Optional

import torch

from config import ALL_EXPERIMENTS
from datasets import get_cifar10_loaders
from metrics import measure_synops
from models import build_model
from utils import get_device, set_seed


def read_importance_rows(path: str) -> Dict[float, Dict[str, float]]:
    """{budget: row} for the cost-blind (importance) rows of one summary.csv."""
    if not os.path.isfile(path):
        return {}
    rows: Dict[float, Dict[str, float]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("Selection") != "importance" or not r.get("SynOps Budget"):
                continue
            rows[float(r["SynOps Budget"])] = {
                "accuracy": float(r["Accuracy After"]),
                "synops": float(r["SynOps After"]),
                "params": float(r["Remaining Parameters"]),
            }
    return rows


def measure_dense_synops(model_name: str, batches: int) -> Optional[float]:
    """SynOps per sample of the unpruned trained baseline, same instrument
    as every pruned number in the summaries."""
    cfg = ALL_EXPERIMENTS[model_name]()
    ckpt = os.path.join(cfg.output_dir, "trained_model.pt")
    if not os.path.isfile(ckpt):
        print(f"(no baseline checkpoint at {ckpt}; skipping dense measurement)")
        return None
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, _, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    return measure_synops(model, test_loader, device, max_batches=batches)["synops_per_sample"]


def fmt(v: Optional[float], dense: Optional[float]) -> str:
    if v is None:
        return f"{'--':>10} {'--':>8}"
    pct = f"{100 * v / dense:7.1f}%" if dense else f"{'':8}"
    return f"{v / 1e6:9.1f}M {pct}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpap_repl", choices=list(ALL_EXPERIMENTS))
    parser.add_argument("--kl-only-tag", default="synops",
                        help="tag of the run with KL-only gates + budgeted selection")
    parser.add_argument("--uniform-tag", default="uniformcost0.5",
                        help="tag of the uniform-cost ablation run")
    parser.add_argument("--real-tag", default="lagrangian0.5",
                        help="tag of the real-SynOps-cost budget run")
    parser.add_argument("--dense-batches", type=int, default=4,
                        help="test batches used to measure the dense baseline's SynOps")
    parser.add_argument("--skip-dense", action="store_true",
                        help="skip the dense measurement (no %%-of-dense column)")
    args = parser.parse_args()

    cfg = ALL_EXPERIMENTS[args.model]()
    base = cfg.output_dir
    arms = [
        ("KL only (budget at selection)", args.kl_only_tag),
        ("+ budget in loss, uniform costs", args.uniform_tag),
        ("+ budget in loss, real SynOps", args.real_tag),
    ]
    data = {
        label: read_importance_rows(os.path.join(base, f"sparsity_curve_{tag}", "summary.csv"))
        for label, tag in arms
    }
    for (label, tag) in arms:
        if not data[label]:
            print(f"MISSING: no importance rows for '{label}' "
                  f"(expected {base}/sparsity_curve_{tag}/summary.csv)")
    if not data[arms[1][0]]:
        print(
            "\nThe uniform-cost ablation has not been run. Submit it with:\n"
            f"  sbatch slurm_curve.sh --model {args.model} --tag {args.uniform_tag} "
            "--synops-loss-budget 0.5 --synops-uniform-costs "
            "--synops-budgets 0.5 0.3 --synops-rank-by importance\n"
        )

    dense = None if args.skip_dense else measure_dense_synops(args.model, args.dense_batches)
    if dense:
        print(f"\nDense baseline SynOps/sample (measured, {args.dense_batches} test batches): "
              f"{dense / 1e6:.1f}M")

    budgets = sorted({b for rows in data.values() for b in rows}, reverse=True)
    for b in budgets:
        print(f"\n=== budget {b:g} of measured marginal cost ===")
        print(f"{'gates':<34} {'accuracy':>9} {'SynOps/sample':>20}")
        for label, _ in arms:
            row = data[label].get(b)
            acc = f"{100 * row['accuracy']:8.2f}%" if row else f"{'--':>9}"
            print(f"{label:<34} {acc} {fmt(row['synops'] if row else None, dense):>20}")

        kl = data[arms[0][0]].get(b)
        uni = data[arms[1][0]].get(b)
        real = data[arms[2][0]].get(b)
        if kl and real:
            print(f"\n  total effect of budget-in-loss:  "
                  f"{100 * (real['accuracy'] - kl['accuracy']):+6.2f}pp accuracy, "
                  f"{(real['synops'] - kl['synops']) / 1e6:+8.1f}M SynOps")
        if kl and uni:
            print(f"  from generic pressure alone:     "
                  f"{100 * (uni['accuracy'] - kl['accuracy']):+6.2f}pp accuracy, "
                  f"{(uni['synops'] - kl['synops']) / 1e6:+8.1f}M SynOps")
        if uni and real:
            print(f"  from the SynOps information:     "
                  f"{100 * (real['accuracy'] - uni['accuracy']):+6.2f}pp accuracy, "
                  f"{(real['synops'] - uni['synops']) / 1e6:+8.1f}M SynOps")

    print(
        "\nCaveat to carry into the write-up: the uniform-cost arm removes the\n"
        "cost information but its training-time constraint ('at most B of the\n"
        "units') is not numerically the same tightness as 'at most B of the\n"
        "SynOps'. All arms are evaluated at the same measured budgets; the\n"
        "attribution is about what the loss knew during training."
    )


if __name__ == "__main__":
    main()
