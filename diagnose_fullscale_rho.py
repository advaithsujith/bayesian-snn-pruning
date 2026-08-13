"""
Measure whether a full-scale trained gate ranking carries real signal,
regardless of what its pooled std looks like.

KeepPlan.ranking_is_usable refuses a ranking whose log_alpha std is under
0.25, a bar calibrated on dpap_repl's Adam-marched gates. The Adam runs on
the SPEAR platforms produced std 0.05-0.08: too uniform for the bar, but a
tiny spread can still be a *consistent ordering*. This script tests that
directly: it measures ground-truth per-channel importance on the trained
baseline by ablation (zero one channel's hard_mask, measure the val-loss
increase) for a per-layer sample of channels, then Spearman-correlates
those importances with the trained -log_alpha from a gate checkpoint.

rho clearly positive => the ranking is weak but real, the std bar is
miscalibrated for this regime, and the budgeted fine-tunes are worth
running on these gates. rho ~ 0 => the ranking really is arbitrary and the
NOT USABLE verdicts stand.

Runs the ablation on the *pretrained baseline* (trained_model.pt), i.e. the
network the ranking is supposed to describe. Pass --ablate-gates-model to
ablate the gate-phase weights instead (the network the rebuild actually
slices), if the two disagree that is itself worth knowing.

Example:
    python diagnose_fullscale_rho.py --model spear_repl_resnet18 --tag lagrw
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from bayesian_layers import collect_prunable_bayesian_layers
from config import ALL_EXPERIMENTS
from datasets import get_pruning_phase_loaders
from losses import get_task_loss
from models import build_model


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def build_from(model_name, cfg, device, path):
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def batched_loss(model, images, targets, loss_fn, batch_size):
    total, n = 0.0, 0
    for i in range(0, images.shape[0], batch_size):
        xb, yb = images[i : i + batch_size], targets[i : i + batch_size]
        total += float(loss_fn(model(xb), yb)) * xb.shape[0]
        n += xb.shape[0]
    return total / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(ALL_EXPERIMENTS))
    p.add_argument("--tag", required=True, help="gate run tag; reads outputs/<model>/sparsity_curve_<tag>/bayesian_model.pt")
    p.add_argument("--max-per-layer", type=int, default=64, help="channels ablated per layer (all if the layer is smaller)")
    p.add_argument("--val-images", type=int, default=1024, help="held-out images the ablation loss is measured on")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="seed for the per-layer channel subsample")
    p.add_argument("--ablate-gates-model", action="store_true",
                   help="ablate the gate-phase weights instead of the pretrained baseline")
    args = p.parse_args()

    cfg = ALL_EXPERIMENTS[args.model]()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    loss_fn = get_task_loss(cfg.pruning_loss())

    baseline_path = os.path.join(cfg.output_dir, "trained_model.pt")
    gates_path = os.path.join(cfg.output_dir, f"sparsity_curve_{args.tag}", "bayesian_model.pt")
    for path in (baseline_path, gates_path):
        if not os.path.isfile(path):
            raise SystemExit(f"missing checkpoint: {path}")

    gates_model = build_from(args.model, cfg, device, gates_path)
    ablate_model = (
        gates_model if args.ablate_gates_model else build_from(args.model, cfg, device, baseline_path)
    )

    # The same held-out split every pruning decision uses; the test set stays
    # untouched here too.
    _, val_loader, _ = get_pruning_phase_loaders(cfg.data, args.batch_size, cfg.seed)
    xs, ys = [], []
    for xb, yb in val_loader:
        xs.append(xb)
        ys.append(yb)
        if sum(t.shape[0] for t in xs) >= args.val_images:
            break
    images = torch.cat(xs)[: args.val_images].to(device)
    targets = torch.cat(ys)[: args.val_images].to(device)

    base = batched_loss(ablate_model, images, targets, loss_fn, args.batch_size)
    print(f"{args.model} tag={args.tag}: base val loss {base:.4f} on {images.shape[0]} images "
          f"({'gate-phase' if args.ablate_gates_model else 'baseline'} weights ablated)")

    name_of = {m: n for n, m in ablate_model.named_modules()}
    gen = torch.Generator().manual_seed(args.seed)
    rows, all_imp, all_neg_la = [], [], []
    for a_layer, g_layer in zip(
        collect_prunable_bayesian_layers(ablate_model),
        collect_prunable_bayesian_layers(gates_model),
    ):
        n = a_layer.hard_mask.numel()
        chosen = (
            torch.arange(n)
            if n <= args.max_per_layer
            else torch.randperm(n, generator=gen)[: args.max_per_layer].sort().values
        )
        deltas = torch.zeros(chosen.numel())
        for k, j in enumerate(chosen.tolist()):
            a_layer.hard_mask[j] = 0.0
            deltas[k] = batched_loss(ablate_model, images, targets, loss_fn, args.batch_size) - base
            a_layer.hard_mask[j] = 1.0
        neg_la = -g_layer.log_alpha.detach().cpu()[chosen]
        rho = spearman(deltas, neg_la)
        lname = name_of[a_layer]
        print(
            f"  {lname:26s} n={chosen.numel():3d} rho={rho:+.2f} "
            f"imp={deltas.mean():.4f}+-{deltas.std():.4f} "
            f"log_alpha std(layer)={g_layer.log_alpha.detach().std():.3f}",
            flush=True,
        )
        all_imp.append(deltas)
        all_neg_la.append(neg_la)
        for k, j in enumerate(chosen.tolist()):
            rows.append({"layer": lname, "channel": j, "ablation_delta": float(deltas[k]),
                         "log_alpha": float(-neg_la[k])})

    pooled = spearman(torch.cat(all_imp), torch.cat(all_neg_la))
    print(f"\nPOOLED rho over {sum(t.numel() for t in all_imp)} sampled channels: {pooled:+.3f}")

    out_csv = os.path.join(cfg.output_dir, f"fullscale_rho_{args.tag}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "channel", "ablation_delta", "log_alpha"])
        w.writeheader()
        w.writerows(rows)
    print(f"per-channel data written to {out_csv}")


if __name__ == "__main__":
    main()
