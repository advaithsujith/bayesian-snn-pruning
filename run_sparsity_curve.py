"""
Accuracy-vs-sparsity curve from a single gate-training run.

Why this replaces sweeping beta_max
------------------------------------
Under the threshold rule (`log_alpha > 3.0`) the sparsity a run lands on is
an *outcome*: the only way to reach a different one is to re-run gate
training at a different `beta_max` and hope. That is what `sweep_beta.py`
was for, at roughly two GPU-hours per point, and it cannot hit a specified
number even then -- the same beta_max produced 27.7% on LeNet and 98.8% on
VGG9.

Ranking the gates instead makes sparsity an *input*. The criterion is
unchanged (still posterior uncertainty, still the same learned log_alpha);
only the cut point moves. So one gate-training run yields every operating
point on the curve, each costing a fine-tune rather than a fine-tune plus
gate training. It also makes three things true that the dissertation's
argument needs:

* **Matched sparsity by construction.** `--mode uniform` keeps the same
  fraction of units per layer as `activity_pruning.select_keep_mask` does,
  using the same rounding, so a Bayesian and a bio-inspired run at the
  same keep_fraction produce networks of identical width. The only
  difference left between them is which units were chosen -- which is the
  comparison the project claims to be making.
* **Published operating points are reachable.** DPAP reports 94.27% at
  33.46% pruned and 93.83% at 50.80%; `--targets 33.46 50.80` lands on
  those directly instead of near them.
* **`beta_max` stops needing to be correct.** It no longer has to land the
  sparsity on a target -- it only has to make the gates *differentiate*
  from each other. That is a far weaker requirement, but not a vacuous
  one: a run whose gates all saturate at the clamp has no usable ordering
  left, so this script refuses to interpret one. See `--plan-only` and the
  saturation warning.

Usage
-----
    # zero GPU: what keep_fraction / layer widths does each target imply?
    python run_sparsity_curve.py --model lenet --targets 27.74 50 90 --plan-only

    # the real thing: gate-train once, then rebuild + fine-tune per target
    python run_sparsity_curve.py --model dpap_repl --targets 33.46 50.80

    # reuse gates trained by an earlier invocation or by run_all.py
    python run_sparsity_curve.py --model dpap_repl --targets 20 40 60 --reuse-gates

Writes `outputs/<model>/sparsity_curve/summary.csv` plus per-target
per-layer structure reports.
"""

import argparse
import os
from typing import Any, Dict, List

import torch

from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS, ExperimentConfig
from datasets import get_cifar10_loaders, get_pruning_phase_loaders
from evaluate import full_evaluation
from metrics import (
    compute_and_set_unit_costs,
    count_parameters,
    pruning_percentage,
    write_csv,
)
from models import build_model
from pruning import (
    compute_uncertainty_report,
    param_target_plan,
    prune_model,
    remaining_structures_report,
)
from train import run_training
from utils import ensure_dirs, get_device, print_banner, set_seed, setup_logger

GATE_CHECKPOINT = "bayesian_model.pt"


def build_and_load(model_name: str, cfg: ExperimentConfig, device: torch.device, path: str):
    """Build the architecture and load a checkpoint into it."""
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    compute_and_set_unit_costs(model, (cfg.latency_batch_size, 3, 32, 32), device)
    return model


def plan_only(model_name: str, cfg: ExperimentConfig, targets: List[float], mode: str) -> None:
    """Report what each target sparsity implies, without touching a GPU or
    any data: the keep_fraction, the realised parameter percentage, and the
    resulting per-layer widths.

    The keep_fraction is the number to hand to `BioPruningConfig.keep_fractions`
    for a matched-sparsity bio run. Reading it off by eye is what produced
    the current mismatch, where every bio run sits at 98.5% pruned and is
    compared against Bayesian runs at 27.7%, 90.6% and 98.8%.
    """
    device = torch.device("cpu")
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    total = count_parameters(model, exclude_gates=True)
    print(f"\n{model_name}: {total:,} parameters\n")

    if mode == "global":
        # A freshly built model has every log_alpha at log_alpha_init, so a
        # global ranking is pure index order and will happily assign an
        # entire layer's worth of pruning to whichever layer comes last --
        # on LeNet it takes fc2 down to a single unit, reproducing the
        # collapse in HANDOFF.md bug #6. Layer *widths* under --mode uniform
        # are geometry and do not depend on the gates at all, so those stay
        # meaningful here; a global plan does not.
        print(
            "NOTE: --mode global on an untrained model ranks by index order, since\n"
            "      every gate starts at the same log_alpha. The widths below are\n"
            "      arbitrary. Use --mode uniform for planning, and read global\n"
            "      allocations off a real gate-trained checkpoint instead.\n"
        )

    for target in targets:
        plan, achieved = param_target_plan(
            model, model_name, cfg.snn, target, mode=mode,
            min_keep=cfg.bayesian.min_keep_per_layer,
        )
        print(f"target {target:.2f}% pruned -> {plan.describe()}")
        report = remaining_structures_report(model, plan)
        for name, stats in report.items():
            flag = "" if stats["structurally_prunable"] else "   (not prunable)"
            print(f"    {name:<28} {stats['remaining']:>5} / {stats['total']:<5}{flag}")
        print()


def diagnose_only(model_name: str, cfg: ExperimentConfig) -> None:
    """
    Pre-flight: on the real pretrained baseline, is the task loss able to
    push back against the KL at all?

    One batch, well under a minute. Worth running before every curve
    submission, because it is the difference between a wasted queue slot
    and a diagnosis. A ratio of order 1 means the two terms are in a
    contest and gates will settle somewhere informative. A ratio several
    orders of magnitude below 1 means the KL is unopposed in that layer,
    its gates will run to the clamp ceiling whatever `beta_max` is, and the
    ranking the curve depends on will be ties broken by index order. That
    is what a gate misplaced before a normalisation looks like, and it is
    what three collapsed DPAP runs and a four-point beta sweep were spent
    discovering the expensive way.
    """
    device = get_device(cfg.device)
    pretrained_path = os.path.join(cfg.output_dir, "trained_model.pt")
    if not os.path.isfile(pretrained_path):
        raise SystemExit(f"No pretrained baseline at '{pretrained_path}'.")

    set_seed(cfg.seed)
    model = build_and_load(model_name, cfg, device, pretrained_path)
    set_bayesian_mode(model, True)
    train_loader, _, _ = get_pruning_phase_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    images, targets = next(iter(train_loader))

    from train import gate_pressure_diagnostic

    pressure = gate_pressure_diagnostic(
        model, images.to(device), targets.to(device),
        beta=cfg.bayesian.beta_max, loss_type=cfg.pruning_loss(),
    )
    print(f"\nGate pressure on the pretrained baseline, at beta_max={cfg.bayesian.beta_max:g}:\n")
    print(f"{'layer':<28} {'|d task/d la|':>15} {'beta*|d KL/d la|':>18} {'ratio':>10}")
    for name, s in pressure.items():
        print(
            f"{name:<28} {s['task_grad']:>15.3e} {s['kl_grad_scaled']:>18.3e} "
            f"{s['ratio']:>10.4f}"
        )
    worst = min(pressure.values(), key=lambda s: s["ratio"])["ratio"] if pressure else 0.0
    print(
        f"\nWeakest layer ratio: {worst:.3e}. Below ~1e-3 the KL is effectively "
        "unopposed there\nand no beta_max will help -- diagnose before spending a "
        "queue slot on the curve."
    )


def run_curve(args: argparse.Namespace) -> None:
    cfg: ExperimentConfig = ALL_EXPERIMENTS[args.model]()
    if args.plan_only:
        plan_only(args.model, cfg, args.targets, args.mode)
        return
    if args.diagnose_only:
        diagnose_only(args.model, cfg)
        return

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = os.path.join(cfg.output_dir, "sparsity_curve")
    ensure_dirs(cfg.checkpoint_dir, cfg.output_dir, cfg.log_dir, out_dir)
    logger = setup_logger(f"{args.model}_curve", cfg.log_dir, f"{args.model}_curve.log")

    pretrained_path = os.path.join(cfg.output_dir, "trained_model.pt")
    gate_path = os.path.join(out_dir, GATE_CHECKPOINT)

    # Test-set evaluation only; every training-time decision uses the
    # pruning-phase split (see datasets.get_pruning_phase_loaders).
    _, _, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    train_loader, val_loader, _ = get_pruning_phase_loaders(
        cfg.data, cfg.train.batch_size, cfg.seed
    )

    csv_rows: List[Dict[str, Any]] = []

    if args.reuse_gates and os.path.isfile(gate_path):
        print(f"\nReusing trained gates: {gate_path}")
        logger.info(f"Reusing trained gates from {gate_path}")
        model = build_and_load(args.model, cfg, device, gate_path)
        set_bayesian_mode(model, True)
    else:
        if not os.path.isfile(pretrained_path):
            raise SystemExit(
                f"No pretrained baseline at '{pretrained_path}'. Run run_all.py for "
                f"'{args.model}' first -- this script deliberately does not pretrain."
            )
        print(f"\nLoading pretrained baseline: {pretrained_path}")
        model = build_and_load(args.model, cfg, device, pretrained_path)

        print("\nTraining Bayesian gates (once, for every target on the curve)...")
        set_bayesian_mode(model, True)
        run_training(
            model, train_loader, val_loader,
            epochs=cfg.bayesian.bayesian_train_epochs,
            lr=cfg.bayesian.bayesian_train_lr,
            weight_decay=(
                cfg.train.weight_decay
                if cfg.bayesian.bayesian_train_weight_decay is None
                else cfg.bayesian.bayesian_train_weight_decay
            ),
            optimizer_name=cfg.train.optimizer,
            scheduler_name=cfg.train.lr_scheduler,
            grad_clip_norm=cfg.train.grad_clip_norm,
            use_amp=cfg.train.use_amp, device=device,
            beta_max=cfg.bayesian.beta_max,
            kl_warmup_epochs=cfg.bayesian.kl_warmup_epochs,
            gamma_max=cfg.bayesian.gamma_max,
            cost_warmup_epochs=cfg.bayesian.cost_warmup_epochs,
            loss_type=cfg.pruning_loss(),
            logger=logger,
            checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{args.model}_curve_gates.pt"),
            csv_log_rows=csv_rows, phase_name="bayesian_train",
            prune_threshold=cfg.bayesian.prune_threshold,
            restore_best_checkpoint=False,
            gate_diagnostic_every=5,
        )
        torch.save(model.state_dict(), gate_path)

    write_csv(csv_rows, os.path.join(out_dir, "gate_training_log.csv"))

    report = compute_uncertainty_report(model, cfg.bayesian.prune_threshold)
    for name, stats in report.items():
        logger.info(
            f"  {name}: median_log_alpha={stats['median']:.2f} std={stats['std']:.2f} "
            f"frac_saturated={stats['frac_saturated']:.3f} "
            f"structurally_prunable={stats['structurally_prunable']}"
        )

    original_params = count_parameters(model, exclude_gates=True)
    results: List[Dict[str, Any]] = []

    for i, target in enumerate(args.targets, start=1):
        tag = f"target{target:g}"
        print(f"\n--- point {i}/{len(args.targets)}: {target:g}% of parameters pruned ---")

        plan, achieved = param_target_plan(
            model, args.model, cfg.snn, target, mode=args.mode,
            min_keep=cfg.bayesian.min_keep_per_layer,
        )
        saturation = plan.saturation_report(model)
        logger.info(f"[{tag}] {plan.describe()}")
        if saturation["frac_saturated_high"] > 0.5:
            # Not fatal, but the resulting point is not evidence about the
            # criterion -- the ranking that produced it was mostly ties.
            logger.warning(
                f"[{tag}] {saturation['frac_saturated_high']:.1%} of gates are pinned at "
                "the clamp ceiling; this keep-set is largely index order, not a "
                "learned ranking. Fix gate training before reporting this point."
            )

        structures = remaining_structures_report(model, plan)
        dead_layers = sum(
            1
            for s in structures.values()
            if s["structurally_prunable"] and s["remaining"] <= cfg.bayesian.min_keep_per_layer
        )

        pruned = prune_model(model, args.model, plan, cfg.snn).to(device)
        remaining_params = count_parameters(pruned, exclude_gates=True)

        finetune_rows: List[Dict[str, Any]] = []
        finetune = run_training(
            pruned, train_loader, val_loader,
            epochs=cfg.finetune.epochs, lr=cfg.finetune.lr,
            weight_decay=cfg.finetune.weight_decay,
            optimizer_name=cfg.finetune.optimizer,
            scheduler_name=cfg.finetune.lr_scheduler,
            grad_clip_norm=cfg.finetune.grad_clip_norm,
            use_amp=cfg.finetune.use_amp, device=device,
            beta_max=0.0, kl_warmup_epochs=1, gamma_max=0.0, cost_warmup_epochs=1,
            loss_type=cfg.pruning_loss(),
            logger=logger,
            checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{args.model}_{tag}_finetuned.pt"),
            csv_log_rows=finetune_rows, phase_name="finetune",
        )

        eval_after = full_evaluation(
            pruned, test_loader, device,
            (cfg.latency_batch_size, 3, 32, 32),
            cfg.latency_num_warmup, cfg.latency_num_runs,
        )
        logger.info(f"[{tag}] test accuracy {eval_after['test_accuracy']:.4f}")

        point_dir = os.path.join(out_dir, tag)
        ensure_dirs(point_dir)
        write_csv(finetune_rows, os.path.join(point_dir, "finetune_log.csv"))
        write_csv(
            [{"layer": n, **s} for n, s in structures.items()],
            os.path.join(point_dir, "remaining_structures.csv"),
        )

        results.append({
            "Target Pruned %": target,
            "Achieved Pruned %": pruning_percentage(original_params, remaining_params),
            "Keep Plan": plan.describe(),
            "Mode": args.mode,
            "Original Parameters": original_params,
            "Remaining Parameters": remaining_params,
            "Accuracy After": eval_after["test_accuracy"],
            "Fine Tune Best Val": finetune["best_val_acc"],
            "Gate Saturation": saturation["frac_saturated_high"],
            "Layers At Min Width": dead_layers,
            "FLOPs": eval_after["flops"],
            "Latency": eval_after["latency_ms"],
        })
        # Written after every point, so a job that runs out of wallclock
        # still leaves the completed points behind.
        write_csv(results, os.path.join(out_dir, "summary.csv"))

    print_banner("SPARSITY CURVE FINISHED")
    print(f"{'pruned %':>10} {'accuracy':>10} {'saturation':>12} {'min-width layers':>18}")
    for r in sorted(results, key=lambda r: r["Achieved Pruned %"]):
        print(
            f"{r['Achieved Pruned %']:>10.2f} {r['Accuracy After']:>10.4f} "
            f"{r['Gate Saturation']:>12.3f} {r['Layers At Min Width']:>18d}"
        )
    print(f"\nWrote {os.path.join(out_dir, 'summary.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpap_repl", choices=list(ALL_EXPERIMENTS))
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[],
        help="target percentages of PARAMETERS to prune (not units, not keep_fractions)",
    )
    parser.add_argument(
        "--mode", default="uniform", choices=["uniform", "global"],
        help="uniform: same keep-fraction in every layer (matches the bio criteria's "
             "widths exactly). global: rank across the whole network, letting the "
             "criterion allocate sparsity between layers.",
    )
    parser.add_argument(
        "--reuse-gates", action="store_true",
        help="skip gate training and load outputs/<model>/sparsity_curve/bayesian_model.pt",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="print the keep_fraction and per-layer widths each target implies, then "
             "exit. No data, no GPU, no training.",
    )
    parser.add_argument(
        "--diagnose-only", action="store_true",
        help="print the per-layer task-vs-KL gradient balance on the pretrained "
             "baseline, then exit. One batch. Run this before submitting a curve.",
    )
    args = parser.parse_args()
    if not args.targets and not args.diagnose_only:
        parser.error("--targets is required unless --diagnose-only is given")
    run_curve(args)


if __name__ == "__main__":
    main()
