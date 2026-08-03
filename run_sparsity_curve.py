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

    # SynOps-budgeted selection: each budget runs cost-blind AND cost-aware
    # from the same gates -- the paired comparison at matched compute budget
    python run_sparsity_curve.py --model dpap_repl --synops-budgets 0.5 0.3 --reuse-gates

    # the two-arm gate-optimizer experiment (submit both, different tags)
    python run_sparsity_curve.py --model dpap_repl --targets 33.46 50.80 --tag adam
    python run_sparsity_curve.py --model dpap_repl --targets 33.46 50.80 \
        --gate-optimizer sgd --gate-lr 0.05 --tag sgd

    # SynOps budget inside the gate-training loss (dual-ascent Lagrangian)
    python run_sparsity_curve.py --model dpap_repl --synops-budgets 0.5 \
        --synops-loss-budget 0.5 --tag synopsloss

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
    measure_synops_unit_costs,
    pruning_percentage,
    set_synops_unit_costs,
    write_csv,
)
from models import build_model
from pruning import (
    KeepPlan,
    compute_uncertainty_report,
    param_target_plan,
    plan_synops_fraction,
    prune_model,
    remaining_structures_report,
    synops_budget_plan,
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
    # Tagged output directories, because successive runs of this script differ
    # by exactly the hyperparameters under investigation and a shared path
    # means each one silently deletes the last. That has already cost this
    # project one set of quoted VGG9 figures.
    out_dir = os.path.join(cfg.output_dir, f"sparsity_curve_{args.tag}" if args.tag else "sparsity_curve")
    ensure_dirs(cfg.checkpoint_dir, cfg.output_dir, cfg.log_dir, out_dir)
    logger = setup_logger(f"{args.model}_curve", cfg.log_dir, f"{args.model}_curve.log")

    pretrained_path = os.path.join(cfg.output_dir, "trained_model.pt")
    # Gates are always *saved* into this run's own directory; they may be
    # *loaded* from another run's, so a follow-up analysis (e.g. the SynOps
    # budget pair) can reuse an expensive gate phase without sharing an
    # output directory with it -- sharing one would overwrite that run's
    # summary.csv, which is how this project previously lost a set of
    # quoted figures.
    gate_path = os.path.join(out_dir, GATE_CHECKPOINT)
    reuse_gate_path = (
        os.path.join(cfg.output_dir, f"sparsity_curve_{args.gates_from}", GATE_CHECKPOINT)
        if args.gates_from
        else gate_path
    )

    # Test-set evaluation only; every training-time decision uses the
    # pruning-phase split (see datasets.get_pruning_phase_loaders).
    _, _, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    train_loader, val_loader, _ = get_pruning_phase_loaders(
        cfg.data, cfg.train.batch_size, cfg.seed
    )

    csv_rows: List[Dict[str, Any]] = []

    # CLI overrides fall back to the config's values, so a submitted job can
    # flip one arm's optimizer or budget without editing config.py -- that is
    # what makes a two-arm (Adam vs SGD-gates) submission two sbatch lines.
    gate_optimizer_name = args.gate_optimizer or cfg.bayesian.gate_optimizer
    gate_lr = args.gate_lr if args.gate_lr is not None else cfg.bayesian.gate_lr
    synops_loss_budget = (
        args.synops_loss_budget
        if args.synops_loss_budget is not None
        else cfg.bayesian.synops_budget_fraction
    )
    synops_refresh_fn = None
    if synops_loss_budget is not None:
        if cfg.bayesian.gamma_max > 0.0:
            raise SystemExit(
                "The SynOps budget term and gamma_max are mutually exclusive: both "
                "write to the same unit_cost buffers. Set gamma_max=0 for this run."
            )

        def synops_refresh_fn(m):
            set_synops_unit_costs(
                m,
                measure_synops_unit_costs(
                    m, args.model, train_loader, device,
                    max_batches=cfg.bayesian.synops_measure_batches,
                ),
            )

    if args.reuse_gates and os.path.isfile(reuse_gate_path):
        print(f"\nReusing trained gates: {reuse_gate_path}")
        logger.info(f"Reusing trained gates from {reuse_gate_path}")
        model = build_and_load(args.model, cfg, device, reuse_gate_path)
        set_bayesian_mode(model, True)
    elif args.gates_from:
        # Explicitly named a source and it is not there: training fresh gates
        # instead would silently produce a different experiment.
        raise SystemExit(
            f"No gate checkpoint at '{reuse_gate_path}' (from --gates-from "
            f"{args.gates_from}). Check the tag; refusing to retrain gates "
            "when reuse was explicitly requested."
        )
    else:
        if not os.path.isfile(pretrained_path):
            raise SystemExit(
                f"No pretrained baseline at '{pretrained_path}'. Run run_all.py for "
                f"'{args.model}' first -- this script deliberately does not pretrain."
            )
        print(f"\nLoading pretrained baseline: {pretrained_path}")
        model = build_and_load(args.model, cfg, device, pretrained_path)

        print("\nTraining Bayesian gates (once, for every target on the curve)...")
        if gate_optimizer_name != "inherit":
            print(f"  gate optimizer: {gate_optimizer_name} (gate_lr={gate_lr})")
        if synops_loss_budget is not None:
            print(f"  SynOps budget term active: {synops_loss_budget:g} of dense SynOps")
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
            checkpoint_path=os.path.join(
                cfg.checkpoint_dir,
                f"{args.model}_{args.tag + '_' if args.tag else ''}curve_gates.pt",
            ),
            csv_log_rows=csv_rows, phase_name="bayesian_train",
            prune_threshold=cfg.bayesian.prune_threshold,
            restore_best_checkpoint=False,
            gate_diagnostic_every=5,
            gate_optimizer_name=gate_optimizer_name,
            gate_lr=gate_lr,
            synops_budget_fraction=synops_loss_budget,
            synops_lambda_lr=cfg.bayesian.synops_lambda_lr,
            synops_recount_every=cfg.bayesian.synops_recount_every,
            synops_refresh_fn=synops_refresh_fn,
        )
        torch.save(model.state_dict(), gate_path)

    write_csv(csv_rows, os.path.join(out_dir, "gate_training_log.csv"))

    # Judge the gates before spending a fine-tune on every target. The curve
    # is built from widths, which are an input, so an unusable ranking still
    # produces a clean monotone curve -- it just measures random structured
    # pruning instead of the criterion. Better to be told here than to read
    # it off five finished points.
    usable_now, reason_now = KeepPlan({}, "probe").ranking_is_usable(model)
    banner = "USABLE" if usable_now else "NOT USABLE"
    print(f"\nGate ranking: {banner} ({reason_now})")
    logger.info(f"Gate ranking after training: {banner} -- {reason_now}")
    if not usable_now and not args.force:
        raise SystemExit(
            f"\nStopping before the fine-tunes: {reason_now}.\n\n"
            "The curve would still come out clean and monotone, because the layer\n"
            "widths are an input rather than something the gates decide. It would\n"
            "just be measuring random structured pruning, which is a legitimate\n"
            "control but is not evidence about the criterion.\n\n"
            "Lower beta_max so the gates settle at an equilibrium instead of\n"
            "marching into the clamp, or pass --force to build the curve anyway\n"
            "and keep it explicitly as a random-pruning baseline."
        )

    report = compute_uncertainty_report(model, cfg.bayesian.prune_threshold)
    for name, stats in report.items():
        logger.info(
            f"  {name}: median_log_alpha={stats['median']:.2f} std={stats['std']:.2f} "
            f"frac_saturated={stats['frac_saturated']:.3f} "
            f"structurally_prunable={stats['structurally_prunable']}"
        )

    original_params = count_parameters(model, exclude_gates=True)
    results: List[Dict[str, Any]] = []

    def execute_point(
        plan: KeepPlan,
        tag: str,
        label: str,
        selection: str,
        synops_budget: "float | str" = "",
        kept_synops_fraction: "float | str" = "",
    ) -> None:
        """Rebuild, fine-tune, evaluate and record one curve point.

        Every row carries the same key set no matter which loop produced
        it: write_csv takes the CSV header from the first row, so rows
        with differing keys would crash the writer mid-run."""
        saturation = plan.saturation_report(model)
        usable, reason = plan.ranking_is_usable(model)
        logger.info(f"[{tag}] {plan.describe()}")
        if not usable:
            # Not fatal, but the resulting point is not evidence about the
            # criterion -- the ranking that produced it was mostly ties.
            logger.warning(f"[{tag}] RANKING NOT USABLE: {reason}.")

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
            # Tag-qualified: two curve runs can differ only in their --tag
            # (e.g. one trained under a SynOps budget, one not) and would
            # otherwise write their per-point checkpoints to the same paths,
            # so concurrent jobs would overwrite each other's.
            checkpoint_path=os.path.join(
                cfg.checkpoint_dir,
                f"{args.model}_{args.tag + '_' if args.tag else ''}{tag}_finetuned.pt",
            ),
            csv_log_rows=finetune_rows, phase_name="finetune",
        )

        eval_after = full_evaluation(
            pruned, test_loader, device,
            (cfg.latency_batch_size, 3, 32, 32),
            cfg.latency_num_warmup, cfg.latency_num_runs,
        )
        logger.info(
            f"[{tag}] test accuracy {eval_after['test_accuracy']:.4f}, "
            f"synops/sample {eval_after['synops']:,.0f}"
        )

        point_dir = os.path.join(out_dir, tag)
        ensure_dirs(point_dir)
        write_csv(finetune_rows, os.path.join(point_dir, "finetune_log.csv"))
        write_csv(
            [{"layer": n, **s} for n, s in structures.items()],
            os.path.join(point_dir, "remaining_structures.csv"),
        )

        results.append({
            "Point": label,
            "Selection": selection,
            "Achieved Pruned %": pruning_percentage(original_params, remaining_params),
            "SynOps Budget": synops_budget,
            "Kept SynOps Fraction": kept_synops_fraction,
            "Keep Plan": plan.describe(),
            "Original Parameters": original_params,
            "Remaining Parameters": remaining_params,
            "Accuracy After": eval_after["test_accuracy"],
            "Fine Tune Best Val": finetune["best_val_acc"],
            "SynOps After": eval_after["synops"],
            "Dense MACs After": eval_after["dense_macs"],
            "Gate Saturation": saturation["frac_saturated_high"],
            "Log Alpha Std": saturation["log_alpha_std"],
            "Ranking Usable": usable,
            "Layers At Min Width": dead_layers,
            "FLOPs": eval_after["flops"],
            "Latency": eval_after["latency_ms"],
        })
        # Written after every point, so a job that runs out of wallclock
        # still leaves the completed points behind.
        write_csv(results, os.path.join(out_dir, "summary.csv"))

    for i, target in enumerate(args.targets, start=1):
        tag = f"target{target:g}"
        print(f"\n--- target point {i}/{len(args.targets)}: {target:g}% of parameters pruned ---")
        plan, achieved = param_target_plan(
            model, args.model, cfg.snn, target, mode=args.mode,
            min_keep=cfg.bayesian.min_keep_per_layer,
        )
        execute_point(plan, tag, f"params {target:g}%", f"log_alpha/{args.mode}")

    if args.synops_budgets:
        # One measurement serves every budget point: the costs describe the
        # gate-trained model being pruned, and both arms must price units
        # identically or the comparison is confounded.
        print("\nMeasuring per-unit SynOps costs on the gate-trained model...")
        unit_costs = measure_synops_unit_costs(
            model, args.model, train_loader, device, max_batches=args.synops_batches
        )
        rank_rules = (
            ("importance", "density") if args.synops_rank_by == "both"
            else (args.synops_rank_by,)
        )
        for i, budget in enumerate(args.synops_budgets, start=1):
            # importance (cost-blind control) first, density (cost-aware)
            # second: if wallclock dies between them, the control exists.
            for rank_by in rank_rules:
                tag = f"synops{budget:g}_{rank_by}"
                print(
                    f"\n--- budget point {i}/{len(args.synops_budgets)} ({rank_by}): "
                    f"{budget:g} of dense SynOps ---"
                )
                plan = synops_budget_plan(
                    model, unit_costs, budget, rank_by=rank_by,
                    min_keep=cfg.bayesian.min_keep_per_layer,
                )
                kept = plan_synops_fraction(model, plan, unit_costs)
                execute_point(plan, tag, f"synops {budget:g}", rank_by, budget, kept)

    print_banner("SPARSITY CURVE FINISHED")
    print(
        f"{'point':>16} {'selection':>18} {'pruned %':>10} {'accuracy':>10} "
        f"{'synops':>14} {'usable':>8}"
    )
    for r in sorted(results, key=lambda r: r["Achieved Pruned %"]):
        print(
            f"{r['Point']:>16} {r['Selection']:>18} {r['Achieved Pruned %']:>10.2f} "
            f"{r['Accuracy After']:>10.4f} {r['SynOps After']:>14,.0f} "
            f"{('yes' if r['Ranking Usable'] else 'NO'):>8}"
        )
    if results and not results[0]["Ranking Usable"]:
        print(
            "\nThe gates did not produce a usable ranking, so the curve above is\n"
            "what random structured pruning of this architecture looks like. That\n"
            "is a legitimate control and worth keeping under that name, but it is\n"
            "not evidence about the pruning criterion. Fix gate training first."
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
        help="skip gate training and load this run's own "
             "outputs/<model>/sparsity_curve_<tag>/bayesian_model.pt",
    )
    parser.add_argument(
        "--gates-from", default="",
        help="reuse gates trained by a DIFFERENT tagged run (give that run's "
             "--tag). Results still go to this run's own --tag directory, so "
             "the source run's summary.csv is not overwritten. Implies "
             "--reuse-gates.",
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
    parser.add_argument(
        "--tag", default="",
        help="suffix for the output directory, e.g. --tag beta0.01. Keeps "
             "successive runs from overwriting each other.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="build the curve even if the gates produced no usable ranking. Only "
             "meaningful for deliberately generating a random-pruning control.",
    )
    parser.add_argument(
        "--synops-budgets", type=float, nargs="+", default=[],
        help="SynOps budgets as fractions of the dense network's measured SynOps "
             "(e.g. 0.5 0.3). Each budget is run twice from the same gates: once "
             "cost-blind (importance ranking) and once cost-aware (density "
             "ranking) -- that pair at matched budget IS the contribution "
             "comparison.",
    )
    parser.add_argument(
        "--synops-rank-by", choices=["both", "importance", "density"], default="both",
        help="which selection rule(s) to run at each budget. 'both' (default) "
             "gives the paired cost-blind/cost-aware comparison. Use "
             "'importance' alone to skip the density arm -- measured on "
             "dpap_repl, density is close to an anti-ranking there because "
             "per-unit cost and posterior importance are positively "
             "correlated across layers (the expensive wide mid-network convs "
             "are also the ones the criterion values most), so it buys cheap "
             "units in the layers that matter least.",
    )
    parser.add_argument(
        "--synops-batches", type=int, default=8,
        help="training-loader batches used to measure per-unit SynOps costs for "
             "the budgeted selection.",
    )
    parser.add_argument(
        "--synops-loss-budget", type=float, default=None,
        help="enable the SynOps-budget Lagrangian term DURING gate training, "
             "constraining expected SynOps to this fraction of the dense "
             "network's (see BayesianConfig.synops_budget_fraction). Overrides "
             "the config value; leave unset to use the config (default: off).",
    )
    parser.add_argument(
        "--gate-optimizer", choices=["inherit", "sgd"], default=None,
        help="override BayesianConfig.gate_optimizer for this run: 'sgd' trains "
             "log_alpha with plain SGD while the weights keep the configured "
             "optimizer. See train.build_gate_split_optimizers for why.",
    )
    parser.add_argument(
        "--gate-lr", type=float, default=None,
        help="learning rate for the split gate optimizer (only meaningful with "
             "--gate-optimizer sgd). Overrides BayesianConfig.gate_lr.",
    )
    args = parser.parse_args()
    if not args.targets and not args.synops_budgets and not args.diagnose_only:
        parser.error(
            "--targets and/or --synops-budgets is required unless --diagnose-only is given"
        )
    if args.gates_from:
        args.reuse_gates = True
        if args.gates_from == args.tag:
            parser.error(
                "--gates-from names the same tag as --tag, which would write this "
                "run's results over the source run's. Use plain --reuse-gates if "
                "that is intended."
            )
    run_curve(args)


if __name__ == "__main__":
    main()
