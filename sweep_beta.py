"""
Sweep `beta_max` against a frozen pretrained baseline.

Why a sweep rather than repeated full runs
-------------------------------------------
`beta_max` sets how hard the KL term pushes gates toward pruning, and it
does not transfer between settings -- it has to be tuned per architecture
(HANDOFF.md records the original values needing exactly this, and the DPAP
replication needed it again for a further reason: gate noise is resampled
every timestep and the output sums over `num_steps`, so signal grows as T
while noise grows as sqrt(T). Tolerable alpha therefore scales with T, and
DPAP's 8 timesteps absorb only 8/25 of what VGG9's 25 do).

Pretraining is by far the most expensive stage and is completely
independent of `beta_max`, so this script loads one saved baseline and runs
only the stages that actually depend on it -- gate training, pruning,
fine-tuning, evaluation -- once per candidate value. Every trial therefore
forks from a bit-identical baseline, which is also what makes the trials
comparable to each other rather than merely to the paper.

Usage
-----
    python sweep_beta.py --model dpap_repl --betas 0.4 0.05 0.02 0.005

Writes `outputs/<model>/sweep_beta/summary.csv` plus a per-trial log, and
prints a ranked table at the end. Requires `outputs/<model>/trained_model.pt`
to exist (run_all.py writes it after the pretrain phase).
"""

import argparse
import copy
import os
from typing import Any, Dict, List

import torch

from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS, ExperimentConfig
from datasets import get_cifar10_loaders
from evaluate import full_evaluation
from metrics import (
    compute_and_set_unit_costs,
    count_parameters,
    count_remaining_structures,
    pruning_percentage,
    write_csv,
)
from models import build_model
from pruning import compute_uncertainty_report, prune_model
from train import run_training
from utils import ensure_dirs, get_device, print_banner, set_seed, setup_logger


def run_one_trial(
    model_name: str,
    cfg: ExperimentConfig,
    beta_max: float,
    pretrained_path: str,
    loaders,
    logger,
    out_dir: str,
) -> Dict[str, Any]:
    """Run gate-training -> prune -> fine-tune -> evaluate for one beta_max."""
    train_loader, val_loader, test_loader = loaders
    device = get_device(cfg.device)
    tag = f"beta{beta_max:g}"

    # Reset to the identical starting point for every trial, so differences
    # between trials are attributable to beta_max and nothing else.
    set_seed(cfg.seed)
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    input_shape = (cfg.latency_batch_size, 3, 32, 32)
    compute_and_set_unit_costs(model, input_shape, device)
    original_params = count_parameters(model, exclude_gates=True)

    csv_rows: List[Dict[str, Any]] = []
    set_bayesian_mode(model, True)

    logger.info(f"[{tag}] training gates at beta_max={beta_max}")
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
        beta_max=beta_max,
        kl_warmup_epochs=cfg.bayesian.kl_warmup_epochs,
        gamma_max=cfg.bayesian.gamma_max,
        cost_warmup_epochs=cfg.bayesian.cost_warmup_epochs,
        loss_type=cfg.pruning_loss(),
        logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_{tag}_bayesian.pt"),
        csv_log_rows=csv_rows, phase_name="bayesian_train",
        prune_threshold=cfg.bayesian.prune_threshold,
        restore_best_checkpoint=False,
    )

    report = compute_uncertainty_report(model, cfg.bayesian.prune_threshold)
    for name, s in report.items():
        logger.info(
            f"[{tag}]   {name}: median_log_alpha={s['median']:.2f} "
            f"frac_prunable={s['frac_prunable']:.3f}"
        )
    remaining_report = count_remaining_structures(model, cfg.bayesian.prune_threshold)

    pruned = prune_model(model, model_name, cfg.bayesian.prune_threshold, cfg.snn).to(device)
    remaining_params = count_parameters(pruned, exclude_gates=True)
    pct = pruning_percentage(original_params, remaining_params)
    logger.info(f"[{tag}] params {original_params} -> {remaining_params} ({pct:.1f}% pruned)")

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
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_{tag}_finetuned.pt"),
        csv_log_rows=csv_rows, phase_name="finetune",
    )

    eval_after = full_evaluation(
        pruned, test_loader, device, input_shape,
        cfg.latency_num_warmup, cfg.latency_num_runs,
    )
    logger.info(f"[{tag}] final test accuracy {eval_after['test_accuracy']:.4f}")

    trial_dir = os.path.join(out_dir, tag)
    ensure_dirs(trial_dir)
    write_csv(csv_rows, os.path.join(trial_dir, "training_log.csv"))
    write_csv(
        [{"layer": n, **s} for n, s in remaining_report.items()],
        os.path.join(trial_dir, "remaining_structures.csv"),
    )

    return {
        "beta_max": beta_max,
        "Original Parameters": original_params,
        "Remaining Parameters": remaining_params,
        "Pruning Percentage": pct,
        "Accuracy After": eval_after["test_accuracy"],
        "Fine Tune Best Val": finetune["best_val_acc"],
        "FLOPs": eval_after["flops"],
        "Latency": eval_after["latency_ms"],
        "Layers Fully Pruned": sum(
            1 for s in remaining_report.values() if s["remaining"] == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dpap_repl", choices=list(ALL_EXPERIMENTS))
    parser.add_argument(
        "--betas", type=float, nargs="+", required=True,
        help="beta_max values to try, in the order they should run",
    )
    args = parser.parse_args()

    base_cfg: ExperimentConfig = ALL_EXPERIMENTS[args.model]()
    ensure_dirs(base_cfg.checkpoint_dir, base_cfg.output_dir, base_cfg.log_dir)
    out_dir = os.path.join(base_cfg.output_dir, "sweep_beta")
    ensure_dirs(out_dir)
    logger = setup_logger(f"{args.model}_sweep", base_cfg.log_dir, f"{args.model}_sweep.log")

    pretrained_path = os.path.join(base_cfg.output_dir, "trained_model.pt")
    if not os.path.isfile(pretrained_path):
        raise SystemExit(
            f"No pretrained baseline at '{pretrained_path}'. Run run_all.py for "
            f"'{args.model}' first -- this script deliberately does not retrain, "
            "since that is the expensive stage the sweep exists to avoid repeating."
        )

    print_banner(f"beta_max sweep: {args.model}\n{len(args.betas)} trials from one frozen baseline")
    set_seed(base_cfg.seed)
    loaders = get_cifar10_loaders(base_cfg.data, base_cfg.train.batch_size, base_cfg.seed)

    results: List[Dict[str, Any]] = []
    for i, beta_max in enumerate(args.betas, start=1):
        print(f"\n--- trial {i}/{len(args.betas)}: beta_max={beta_max:g} ---")
        cfg = copy.deepcopy(base_cfg)
        results.append(
            run_one_trial(args.model, cfg, beta_max, pretrained_path, loaders, logger, out_dir)
        )
        # Written after every trial, so a job that runs out of wallclock still
        # leaves the completed trials behind.
        write_csv(results, os.path.join(out_dir, "summary.csv"))

    print_banner("SWEEP FINISHED")
    print(f"{'beta_max':>10} {'pruned %':>10} {'accuracy':>10} {'dead layers':>12}")
    for r in sorted(results, key=lambda r: -r["Accuracy After"]):
        print(
            f"{r['beta_max']:>10g} {r['Pruning Percentage']:>10.2f} "
            f"{r['Accuracy After']:>10.4f} {r['Layers Fully Pruned']:>12d}"
        )
    print(f"\nWrote {os.path.join(out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
