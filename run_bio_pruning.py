"""
Top-level orchestrator for the bio-inspired pruning baselines
(activity_pruning.py): naive static firing-rate, SCA, and DPAP-structured.

For each architecture, this script reuses (or trains once and saves, if
run before run_all.py) the exact same deterministic pretrained checkpoint
the Bayesian pipeline forks from (`outputs/<model>/trained_model.pt`), then
runs each of the three criteria across every sparsity point in
`cfg.bio.keep_fractions`, fine-tunes, and evaluates -- mirroring
run_all.py's Train -> prune -> fine-tune -> evaluate structure so results
are directly comparable at matched sparsity. Writes
`outputs/bio_results.csv` and, if `outputs/final_results.csv` (the
Bayesian side's output) is present, overlays the Bayesian point on the
same accuracy-vs-sparsity axes for a direct comparison plot.
"""

import csv
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from activity_pruning import (
    prune_model_activity,
    run_dpap_pruning,
    run_naive_firing_rate_pruning,
    run_sca_pruning,
)
from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS, ExperimentConfig
from datasets import get_cifar10_loaders
from evaluate import full_evaluation
from metrics import count_parameters, pruning_percentage, write_csv
from models import build_model
from train import run_training
from utils import ensure_dirs, get_device, print_banner, set_seed, setup_logger

MODEL_ORDER = ["lenet", "vgg9", "resnet18"]
CRITERIA = ["naive_firing_rate", "sca", "dpap"]


def _ensure_pretrained_checkpoint(
    model_name: str, cfg: ExperimentConfig, device: torch.device, train_loader, val_loader, logger
) -> str:
    """
    Return the path to the deterministic pretrained baseline checkpoint,
    training and saving one if it doesn't already exist. If run_all.py has
    already been run for this architecture, `trained_model.pt` already
    exists and is reused as-is, guaranteeing the Bayesian and bio-inspired
    pipelines fork from bit-identical weights.

    Run-order caveat: run_all.py's own pretrain phase does not check for
    an existing `trained_model.pt` before unconditionally retraining and
    overwriting it. If this script is ever run *before* run_all.py on a
    fresh output directory, a later run_all.py run would silently replace
    the exact checkpoint this script's bio results were computed from. To
    keep that scenario's provenance recoverable rather than silently lost,
    a self-trained fallback baseline is additionally archived under
    `bio_pretrained_baseline.pt` and never overwritten by either script.
    In this user's normal workflow (run_all.py already run first, per
    HANDOFF.md), `trained_model.pt` already exists and this whole fallback
    path is never exercised.
    """
    pretrained_path = os.path.join(cfg.output_dir, "trained_model.pt")
    if os.path.isfile(pretrained_path):
        logger.info(f"Reusing existing pretrained baseline: {pretrained_path}")
        return pretrained_path

    logger.info(
        "No pretrained baseline found for this architecture -- training one now "
        "(this becomes the shared starting point for both Bayesian and bio-inspired pruning)."
    )
    model = build_model(model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes).to(device)
    set_bayesian_mode(model, False)
    pretrain_csv_rows: List[Dict[str, Any]] = []
    run_training(
        model, train_loader, val_loader,
        epochs=cfg.train.epochs, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        optimizer_name=cfg.train.optimizer, scheduler_name=cfg.train.lr_scheduler,
        grad_clip_norm=cfg.train.grad_clip_norm, use_amp=cfg.train.use_amp, device=device,
        beta_max=0.0, kl_warmup_epochs=1, logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_bio_pretrained.pt"),
        csv_log_rows=pretrain_csv_rows, phase_name="pretrain",
    )
    ensure_dirs(cfg.output_dir)
    torch.save(model.state_dict(), pretrained_path)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "bio_pretrained_baseline.pt"))
    write_csv(pretrain_csv_rows, os.path.join(cfg.output_dir, "pretrain_training_log.csv"))
    return pretrained_path


def run_bio_experiment_point(
    model_name: str,
    criterion: str,
    keep_fraction: float,
    cfg: ExperimentConfig,
    device: torch.device,
    pretrained_path: str,
    train_loader,
    val_loader,
    test_loader,
    original_params: int,
    logger,
) -> Dict[str, Any]:
    """Run one (architecture, criterion, target keep_fraction) point end to end."""
    model = build_model(model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    set_bayesian_mode(model, False)

    tag = f"{criterion}_kf{keep_fraction:.3f}"
    out_dir = os.path.join(cfg.output_dir, "bio", criterion, f"keep_{keep_fraction:.3f}")
    ensure_dirs(out_dir)
    csv_rows: List[Dict[str, Any]] = []

    if criterion == "naive_firing_rate":
        keep_masks = run_naive_firing_rate_pruning(
            model, model_name, train_loader, device, keep_fraction, cfg.bio, logger
        )
    elif criterion == "sca":
        keep_masks = run_sca_pruning(
            model, model_name, train_loader, val_loader, device, keep_fraction, cfg.bio,
            cfg.train.grad_clip_norm, cfg.train.use_amp, logger, csv_rows,
        )
    elif criterion == "dpap":
        keep_masks = run_dpap_pruning(
            model, model_name, train_loader, val_loader, device, keep_fraction, cfg.bio,
            cfg.train.grad_clip_norm, cfg.train.use_amp, logger, csv_rows,
        )
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Options: {CRITERIA}")

    pruned_model = prune_model_activity(model, model_name, keep_masks, cfg.snn).to(device)
    remaining_params = count_parameters(pruned_model, exclude_gates=True)
    logger.info(
        f"[{tag}] Parameters: {original_params} -> {remaining_params} "
        f"({pruning_percentage(original_params, remaining_params):.1f}% pruned)"
    )

    finetune_result = run_training(
        pruned_model, train_loader, val_loader,
        epochs=cfg.finetune.epochs, lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay,
        optimizer_name=cfg.finetune.optimizer, scheduler_name=cfg.finetune.lr_scheduler,
        grad_clip_norm=cfg.finetune.grad_clip_norm, use_amp=cfg.finetune.use_amp, device=device,
        beta_max=0.0, kl_warmup_epochs=1, logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_{tag}_finetuned.pt"),
        csv_log_rows=csv_rows, phase_name="finetune",
    )

    input_shape = (cfg.latency_batch_size, 3, 32, 32)
    eval_after = full_evaluation(
        pruned_model, test_loader, device, input_shape, cfg.latency_num_warmup, cfg.latency_num_runs
    )
    logger.info(f"[{tag}] Final test accuracy: {eval_after['test_accuracy']:.4f}")

    torch.save(pruned_model.state_dict(), os.path.join(out_dir, "pruned_model.pt"))
    write_csv(csv_rows, os.path.join(out_dir, "training_log.csv"))

    summary_lines = [
        f"Bio-inspired pruning: {model_name} / {criterion} / target keep_fraction={keep_fraction}",
        "=" * 60,
        f"Original parameters: {original_params}",
        f"Remaining parameters: {remaining_params}",
        f"Pruning percentage: {pruning_percentage(original_params, remaining_params):.2f}%",
        f"Test accuracy after pruning + fine-tuning: {eval_after['test_accuracy']:.4f}",
        f"Estimated FLOPs (post-pruning): {eval_after['flops']:,}",
        f"Inference latency (post-pruning): {eval_after['latency_ms']:.3f} ms",
        f"Fine-tuning time: {finetune_result['total_time_sec']:.1f}s "
        f"(best val acc {finetune_result['best_val_acc']:.4f})",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    return {
        "Model": model_name,
        "Criterion": criterion,
        "Target Keep Fraction": keep_fraction,
        "Original Parameters": original_params,
        "Remaining Parameters": remaining_params,
        "Pruning Percentage": pruning_percentage(original_params, remaining_params),
        "Accuracy After": eval_after["test_accuracy"],
        "Fine Tune Time": finetune_result["total_time_sec"],
        "FLOPs": eval_after["flops"],
        "Latency": eval_after["latency_ms"],
        "GPU Memory": eval_after["gpu_memory_mb"],
    }


def run_bio_experiments_for_model(model_name: str, experiment_index: int, total_experiments: int) -> List[Dict[str, Any]]:
    """Run every (criterion, keep_fraction) combination for one architecture."""
    cfg: ExperimentConfig = ALL_EXPERIMENTS[model_name]()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.checkpoint_dir, cfg.output_dir, cfg.log_dir, cfg.plot_dir)
    logger = setup_logger(f"{model_name}_bio", cfg.log_dir, f"{model_name}_bio.log")

    print_banner(f"Bio-Inspired Pruning: Experiment {experiment_index} / {total_experiments}\n{model_name.upper()}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)

    pretrained_path = _ensure_pretrained_checkpoint(model_name, cfg, device, train_loader, val_loader, logger)

    probe_model = build_model(model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes)
    original_params = count_parameters(probe_model, exclude_gates=True)
    del probe_model

    results: List[Dict[str, Any]] = []
    for criterion in CRITERIA:
        for keep_fraction in cfg.bio.keep_fractions:
            set_seed(cfg.seed)  # identical data order / stochastic state for every sweep point
            print(f"\nRunning {model_name} / {criterion} / keep_fraction={keep_fraction} ...")
            result = run_bio_experiment_point(
                model_name, criterion, keep_fraction, cfg, device,
                pretrained_path, train_loader, val_loader, test_loader,
                original_params, logger,
            )
            results.append(result)

    return results


def _load_bayesian_results(path: str = "./outputs/final_results.csv") -> Dict[str, Dict[str, float]]:
    """Read run_all.py's per-architecture Bayesian result, if it exists, so
    it can be overlaid on the bio-inspired accuracy-vs-sparsity curves."""
    if not os.path.isfile(path):
        return {}
    by_model: Dict[str, Dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_model[row["Model"]] = {
                "pruning_pct": float(row["Pruning Percentage"]),
                "accuracy": float(row["Accuracy After"]),
            }
    return by_model


def make_bio_comparison_plots(results: List[Dict[str, Any]], plot_dir: str) -> None:
    """One accuracy-vs-sparsity subplot per architecture, one line per
    bio-inspired criterion, with the Bayesian point overlaid as a star if
    outputs/final_results.csv is available."""
    bayesian_by_model = _load_bayesian_results()
    models = [m for m in MODEL_ORDER if any(r["Model"] == m for r in results)]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model_name in zip(axes[0], models):
        for criterion in CRITERIA:
            points = sorted(
                (r for r in results if r["Model"] == model_name and r["Criterion"] == criterion),
                key=lambda r: r["Pruning Percentage"],
            )
            if not points:
                continue
            ax.plot(
                [p["Pruning Percentage"] for p in points],
                [p["Accuracy After"] for p in points],
                marker="o", label=criterion,
            )
        if model_name in bayesian_by_model:
            b = bayesian_by_model[model_name]
            ax.scatter(
                [b["pruning_pct"]], [b["accuracy"]],
                marker="*", s=220, color="red", label="bayesian", zorder=5,
            )
        ax.set_title(model_name)
        ax.set_xlabel("Pruning percentage (%)")
        ax.set_ylabel("Test accuracy after pruning + fine-tuning")
        ax.legend(fontsize=8)

    fig.suptitle("Accuracy vs. sparsity: bio-inspired criteria vs. Bayesian")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(plot_dir, "bio_vs_bayesian_sparsity_curves.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    """Run every (architecture, criterion, keep_fraction) combination, then
    build the cross-architecture comparison figure."""
    ensure_dirs("./checkpoints", "./outputs", "./plots", "./logs")

    all_results: List[Dict[str, Any]] = []
    for i, model_name in enumerate(MODEL_ORDER, start=1):
        all_results.extend(run_bio_experiments_for_model(model_name, i, len(MODEL_ORDER)))

    write_csv(all_results, "./outputs/bio_results.csv")
    make_bio_comparison_plots(all_results, "./plots")

    print_banner("ALL BIO-INSPIRED EXPERIMENTS FINISHED")
    for r in all_results:
        print(
            f"{r['Model']:>10s} / {r['Criterion']:<18s} kf={r['Target Keep Fraction']:.3f}: "
            f"{r['Pruning Percentage']:5.1f}% pruned, accuracy {r['Accuracy After']:.4f}"
        )


if __name__ == "__main__":
    main()
