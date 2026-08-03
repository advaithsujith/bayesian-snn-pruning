"""
Top-level orchestrator. Running `python run_all.py` performs the complete
Train -> Evaluate -> Convert-to-Bayesian -> Bayesian-train -> Compute
uncertainty -> Structured-prune -> Fine-tune -> Final-evaluate -> Save
pipeline for LeNet-SNN, then VGG9-SNN, then Spiking ResNet-18, and finally
writes outputs/final_results.csv plus every comparison figure.
"""

import csv
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS, ExperimentConfig
from datasets import get_cifar10_loaders, get_pruning_phase_loaders
from evaluate import full_evaluation
from metrics import (
    compression_ratio,
    compute_and_set_unit_costs,
    count_parameters,
    gather_log_alpha_values,
    pruning_percentage,
    write_csv,
)
from models import build_model
from pruning import (
    build_keep_plan,
    compute_uncertainty_report,
    prune_model,
    remaining_structures_report,
)
from train import evaluate_loader, run_training
from utils import ensure_dirs, get_device, print_banner, save_config_json, set_seed, setup_logger

MODEL_ORDER = ["dpap_repl"]


def make_experiment_plots(
    model_name: str,
    csv_rows: List[Dict[str, Any]],
    log_alpha_values: Dict[str, torch.Tensor],
    output_dir: str,
) -> None:
    """
    Build the single per-experiment `plots.png` containing: training loss,
    validation loss, accuracy curves, an uncertainty (log_alpha) histogram
    pooled across all gated layers, and a per-layer posterior-variance
    (log_alpha) distribution.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model_name} -- training and pruning diagnostics")

    phases: List[str] = []
    for row in csv_rows:
        if row["phase"] not in phases:
            phases.append(row["phase"])

    ax = axes[0, 0]
    for phase in phases:
        rows = [r for r in csv_rows if r["phase"] == phase]
        ax.plot([r["epoch"] for r in rows], [r["train_task_loss"] for r in rows], label=f"{phase} train")
    ax.set_title("Training task loss")
    ax.set_xlabel("epoch (within phase)")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0, 1]
    for phase in phases:
        rows = [r for r in csv_rows if r["phase"] == phase]
        ax.plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label=f"{phase} val")
    ax.set_title("Validation loss")
    ax.set_xlabel("epoch (within phase)")
    ax.set_ylabel("loss")
    ax.legend()

    ax = axes[0, 2]
    for phase in phases:
        rows = [r for r in csv_rows if r["phase"] == phase]
        ax.plot([r["epoch"] for r in rows], [r["train_accuracy"] for r in rows], label=f"{phase} train acc")
        ax.plot([r["epoch"] for r in rows], [r["val_accuracy"] for r in rows], linestyle="--", label=f"{phase} val acc")
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch (within phase)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    bayes_rows = [r for r in csv_rows if r["phase"] == "bayesian_train"]
    if bayes_rows:
        ax.plot([r["epoch"] for r in bayes_rows], [r["train_kl"] for r in bayes_rows])
    ax.set_title("KL divergence over Bayesian training")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total KL")

    ax = axes[1, 1]
    all_log_alpha = torch.cat([v.flatten() for v in log_alpha_values.values()]) if log_alpha_values else torch.tensor([])
    if all_log_alpha.numel() > 0:
        ax.hist(all_log_alpha.numpy(), bins=50)
        ax.axvline(3.0, color="red", linestyle="--", label="prune threshold")
    ax.set_title("Posterior uncertainty histogram (log alpha, all gates)")
    ax.set_xlabel("log alpha")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[1, 2]
    layer_names = list(log_alpha_values.keys())
    layer_medians = [float(v.median()) for v in log_alpha_values.values()]
    if layer_names:
        ax.barh(layer_names, layer_medians)
        ax.axvline(3.0, color="red", linestyle="--")
    ax.set_title("Median log alpha per layer")
    ax.set_xlabel("median log alpha")
    ax.tick_params(axis="y", labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "plots.png"), dpi=150)
    plt.close(fig)


def build_summary_text(
    model_name: str,
    original_params: int,
    remaining_params: int,
    eval_before: Dict[str, Any],
    eval_after: Dict[str, Any],
    pretrain_result: Dict[str, Any],
    bayes_result: Dict[str, Any],
    finetune_result: Dict[str, Any],
    keep_plan_description: str,
    saturation: Dict[str, float],
) -> str:
    """Build the plain-text summary.txt content for one experiment."""
    lines = [
        f"Experiment summary: {model_name}",
        "=" * 50,
        f"Original parameters (deterministic baseline): {original_params}",
        f"Remaining parameters (after structured pruning): {remaining_params}",
        f"Compression ratio: {compression_ratio(original_params, remaining_params):.3f}x",
        f"Pruning percentage: {pruning_percentage(original_params, remaining_params):.2f}%",
        f"Keep-set decided by: {keep_plan_description}",
        (
            f"Gate saturation at clamp ceiling: {saturation['frac_saturated_high']:.3f} "
            f"of {saturation['total_gates']} prunable gates"
        ),
        "",
        f"Test accuracy before Bayesian pruning: {eval_before['test_accuracy']:.4f}",
        f"Test accuracy after pruning + fine-tuning: {eval_after['test_accuracy']:.4f}",
        f"Accuracy change: {eval_after['test_accuracy'] - eval_before['test_accuracy']:+.4f}",
        "",
        f"Estimated FLOPs (post-pruning): {eval_after['flops']:,}",
        f"Inference latency (post-pruning): {eval_after['latency_ms']:.3f} ms",
        f"Peak GPU memory (post-pruning): {eval_after['gpu_memory_mb']:.1f} MB",
        "",
        (
            "Pretraining: skipped, reused an existing baseline checkpoint"
            if pretrain_result["total_time_sec"] == 0.0
            else f"Pretraining time: {pretrain_result['total_time_sec']:.1f}s "
            f"(best val acc {pretrain_result['best_val_acc']:.4f})"
        ),
        f"Bayesian gate training time: {bayes_result['total_time_sec']:.1f}s (best val acc {bayes_result['best_val_acc']:.4f})",
        f"Fine-tuning time: {finetune_result['total_time_sec']:.1f}s (best val acc {finetune_result['best_val_acc']:.4f})",
    ]
    return "\n".join(lines) + "\n"


def run_experiment(model_name: str, experiment_index: int, total_experiments: int) -> Dict[str, Any]:
    """Run the full pipeline for a single architecture end to end."""
    cfg: ExperimentConfig = ALL_EXPERIMENTS[model_name]()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.checkpoint_dir, cfg.output_dir, cfg.log_dir, cfg.plot_dir)
    logger = setup_logger(model_name, cfg.log_dir, f"{model_name}.log")

    print_banner(f"Running Experiment {experiment_index} / {total_experiments}\n{model_name.upper()}")

    print("\nBuilding model...")
    model = build_model(
        model_name, cfg.snn, cfg.bayesian, num_classes=cfg.num_classes, arch_cfg=cfg.arch
    ).to(device)
    set_bayesian_mode(model, False)
    original_params = count_parameters(model, exclude_gates=True)
    logger.info(f"Model built with {original_params} parameters (gates excluded from count).")

    input_shape = (cfg.latency_batch_size, 3, 32, 32)
    compute_and_set_unit_costs(model, input_shape, device)
    logger.info("Per-unit expected-FLOPs costs computed and set on every Bayesian layer.")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    # The pruning stages use their own loaders whenever the pretrain recipe
    # validates on the test set (replication mode) -- otherwise every gate
    # decision, checkpoint and hyperparameter would be selected on the test
    # set. Identical to the loaders above for every non-replication config.
    # See datasets.get_pruning_phase_loaders.
    prune_train_loader, prune_val_loader, _ = get_pruning_phase_loaders(
        cfg.data, cfg.train.batch_size, cfg.seed
    )
    if cfg.data.pruning_val_fraction is not None:
        logger.info(
            "Pruning stages use a held-out validation split "
            f"(val_fraction={cfg.data.pruning_val_fraction}) rather than the "
            "pretrain recipe's test-set validation."
        )
    csv_rows: List[Dict[str, Any]] = []

    pretrained_path = os.path.join(cfg.output_dir, "trained_model.pt")
    reuse_pretrained = cfg.reuse_pretrained and os.path.isfile(pretrained_path)

    if reuse_pretrained:
        # Skip straight to the pruning stages using the saved baseline.
        # Pretraining dominates runtime (5.4h of a ~7h DPAP run), and it is
        # entirely independent of every pruning hyperparameter, so tuning
        # beta_max or gamma_max does not need it repeated. The baseline is
        # bit-identical to the one the reused checkpoint was measured on,
        # which also makes successive pruning runs exactly comparable.
        print(f"\nReusing pretrained baseline: {pretrained_path}")
        logger.info(f"Reusing existing pretrained baseline, skipping pretrain: {pretrained_path}")
        try:
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        except RuntimeError as exc:
            # Loudly, rather than silently training on a mismatched baseline.
            raise RuntimeError(
                f"Could not load '{pretrained_path}' into the current {model_name} model. "
                "The checkpoint was almost certainly saved from a different architecture "
                "or config. Delete it, or set reuse_pretrained=False to retrain from "
                f"scratch.\nUnderlying error: {exc}"
            ) from exc
        pretrain_result = {"best_val_acc": float("nan"), "total_time_sec": 0.0}
    else:
        print("\nTraining...")
        pretrain_result = run_training(
            model,
            train_loader,
            val_loader,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            optimizer_name=cfg.train.optimizer,
            scheduler_name=cfg.train.lr_scheduler,
            grad_clip_norm=cfg.train.grad_clip_norm,
            use_amp=cfg.train.use_amp,
            device=device,
            beta_max=0.0,
            kl_warmup_epochs=1,
            gamma_max=0.0,
            cost_warmup_epochs=1,
            loss_type=cfg.loss_type,
            lr_warmup_epochs=cfg.train.lr_warmup_epochs,
            min_lr=cfg.train.min_lr,
            logger=logger,
            checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_trained.pt"),
            csv_log_rows=csv_rows,
            phase_name="pretrain",
        )

    print("\nValidation...")
    val_stats = evaluate_loader(model, val_loader, device, loss_type=cfg.loss_type)
    logger.info(f"Post-pretrain validation accuracy: {val_stats['accuracy']:.4f}")

    eval_before = full_evaluation(
        model, test_loader, device, input_shape, cfg.latency_num_warmup, cfg.latency_num_runs
    )
    logger.info(f"Test accuracy before Bayesian conversion: {eval_before['test_accuracy']:.4f}")
    if not reuse_pretrained:
        torch.save(model.state_dict(), pretrained_path)

    print("\nConverting to Bayesian...")
    set_bayesian_mode(model, True)
    logger.info("Bayesian gates activated on every structurally-prunable layer.")

    print("\nTraining Bayesian gates...")
    bayes_result = run_training(
        model,
        prune_train_loader,
        prune_val_loader,
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
        use_amp=cfg.train.use_amp,
        device=device,
        beta_max=cfg.bayesian.beta_max,
        kl_warmup_epochs=cfg.bayesian.kl_warmup_epochs,
        gamma_max=cfg.bayesian.gamma_max,
        cost_warmup_epochs=cfg.bayesian.cost_warmup_epochs,
        loss_type=cfg.pruning_loss(),
        logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_bayesian.pt"),
        csv_log_rows=csv_rows,
        phase_name="bayesian_train",
        prune_threshold=cfg.bayesian.prune_threshold,
        restore_best_checkpoint=False,
        gate_diagnostic_every=5,
        gate_optimizer_name=cfg.bayesian.gate_optimizer,
        gate_lr=cfg.bayesian.gate_lr,
    )
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "bayesian_model.pt"))

    print("\nComputing posterior uncertainty...")
    uncertainty_report = compute_uncertainty_report(model, cfg.bayesian.prune_threshold)
    for layer_name, stats in uncertainty_report.items():
        logger.info(
            f"  {layer_name}: median_log_alpha={stats['median']:.2f} "
            f"std={stats['std']:.2f} frac_prunable={stats['frac_prunable']:.3f} "
            f"frac_saturated={stats['frac_saturated']:.3f} "
            f"structurally_prunable={stats['structurally_prunable']}"
        )
    log_alpha_values = gather_log_alpha_values(model)

    print("\nStructured pruning...")
    keep_plan = build_keep_plan(model, model_name, cfg.snn, cfg.bayesian)
    saturation = keep_plan.saturation_report(model)
    logger.info(f"Keep-set decided by: {keep_plan.describe()}")
    logger.info(
        f"Gate saturation: {saturation['frac_saturated_high']:.3f} at the ceiling, "
        f"{saturation['frac_saturated_low']:.3f} at the floor, of "
        f"{saturation['total_gates']} prunable gates."
    )
    if keep_plan.mode != "threshold" and saturation["frac_saturated_high"] > 0.5:
        logger.warning(
            "Over half this model's gates are pinned at the clamp ceiling, so the "
            "ranking that chose the keep-set is mostly ties broken by index order. "
            "Treat this run's sparsity/accuracy point as uninterpretable."
        )
    # Derived from the same plan the rebuild uses, so the per-layer report
    # cannot disagree with what was actually built.
    remaining_report = remaining_structures_report(model, keep_plan)
    pruned_model = prune_model(model, model_name, keep_plan, cfg.snn).to(device)
    remaining_params = count_parameters(pruned_model, exclude_gates=True)
    logger.info(
        f"Parameters: {original_params} -> {remaining_params} "
        f"({pruning_percentage(original_params, remaining_params):.1f}% pruned)"
    )

    print("\nFine tuning...")
    finetune_result = run_training(
        pruned_model,
        prune_train_loader,
        prune_val_loader,
        epochs=cfg.finetune.epochs,
        lr=cfg.finetune.lr,
        weight_decay=cfg.finetune.weight_decay,
        optimizer_name=cfg.finetune.optimizer,
        scheduler_name=cfg.finetune.lr_scheduler,
        grad_clip_norm=cfg.finetune.grad_clip_norm,
        use_amp=cfg.finetune.use_amp,
        device=device,
        beta_max=0.0,
        kl_warmup_epochs=1,
        gamma_max=0.0,
        cost_warmup_epochs=1,
        loss_type=cfg.pruning_loss(),
        logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"{model_name}_pruned_finetuned.pt"),
        csv_log_rows=csv_rows,
        phase_name="finetune",
    )

    print("\nTesting...")
    eval_after = full_evaluation(
        pruned_model, test_loader, device, input_shape, cfg.latency_num_warmup, cfg.latency_num_runs
    )
    logger.info(f"Final test accuracy after pruning + fine-tuning: {eval_after['test_accuracy']:.4f}")

    print("\nSaving checkpoints...")
    torch.save(pruned_model.state_dict(), os.path.join(cfg.output_dir, "pruned_model.pt"))
    write_csv(csv_rows, os.path.join(cfg.output_dir, "training_log.csv"))
    save_config_json(cfg, os.path.join(cfg.output_dir, "config.json"))

    metrics_rows = [
        {"stage": "before_pruning", **eval_before},
        {"stage": "after_pruning", **eval_after},
    ]
    write_csv(metrics_rows, os.path.join(cfg.output_dir, "metrics.csv"))

    structures_rows = [
        {"layer": name, **stats} for name, stats in remaining_report.items()
    ]
    write_csv(structures_rows, os.path.join(cfg.output_dir, "remaining_structures.csv"))

    make_experiment_plots(model_name, csv_rows, log_alpha_values, cfg.output_dir)

    summary_text = build_summary_text(
        model_name, original_params, remaining_params, eval_before, eval_after,
        pretrain_result, bayes_result, finetune_result,
        keep_plan.describe(), saturation,
    )
    with open(os.path.join(cfg.output_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    print("\nFinished.")

    return {
        "Model": model_name,
        "Keep Plan": keep_plan.describe(),
        "Gate Saturation": saturation["frac_saturated_high"],
        "Original Parameters": original_params,
        "Remaining Parameters": remaining_params,
        "Compression Ratio": compression_ratio(original_params, remaining_params),
        "Pruning Percentage": pruning_percentage(original_params, remaining_params),
        "Accuracy Before": eval_before["test_accuracy"],
        "Accuracy After": eval_after["test_accuracy"],
        "Training Time": pretrain_result["total_time_sec"] + bayes_result["total_time_sec"],
        "Fine Tune Time": finetune_result["total_time_sec"],
        "FLOPs": eval_after["flops"],
        "Latency": eval_after["latency_ms"],
        "GPU Memory": eval_after["gpu_memory_mb"],
    }


def make_comparison_plots(results: List[Dict[str, Any]], plot_dir: str) -> None:
    """Build the cross-model comparison figures saved under ./plots/."""
    models = [r["Model"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter([r["Pruning Percentage"] for r in results], [r["Accuracy After"] for r in results])
    for r in results:
        ax.annotate(r["Model"], (r["Pruning Percentage"], r["Accuracy After"]))
    ax.set_xlabel("Pruning percentage (%)")
    ax.set_ylabel("Accuracy after pruning + fine-tuning")
    ax.set_title("Compression vs. accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "compression_vs_accuracy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(models, [r["Pruning Percentage"] for r in results])
    ax.set_ylabel("Pruning percentage (%)")
    ax.set_title("Pruning percentage by model")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "pruning_percentage.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(models))
    ax.bar([i - 0.2 for i in x], [r["Original Parameters"] for r in results], width=0.4, label="original")
    ax.bar([i + 0.2 for i in x], [r["Remaining Parameters"] for r in results], width=0.4, label="remaining")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Parameters")
    ax.set_title("Remaining parameters by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "remaining_parameters.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(models))
    ax.bar([i - 0.2 for i in x], [r["Accuracy Before"] for r in results], width=0.4, label="before pruning")
    ax.bar([i + 0.2 for i in x], [r["Accuracy After"] for r in results], width=0.4, label="after pruning")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy before vs. after pruning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "accuracy_comparison.png"), dpi=150)
    plt.close(fig)


def merge_final_results(new_rows: List[Dict[str, Any]], path: str) -> List[Dict[str, Any]]:
    """Merge this run's rows into `path`, replacing rows for the models that
    were re-run and keeping every other model's row untouched.

    `MODEL_ORDER` is routinely narrowed to a single architecture while
    iterating on it, and a plain overwrite then deletes the other
    architectures' results from the cross-model table. That is how the
    quoted VGG9 figures (98.71% / 86.08%) stopped existing anywhere in the
    repo, leaving HANDOFF.md citing numbers no file could confirm. Rows are
    cheap; recovering a deleted 3-hour run is not.
    """
    existing: List[Dict[str, Any]] = []
    if os.path.isfile(path):
        with open(path, newline="") as f:
            existing = [dict(row) for row in csv.DictReader(f)]

    replaced = {row["Model"] for row in new_rows}
    merged = [row for row in existing if row.get("Model") not in replaced]
    merged.extend(new_rows)
    return merged


def main() -> None:
    """Run every experiment in MODEL_ORDER, then build the cross-model summary."""
    ensure_dirs("./checkpoints", "./outputs", "./plots", "./logs")

    results: List[Dict[str, Any]] = []
    for i, model_name in enumerate(MODEL_ORDER, start=1):
        result = run_experiment(model_name, i, len(MODEL_ORDER))
        results.append(result)

    final_path = "./outputs/final_results.csv"
    write_csv(merge_final_results(results, final_path), final_path)
    make_comparison_plots(results, "./plots")

    print_banner("ALL EXPERIMENTS FINISHED")
    for r in results:
        print(
            f"{r['Model']:>10s}: {r['Pruning Percentage']:5.1f}% pruned, "
            f"accuracy {r['Accuracy Before']:.4f} -> {r['Accuracy After']:.4f}"
        )


if __name__ == "__main__":
    main()
