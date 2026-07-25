"""
Multi-sampler hyperparameter search for a fair Bayesian-vs-bio-inspired comparison.

Motivation
----------
Bayesian pruning's hyperparameters (beta_max, kl_warmup_epochs) were tuned through
real, iterative CSF3 runs (see HANDOFF.md). The bio-inspired criteria in
activity_pruning.py (SCA, DPAP) originally shipped with untuned first-guess defaults,
and LeNet's Bayesian beta_max was "fixed" from a catastrophic collapse (fc2 pruned to a
single surviving neuron) with a one-off manual guess rather than anything principled.
Both are a threat to the validity of the comparison this project's dissertation is
built on: if one method is tuned and another isn't, "method A beats method B" may just
mean "method A got more tuning effort."

This script runs the same search procedure -- three independent samplers (random
search, TPE, CMA-ES via Optuna), 15 trials each -- against all three pruning criteria
(Bayesian's beta_max, SCA's, DPAP's), entirely on LeNet (cheapest architecture to
iterate on). Every trial is a real, shortened (~1/3 epoch budget) run of the actual
pipeline in train.py/pruning.py/activity_pruning.py -- nothing here reimplements the
pruning mechanics, it only orchestrates them with different, trial-sampled parameters.
Independent samplers landing on similar hyperparameters is the actual fairness claim:
not "these numbers are correct" but "multiple different search strategies agree this is
a good operating point," which is symmetric across all three methods by construction.

Once the search finishes, the winning hyperparameters per method get one full-length
confirmation run (same epoch budgets as run_all.py / run_bio_pruning.py use elsewhere),
evaluated on the held-out test set rather than the validation set the search itself
optimizes against.

Transfer to VGG9/ResNet18 (not performed by this script -- see README): only LeNet's
Bayesian beta_max should be replaced (it was the broken one; VGG9/ResNet18's Bayesian
beta_max already produced validated, working results and are left untouched). SCA's and
DPAP's searched hyperparameters transfer as-is to all three architectures' configs,
matching this codebase's existing convention that bayesian_train_epochs is already
shared identically across architectures.
"""

import argparse
import os
from typing import Any, Callable, Dict, List

import optuna
import torch

from activity_pruning import prune_model_activity, run_dpap_pruning, run_sca_pruning
from bayesian_layers import set_bayesian_mode
from config import ALL_EXPERIMENTS, BioPruningConfig, ExperimentConfig
from datasets import get_cifar10_loaders
from evaluate import full_evaluation
from metrics import count_parameters, pruning_percentage, write_csv
from models import build_model
from pruning import prune_model
from run_bio_pruning import _ensure_pretrained_checkpoint
from train import evaluate_loader, run_training
from utils import ensure_dirs, get_device, print_banner, set_seed, setup_logger

METHODS = ["bayesian", "sca", "dpap"]
SAMPLERS: Dict[str, Callable[[int], optuna.samplers.BaseSampler]] = {
    "random": lambda seed: optuna.samplers.RandomSampler(seed=seed),
    "tpe": lambda seed: optuna.samplers.TPESampler(seed=seed),
    "cmaes": lambda seed: optuna.samplers.CmaEsSampler(seed=seed),
}
N_TRIALS_PER_SAMPLER = 15
KEEP_FRACTION = 0.1  # matches the sparsity point already used elsewhere for bio methods

# Shortened-proxy budget used during search trials only (~1/3 of the full pipeline's
# epoch counts) -- cheap enough to make 135 total trials feasible on LeNet.
TRIAL_BAYESIAN_EPOCHS = 25
TRIAL_KL_WARMUP_EPOCHS = 17
TRIAL_FINETUNE_EPOCHS = 7

# Bayesian's sparsity is emergent (a threshold, not a dial), so the objective
# penalizes trials that dodge pruning almost entirely instead of finding a genuine
# accuracy/sparsity tradeoff.
MIN_PRUNING_PCT_FOR_BAYESIAN = 20.0
BAYESIAN_PENALTY = 0.5

OUTPUT_DIR = "./outputs/hpo"


def _run_bayesian_pipeline(
    cfg: ExperimentConfig,
    device: torch.device,
    pretrained_path: str,
    train_loader,
    val_loader,
    beta_max: float,
    bayesian_epochs: int,
    kl_warmup_epochs: int,
    finetune_epochs: int,
    logger,
    tag: str,
) -> torch.nn.Module:
    """Fork from the shared pretrained checkpoint, train Bayesian gates, physically
    prune, fine-tune. Returns the finished pruned model. Shared by both the search
    objective (shortened epochs) and the final confirmation run (full epochs)."""
    model = build_model("lenet", cfg.snn, cfg.bayesian, num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    set_bayesian_mode(model, True)

    run_training(
        model, train_loader, val_loader,
        epochs=bayesian_epochs, lr=cfg.bayesian.bayesian_train_lr, weight_decay=cfg.train.weight_decay,
        optimizer_name=cfg.train.optimizer, scheduler_name=cfg.train.lr_scheduler,
        grad_clip_norm=cfg.train.grad_clip_norm, use_amp=cfg.train.use_amp, device=device,
        beta_max=beta_max, kl_warmup_epochs=kl_warmup_epochs, logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"hpo_{tag}_bayesian.pt"),
        csv_log_rows=[], phase_name="bayesian_train",
        prune_threshold=cfg.bayesian.prune_threshold, restore_best_checkpoint=False,
    )

    pruned = prune_model(model, "lenet", cfg.bayesian.prune_threshold, cfg.snn).to(device)

    run_training(
        pruned, train_loader, val_loader,
        epochs=finetune_epochs, lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay,
        optimizer_name=cfg.finetune.optimizer, scheduler_name=cfg.finetune.lr_scheduler,
        grad_clip_norm=cfg.finetune.grad_clip_norm, use_amp=cfg.finetune.use_amp, device=device,
        beta_max=0.0, kl_warmup_epochs=1, logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"hpo_{tag}_finetuned.pt"),
        csv_log_rows=[], phase_name="finetune",
    )
    return pruned


def _run_bio_pipeline(
    criterion: str,
    cfg: ExperimentConfig,
    device: torch.device,
    pretrained_path: str,
    train_loader,
    val_loader,
    bio_cfg: BioPruningConfig,
    finetune_epochs: int,
    logger,
    tag: str,
) -> torch.nn.Module:
    """Fork from the shared pretrained checkpoint, run SCA or DPAP's dynamic pruning,
    physically prune, fine-tune. Shared by both search objective and confirmation run."""
    model = build_model("lenet", cfg.snn, cfg.bayesian, num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    set_bayesian_mode(model, False)

    csv_rows: List[Dict[str, Any]] = []
    if criterion == "sca":
        keep_masks = run_sca_pruning(
            model, "lenet", train_loader, val_loader, device, KEEP_FRACTION, bio_cfg,
            cfg.train.grad_clip_norm, cfg.train.use_amp, logger, csv_rows,
        )
    elif criterion == "dpap":
        keep_masks = run_dpap_pruning(
            model, "lenet", train_loader, val_loader, device, KEEP_FRACTION, bio_cfg,
            cfg.train.grad_clip_norm, cfg.train.use_amp, logger, csv_rows,
        )
    else:
        raise ValueError(f"Unknown bio criterion '{criterion}'")

    pruned = prune_model_activity(model, "lenet", keep_masks, cfg.snn).to(device)

    run_training(
        pruned, train_loader, val_loader,
        epochs=finetune_epochs, lr=cfg.finetune.lr, weight_decay=cfg.finetune.weight_decay,
        optimizer_name=cfg.finetune.optimizer, scheduler_name=cfg.finetune.lr_scheduler,
        grad_clip_norm=cfg.finetune.grad_clip_norm, use_amp=cfg.finetune.use_amp, device=device,
        beta_max=0.0, kl_warmup_epochs=1, logger=logger,
        checkpoint_path=os.path.join(cfg.checkpoint_dir, f"hpo_{tag}_finetuned.pt"),
        csv_log_rows=[], phase_name="finetune",
    )
    return pruned


# ---------------------------------------------------------------------------
# Trial objective factories
# ---------------------------------------------------------------------------


def bayesian_objective_factory(
    cfg, device, pretrained_path, train_loader, val_loader, original_params, results_rows, logger, sampler_name,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        beta_max = trial.suggest_float("beta_max", 0.02, 2.0, log=True)

        pruned = _run_bayesian_pipeline(
            cfg, device, pretrained_path, train_loader, val_loader,
            beta_max=beta_max, bayesian_epochs=TRIAL_BAYESIAN_EPOCHS,
            kl_warmup_epochs=TRIAL_KL_WARMUP_EPOCHS, finetune_epochs=TRIAL_FINETUNE_EPOCHS,
            logger=logger, tag="bayesian_trial",
        )

        val_stats = evaluate_loader(pruned, val_loader, device)
        remaining = count_parameters(pruned, exclude_gates=True)
        prune_pct = pruning_percentage(original_params, remaining)
        objective_value = val_stats["accuracy"]
        if prune_pct < MIN_PRUNING_PCT_FOR_BAYESIAN:
            objective_value -= BAYESIAN_PENALTY

        logger.info(
            f"[bayesian/{sampler_name}] trial {trial.number}: beta_max={beta_max:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} pruning_pct={prune_pct:.1f}% "
            f"objective={objective_value:.4f}"
        )
        results_rows.append({
            "sampler": sampler_name, "trial": trial.number, "beta_max": beta_max,
            "val_accuracy": val_stats["accuracy"], "pruning_percentage": prune_pct,
            "objective_value": objective_value,
        })
        return objective_value

    return objective


def _bio_objective_factory(
    criterion: str, cfg, device, pretrained_path, train_loader, val_loader, original_params,
    results_rows, logger, sampler_name,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        if criterion == "sca":
            total_epochs = trial.suggest_int("total_epochs", 20, 100)
            num_cycles = trial.suggest_int("sca_num_cycles", 3, 10)
            epochs_per_cycle = max(1, total_epochs // num_cycles)
            lr = trial.suggest_float("sca_lr", 1e-4, 2e-3, log=True)
            bio_cfg = BioPruningConfig(
                sca_epochs_per_cycle=epochs_per_cycle, sca_num_cycles=num_cycles, sca_lr=lr,
            )
            param_summary = f"total_epochs={total_epochs} num_cycles={num_cycles} lr={lr:.5f}"
        else:
            dpap_epochs = trial.suggest_int("dpap_train_epochs", 20, 100)
            ema_decay = trial.suggest_float("dpap_ema_decay", 0.8, 0.999)
            survival_decay = trial.suggest_float("dpap_survival_decay", 0.0, 0.1)
            lr = trial.suggest_float("dpap_lr", 1e-4, 2e-3, log=True)
            bio_cfg = BioPruningConfig(
                dpap_train_epochs=dpap_epochs, dpap_ema_decay=ema_decay,
                dpap_survival_decay=survival_decay, dpap_lr=lr,
            )
            param_summary = (
                f"train_epochs={dpap_epochs} ema_decay={ema_decay:.4f} "
                f"survival_decay={survival_decay:.4f} lr={lr:.5f}"
            )

        pruned = _run_bio_pipeline(
            criterion, cfg, device, pretrained_path, train_loader, val_loader,
            bio_cfg, finetune_epochs=TRIAL_FINETUNE_EPOCHS, logger=logger,
            tag=f"{criterion}_trial",
        )

        val_stats = evaluate_loader(pruned, val_loader, device)
        remaining = count_parameters(pruned, exclude_gates=True)
        prune_pct = pruning_percentage(original_params, remaining)

        logger.info(
            f"[{criterion}/{sampler_name}] trial {trial.number}: {param_summary} "
            f"val_acc={val_stats['accuracy']:.4f} pruning_pct={prune_pct:.1f}%"
        )
        row = {"sampler": sampler_name, "trial": trial.number, **trial.params,
               "val_accuracy": val_stats["accuracy"], "pruning_percentage": prune_pct,
               "objective_value": val_stats["accuracy"]}
        results_rows.append(row)
        return val_stats["accuracy"]

    return objective


def sca_objective_factory(*args, **kwargs):
    return _bio_objective_factory("sca", *args, **kwargs)


def dpap_objective_factory(*args, **kwargs):
    return _bio_objective_factory("dpap", *args, **kwargs)


OBJECTIVE_FACTORIES = {
    "bayesian": bayesian_objective_factory,
    "sca": sca_objective_factory,
    "dpap": dpap_objective_factory,
}


# ---------------------------------------------------------------------------
# Search orchestration
# ---------------------------------------------------------------------------


def run_search_for_method(
    method: str, cfg, device, pretrained_path, train_loader, val_loader, original_params, logger,
) -> Dict[str, Any]:
    print_banner(f"HPO search: {method}")
    factory = OBJECTIVE_FACTORIES[method]

    per_sampler_best: Dict[str, Dict[str, Any]] = {}
    all_trial_rows: List[Dict[str, Any]] = []

    for sampler_name, sampler_ctor in SAMPLERS.items():
        print(f"\n  sampler={sampler_name} ({N_TRIALS_PER_SAMPLER} trials)...")
        results_rows: List[Dict[str, Any]] = []
        objective = factory(
            cfg, device, pretrained_path, train_loader, val_loader, original_params,
            results_rows, logger, sampler_name,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler_ctor(cfg.seed))
        study.optimize(objective, n_trials=N_TRIALS_PER_SAMPLER)

        ensure_dirs(OUTPUT_DIR)
        write_csv(results_rows, os.path.join(OUTPUT_DIR, f"{method}_{sampler_name}_trials.csv"))
        all_trial_rows.extend(results_rows)

        per_sampler_best[sampler_name] = {"params": study.best_trial.params, "value": study.best_value}
        logger.info(f"[{method}/{sampler_name}] best: params={study.best_trial.params} value={study.best_value:.4f}")

    overall_best_row = max(all_trial_rows, key=lambda r: r["objective_value"])
    return {"method": method, "per_sampler_best": per_sampler_best, "overall_best": overall_best_row}


def write_convergence_summary(all_results: List[Dict[str, Any]], path: str) -> None:
    lines = ["Multi-sampler HPO convergence summary", "=" * 60, ""]
    for result in all_results:
        lines.append(f"Method: {result['method']}")
        for sampler_name, best in result["per_sampler_best"].items():
            lines.append(f"  {sampler_name:>8s}: value={best['value']:.4f} params={best['params']}")
        lines.append(f"  overall best across samplers: {result['overall_best']}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_confirmation(
    method: str, best_row: Dict[str, Any], cfg, device, pretrained_path,
    train_loader, val_loader, test_loader, original_params, logger,
) -> Dict[str, Any]:
    """One full-length run (same epoch budgets used elsewhere in the codebase) with the
    winning hyperparameters, evaluated on the held-out test set."""
    print_banner(f"Confirmation run: {method}")
    input_shape = (cfg.latency_batch_size, 3, 32, 32)

    if method == "bayesian":
        pruned = _run_bayesian_pipeline(
            cfg, device, pretrained_path, train_loader, val_loader,
            beta_max=best_row["beta_max"], bayesian_epochs=cfg.bayesian.bayesian_train_epochs,
            kl_warmup_epochs=cfg.bayesian.kl_warmup_epochs, finetune_epochs=cfg.finetune.epochs,
            logger=logger, tag="bayesian_confirm",
        )
    else:
        if method == "sca":
            bio_cfg = BioPruningConfig(
                sca_epochs_per_cycle=max(1, int(best_row["total_epochs"]) // int(best_row["sca_num_cycles"])),
                sca_num_cycles=int(best_row["sca_num_cycles"]), sca_lr=best_row["sca_lr"],
            )
        else:
            bio_cfg = BioPruningConfig(
                dpap_train_epochs=int(best_row["dpap_train_epochs"]),
                dpap_ema_decay=best_row["dpap_ema_decay"],
                dpap_survival_decay=best_row["dpap_survival_decay"],
                dpap_lr=best_row["dpap_lr"],
            )
        pruned = _run_bio_pipeline(
            method, cfg, device, pretrained_path, train_loader, val_loader,
            bio_cfg, finetune_epochs=cfg.finetune.epochs, logger=logger, tag=f"{method}_confirm",
        )

    eval_result = full_evaluation(pruned, test_loader, device, input_shape, cfg.latency_num_warmup, cfg.latency_num_runs)
    remaining = count_parameters(pruned, exclude_gates=True)
    prune_pct = pruning_percentage(original_params, remaining)
    logger.info(
        f"[{method}] confirmation: test_accuracy={eval_result['test_accuracy']:.4f} "
        f"pruning_pct={prune_pct:.1f}%"
    )
    return {"method": method, "test_accuracy": eval_result["test_accuracy"], "pruning_percentage": prune_pct,
            "hyperparameters": best_row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=METHODS,
        help="Subset of methods to search, for splitting the run across multiple sbatch submissions "
             "if the combined job doesn't fit CSF3's wallclock limit.",
    )
    args = parser.parse_args()

    cfg: ExperimentConfig = ALL_EXPERIMENTS["lenet"]()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ensure_dirs(cfg.checkpoint_dir, cfg.output_dir, cfg.log_dir, cfg.plot_dir, OUTPUT_DIR)
    logger = setup_logger("hpo", cfg.log_dir, "hpo.log")

    train_loader, val_loader, test_loader = get_cifar10_loaders(cfg.data, cfg.train.batch_size, cfg.seed)
    pretrained_path = _ensure_pretrained_checkpoint("lenet", cfg, device, train_loader, val_loader, logger)

    probe_model = build_model("lenet", cfg.snn, cfg.bayesian, num_classes=cfg.num_classes)
    original_params = count_parameters(probe_model, exclude_gates=True)
    del probe_model

    all_results: List[Dict[str, Any]] = []
    for method in args.methods:
        set_seed(cfg.seed)
        result = run_search_for_method(method, cfg, device, pretrained_path, train_loader, val_loader, original_params, logger)
        all_results.append(result)

    write_convergence_summary(all_results, os.path.join(OUTPUT_DIR, "convergence_summary.txt"))

    print_banner("Running full-length confirmation runs for the winning hyperparameters")
    confirmation_rows: List[Dict[str, Any]] = []
    for result in all_results:
        set_seed(cfg.seed)
        confirmation = run_confirmation(
            result["method"], result["overall_best"], cfg, device, pretrained_path,
            train_loader, val_loader, test_loader, original_params, logger,
        )
        confirmation_rows.append(confirmation)

    write_csv(confirmation_rows, os.path.join(OUTPUT_DIR, "confirmation_results.csv"))

    print_banner("HPO SEARCH FINISHED")
    for row in confirmation_rows:
        print(f"{row['method']:>10s}: test_accuracy={row['test_accuracy']:.4f} pruning_pct={row['pruning_percentage']:.1f}%")
    print(f"\nSee {OUTPUT_DIR}/convergence_summary.txt for per-sampler agreement and "
          f"{OUTPUT_DIR}/confirmation_results.csv for final numbers.")


if __name__ == "__main__":
    main()
