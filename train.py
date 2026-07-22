"""
Training loops shared by both the initial Bayesian training phase and the
post-pruning fine-tuning phase (fine-tuning is just training with beta
forced to 0, since a physically-pruned model has no Bayesian gates left).
"""

import logging
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bayesian_layers import collect_bayesian_layers
from losses import bayesian_snn_loss, kl_beta_schedule, spike_rate_cross_entropy
from utils import save_checkpoint


def _spike_accuracy(spk_rec: torch.Tensor, targets: torch.Tensor) -> float:
    """Classification accuracy from summed spike counts over time."""
    spike_counts = spk_rec.sum(dim=0)
    predictions = spike_counts.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    grad_clip_norm: float,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Dict[str, float]:
    """Run one training epoch. Returns average task loss, KL, total loss, and accuracy."""
    model.train()
    is_bayesian = len(collect_bayesian_layers(model)) > 0

    total_task_loss, total_kl, total_loss, total_acc, n_batches = 0.0, 0.0, 0.0, 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            spk_rec = model(images)
            if is_bayesian:
                loss, task_loss, kl_term = bayesian_snn_loss(spk_rec, targets, model, beta)
            else:
                task_loss = spike_rate_cross_entropy(spk_rec, targets)
                kl_term = torch.tensor(0.0, device=device)
                loss = task_loss

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_task_loss += task_loss.item()
        total_kl += float(kl_term)
        total_loss += loss.item()
        total_acc += _spike_accuracy(spk_rec.detach(), targets)
        n_batches += 1

    return {
        "task_loss": total_task_loss / n_batches,
        "kl": total_kl / n_batches,
        "total_loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate task loss and accuracy on a loader, with no gradient/noise-free gates."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        spk_rec = model(images)
        loss = spike_rate_cross_entropy(spk_rec, targets)
        total_loss += loss.item()
        total_acc += _spike_accuracy(spk_rec, targets)
        n_batches += 1

    return {"loss": total_loss / n_batches, "accuracy": total_acc / n_batches}


def _param_groups_excluding_gates(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """
    Split parameters into two weight-decay groups: `log_alpha` gate
    parameters get weight_decay=0, everything else gets the requested
    value.

    L2 weight decay pulls every parameter toward zero regardless of the
    loss gradient. Applying it to log_alpha (initialised at a negative
    value, e.g. -3.0) would silently drag every gate toward log_alpha=0
    every step -- during "Train" this defeats the intended frozen/inert
    behaviour of a disabled gate, and during "Bayesian train" it fights
    against the KL term's pruning pressure by resisting exactly the
    positive log_alpha growth that pruning is supposed to encourage.
    Gate parameters already have their own dedicated regulariser (the KL
    term against the log-uniform prior); they must not also receive
    generic L2 decay.
    """
    gate_params = [p for name, p in model.named_parameters() if "log_alpha" in name]
    other_params = [p for name, p in model.named_parameters() if "log_alpha" not in name]
    return [
        {"params": other_params, "weight_decay": weight_decay},
        {"params": gate_params, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Construct an optimizer by name ('adam' or 'sgd'). Excludes log_alpha
    gate parameters from weight decay -- see _param_groups_excluding_gates."""
    param_groups = _param_groups_excluding_gates(model, weight_decay)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, epochs: int):
    """Construct an LR scheduler by name ('cosine' or 'none')."""
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'")


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    scheduler_name: str,
    grad_clip_norm: float,
    use_amp: bool,
    device: torch.device,
    beta_max: float,
    kl_warmup_epochs: int,
    logger: logging.Logger,
    checkpoint_path: str,
    csv_log_rows: List[Dict[str, Any]],
    phase_name: str,
    prune_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Full multi-epoch training/fine-tuning loop with validation, cosine LR
    scheduling, best-checkpoint saving, and per-epoch CSV row logging.

    Setting `beta_max=0.0` (as run_all.py does for the fine-tuning phase)
    makes this a plain SNN training loop with no Bayesian regularisation,
    since a physically-pruned model has no gates left to regularise.

    During the `"bayesian_train"` phase, an extra diagnostic line reports
    the min/median/max `log_alpha` and the fraction of gates already past
    `prune_threshold`, pooled across every gated layer -- this is the
    live signal for whether the KL pressure (`beta_max`) is actually
    strong enough to drive gates to collapse, rather than only finding
    out after the full phase completes.
    """
    model.to(device)
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        beta = kl_beta_schedule(epoch, kl_warmup_epochs, beta_max)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, beta, grad_clip_norm, use_amp, scaler
        )
        val_stats = evaluate_loader(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        logger.info(
            f"[{phase_name}] epoch {epoch + 1}/{epochs} "
            f"train_task_loss={train_stats['task_loss']:.4f} "
            f"train_kl={train_stats['kl']:.2f} beta={beta:.5f} "
            f"train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if phase_name == "bayesian_train":
            gated_layers = collect_bayesian_layers(model)
            if gated_layers:
                all_log_alpha = torch.cat([layer.log_alpha.detach().flatten() for layer in gated_layers])
                logger.info(
                    f"    log_alpha: min={all_log_alpha.min():.2f} "
                    f"median={all_log_alpha.median():.2f} max={all_log_alpha.max():.2f} "
                    f"frac_prunable={(all_log_alpha > prune_threshold).float().mean():.3f}"
                )

        csv_log_rows.append(
            {
                "phase": phase_name,
                "epoch": epoch + 1,
                "train_task_loss": train_stats["task_loss"],
                "train_kl": train_stats["kl"],
                "train_total_loss": train_stats["total_loss"],
                "beta": beta,
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "epoch_time_sec": epoch_time,
            }
        )

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_time
    save_checkpoint({"model_state_dict": model.state_dict(), "best_val_acc": best_val_acc}, checkpoint_path)

    return {"best_val_acc": best_val_acc, "total_time_sec": total_time}
