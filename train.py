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

from bayesian_layers import collect_bayesian_layers, collect_prunable_bayesian_layers, total_kl
from losses import bayesian_snn_loss, get_task_loss, linear_warmup_schedule, spike_rate_cross_entropy
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
    gamma: float = 0.0,
    prune_threshold: float = 3.0,
    loss_type: str = "spike_rate_ce",
    gate_optimizer: Optional[torch.optim.Optimizer] = None,
    lam: float = 0.0,
    synops_budget: float = 0.0,
) -> Dict[str, float]:
    """Run one training epoch. Returns average task loss, KL, expected-cost,
    expected-SynOps, total loss, and accuracy. `gamma`/`prune_threshold` only
    take effect for Bayesian models (bio-inspired pruning models have no
    gates, so the non-bayesian branch below never uses them). `loss_type`
    selects the task loss and applies to both branches.

    `gate_optimizer`, when given, holds the log_alpha parameters *instead of*
    `optimizer` (see build_gate_split_optimizers); both are zeroed and
    stepped together on every batch, from the same backward pass.
    `lam`/`synops_budget` parameterise the SynOps-budget Lagrangian term
    (see losses.bayesian_snn_loss); both default to inert."""
    model.train()
    is_bayesian = len(collect_bayesian_layers(model)) > 0
    optimizers = [optimizer] + ([gate_optimizer] if gate_optimizer is not None else [])

    total_task_loss, total_kl, total_cost, total_synops, total_loss, total_acc, n_batches = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
    )

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            spk_rec = model(images)
            if is_bayesian:
                loss, task_loss, kl_term, cost_term, synops_term = bayesian_snn_loss(
                    spk_rec, targets, model, beta, gamma, prune_threshold, loss_type,
                    lam=lam, synops_budget=synops_budget,
                )
            else:
                task_loss = get_task_loss(loss_type)(spk_rec, targets)
                kl_term = torch.tensor(0.0, device=device)
                cost_term = torch.tensor(0.0, device=device)
                synops_term = torch.tensor(0.0, device=device)
                loss = task_loss

        if amp_enabled:
            scaler.scale(loss).backward()
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            for opt in optimizers:
                opt.step()

        total_task_loss += task_loss.item()
        total_kl += float(kl_term)
        total_cost += float(cost_term)
        total_synops += float(synops_term)
        total_loss += loss.item()
        total_acc += _spike_accuracy(spk_rec.detach(), targets)
        n_batches += 1

    return {
        "task_loss": total_task_loss / n_batches,
        "kl": total_kl / n_batches,
        "cost": total_cost / n_batches,
        "synops": total_synops / n_batches,
        "total_loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_type: str = "spike_rate_ce",
) -> Dict[str, float]:
    """Evaluate task loss and accuracy on a loader, with no gradient/noise-free gates.

    `loss_type` must match the one used for training, or the reported
    validation loss is on a different scale to the training loss and the
    two curves cannot be read together. Accuracy is unaffected either way,
    since it ranks by summed spike count.
    """
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        spk_rec = model(images)
        loss = get_task_loss(loss_type)(spk_rec, targets)
        total_loss += loss.item()
        total_acc += _spike_accuracy(spk_rec, targets)
        n_batches += 1

    return {"loss": total_loss / n_batches, "accuracy": total_acc / n_batches}


_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)


def gate_pressure_diagnostic(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
    loss_type: str = "spike_rate_ce",
) -> Dict[str, Dict[str, float]]:
    """
    Per-layer tug-of-war on `log_alpha`, measured on a single batch.

    For each gated layer this reports the mean magnitude of
    `d(task_loss)/d(log_alpha)` against `beta * d(KL)/d(log_alpha)`, and
    their ratio. A ratio far below 1 means the KL faces essentially no
    opposition in that layer, so its gates will run to the clamp ceiling
    whatever `beta_max` is -- which is exactly what a gate misplaced before
    a BatchNorm looks like, and what three collapsed DPAP runs and a
    four-point beta sweep were spent discovering the expensive way.

    This exists because `beta_max` is *not* a speed control under Adam.
    Adam divides each update by that parameter's own running gradient
    magnitude, so wherever one term dominates the update on log_alpha is
    roughly `lr * sign(grad)` -- independent of how large the gradient, and
    therefore of beta_max. Scaling beta_max up or down does not make gates
    move faster or slower; it moves the point where the two gradients
    cancel. (The observed DPAP trajectory bears this out: log_alpha went
    -3.05 -> 3.54 in 17 epochs, and 1000 steps/epoch at lr=5e-4 gives 0.5
    per epoch of pure sign-following.) So the informative measurement is
    the *balance* between the two terms, which this reads directly in
    seconds, rather than the outcome of a sweep costing GPU-hours.

    Costs two extra backward passes on one batch. `torch.autograd.grad` is
    used rather than `.backward()`, so no `.grad` buffer and no optimizer
    state is touched. Normalisation layers are held in eval mode for the
    duration so this measurement cannot perturb their running statistics.
    """
    gated = [layer for layer in collect_bayesian_layers(model) if layer.enable_gate_noise]
    if not gated:
        return {}

    name_of = {module: name for name, module in model.named_modules()}
    was_training = model.training
    model.train()
    norms = [m for m in model.modules() if isinstance(m, _NORM_TYPES)]
    norm_modes = [m.training for m in norms]
    for m in norms:
        m.eval()

    try:
        log_alphas = [layer.log_alpha for layer in gated]
        # Full precision deliberately: this is a gradient-magnitude
        # comparison, and fp16 underflow is exactly the sort of artefact
        # that would make a real imbalance look like a measurement error.
        spk_rec = model(images)
        task_loss = get_task_loss(loss_type)(spk_rec, targets)
        task_grads = torch.autograd.grad(task_loss, log_alphas, allow_unused=True)
        kl_grads = torch.autograd.grad(total_kl(model), log_alphas, allow_unused=True)
    finally:
        for m, mode in zip(norms, norm_modes):
            m.train(mode)
        model.train(was_training)

    report: Dict[str, Dict[str, float]] = {}
    for layer, t_grad, k_grad in zip(gated, task_grads, kl_grads):
        task_mag = 0.0 if t_grad is None else float(t_grad.abs().mean())
        kl_mag = 0.0 if k_grad is None else float(k_grad.abs().mean())
        pressure = beta * kl_mag
        report[name_of.get(layer, repr(layer))] = {
            "task_grad": task_mag,
            "kl_grad_scaled": pressure,
            "ratio": task_mag / pressure if pressure > 0 else float("inf"),
        }
    return report


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
    """Construct an optimizer by name ('adam', 'adamw' or 'sgd'). Excludes
    log_alpha gate parameters from weight decay -- see
    _param_groups_excluding_gates.

    'adamw' is needed for the DPAP replication, whose released code trains
    with AdamW at weight_decay=0.01 (docs/replication_targets.md). The
    distinction matters more than usual here: Adam couples weight decay into
    the gradient, whereas AdamW applies it as a separate decoupled step, and
    at DPAP's decay of 0.01 -- 200x this project's default -- treating one
    as the other would be a materially different optimiser.
    """
    param_groups = _param_groups_excluding_gates(model, weight_decay)
    return _optimizer_by_name(name, param_groups, lr)


def _optimizer_by_name(name: str, param_groups: List[Dict[str, Any]], lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_gate_split_optimizers(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
    gate_optimizer: str = "inherit",
    gate_lr: Optional[float] = None,
) -> "tuple[torch.optim.Optimizer, Optional[torch.optim.Optimizer]]":
    """
    Build (main_optimizer, gate_optimizer_or_None) for a training phase.

    `gate_optimizer="inherit"` reproduces build_optimizer exactly: one
    optimizer, gates in their own weight_decay=0 group. `"sgd"` moves every
    `log_alpha` parameter out of the main optimizer and into a plain SGD
    (momentum 0, weight decay 0) at `gate_lr` (None => `lr`).

    Why plain SGD, and why no momentum: Adam divides each update by that
    parameter's own running gradient magnitude, so while one loss term
    dominates the update on log_alpha is ~lr * sign(grad) -- a constant
    march per step that is independent of how large the opposing gradient
    is. The task-vs-KL equilibrium the gate mechanism relies on assumes the
    step *shrinks* as the two gradients approach balance, which is exactly
    the gradient-proportional stepping SGD provides. Momentum is left off
    because it reintroduces a (milder) version of the same insensitivity:
    accumulated velocity carries a gate past the balance point the current
    gradient no longer supports.

    Note the scheduler consequence: run_training's LR scheduler is attached
    to the main optimizer only, so under "sgd" the gate LR is constant for
    the whole phase. That is deliberate -- a decaying gate LR would make
    late-phase gate movement vanish for scheduling reasons that have
    nothing to do with the loss landscape, muddying exactly the
    "did the gates settle or were they frozen" reading the phase exists
    to produce.
    """
    if gate_optimizer == "inherit":
        return build_optimizer(model, name, lr, weight_decay), None
    if gate_optimizer != "sgd":
        raise ValueError(f"Unknown gate_optimizer '{gate_optimizer}'. Options: inherit, sgd")

    gate_params = [p for pname, p in model.named_parameters() if "log_alpha" in pname]
    other_params = [p for pname, p in model.named_parameters() if "log_alpha" not in pname]
    main = _optimizer_by_name(name, [{"params": other_params, "weight_decay": weight_decay}], lr)
    if not gate_params:
        # A physically-pruned (non-Bayesian) model has no gates left; the
        # split silently degenerates to the main optimizer alone.
        return main, None
    gates = torch.optim.SGD(
        [{"params": gate_params, "weight_decay": 0.0}],
        lr=(lr if gate_lr is None else gate_lr),
        momentum=0.0,
    )
    return main, gates


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    warmup_epochs: int = 0,
    min_lr: float = 0.0,
):
    """Construct an LR scheduler by name ('cosine', 'cosine_warmup', 'none').

    'cosine_warmup' linearly ramps the learning rate from ~0 over
    `warmup_epochs` before cosine-annealing to `min_lr`, matching the
    schedule DPAP's code configures (warmup 5 epochs, min_lr 1e-5). Warmup
    matters for AdamW specifically: its decoupled weight decay is applied
    every step from the start, so a cold high-LR start can shrink weights
    before the gradient signal has stabilised.
    """
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "cosine_warmup":
        cosine_epochs = max(1, epochs - warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=min_lr
        )
        if warmup_epochs <= 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
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
    restore_best_checkpoint: bool = True,
    gamma_max: float = 0.0,
    cost_warmup_epochs: int = 1,
    loss_type: str = "spike_rate_ce",
    lr_warmup_epochs: int = 0,
    min_lr: float = 0.0,
    gate_diagnostic_every: int = 0,
    gate_optimizer_name: str = "inherit",
    gate_lr: Optional[float] = None,
    synops_budget_fraction: Optional[float] = None,
    synops_lambda_lr: float = 0.05,
    synops_recount_every: int = 1,
    synops_refresh_fn: Optional[Any] = None,
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

    `restore_best_checkpoint` selects whether the model is reverted to
    its best-validation-accuracy epoch at the end (the right choice for
    ordinary training/fine-tuning, to avoid overfitting). For
    `"bayesian_train"` this must be False: validation accuracy declines
    monotonically as the KL term pushes gates toward collapse, so "best
    accuracy" always lands near epoch 1, before pruning has had any
    effect -- reverting to it would silently undo the entire phase.

    `gamma_max`/`cost_warmup_epochs` control the optional expected-FLOPs
    cost term (see losses.bayesian_snn_loss) the same way `beta_max`/
    `kl_warmup_epochs` control the KL term. Default `gamma_max=0.0` makes
    this term inert, matching pre-existing behaviour.

    `gate_diagnostic_every` (0 = off) logs the per-layer task-vs-KL
    gradient balance on a fixed batch every N epochs -- see
    gate_pressure_diagnostic for why that measurement, and not a beta_max
    sweep, is what diagnoses a collapse.

    `gate_optimizer_name`/`gate_lr` select a separate plain-SGD optimizer
    for the log_alpha parameters (see build_gate_split_optimizers). The LR
    scheduler applies to the main optimizer only.

    `synops_budget_fraction` (with `synops_refresh_fn`) enables the
    SynOps-budget Lagrangian: the refresh callable re-measures per-unit
    SynOps costs into every gated layer's `unit_cost` buffer (it is given
    the model and must handle its own data/device), and is invoked at
    phase start and every `synops_recount_every` epochs after. The budget
    itself is frozen at phase start -- `fraction * total_unit_cost` of the
    *initial* measurement -- so the target does not drift as the network's
    firing rates change. `lam` starts at 0 and follows dual ascent on the
    normalised violation each epoch. Returns the final `lam` in the result
    dict as "final_lambda" when active.
    """
    from bayesian_layers import total_unit_cost

    model.to(device)
    optimizer, gate_opt = build_gate_split_optimizers(
        model, optimizer_name, lr, weight_decay, gate_optimizer_name, gate_lr
    )
    scheduler = build_scheduler(optimizer, scheduler_name, epochs, lr_warmup_epochs, min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    use_synops = synops_budget_fraction is not None
    lam = 0.0
    synops_budget = 0.0
    if use_synops:
        if synops_refresh_fn is None:
            raise ValueError(
                "synops_budget_fraction is set but no synops_refresh_fn was given; "
                "without a measurement callable every unit_cost buffer stays zero "
                "and the budget term would be silently inert"
            )
        synops_refresh_fn(model)
        dense_synops = total_unit_cost(model)
        if dense_synops <= 0.0:
            raise ValueError(
                "measured total SynOps is zero after the initial refresh -- the "
                "cost measurement found no events, so a budget cannot be defined"
            )
        synops_budget = synops_budget_fraction * dense_synops
        logger.info(
            f"[{phase_name}] SynOps budget: {synops_budget:,.0f} "
            f"({synops_budget_fraction:g} of measured dense {dense_synops:,.0f}); "
            f"lambda_lr={synops_lambda_lr:g}, recount every {synops_recount_every} epoch(s)"
        )

    # One fixed batch, held for the whole phase, so successive diagnostics
    # differ because the model changed rather than because the data did.
    diag_batch = None
    if gate_diagnostic_every > 0:
        diag_images, diag_targets = next(iter(train_loader))
        diag_batch = (diag_images.to(device), diag_targets.to(device))

    best_val_acc = -1.0
    best_state = None
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        beta = linear_warmup_schedule(epoch, kl_warmup_epochs, beta_max)
        gamma = linear_warmup_schedule(epoch, cost_warmup_epochs, gamma_max)

        if use_synops and synops_recount_every > 0 and epoch > 0 and epoch % synops_recount_every == 0:
            synops_refresh_fn(model)

        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=beta,
            grad_clip_norm=grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
            gamma=gamma,
            prune_threshold=prune_threshold,
            loss_type=loss_type,
            gate_optimizer=gate_opt,
            lam=lam,
            synops_budget=synops_budget,
        )
        val_stats = evaluate_loader(model, val_loader, device, loss_type=loss_type)

        if scheduler is not None:
            scheduler.step()

        synops_line = ""
        epoch_lambda = lam
        if use_synops:
            violation = (train_stats["synops"] - synops_budget) / synops_budget
            synops_line = (
                f" E_synops={train_stats['synops']:,.0f} "
                f"({train_stats['synops'] / synops_budget:.2f}x budget) lambda={epoch_lambda:.4f}"
            )
            # Dual ascent *after* the epoch that trained under the current
            # lam, so the logged lambda is the one the epoch actually used.
            lam = max(0.0, lam + synops_lambda_lr * violation)

        epoch_time = time.time() - epoch_start
        logger.info(
            f"[{phase_name}] epoch {epoch + 1}/{epochs} "
            f"train_task_loss={train_stats['task_loss']:.4f} "
            f"train_kl={train_stats['kl']:.2f} beta={beta:.5f} "
            f"train_cost={train_stats['cost']:.2f} gamma={gamma:.6f} "
            f"train_acc={train_stats['accuracy']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"time={epoch_time:.1f}s{synops_line}"
        )

        if phase_name == "bayesian_train":
            gated_layers = collect_prunable_bayesian_layers(model)
            if gated_layers:
                all_log_alpha = torch.cat([layer.log_alpha.detach().flatten() for layer in gated_layers])
                ceiling = max(layer.log_alpha_clamp_max for layer in gated_layers)
                # frac_saturated is the number that matters under ranked
                # pruning: gates pinned at the clamp are tied, and ties are
                # broken by index order, so a run with frac_saturated near 1
                # produces a keep-set that is arbitrary rather than learned
                # -- and still hits its sparsity target, so nothing else in
                # the output looks wrong.
                logger.info(
                    f"    log_alpha: min={all_log_alpha.min():.2f} "
                    f"median={all_log_alpha.median():.2f} max={all_log_alpha.max():.2f} "
                    f"std={all_log_alpha.std():.2f} "
                    f"frac_prunable={(all_log_alpha > prune_threshold).float().mean():.3f} "
                    f"frac_saturated={(all_log_alpha >= ceiling - 1e-3).float().mean():.3f}"
                )

        if (
            diag_batch is not None
            and gate_diagnostic_every > 0
            and epoch % gate_diagnostic_every == 0
        ):
            pressure = gate_pressure_diagnostic(
                model, diag_batch[0], diag_batch[1], beta, loss_type
            )
            for layer_name, stats in pressure.items():
                logger.info(
                    f"    gate pressure {layer_name}: "
                    f"|d task/d log_alpha|={stats['task_grad']:.3e} "
                    f"beta*|d KL/d log_alpha|={stats['kl_grad_scaled']:.3e} "
                    f"ratio={stats['ratio']:.3f}"
                )

        csv_log_rows.append(
            {
                "phase": phase_name,
                "epoch": epoch + 1,
                "train_task_loss": train_stats["task_loss"],
                "train_kl": train_stats["kl"],
                "train_cost": train_stats["cost"],
                "train_total_loss": train_stats["total_loss"],
                "beta": beta,
                "gamma": gamma,
                # Always present (0.0 when the budget term is off) so every
                # phase's rows share one column set -- write_csv takes its
                # header from the first row, and run_all.py accumulates all
                # phases into a single training_log.csv.
                "train_synops": train_stats["synops"],
                "lambda": epoch_lambda,
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_accuracy": val_stats["accuracy"],
                "epoch_time_sec": epoch_time,
            }
        )

        if val_stats["accuracy"] > best_val_acc:
            best_val_acc = val_stats["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if restore_best_checkpoint and best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - start_time
    save_checkpoint({"model_state_dict": model.state_dict(), "best_val_acc": best_val_acc}, checkpoint_path)

    result = {"best_val_acc": best_val_acc, "total_time_sec": total_time}
    if use_synops:
        result["final_lambda"] = lam
    return result
