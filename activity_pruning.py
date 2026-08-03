"""
Bio-inspired (activity-based) structured pruning: three criteria, contrasted
against the posterior-uncertainty criterion in pruning.py.

All three criteria fork from the exact same deterministic pretrained
checkpoint the Bayesian pipeline saves as `trained_model.pt` (see
run_all.py's "Train" phase, `set_bayesian_mode(model, False)`), reuse the
same three architectures (models.py), the same CIFAR-10 pipeline
(datasets.py), the same optimizer/training-loop code (train.py), and the
same evaluation/metrics infrastructure (evaluate.py, metrics.py) as the
Bayesian side. Only the pruning *criterion* -- what gets scored, and how
the keep/drop decision is made -- differs, per the dissertation's
controlled-comparison framing (see HANDOFF.md). Physical rebuilding reuses
pruning.py's `Pruned*` container classes and slicing utilities unchanged;
this file only supplies alternative keep-index decisions and the
architecture-specific glue to apply them, so the already-verified Bayesian
pipeline in pruning.py is never modified.

Gate machinery reuse
---------------------
Every layer here is still a `BayesianConv2d` / `BayesianLinear` (gates are
part of the architecture in models.py), but `enable_gate_noise` is left
permanently False for every criterion in this file -- these methods never
"convert to Bayesian". With gate noise disabled, `forward()` always takes
the `h * self.hard_mask` branch (see bayesian_layers.py), so the existing
`hard_mask` buffer doubles as this file's "currently live channel" switch:
setting it to zero for a unit deterministically removes that unit's
contribution to every downstream layer, with no changes needed to
bayesian_layers.py.

The three criteria
-------------------
1. **Naive static firing-rate pruning.** One-shot: a single forward pass
   over the training set accumulates mean spike rate per channel/neuron,
   the bottom units are masked, done. No training-time dynamics at all --
   deliberately the cheapest and simplest baseline, contrasted against the
   two dynamic methods below.

2. **SCA (Spiking Channel Activity-based pruning), Li et al., ICML 2024,
   "Towards Efficient Deep Spiking Neural Networks Construction with
   Spiking Activity based Pruning"** (proceedings.mlr.press/v235/li24bz).
   Criterion: mean |membrane potential| per channel/neuron (the paper's
   own justification draws a direct analogy to biological
   depolarization/hyperpolarization levels). Dynamic: training is split
   into `sca_num_cycles` cycles; each cycle trains for
   `sca_epochs_per_cycle` epochs, then narrows the alive set to a smaller
   target keep-count using freshly accumulated activity, so the network
   adapts to each cut before the next one is applied.

3. **DPAP-structured (Developmental Plasticity-inspired Adaptive
   Pruning), arXiv 2211.12714** -- neuron/channel branch only (the paper
   also defines an unstructured synapse-level branch via BCM plasticity,
   which is out of scope here; see HANDOFF.md for why unstructured
   criteria don't fit this project's physical-rebuild comparison).
   Criterion: an exponential-moving-average "survival score" per
   channel/neuron driven by spike-rate activity, with a small constant
   subtracted every epoch (the paper's "use it or lose it, gradually
   decay" principle) so units drift toward elimination unless activity is
   strong enough to offset the decay. Dynamic at per-epoch granularity
   (finer-grained than SCA's per-cycle schedule).

Deliberate simplifications versus the source papers
-----------------------------------------------------
- **No genuine "regrow".** Once a unit's `hard_mask` is set to 0, the
  multiplicative-zero forward path also zeroes the gradient reaching that
  unit's underlying conv/linear weights (chain rule through `h *
  hard_mask`), so a masked-off unit is permanently inert -- it cannot
  organically "recover" and re-enter the keep set. Both SCA and DPAP here
  are therefore implemented as *monotonic* schedules (each step narrows
  the alive set, ranked among currently-alive units only), not the
  literature's full prune+regrow search, which would require a
  dense-gradient or periodic-reinit mechanism to keep masked weights
  trainable. This is flagged explicitly rather than silently claimed.
- **DPAP's threshold rule is replaced with explicit target-fraction
  selection.** The paper eliminates a unit once its survival score drops
  below zero, an emergent (uncontrolled) final sparsity. This project
  needs matched-sparsity comparison points against the Bayesian side, so
  DPAP here ramps a target *keep-count* schedule (mirroring SCA's ramp)
  and selects top-k by survival score at each step instead of waiting for
  an uncontrolled zero-crossing.
- **Only each criterion's structured/channel-level signal is used.**
  Where a source paper mixes structured and unstructured pruning (DPAP),
  only the structured branch is reproduced, for the same
  physical-rebuild-comparability reason.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bayesian_layers import BayesianConv2d, BayesianLinear
from config import BioPruningConfig, SNNConfig
from models import LeNetSNN, VGG9SNN, VGGStyleSNN, SpikingResNet18
from pruning import (
    PrunedBasicBlock,
    PrunedLeNetSNN,
    PrunedSpikingResNet18,
    PrunedVGG9SNN,
    PrunedVGGStyleSNN,
    _rebuild_vgg_style,
    channel_keep_to_flat_indices,
    slice_batchnorm,
)
from train import build_optimizer, build_scheduler, evaluate_loader, train_one_epoch

# ---------------------------------------------------------------------------
# Layer <-> LIF-neuron mapping (needed to hook the activity signal that sits
# immediately downstream of each structurally-prunable layer)
# ---------------------------------------------------------------------------

LayerLifPair = Tuple[str, "BayesianConv2d | BayesianLinear", nn.Module]


def _lenet_layer_lif_pairs(model: LeNetSNN) -> List[LayerLifPair]:
    return [
        ("conv1", model.conv1, model.lif1),
        ("conv2", model.conv2, model.lif2),
        ("fc1", model.fc1, model.lif3),
        ("fc2", model.fc2, model.lif4),
    ]


def _vgg9_layer_lif_pairs(model: VGG9SNN) -> List[LayerLifPair]:
    pairs = [
        (f"conv_layers.{i}", conv, lif)
        for i, (conv, lif) in enumerate(zip(model.conv_layers, model.lif_layers))
    ]
    pairs.append(("fc1", model.fc1, model.lif_fc1))
    return pairs


def _resnet18_layer_lif_pairs(model: SpikingResNet18) -> List[LayerLifPair]:
    pairs: List[LayerLifPair] = []
    for stage_idx, stage in enumerate(model._all_stages(), start=1):
        for block_idx, block in enumerate(stage):
            # Only conv1 is structurally prunable -- conv2 is residual-tied
            # (see models.py's residual pruning caveat), so it is excluded
            # here exactly as pruning.py excludes it for the Bayesian side.
            pairs.append((f"stage{stage_idx}.{block_idx}.conv1", block.conv1, block.lif1))
    return pairs


def _vggstyle_layer_lif_pairs(model: "VGGStyleSNN") -> List[LayerLifPair]:
    """(layer, lif) pairs for the configurable conv-stack family.

    Generalises _vgg9_layer_lif_pairs to any number of hidden fc layers,
    including none at all (Chowdhury's VGG9 collapses the usual 3 VGG fc
    layers into the single unpruned classifier). Names match the ones
    pruning.py reports, so the two criteria's per-layer CSVs line up.
    """
    pairs: List[LayerLifPair] = [
        (f"conv_layers.{i}", conv, lif)
        for i, (conv, lif) in enumerate(zip(model.conv_layers, model.lif_layers))
    ]
    pairs.extend(
        (f"fc_layers.{i}", fc, lif)
        for i, (fc, lif) in enumerate(zip(model.fc_layers, model.lif_fc_layers))
    )
    return pairs


def get_prunable_layer_lif_pairs(model: nn.Module, model_name: str) -> List[LayerLifPair]:
    """Dispatch to the correct architecture-specific (layer, lif) pairing."""
    dispatch = {
        "lenet": _lenet_layer_lif_pairs,
        "vgg9": _vgg9_layer_lif_pairs,
        "resnet18": _resnet18_layer_lif_pairs,
    }
    if isinstance(model, VGGStyleSNN):
        return _vggstyle_layer_lif_pairs(model)
    if model_name not in dispatch:
        raise ValueError(
            f"Unknown model name '{model_name}'. Built-in options: {list(dispatch)}; "
            "any other name requires the model to be a VGGStyleSNN."
        )
    return dispatch[model_name](model)


# ---------------------------------------------------------------------------
# Activity accumulation via forward hooks on the LIF neuron immediately
# downstream of each prunable layer
# ---------------------------------------------------------------------------


class ActivityAccumulator:
    """
    Accumulates a running mean |signal| per channel/neuron for a set of
    (layer_name, bayesian_layer, lif_module) triples, via forward hooks on
    each `lif_module`.

    `snn.Leaky.forward` returns `(spk, mem)`; a forward hook receives that
    full tuple as `output`, so no changes to models.py are needed to read
    either spikes or membrane potentials. Since each model's own
    `forward()` internally loops over `num_steps` timesteps, a hook fires
    once per timestep per batch automatically -- accumulated statistics
    are therefore already averaged over both samples and time, matching
    how the source papers describe their activity signals.
    """

    def __init__(self, pairs: List[LayerLifPair], signal: str) -> None:
        if signal not in ("membrane_potential", "spike_rate"):
            raise ValueError(f"Unknown signal '{signal}'. Options: membrane_potential, spike_rate")
        self._signal = signal
        self._pairs = pairs
        self._sums: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}
        self._handles: List[Any] = []

    def _make_hook(self, name: str, is_conv: bool):
        def hook(_module: nn.Module, _inputs: Any, output: "tuple[torch.Tensor, torch.Tensor]") -> None:
            spk, mem = output
            signal_tensor = (mem if self._signal == "membrane_potential" else spk).detach().abs()
            per_unit = signal_tensor.mean(dim=(0, 2, 3)) if is_conv else signal_tensor.mean(dim=0)
            per_unit = per_unit.float().cpu()
            if name not in self._sums:
                self._sums[name] = torch.zeros_like(per_unit)
                self._counts[name] = 0
            self._sums[name] += per_unit
            self._counts[name] += 1

        return hook

    def register(self) -> None:
        for name, layer, lif in self._pairs:
            is_conv = isinstance(layer, BayesianConv2d)
            self._handles.append(lif.register_forward_hook(self._make_hook(name, is_conv)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def reset(self) -> None:
        self._sums = {}
        self._counts = {}

    def scores(self) -> Dict[str, torch.Tensor]:
        """Mean |signal| per unit, accumulated since the last reset()."""
        return {name: self._sums[name] / max(self._counts[name], 1) for name in self._sums}


# ---------------------------------------------------------------------------
# Generic keep-mask selection (shared across all three criteria)
# ---------------------------------------------------------------------------


def select_keep_mask(scores: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """
    Boolean mask, True for the highest-scoring units to KEEP. Selects the
    top `max(1, round(keep_fraction * n))` units by score (higher = more
    important, true for every criterion in this file), matching the
    "never fully zero a layer" guarantee `pruning.compute_keep_indices`
    uses on the Bayesian side.
    """
    n = scores.numel()
    keep_count = min(max(1, round(keep_fraction * n)), n)
    _, top_idx = torch.topk(scores, keep_count)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[top_idx] = True
    return mask


def _select_top_k_within_alive(scores: torch.Tensor, alive_mask: torch.Tensor, target_count: int) -> torch.Tensor:
    """
    Return a new boolean mask that is a subset of `alive_mask`, keeping the
    `target_count` highest-scoring currently-alive units. Used by both SCA
    and DPAP to implement a monotonic (prune-only) schedule -- see the
    module docstring for why true regrowth isn't supported.
    """
    target_count = max(1, min(target_count, int(alive_mask.sum().item())))
    masked_scores = scores.clone()
    masked_scores[~alive_mask] = float("-inf")
    _, top_idx = torch.topk(masked_scores, target_count)
    new_mask = torch.zeros_like(alive_mask)
    new_mask[top_idx] = True
    return new_mask


def keep_indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a boolean keep-mask to sorted integer indices (nonzero() on a
    1-D tensor is already ascending)."""
    return mask.nonzero(as_tuple=True)[0].long()


def _apply_hard_masks(pairs: List[LayerLifPair], keep_masks: Dict[str, torch.Tensor]) -> None:
    """Freeze the current keep/drop decision into every layer's hard_mask
    buffer, so subsequent forward passes reflect the pruning-in-progress
    state (see the module docstring's "Gate machinery reuse" section)."""
    for name, layer, _ in pairs:
        layer.set_hard_mask(keep_masks[name])


def _register_bn_remask_hooks(model: nn.Module, model_name: str) -> List[Any]:
    """
    Fix BatchNorm mask leakage in the dynamic (SCA/DPAP) training loops.

    Wherever the forward is `bn(conv(x))`, the conv's hard_mask zeroes a
    channel's *pre-BatchNorm* output, but BatchNorm2d in training mode
    normalises an all-zero-input channel to exactly `bn.bias[c]` -- a
    nonzero constant that keeps receiving gradient and keeps changing every
    step, since `bn.bias` sits downstream of where the mask is applied.
    Left unfixed, a channel this file intends to treat as permanently dead
    instead leaks a live, still-training signal into whatever consumes it
    and into later layers' activity statistics. This registers a forward
    hook on each such BatchNorm that re-applies its conv's *current*
    hard_mask to the BatchNorm output, so a masked channel is genuinely
    zero downstream regardless of what BatchNorm does to it.

    Applies to two architecture families:
      - SpikingResNet18: every BasicBlock's `bn1` (paired with `conv1`).
      - VGGStyleSNN with `norm_type="batch"`: every `norm_layers[i]`
        (paired with `conv_layers[i]`). Both the SCA and DPAP replications
        use batch-norm in exactly this conv->BN->LIF position, so omitting
        them here would reintroduce the same leakage bug this hook exists
        to fix -- and silently, since the criterion would then be scoring
        channels that are not actually dead.

    LeNet/VGG9 place no BatchNorm between a prunable layer and its LIF
    neuron, so this stays a genuine no-op for them.
    """
    handles: List[Any] = []

    if isinstance(model, VGGStyleSNN):
        for conv, norm in zip(model.conv_layers, model.norm_layers):
            if not isinstance(norm, nn.BatchNorm2d):
                continue
            if conv.defer_gate:
                # The gate -- and therefore hard_mask -- is already applied
                # after this BatchNorm (see BayesianConv2d.defer_gate), so a
                # masked channel is zero downstream by construction and this
                # hook would only re-apply the same mask a second time.
                continue

            def _hook(_module, _inputs, output, _conv=conv):
                return output * _conv.hard_mask.view(1, -1, 1, 1)

            handles.append(norm.register_forward_hook(_hook))
        return handles

    if model_name != "resnet18":
        return []
    for stage in model._all_stages():
        for block in stage:
            def _hook(_module, _inputs, output, _conv=block.conv1):
                return output * _conv.hard_mask.view(1, -1, 1, 1)

            handles.append(block.bn1.register_forward_hook(_hook))
    return handles


def _remove_hooks(handles: List[Any]) -> None:
    for handle in handles:
        handle.remove()


# ---------------------------------------------------------------------------
# Criterion 1: naive static firing-rate pruning
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_naive_firing_rate_pruning(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    device: torch.device,
    keep_fraction: float,
    bio_cfg: BioPruningConfig,
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """
    One-shot: accumulate mean spike rate per channel/neuron over
    `bio_cfg.naive_num_passes` passes through `train_loader`, then keep the
    top `keep_fraction` units by that score. No gradient, no training-time
    dynamics -- this is the naive/static contrast against SCA and DPAP.
    """
    pairs = get_prunable_layer_lif_pairs(model, model_name)
    model.eval()
    accumulator = ActivityAccumulator(pairs, signal="spike_rate")
    accumulator.register()

    for pass_idx in range(bio_cfg.naive_num_passes):
        for images, _ in train_loader:
            images = images.to(device)
            model(images)
        logger.info(f"[naive_firing_rate] activity pass {pass_idx + 1}/{bio_cfg.naive_num_passes} done")

    accumulator.remove()
    scores = accumulator.scores()
    keep_masks = {name: select_keep_mask(scores[name], keep_fraction) for name in scores}
    _apply_hard_masks(pairs, keep_masks)
    return keep_masks


# ---------------------------------------------------------------------------
# Criterion 2: SCA (dynamic, cycle-based, membrane-potential criterion)
# ---------------------------------------------------------------------------


def run_sca_pruning(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    keep_fraction: float,
    bio_cfg: BioPruningConfig,
    grad_clip_norm: float,
    use_amp: bool,
    logger: logging.Logger,
    csv_log_rows: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    SCA (Li et al., ICML 2024): `bio_cfg.sca_num_cycles` cycles of
    `bio_cfg.sca_epochs_per_cycle` epochs each. Every cycle, the target
    alive-count ramps linearly from every unit down to `keep_fraction`,
    and the new alive set is the top-scoring subset (by mean |membrane
    potential| accumulated over that cycle) of the *previous* alive set
    (monotonic -- see module docstring).
    """
    pairs = get_prunable_layer_lif_pairs(model, model_name)
    layer_sizes = {name: layer.log_alpha.numel() for name, layer, _ in pairs}
    alive_mask = {name: torch.ones(size, dtype=torch.bool) for name, size in layer_sizes.items()}

    optimizer = build_optimizer(model, "adam", bio_cfg.sca_lr, bio_cfg.sca_weight_decay)
    total_epochs = bio_cfg.sca_epochs_per_cycle * bio_cfg.sca_num_cycles
    scheduler = build_scheduler(optimizer, "cosine", total_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    bn_remask_handles = _register_bn_remask_hooks(model, model_name)

    global_epoch = 0
    for cycle in range(bio_cfg.sca_num_cycles):
        accumulator = ActivityAccumulator(pairs, signal="membrane_potential")

        for _ in range(bio_cfg.sca_epochs_per_cycle):
            epoch_start = time.time()
            # Registered only around train_one_epoch, not evaluate_loader,
            # so validation-set forward passes don't contaminate the
            # activity statistics used to pick the next alive set.
            accumulator.register()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, device, beta=0.0,
                grad_clip_norm=grad_clip_norm, use_amp=use_amp, scaler=scaler,
            )
            accumulator.remove()
            val_stats = evaluate_loader(model, val_loader, device)
            if scheduler is not None:
                scheduler.step()
            global_epoch += 1
            epoch_time = time.time() - epoch_start

            logger.info(
                f"[sca_train] cycle {cycle + 1}/{bio_cfg.sca_num_cycles} "
                f"epoch {global_epoch}/{total_epochs} "
                f"train_acc={train_stats['accuracy']:.4f} val_acc={val_stats['accuracy']:.4f} "
                f"time={epoch_time:.1f}s"
            )
            csv_log_rows.append({
                "phase": "sca_train", "epoch": global_epoch,
                "train_task_loss": train_stats["task_loss"], "train_kl": 0.0,
                "train_total_loss": train_stats["total_loss"], "beta": 0.0,
                "train_accuracy": train_stats["accuracy"],
                "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
                "epoch_time_sec": epoch_time,
            })

        scores = accumulator.scores()

        target_frac = 1.0 - (1.0 - keep_fraction) * (cycle + 1) / bio_cfg.sca_num_cycles
        new_alive: Dict[str, torch.Tensor] = {}
        for name, size in layer_sizes.items():
            target_count = max(1, round(target_frac * size))
            new_alive[name] = _select_top_k_within_alive(scores[name], alive_mask[name], target_count)
        alive_mask = new_alive
        _apply_hard_masks(pairs, alive_mask)

        total_alive = sum(int(m.sum().item()) for m in alive_mask.values())
        total_units = sum(layer_sizes.values())
        logger.info(
            f"[sca_train]   end of cycle {cycle + 1}: target_frac={target_frac:.3f} "
            f"alive={total_alive}/{total_units} ({100.0 * total_alive / total_units:.1f}%)"
        )

    _remove_hooks(bn_remask_handles)
    return alive_mask


# ---------------------------------------------------------------------------
# Criterion 3: DPAP-structured (per-epoch EMA survival-trace criterion)
# ---------------------------------------------------------------------------


def run_dpap_pruning(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    keep_fraction: float,
    bio_cfg: BioPruningConfig,
    grad_clip_norm: float,
    use_amp: bool,
    logger: logging.Logger,
    csv_log_rows: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    DPAP-structured (arXiv 2211.12714), neuron/channel branch: an
    exponential-moving-average survival score per unit, driven by
    spike-rate activity with a small constant subtracted every epoch
    ("use it or lose it, gradually decay"). The alive set narrows every
    epoch (finer-grained than SCA's per-cycle schedule) toward the same
    linear target-count ramp, ranked by the running survival score.
    """
    pairs = get_prunable_layer_lif_pairs(model, model_name)
    layer_sizes = {name: layer.log_alpha.numel() for name, layer, _ in pairs}
    alive_mask = {name: torch.ones(size, dtype=torch.bool) for name, size in layer_sizes.items()}
    survival_score = {name: torch.zeros(size) for name, size in layer_sizes.items()}

    optimizer = build_optimizer(model, "adam", bio_cfg.dpap_lr, bio_cfg.dpap_weight_decay)
    scheduler = build_scheduler(optimizer, "cosine", bio_cfg.dpap_train_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    bn_remask_handles = _register_bn_remask_hooks(model, model_name)

    for epoch in range(bio_cfg.dpap_train_epochs):
        epoch_start = time.time()
        accumulator = ActivityAccumulator(pairs, signal="spike_rate")
        accumulator.register()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, beta=0.0,
            grad_clip_norm=grad_clip_norm, use_amp=use_amp, scaler=scaler,
        )

        accumulator.remove()
        epoch_activity = accumulator.scores()

        target_frac = 1.0 - (1.0 - keep_fraction) * (epoch + 1) / bio_cfg.dpap_train_epochs
        new_alive: Dict[str, torch.Tensor] = {}
        for name, size in layer_sizes.items():
            survival_score[name] = (
                bio_cfg.dpap_ema_decay * survival_score[name]
                + (1.0 - bio_cfg.dpap_ema_decay) * epoch_activity[name]
                - bio_cfg.dpap_survival_decay
            )
            target_count = max(1, round(target_frac * size))
            new_alive[name] = _select_top_k_within_alive(survival_score[name], alive_mask[name], target_count)
        alive_mask = new_alive
        _apply_hard_masks(pairs, alive_mask)

        val_stats = evaluate_loader(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - epoch_start

        total_alive = sum(int(m.sum().item()) for m in alive_mask.values())
        total_units = sum(layer_sizes.values())
        logger.info(
            f"[dpap_train] epoch {epoch + 1}/{bio_cfg.dpap_train_epochs} "
            f"train_acc={train_stats['accuracy']:.4f} val_acc={val_stats['accuracy']:.4f} "
            f"alive={total_alive}/{total_units} ({100.0 * total_alive / total_units:.1f}%) "
            f"time={epoch_time:.1f}s"
        )
        csv_log_rows.append({
            "phase": "dpap_train", "epoch": epoch + 1,
            "train_task_loss": train_stats["task_loss"], "train_kl": 0.0,
            "train_total_loss": train_stats["total_loss"], "beta": 0.0,
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"], "val_accuracy": val_stats["accuracy"],
            "epoch_time_sec": epoch_time,
        })

    _remove_hooks(bn_remask_handles)
    return alive_mask


# ---------------------------------------------------------------------------
# Physical rebuilding -- reuses pruning.py's Pruned* classes and slicing
# utilities unchanged; only the keep-index source differs from pruning.py's
# threshold-on-log_alpha logic.
# ---------------------------------------------------------------------------


def prune_lenet_activity(model: LeNetSNN, keep_masks: Dict[str, torch.Tensor], snn_cfg: SNNConfig) -> PrunedLeNetSNN:
    keep1 = keep_indices_from_mask(keep_masks["conv1"])
    keep2 = keep_indices_from_mask(keep_masks["conv2"])
    keep_fc1 = keep_indices_from_mask(keep_masks["fc1"])
    keep_fc2 = keep_indices_from_mask(keep_masks["fc2"])

    new_conv1 = nn.Conv2d(3, len(keep1), kernel_size=5)
    with torch.no_grad():
        new_conv1.weight.copy_(model.conv1.conv.weight[keep1])
        new_conv1.bias.copy_(model.conv1.conv.bias[keep1])

    new_conv2 = nn.Conv2d(len(keep1), len(keep2), kernel_size=5)
    with torch.no_grad():
        new_conv2.weight.copy_(model.conv2.conv.weight[keep2][:, keep1])
        new_conv2.bias.copy_(model.conv2.conv.bias[keep2])

    flat_idx = channel_keep_to_flat_indices(keep2, spatial_size=5 * 5)
    new_fc1 = nn.Linear(len(flat_idx), len(keep_fc1))
    with torch.no_grad():
        new_fc1.weight.copy_(model.fc1.linear.weight[keep_fc1][:, flat_idx])
        new_fc1.bias.copy_(model.fc1.linear.bias[keep_fc1])

    new_fc2 = nn.Linear(len(keep_fc1), len(keep_fc2))
    with torch.no_grad():
        new_fc2.weight.copy_(model.fc2.linear.weight[keep_fc2][:, keep_fc1])
        new_fc2.bias.copy_(model.fc2.linear.bias[keep_fc2])

    new_fc_out = nn.Linear(len(keep_fc2), model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight[:, keep_fc2])
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedLeNetSNN(new_conv1, new_conv2, new_fc1, new_fc2, new_fc_out, snn_cfg)


def prune_vgg9_activity(model: VGG9SNN, keep_masks: Dict[str, torch.Tensor], snn_cfg: SNNConfig) -> PrunedVGG9SNN:
    new_convs: List[nn.Conv2d] = []
    prev_keep = torch.arange(3)

    for i, conv in enumerate(model.conv_layers):
        keep_out = keep_indices_from_mask(keep_masks[f"conv_layers.{i}"])
        new_conv = nn.Conv2d(
            in_channels=len(prev_keep), out_channels=len(keep_out),
            kernel_size=conv.conv.kernel_size, padding=conv.conv.padding,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.conv.weight[keep_out][:, prev_keep])
            new_conv.bias.copy_(conv.conv.bias[keep_out])
        new_convs.append(new_conv)
        prev_keep = keep_out

    spatial = 4  # 3 max-pools: 32 -> 16 -> 8 -> 4
    flat_idx = channel_keep_to_flat_indices(prev_keep, spatial_size=spatial * spatial)

    keep_fc1 = keep_indices_from_mask(keep_masks["fc1"])
    new_fc1 = nn.Linear(len(flat_idx), len(keep_fc1))
    with torch.no_grad():
        new_fc1.weight.copy_(model.fc1.linear.weight[keep_fc1][:, flat_idx])
        new_fc1.bias.copy_(model.fc1.linear.bias[keep_fc1])

    new_fc_out = nn.Linear(len(keep_fc1), model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight[:, keep_fc1])
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedVGG9SNN(new_convs, list(model.pool_flags), new_fc1, new_fc_out, snn_cfg)


def prune_vggstyle_activity(
    model: "VGGStyleSNN", keep_masks: Dict[str, torch.Tensor], snn_cfg: SNNConfig
) -> "PrunedVGGStyleSNN":
    """Physically prune a VGGStyleSNN from bio-inspired activity keep-masks.

    Delegates the actual index arithmetic to pruning._rebuild_vgg_style --
    the same routine the Bayesian criterion uses -- so the two pruning
    criteria are guaranteed to rebuild an identical architecture given
    identical keep-sets, and only the *choice* of keep-set differs. That is
    exactly the controlled comparison the dissertation rests on.
    """
    conv_keeps = [
        keep_indices_from_mask(keep_masks[f"conv_layers.{i}"])
        for i in range(len(model.conv_layers))
    ]
    fc_keeps = [
        keep_indices_from_mask(keep_masks[f"fc_layers.{i}"]) for i in range(len(model.fc_layers))
    ]
    return _rebuild_vgg_style(model, model.arch_cfg, snn_cfg, conv_keeps, fc_keeps)


def _prune_basic_block_activity(block, keep_mask: torch.Tensor, snn_cfg: SNNConfig) -> PrunedBasicBlock:
    keep_mid = keep_indices_from_mask(keep_mask)

    new_conv1 = nn.Conv2d(
        in_channels=block.conv1.conv.in_channels, out_channels=len(keep_mid),
        kernel_size=block.conv1.conv.kernel_size, stride=block.conv1.conv.stride,
        padding=block.conv1.conv.padding,
    )
    with torch.no_grad():
        new_conv1.weight.copy_(block.conv1.conv.weight[keep_mid])
        new_conv1.bias.copy_(block.conv1.conv.bias[keep_mid])
    new_bn1 = slice_batchnorm(block.bn1, keep_mid)

    fixed_out_channels = block.conv2.conv.out_channels
    new_conv2 = nn.Conv2d(
        in_channels=len(keep_mid), out_channels=fixed_out_channels,
        kernel_size=block.conv2.conv.kernel_size, stride=block.conv2.conv.stride,
        padding=block.conv2.conv.padding,
    )
    with torch.no_grad():
        new_conv2.weight.copy_(block.conv2.conv.weight[:, keep_mid])
        new_conv2.bias.copy_(block.conv2.conv.bias)
    new_bn2 = slice_batchnorm(block.bn2, torch.arange(fixed_out_channels))

    new_downsample = None
    if block.downsample is not None:
        old_conv, old_bn = block.downsample[0], block.downsample[1]
        new_downsample = nn.Sequential(
            nn.Conv2d(
                old_conv.in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, bias=False,
            ),
            slice_batchnorm(old_bn, torch.arange(old_bn.num_features)),
        )
        with torch.no_grad():
            new_downsample[0].weight.copy_(old_conv.weight)

    return PrunedBasicBlock(new_conv1, new_bn1, new_conv2, new_bn2, new_downsample, snn_cfg)


def prune_resnet18_activity(
    model: SpikingResNet18, keep_masks: Dict[str, torch.Tensor], snn_cfg: SNNConfig
) -> PrunedSpikingResNet18:
    new_stem = nn.Conv2d(3, model.stem_conv.conv.out_channels, kernel_size=3, padding=1)
    with torch.no_grad():
        new_stem.weight.copy_(model.stem_conv.conv.weight)
        new_stem.bias.copy_(model.stem_conv.conv.bias)
    new_stem_bn = slice_batchnorm(model.stem_bn, torch.arange(model.stem_bn.num_features))

    new_stages: List[List[PrunedBasicBlock]] = []
    for stage_idx, stage in enumerate(model._all_stages(), start=1):
        new_blocks = [
            _prune_basic_block_activity(block, keep_masks[f"stage{stage_idx}.{block_idx}.conv1"], snn_cfg)
            for block_idx, block in enumerate(stage)
        ]
        new_stages.append(new_blocks)

    new_fc_out = nn.Linear(model.fc_out.in_features, model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight)
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedSpikingResNet18(new_stem, new_stem_bn, new_stages, new_fc_out, snn_cfg)


def prune_model_activity(
    model: nn.Module, model_name: str, keep_masks: Dict[str, torch.Tensor], snn_cfg: SNNConfig
) -> nn.Module:
    """Dispatch to the correct architecture-specific physical pruning routine."""
    dispatch = {
        "lenet": prune_lenet_activity,
        "vgg9": prune_vgg9_activity,
        "resnet18": prune_resnet18_activity,
    }
    is_vgg_style = isinstance(model, VGGStyleSNN)
    if not is_vgg_style and model_name not in dispatch:
        # Validate before mutating anything, so a rejected call leaves the
        # model's train/eval state exactly as the caller left it.
        raise ValueError(
            f"Unknown model name '{model_name}'. Built-in options: {list(dispatch)}; "
            "any other name requires the model to be a VGGStyleSNN."
        )
    model.eval()
    with torch.no_grad():
        if is_vgg_style:
            return prune_vggstyle_activity(model, keep_masks, snn_cfg)
        return dispatch[model_name](model, keep_masks, snn_cfg)
