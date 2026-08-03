"""
Posterior-uncertainty-based structured pruning.

Pruning criterion
------------------
Every structurally-prunable BayesianConv2d / BayesianLinear layer is
scored by its learned posterior gate: `log_alpha_j` is unit j's
noise-to-signal ratio, so a *low* log_alpha means the network kept that
unit's output crisp and a high one means it learned the output is
indistinguishable from noise. This is a posterior-uncertainty criterion --
derived purely from the learned gate, not from weight magnitude,
activation statistics, or any hand-designed importance heuristic.

Turning that score into a keep/drop decision is a separate choice, and
this module supports two (see KeepPlan):

* **Threshold** (`log_alpha > 3.0`, following Molchanov, Ashukha &
  Vetrov, 2017 -- an effective binary dropout rate above 95%). Faithful
  to the source method, but the resulting sparsity is *emergent*: it is
  whatever the threshold happens to cut, which is why the same
  `beta_max` gives 27.7% on LeNet and 98.8% on VGG9, and why hitting any
  particular sparsity meant re-running gate training at a new beta_max.

* **Ranked to a target sparsity** -- sort by log_alpha and keep the best
  k. The criterion is unchanged (still posterior uncertainty, still the
  same learned gates); only the cut point becomes an explicit input
  rather than an outcome. This is what makes the comparison the
  dissertation actually claims possible: one gate-training run yields a
  whole accuracy-vs-sparsity curve, the bio-inspired criteria in
  activity_pruning.py already select their keep-sets exactly this way
  (`select_keep_mask`, same `round`-based counting), so matched sparsity
  becomes true by construction rather than by coincidence, and published
  operating points (e.g. DPAP's 33.46% / 50.80%) can be hit directly.

  The failure mode to watch is gate *saturation*: units pinned at the
  clamp ceiling are tied, and ties are broken by index order, which is
  arbitrary. A run whose gates all saturated will still produce a
  network at exactly the requested sparsity and will not look wrong from
  the outside. `KeepPlan.saturation_report` exists to make that visible;
  check it before believing a ranked result.

Physical rebuilding
--------------------
Unlike masking, this module builds brand-new, smaller, purely
deterministic (non-Bayesian) modules containing only the surviving
rows/channels of each weight tensor, so the pruned model is an ordinary,
smaller SNN with no gates, no noise, and no wasted compute on removed
structures.

Residual networks are a special case: see the "Residual pruning caveat"
in models.py's module docstring for why only each BasicBlock's internal
`conv1` is physically prunable.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from bayesian_layers import (
    BayesianConv2d,
    BayesianLinear,
    collect_bayesian_layers,
    collect_prunable_bayesian_layers,
)
from config import ArchConfig, BayesianConfig, SNNConfig
from encoding import encode_timestep
from models import (
    LeNetSNN,
    VGG9SNN,
    VGGStyleSNN,
    SpikingResNet18,
    SpikingBasicBlock,
    _make_leaky,
    _VGG9_CONV_CHANNELS,
)

GatedLayer = "BayesianConv2d | BayesianLinear"


# ---------------------------------------------------------------------------
# Keep-set decisions
#
# Every architecture's rebuild routine consults a KeepPlan rather than
# recomputing a criterion inline. That indirection is the point: the plan is
# built once, the diagnostics are read off the same object the rebuild uses,
# and the two therefore cannot disagree. They previously could -- and did:
# metrics.count_remaining_structures reported a layer as 0-remaining while
# compute_keep_indices was quietly keeping one unit to avoid a zero-width
# layer, and it reported ResNet18's conv2 channels as pruned when
# _prune_basic_block never removes them.
# ---------------------------------------------------------------------------


class KeepPlan:
    """Which units each gated layer keeps, decided once for a whole model.

    Keyed by the layer module itself (nn.Module hashes by identity), so a
    plan is bound to the exact model it was built from and cannot be
    silently applied to another.
    """

    def __init__(self, keeps: Dict[nn.Module, torch.Tensor], mode: str, detail: str = "") -> None:
        self.keeps = keeps
        self.mode = mode
        self.detail = detail

    def indices(self, layer: nn.Module) -> torch.Tensor:
        """Sorted keep-indices for one layer."""
        if layer not in self.keeps:
            raise KeyError(
                "this KeepPlan has no entry for the given layer; it was almost "
                "certainly built from a different model instance"
            )
        return self.keeps[layer]

    def describe(self) -> str:
        return f"{self.mode}({self.detail})" if self.detail else self.mode

    def keep_counts(self) -> Tuple[int, ...]:
        return tuple(int(idx.numel()) for idx in self.keeps.values())

    def saturation_report(self, model: nn.Module, tol: float = 1e-3) -> Dict[str, float]:
        """
        How much of the ranking this plan used was decided by ties.

        Two ways a ranking can be vacuous, and they need separate checks:

        * **Saturation.** A gate clamped at `log_alpha_clamp_max` receives
          no gradient, so once several pin there they are indistinguishable
          and the sort falls back to index order. `frac_saturated_high`
          near 1.0 means the keep-set is arbitrary.

        * **Dispersion.** Gates can also be undifferentiated *without*
          being saturated, if the KL drags them all upward in lockstep and
          the task loss never separates them. That is not hypothetical: a
          `dpap_repl` run sat at `std(log_alpha) = 0.12` across 2304 gates
          for twenty-odd epochs with `frac_saturated = 0.000` the whole
          time, so the saturation check alone would have passed it. Only
          later did it march into the clamp. `log_alpha_std` catches the
          earlier state.

        Neither shows up in accuracy or in the sparsity figure. A run that
        fails both still produces a clean, monotone accuracy-vs-sparsity
        curve, because the widths are an input. What it measures then is
        random structured pruning, which is a useful control but not
        evidence about the criterion.
        """
        high = low = total = 0
        values = []
        for layer in collect_prunable_bayesian_layers(model):
            la = layer.log_alpha.detach()
            high += int((la >= layer.log_alpha_clamp_max - tol).sum())
            low += int((la <= layer.log_alpha_clamp_min + tol).sum())
            total += int(la.numel())
            values.append(la.flatten())
        if total == 0:
            return {
                "frac_saturated_high": 0.0,
                "frac_saturated_low": 0.0,
                "log_alpha_std": 0.0,
                "total_gates": 0,
            }
        pooled = torch.cat(values)
        return {
            "frac_saturated_high": high / total,
            "frac_saturated_low": low / total,
            "log_alpha_std": float(pooled.std()) if pooled.numel() > 1 else 0.0,
            "total_gates": total,
        }

    def ranking_is_usable(
        self, model: nn.Module, max_saturated: float = 0.5, min_std: float = 0.25
    ) -> Tuple[bool, str]:
        """(usable, reason). See saturation_report for what each bound means."""
        r = self.saturation_report(model)
        if r["frac_saturated_high"] > max_saturated:
            return False, (
                f"{r['frac_saturated_high']:.1%} of gates are pinned at the clamp "
                "ceiling, so the keep-set is largely index order among ties"
            )
        if r["log_alpha_std"] < min_std:
            return False, (
                f"log_alpha std is {r['log_alpha_std']:.3f} across {r['total_gates']} "
                "gates, so they never differentiated from each other and the "
                "ranking is close to arbitrary"
            )
        return True, (
            f"log_alpha std {r['log_alpha_std']:.3f}, "
            f"{r['frac_saturated_high']:.1%} saturated"
        )


def _ranked_keep_indices(layer: GatedLayer, keep_count: int) -> torch.Tensor:
    """Sorted indices of the `keep_count` units with the lowest log_alpha.

    Ranking uses the *raw* parameter rather than `_clamped_log_alpha()`:
    the clamp is a forward-pass guard, and collapsing everything beyond it
    to one value would throw away whatever ordering momentum did manage to
    establish among saturated gates.
    """
    n = layer.log_alpha.numel()
    keep_count = min(max(1, int(keep_count)), n)
    order = torch.argsort(layer.log_alpha.detach())
    return order[:keep_count].sort().values.long()


def _keep_count_for_fraction(n: int, keep_fraction: float, min_keep: int = 1) -> int:
    """Units to keep in a layer of `n` at a target fraction.

    Deliberately identical to activity_pruning.select_keep_mask's
    `min(max(1, round(f * n)), n)` -- if the two rounded differently, a
    "matched sparsity" comparison would silently compare networks of
    different widths.
    """
    return min(max(min_keep, round(keep_fraction * n)), n)


def compute_keep_indices(layer: GatedLayer, threshold: float) -> torch.Tensor:
    """
    Return the (sorted) indices of channels/neurons to KEEP for one layer.

    If every unit in a layer would be pruned, the single most confident
    unit (smallest log_alpha) is kept regardless, so a layer is never
    reduced to zero width.
    """
    prunable = layer.prunable_mask(threshold)
    keep_bool = ~prunable
    keep = keep_bool.nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        keep = layer.log_alpha.detach().argmin().unsqueeze(0)
    return keep.long()


def _all_gated_layers(model: nn.Module) -> List[GatedLayer]:
    return collect_bayesian_layers(model)


def _plan_from_counts(
    model: nn.Module, counts: Dict[nn.Module, int], mode: str, detail: str = ""
) -> KeepPlan:
    """Build a plan that keeps the best `counts[layer]` units of each prunable
    layer, and every unit of every non-prunable one."""
    keeps: Dict[nn.Module, torch.Tensor] = {}
    for layer in _all_gated_layers(model):
        if layer.structurally_prunable:
            keeps[layer] = _ranked_keep_indices(layer, counts[layer])
        else:
            keeps[layer] = torch.arange(layer.log_alpha.numel(), dtype=torch.long)
    return KeepPlan(keeps, mode, detail)


def threshold_plan(model: nn.Module, threshold: float) -> KeepPlan:
    """Molchanov-style plan: drop every unit with log_alpha above `threshold`.

    Sparsity is an outcome, not an input. Reproduces the original
    behaviour of this module exactly, including the keep-at-least-one
    fallback.
    """
    keeps: Dict[nn.Module, torch.Tensor] = {}
    for layer in _all_gated_layers(model):
        if layer.structurally_prunable:
            keeps[layer] = compute_keep_indices(layer, threshold)
        else:
            keeps[layer] = torch.arange(layer.log_alpha.numel(), dtype=torch.long)
    return KeepPlan(keeps, "threshold", f"log_alpha>{threshold:g}")


def uniform_ratio_plan(model: nn.Module, keep_fraction: float, min_keep: int = 1) -> KeepPlan:
    """Keep the same *fraction* of units in every prunable layer.

    This is the strictly-controlled comparison point: run at the same
    keep_fraction as a bio-inspired criterion and the two pruned networks
    have identical layer widths, so the only thing that differs between
    them is which units were chosen. Nothing about the architecture, the
    parameter count or the FLOPs can confound the accuracy difference.

    It also denies the Bayesian criterion its allocation freedom, which is
    a real part of what it does (VGG9's fc1 collapsing 800 -> 21 while the
    conv stack stayed wide). Use global_ratio_plan to measure that, and
    this one to isolate the selection quality.
    """
    counts = {
        layer: _keep_count_for_fraction(layer.log_alpha.numel(), keep_fraction, min_keep)
        for layer in collect_prunable_bayesian_layers(model)
    }
    return _plan_from_counts(model, counts, "uniform_ratio", f"keep_fraction={keep_fraction:g}")


def global_ratio_plan(model: nn.Module, keep_fraction: float, min_keep: int = 1) -> KeepPlan:
    """Keep the best `keep_fraction` of all prunable units, ranked across the
    whole network rather than within each layer.

    log_alpha is a noise-to-signal *ratio* and therefore dimensionless, so
    comparing it across layers is meaningful in a way that comparing raw
    activations or weight norms would not be. This lets the criterion
    decide where the redundancy is, which is the behaviour the threshold
    rule had and the per-layer version gives up.

    `min_keep` is a floor, not a suggestion: a global ranking is free to
    empty a whole layer, and a severed layer is not a sparse network but a
    broken one (LeNet's fc2 collapsing to a single unit cost ~43
    accuracy points and no amount of fine-tuning recovered it). Promoting
    units to satisfy the floor can push the realised keep-count slightly
    above the target; the promotion count is recorded in the plan detail.
    """
    layers = collect_prunable_bayesian_layers(model)
    if not layers:
        return _plan_from_counts(model, {}, "global_ratio", f"keep_fraction={keep_fraction:g}")

    scores = torch.cat([layer.log_alpha.detach().flatten() for layer in layers])
    owner = torch.cat(
        [
            torch.full((layer.log_alpha.numel(),), i, dtype=torch.long)
            for i, layer in enumerate(layers)
        ]
    )
    total = int(scores.numel())
    target = min(max(1, round(keep_fraction * total)), total)

    order = torch.argsort(scores)
    selected = torch.zeros(total, dtype=torch.bool)
    selected[order[:target]] = True

    counts: Dict[nn.Module, int] = {}
    promoted = 0
    for i, layer in enumerate(layers):
        n = int((owner == i).sum())
        kept = int(selected[owner == i].sum())
        floor = min(min_keep, n)
        if kept < floor:
            promoted += floor - kept
            kept = floor
        counts[layer] = kept

    detail = f"keep_fraction={keep_fraction:g}"
    if promoted:
        detail += f", {promoted} unit(s) promoted to satisfy min_keep={min_keep}"
    return _plan_from_counts(model, counts, "global_ratio", detail)


def _pruned_param_count(model: nn.Module, model_name: str, plan: KeepPlan, snn_cfg: SNNConfig) -> int:
    """Parameters remaining if `plan` were applied. Uses the real rebuild
    routines rather than a parallel formula, so the planner and the thing
    it plans for can never drift apart."""
    from metrics import count_parameters  # local: metrics imports bayesian_layers, not this

    was_training = model.training
    pruned = prune_model(model, model_name, plan, snn_cfg)
    model.train(was_training)
    return count_parameters(pruned, exclude_gates=True)


def param_target_plan(
    model: nn.Module,
    model_name: str,
    snn_cfg: SNNConfig,
    target_pruned_pct: float,
    mode: str = "uniform",
    min_keep: int = 1,
    max_iter: int = 30,
) -> Tuple[KeepPlan, float]:
    """
    Find the plan whose *parameter* pruning percentage is closest to
    `target_pruned_pct`, and return it with the percentage it achieves.

    Sparsity is quoted in the literature as a fraction of parameters (or
    connections) removed, but a keep-fraction is a fraction of *units*.
    Those are not the same number and the gap is large: dropping the same
    fraction of units from two adjacent layers shrinks the weight matrix
    between them on both axes, so a uniform per-layer keep_fraction of
    0.72 on LeNet removes 46% of parameters, not 28%. Any comparison
    against a published pruning percentage has to go through this
    conversion or it is comparing different quantities.

    Parameter count is monotonically non-decreasing in the keep fraction
    but takes integer steps, so an exact hit is usually not available;
    bisection runs to `max_iter` and the closest achievable point is
    returned. Rebuilds are memoised on the keep-count vector, so the loop
    costs far fewer than `max_iter` model constructions. Runs entirely on
    whatever device the model is on and touches no data -- no GPU time.
    """
    fraction, pct, plan = _bisect_to_param_target(
        model, model_name, snn_cfg, target_pruned_pct, mode, min_keep, max_iter
    )
    promoted = plan.detail.partition(", ")[2]
    plan.detail = (
        f"keep_fraction={fraction:.4f} -> {pct:.2f}% of parameters pruned "
        f"(target {target_pruned_pct:g}%)"
    )
    if promoted:
        plan.detail += f"; {promoted}"
    return plan, pct


def _bisect_to_param_target(
    model: nn.Module,
    model_name: str,
    snn_cfg: SNNConfig,
    target_pruned_pct: float,
    mode: str,
    min_keep: int,
    max_iter: int,
) -> Tuple[float, float, KeepPlan]:
    """Bisect the keep-fraction until the realised parameter-pruning
    percentage is as close to the target as the integer steps allow.
    Returns (keep_fraction, achieved_pct, plan)."""
    build = {"uniform": uniform_ratio_plan, "global": global_ratio_plan}
    if mode not in build:
        raise ValueError(f"Unknown ranked-pruning mode '{mode}'. Options: {list(build)}")

    from metrics import count_parameters

    total_params = count_parameters(model, exclude_gates=True)
    cache: Dict[Tuple[int, ...], float] = {}

    def evaluate(keep_fraction: float) -> Tuple[float, KeepPlan]:
        plan = build[mode](model, keep_fraction, min_keep)
        key = plan.keep_counts()
        if key not in cache:
            remaining = _pruned_param_count(model, model_name, plan, snn_cfg)
            cache[key] = 100.0 * (1.0 - remaining / total_params)
        return cache[key], plan

    lo, hi = 0.0, 1.0
    best_pct, best_plan = evaluate(1.0)
    best = (1.0, best_pct, best_plan)
    best_err = abs(best_pct - target_pruned_pct)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        pct, plan = evaluate(mid)
        err = abs(pct - target_pruned_pct)
        if err < best_err:
            best_err, best = err, (mid, pct, plan)
        if pct > target_pruned_pct:
            lo = mid  # pruning too hard -> keep more
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return best


def keep_fraction_for_param_target(
    model: nn.Module,
    model_name: str,
    snn_cfg: SNNConfig,
    target_pruned_pct: float,
    min_keep: int = 1,
    max_iter: int = 30,
) -> Tuple[float, float]:
    """
    The uniform per-layer keep_fraction that removes `target_pruned_pct` of
    parameters, plus the percentage it actually achieves.

    Needed to set `BioPruningConfig.keep_fractions` for a matched-sparsity
    comparison: the bio criteria take a keep_fraction as input, while the
    Bayesian side and every published baseline quote parameter
    percentages. Converting between them by eye is how "matched sparsity"
    ended up comparing a 27.7%-pruned network against a 98.5%-pruned one.
    Purely geometric -- no data, no training, no GPU.
    """
    fraction, pct, _ = _bisect_to_param_target(
        model, model_name, snn_cfg, target_pruned_pct, "uniform", min_keep, max_iter
    )
    return fraction, pct


def build_keep_plan(
    model: nn.Module,
    model_name: str,
    snn_cfg: SNNConfig,
    bayesian_cfg: BayesianConfig,
) -> KeepPlan:
    """Build the KeepPlan a config asks for. `prune_mode="threshold"` is the
    default and reproduces every pre-existing experiment unchanged."""
    mode = bayesian_cfg.prune_mode
    if mode == "threshold":
        return threshold_plan(model, bayesian_cfg.prune_threshold)
    if mode == "uniform_ratio":
        return uniform_ratio_plan(model, bayesian_cfg.keep_fraction, bayesian_cfg.min_keep_per_layer)
    if mode == "global_ratio":
        return global_ratio_plan(model, bayesian_cfg.keep_fraction, bayesian_cfg.min_keep_per_layer)
    if mode in ("param_target", "param_target_global"):
        plan, _ = param_target_plan(
            model,
            model_name,
            snn_cfg,
            bayesian_cfg.target_pruned_pct,
            mode="global" if mode.endswith("global") else "uniform",
            min_keep=bayesian_cfg.min_keep_per_layer,
        )
        return plan
    raise ValueError(
        f"Unknown prune_mode '{mode}'. Options: threshold, uniform_ratio, "
        "global_ratio, param_target, param_target_global"
    )


def remaining_structures_report(model: nn.Module, plan: KeepPlan) -> Dict[str, Dict[str, Any]]:
    """
    Per-layer (total, remaining, pruned) counts **as the rebuild will
    actually apply them**.

    Preferred over metrics.count_remaining_structures, which re-derives the
    decision from a threshold and so can disagree with what was built. Two
    ways it did: a layer whose gates all crossed the threshold was reported
    as 0 remaining although compute_keep_indices keeps one unit, and
    ResNet18's residual-tied conv2 was reported as 0/512 remaining although
    _prune_basic_block never removes a single one of its output channels.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (BayesianConv2d, BayesianLinear)):
            continue
        keep = plan.indices(module)
        total = int(module.log_alpha.numel())
        remaining = int(keep.numel())
        report[name] = {
            "total": total,
            "remaining": remaining,
            "pruned": total - remaining,
            "structurally_prunable": module.structurally_prunable,
            "median_log_alpha": float(module.log_alpha.detach().median()),
        }
    return report


def channel_keep_to_flat_indices(keep_channels: torch.Tensor, spatial_size: int) -> torch.Tensor:
    """
    Convert channel indices to keep into flat indices into a
    channel-major-flattened [C, H, W] -> [C*H*W] tensor (i.e. matching
    PyTorch's `tensor.flatten(1)` on a [B, C, H, W] input), so that a
    downstream BayesianLinear's input dimension can be sliced consistently
    with which conv output channels survived pruning.
    """
    flat_indices: List[int] = []
    for c in keep_channels.tolist():
        start = c * spatial_size
        flat_indices.extend(range(start, start + spatial_size))
    return torch.tensor(flat_indices, dtype=torch.long)


def slice_batchnorm(bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    """Build a smaller BatchNorm2d containing only the kept channels' statistics."""
    new_bn = nn.BatchNorm2d(len(keep_idx))
    with torch.no_grad():
        new_bn.weight.copy_(bn.weight[keep_idx])
        new_bn.bias.copy_(bn.bias[keep_idx])
        new_bn.running_mean.copy_(bn.running_mean[keep_idx])
        new_bn.running_var.copy_(bn.running_var[keep_idx])
        new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    return new_bn


def compute_uncertainty_report(model: nn.Module, threshold: float) -> Dict[str, Dict[str, float]]:
    """
    Summarise the learned posterior uncertainty of every Bayesian layer:
    min / median / max log_alpha, the fraction of units past the pruning
    threshold, and the fraction pinned at the clamp ceiling. Used for the
    "Computing posterior uncertainty..." logging step and for the
    uncertainty-histogram plots.

    `frac_saturated` and `std` matter most under ranked pruning: a layer
    whose gates have all saturated has no usable internal ordering left,
    and one whose `std` is ~0 never differentiated its units at all. Both
    look identical to a healthy layer in `frac_prunable` alone.
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            log_alpha = module.log_alpha.detach()
            report[name] = {
                "min": float(log_alpha.min()),
                "median": float(log_alpha.median()),
                "max": float(log_alpha.max()),
                "std": float(log_alpha.std()) if log_alpha.numel() > 1 else 0.0,
                "frac_prunable": float((log_alpha > threshold).float().mean()),
                "frac_saturated": float(
                    (log_alpha >= module.log_alpha_clamp_max - 1e-3).float().mean()
                ),
                "structurally_prunable": module.structurally_prunable,
            }
    return report


# ---------------------------------------------------------------------------
# LeNet-SNN
# ---------------------------------------------------------------------------


class PrunedLeNetSNN(nn.Module):
    """A physically-pruned, non-Bayesian LeNet-SNN. Forward logic mirrors
    LeNetSNN.forward exactly, but every layer is a plain deterministic
    nn.Conv2d / nn.Linear built with only the surviving rows/channels."""

    def __init__(
        self,
        conv1: nn.Conv2d,
        conv2: nn.Conv2d,
        fc1: nn.Linear,
        fc2: nn.Linear,
        fc_out: nn.Linear,
        snn_cfg: SNNConfig,
    ) -> None:
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.conv1 = conv1
        self.lif1 = _make_leaky(snn_cfg)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = conv2
        self.lif2 = _make_leaky(snn_cfg)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = fc1
        self.lif3 = _make_leaky(snn_cfg)
        self.fc2 = fc2
        self.lif4 = _make_leaky(snn_cfg)
        self.fc_out = fc_out
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec: List[torch.Tensor] = []
        for _ in range(self.num_steps):
            cur1 = self.pool1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur_out = self.fc_out(spk4)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)


def prune_lenet(model: LeNetSNN, plan: KeepPlan, snn_cfg: SNNConfig) -> PrunedLeNetSNN:
    """Physically prune a trained LeNetSNN into a smaller PrunedLeNetSNN."""
    keep1 = plan.indices(model.conv1)
    keep2 = plan.indices(model.conv2)
    keep_fc1 = plan.indices(model.fc1)
    keep_fc2 = plan.indices(model.fc2)

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


# ---------------------------------------------------------------------------
# VGG9-SNN
# ---------------------------------------------------------------------------


class PrunedVGG9SNN(nn.Module):
    """A physically-pruned, non-Bayesian VGG9-SNN."""

    def __init__(
        self,
        conv_layers: List[nn.Conv2d],
        pool_flags: List[bool],
        fc1: nn.Linear,
        fc_out: nn.Linear,
        snn_cfg: SNNConfig,
    ) -> None:
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lif_layers = nn.ModuleList([_make_leaky(snn_cfg) for _ in conv_layers])
        self.pool_flags = pool_flags
        self.pool = nn.MaxPool2d(2)
        self.fc1 = fc1
        self.lif_fc1 = _make_leaky(snn_cfg)
        self.fc_out = fc_out
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem_conv = [lif.init_leaky() for lif in self.lif_layers]
        mem_fc1 = self.lif_fc1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec: List[torch.Tensor] = []
        for _ in range(self.num_steps):
            spk = x
            for i, (conv, lif) in enumerate(zip(self.conv_layers, self.lif_layers)):
                cur = conv(spk)
                spk, mem_conv[i] = lif(cur, mem_conv[i])
                if self.pool_flags[i]:
                    spk = self.pool(spk)

            cur_fc1 = self.fc1(spk.flatten(1))
            spk_fc1, mem_fc1 = self.lif_fc1(cur_fc1, mem_fc1)
            cur_out = self.fc_out(spk_fc1)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)


def prune_vgg9(model: VGG9SNN, plan: KeepPlan, snn_cfg: SNNConfig) -> PrunedVGG9SNN:
    """Physically prune a trained VGG9SNN into a smaller PrunedVGG9SNN.

    Channel keep-masks are propagated sequentially through the conv chain:
    layer i's kept input channels are exactly layer (i-1)'s kept output
    channels (the very first layer's input is always the full 3 RGB
    channels, which are never pruned).
    """
    new_convs: List[nn.Conv2d] = []
    prev_keep = torch.arange(3)

    for conv in model.conv_layers:
        keep_out = plan.indices(conv)
        new_conv = nn.Conv2d(
            in_channels=len(prev_keep),
            out_channels=len(keep_out),
            kernel_size=conv.conv.kernel_size,
            padding=conv.conv.padding,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.conv.weight[keep_out][:, prev_keep])
            new_conv.bias.copy_(conv.conv.bias[keep_out])
        new_convs.append(new_conv)
        prev_keep = keep_out

    spatial = 4  # 3 max-pools: 32 -> 16 -> 8 -> 4
    flat_idx = channel_keep_to_flat_indices(prev_keep, spatial_size=spatial * spatial)

    keep_fc1 = plan.indices(model.fc1)
    new_fc1 = nn.Linear(len(flat_idx), len(keep_fc1))
    with torch.no_grad():
        new_fc1.weight.copy_(model.fc1.linear.weight[keep_fc1][:, flat_idx])
        new_fc1.bias.copy_(model.fc1.linear.bias[keep_fc1])

    new_fc_out = nn.Linear(len(keep_fc1), model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight[:, keep_fc1])
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedVGG9SNN(new_convs, list(model.pool_flags), new_fc1, new_fc_out, snn_cfg)


# ---------------------------------------------------------------------------
# Configurable VGG-style conv stack (models.VGGStyleSNN)
# ---------------------------------------------------------------------------


class PrunedVGGStyleSNN(nn.Module):
    """A physically-pruned, non-Bayesian counterpart of VGGStyleSNN.

    Forward logic mirrors VGGStyleSNN.forward exactly -- same encoding,
    pooling, normalisation and constant-across-timesteps dropout -- but
    every gated layer is a plain deterministic nn.Conv2d / nn.Linear
    containing only the surviving channels/neurons.
    """

    def __init__(
        self,
        conv_layers: List[nn.Conv2d],
        norm_layers: List[nn.Module],
        fc_layers: List[nn.Linear],
        fc_out: nn.Linear,
        arch_cfg: ArchConfig,
        snn_cfg: SNNConfig,
    ) -> None:
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.arch_cfg = arch_cfg
        self.encoding = arch_cfg.encoding
        self.dropout_p = arch_cfg.dropout_p
        self.pool_flags = _pool_flags_from_spec(arch_cfg.conv_spec)

        thresholds = arch_cfg.layer_thresholds

        def threshold_for(idx: int) -> "float | None":
            return None if thresholds is None else thresholds[idx]

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.lif_layers = nn.ModuleList(
            [_make_leaky(snn_cfg, threshold_override=threshold_for(i)) for i in range(len(conv_layers))]
        )
        self.pool = (
            nn.AvgPool2d(arch_cfg.pool_size)
            if arch_cfg.pool_type == "avg"
            else nn.MaxPool2d(arch_cfg.pool_size)
        )

        n_conv = len(conv_layers)
        self.fc_layers = nn.ModuleList(fc_layers)
        self.lif_fc_layers = nn.ModuleList(
            [_make_leaky(snn_cfg, threshold_override=threshold_for(n_conv + i)) for i in range(len(fc_layers))]
        )

        self.fc_out = fc_out
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def _dropout(self, spk: torch.Tensor, cache: List["torch.Tensor | None"], idx: int) -> torch.Tensor:
        """See VGGStyleSNN._dropout -- mask held constant across timesteps."""
        if self.dropout_p <= 0.0 or not self.training:
            return spk
        if cache[idx] is None:
            keep_prob = 1.0 - self.dropout_p
            cache[idx] = torch.bernoulli(torch.full_like(spk, keep_prob)) / keep_prob
        return spk * cache[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem_conv = [lif.init_leaky() for lif in self.lif_layers]
        mem_fc = [lif.init_leaky() for lif in self.lif_fc_layers]
        mem_out = self.lif_out.init_leaky()

        conv_masks: List["torch.Tensor | None"] = [None] * len(self.conv_layers)
        fc_masks: List["torch.Tensor | None"] = [None] * len(self.fc_layers)

        spk_out_rec: List[torch.Tensor] = []
        for _ in range(self.num_steps):
            spk = encode_timestep(x, self.encoding)

            for i, (conv, norm, lif) in enumerate(
                zip(self.conv_layers, self.norm_layers, self.lif_layers)
            ):
                cur = norm(conv(spk))
                spk, mem_conv[i] = lif(cur, mem_conv[i])
                if self.pool_flags[i]:
                    spk = self.pool(spk)
                spk = self._dropout(spk, conv_masks, i)

            spk = spk.flatten(1)
            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_fc_layers)):
                cur = fc(spk)
                spk, mem_fc[i] = lif(cur, mem_fc[i])
                spk = self._dropout(spk, fc_masks, i)

            cur_out = self.fc_out(spk)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)


def transfer_leaky_state(src: nn.Module, dst: nn.Module, keep_idx: "torch.Tensor | None" = None) -> None:
    """
    Copy a learned membrane decay from one Leaky neuron to another.

    When `SNNConfig.learn_beta` is True, snnTorch promotes `beta` to a
    trained `nn.Parameter`, so it carries information exactly like a weight
    does. Every physical-rebuild routine constructs *fresh* neurons via
    `_make_leaky`, which re-initialises beta to the config's starting value
    -- silently discarding whatever was learned. That matters most for the
    DPAP replication, whose parametric-LIF membrane time constant is the
    method's central claim (see docs/replication_targets.md), and it fails
    silently: the pruned network simply runs with the wrong dynamics.

    `keep_idx` slices beta when it is per-channel; a scalar beta (snnTorch's
    default, even when learnable) is copied through unchanged.
    """
    src_beta = getattr(src, "beta", None)
    dst_beta = getattr(dst, "beta", None)
    if src_beta is None or dst_beta is None:
        return
    with torch.no_grad():
        value = src_beta.detach()
        if keep_idx is not None and value.ndim > 0 and value.numel() > 1:
            value = value[keep_idx]
        if dst_beta.shape != value.shape:
            raise ValueError(
                f"cannot transfer learned beta: source shape {tuple(value.shape)} "
                f"does not match destination {tuple(dst_beta.shape)}"
            )
        dst_beta.copy_(value)


def _pool_flags_from_spec(conv_spec: List) -> List[bool]:
    """Recover the per-conv "is there a pool after me" flags from a conv spec.

    Mirrors the loop in VGGStyleSNN.__init__ so the pruned model pools at
    exactly the same depths as the model it was built from.
    """
    flags: List[bool] = []
    for entry in conv_spec:
        if entry == "M":
            flags[-1] = True
        else:
            flags.append(False)
    return flags


def _rebuild_vgg_style(
    model: nn.Module,
    arch_cfg: ArchConfig,
    snn_cfg: SNNConfig,
    conv_keeps: List[torch.Tensor],
    fc_keeps: List[torch.Tensor],
) -> PrunedVGGStyleSNN:
    """
    Build a PrunedVGGStyleSNN from explicit per-layer keep-index tensors.

    Shared by both pruning criteria: the Bayesian path (pruning.py) derives
    `conv_keeps`/`fc_keeps` from posterior uncertainty, and the bio-inspired
    path (activity_pruning.py) derives them from activity masks. Keeping the
    index arithmetic in exactly one place means the two criteria cannot
    silently disagree about how a network is physically rebuilt.

    Keep-masks propagate sequentially: layer i's surviving *input* channels
    are layer (i-1)'s surviving *output* channels. The first conv's input is
    the untouched image, and the classifier's output width is never pruned.
    """
    new_convs: List[nn.Conv2d] = []
    new_norms: List[nn.Module] = []
    prev_keep = torch.arange(arch_cfg.in_channels)

    for conv, norm, keep_out in zip(model.conv_layers, model.norm_layers, conv_keeps):
        new_conv = nn.Conv2d(
            in_channels=len(prev_keep),
            out_channels=len(keep_out),
            kernel_size=conv.conv.kernel_size,
            stride=conv.conv.stride,
            padding=conv.conv.padding,
            bias=conv.conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(conv.conv.weight[keep_out][:, prev_keep])
            if conv.conv.bias is not None:
                new_conv.bias.copy_(conv.conv.bias[keep_out])
        new_convs.append(new_conv)

        new_norms.append(
            slice_batchnorm(norm, keep_out) if isinstance(norm, nn.BatchNorm2d) else nn.Identity()
        )
        prev_keep = keep_out

    spatial = arch_cfg.spatial_after_convs()
    flat_idx = channel_keep_to_flat_indices(prev_keep, spatial_size=spatial * spatial)

    new_fcs: List[nn.Linear] = []
    prev_in = flat_idx
    for fc, keep_out in zip(model.fc_layers, fc_keeps):
        new_fc = nn.Linear(len(prev_in), len(keep_out))
        with torch.no_grad():
            new_fc.weight.copy_(fc.linear.weight[keep_out][:, prev_in])
            new_fc.bias.copy_(fc.linear.bias[keep_out])
        new_fcs.append(new_fc)
        prev_in = keep_out

    # With no hidden fc layers (e.g. Chowdhury's VGG9, whose 3 VGG fc layers
    # collapse to a single classifier), the classifier reads the flattened
    # conv output directly, so it is sliced by flat_idx rather than by a
    # preceding fc's keep-set.
    new_fc_out = nn.Linear(len(prev_in), model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight[:, prev_in])
        new_fc_out.bias.copy_(model.fc_out.bias)

    pruned = PrunedVGGStyleSNN(new_convs, new_norms, new_fcs, new_fc_out, arch_cfg, snn_cfg)

    # Carry over any *learned* membrane decay -- the fresh neurons built
    # inside PrunedVGGStyleSNN would otherwise reset it to the config's
    # initial value. See transfer_leaky_state.
    for src, dst, keep in zip(model.lif_layers, pruned.lif_layers, conv_keeps):
        transfer_leaky_state(src, dst, keep)
    for src, dst, keep in zip(model.lif_fc_layers, pruned.lif_fc_layers, fc_keeps):
        transfer_leaky_state(src, dst, keep)
    transfer_leaky_state(model.lif_out, pruned.lif_out)

    return pruned


def prune_vgg_style(model: nn.Module, plan: "KeepPlan | float", snn_cfg: SNNConfig) -> PrunedVGGStyleSNN:
    """Physically prune a trained VGGStyleSNN using posterior uncertainty.

    `plan` may be a bare float for backwards compatibility, in which case
    it is interpreted as the Molchanov log_alpha threshold.
    """
    if not isinstance(plan, KeepPlan):
        plan = threshold_plan(model, float(plan))
    conv_keeps = [plan.indices(conv) for conv in model.conv_layers]
    fc_keeps = [plan.indices(fc) for fc in model.fc_layers]
    return _rebuild_vgg_style(model, model.arch_cfg, snn_cfg, conv_keeps, fc_keeps)


# ---------------------------------------------------------------------------
# Spiking ResNet-18
# ---------------------------------------------------------------------------


class PrunedBasicBlock(nn.Module):
    """A physically-pruned BasicBlock: conv1/bn1 shrink to the surviving
    'mid' channels; conv2/bn2/downsample keep their original fixed width
    (see models.py's residual pruning caveat)."""

    def __init__(
        self,
        conv1: nn.Conv2d,
        bn1: nn.BatchNorm2d,
        conv2: nn.Conv2d,
        bn2: nn.BatchNorm2d,
        downsample: "nn.Sequential | None",
        snn_cfg: SNNConfig,
    ) -> None:
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.lif1 = _make_leaky(snn_cfg)
        self.conv2 = conv2
        self.bn2 = bn2
        self.lif2 = _make_leaky(snn_cfg)
        self.downsample = downsample

    def init_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lif1.init_leaky(), self.lif2.init_leaky()

    def forward(
        self, x: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out, mem1)
        out = self.bn2(self.conv2(spk1))
        out = out + identity
        spk2, mem2 = self.lif2(out, mem2)
        return spk2, mem1, mem2


class PrunedSpikingResNet18(nn.Module):
    """A physically-pruned, non-Bayesian Spiking ResNet-18."""

    def __init__(
        self,
        stem_conv: nn.Conv2d,
        stem_bn: nn.BatchNorm2d,
        stages: List[List[PrunedBasicBlock]],
        fc_out: nn.Linear,
        snn_cfg: SNNConfig,
    ) -> None:
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.stem_conv = stem_conv
        self.stem_bn = stem_bn
        self.stem_lif = _make_leaky(snn_cfg)
        self.stage1 = nn.ModuleList(stages[0])
        self.stage2 = nn.ModuleList(stages[1])
        self.stage3 = nn.ModuleList(stages[2])
        self.stage4 = nn.ModuleList(stages[3])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = fc_out
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def _all_stages(self) -> List[nn.ModuleList]:
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stages = self._all_stages()
        block_mem_states = [[block.init_state() for block in stage] for stage in stages]
        stem_mem = self.stem_lif.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec: List[torch.Tensor] = []
        for _ in range(self.num_steps):
            cur = self.stem_bn(self.stem_conv(x))
            spk, stem_mem = self.stem_lif(cur, stem_mem)

            for stage_idx, stage in enumerate(stages):
                for block_idx, block in enumerate(stage):
                    mem1, mem2 = block_mem_states[stage_idx][block_idx]
                    spk, mem1, mem2 = block(spk, mem1, mem2)
                    block_mem_states[stage_idx][block_idx] = (mem1, mem2)

            pooled = self.global_pool(spk).flatten(1)
            cur_out = self.fc_out(pooled)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)


def _prune_basic_block(block: SpikingBasicBlock, plan: KeepPlan, snn_cfg: SNNConfig) -> PrunedBasicBlock:
    """Prune one BasicBlock's internal (conv1) channels only."""
    keep_mid = plan.indices(block.conv1)

    new_conv1 = nn.Conv2d(
        in_channels=block.conv1.conv.in_channels,
        out_channels=len(keep_mid),
        kernel_size=block.conv1.conv.kernel_size,
        stride=block.conv1.conv.stride,
        padding=block.conv1.conv.padding,
    )
    with torch.no_grad():
        new_conv1.weight.copy_(block.conv1.conv.weight[keep_mid])
        new_conv1.bias.copy_(block.conv1.conv.bias[keep_mid])
    new_bn1 = slice_batchnorm(block.bn1, keep_mid)

    fixed_out_channels = block.conv2.conv.out_channels
    new_conv2 = nn.Conv2d(
        in_channels=len(keep_mid),
        out_channels=fixed_out_channels,
        kernel_size=block.conv2.conv.kernel_size,
        stride=block.conv2.conv.stride,
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
                old_conv.in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                bias=False,
            ),
            slice_batchnorm(old_bn, torch.arange(old_bn.num_features)),
        )
        with torch.no_grad():
            new_downsample[0].weight.copy_(old_conv.weight)

    return PrunedBasicBlock(new_conv1, new_bn1, new_conv2, new_bn2, new_downsample, snn_cfg)


def prune_resnet18(model: SpikingResNet18, plan: KeepPlan, snn_cfg: SNNConfig) -> PrunedSpikingResNet18:
    """
    Physically prune a trained SpikingResNet18.

    Only each BasicBlock's internal `conv1` channels are removed. The
    stem, every block's `conv2` (residual-tied) output, and every
    downsample projection keep their original fixed width, so all
    residual additions remain dimensionally valid without any special
    padding/projection logic. See the module docstring for rationale.
    """
    new_stem = nn.Conv2d(3, model.stem_conv.conv.out_channels, kernel_size=3, padding=1)
    with torch.no_grad():
        new_stem.weight.copy_(model.stem_conv.conv.weight)
        new_stem.bias.copy_(model.stem_conv.conv.bias)
    new_stem_bn = slice_batchnorm(model.stem_bn, torch.arange(model.stem_bn.num_features))

    new_stages: List[List[PrunedBasicBlock]] = []
    for stage in model._all_stages():
        new_stages.append([_prune_basic_block(block, plan, snn_cfg) for block in stage])

    new_fc_out = nn.Linear(model.fc_out.in_features, model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight)
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedSpikingResNet18(new_stem, new_stem_bn, new_stages, new_fc_out, snn_cfg)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def prune_model(
    model: nn.Module, model_name: str, plan: "KeepPlan | float", snn_cfg: SNNConfig
) -> nn.Module:
    """Dispatch to the correct architecture-specific physical pruning routine.

    `plan` is normally a KeepPlan (see build_keep_plan). A bare float is
    accepted and interpreted as the Molchanov log_alpha threshold, so every
    existing call site keeps working unchanged.

    Any VGGStyleSNN (the configurable conv-stack family used for the paper
    replications) routes to `prune_vgg_style` regardless of its config name,
    since that one routine handles every architecture the class can express.
    """
    dispatch = {
        "lenet": prune_lenet,
        "vgg9": prune_vgg9,
        "resnet18": prune_resnet18,
    }
    is_vgg_style = isinstance(model, VGGStyleSNN)
    if not is_vgg_style and model_name not in dispatch:
        # Validate before mutating anything, so a rejected call leaves the
        # model's train/eval state exactly as the caller left it.
        raise ValueError(
            f"Unknown model name '{model_name}'. Built-in options: {list(dispatch)}; "
            "any other name requires the model to be a VGGStyleSNN."
        )
    if not isinstance(plan, KeepPlan):
        plan = threshold_plan(model, float(plan))
    model.eval()
    with torch.no_grad():
        if is_vgg_style:
            return prune_vgg_style(model, plan, snn_cfg)
        return dispatch[model_name](model, plan, snn_cfg)
