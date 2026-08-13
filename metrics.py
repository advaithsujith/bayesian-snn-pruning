"""
Compression, resource, and timing metrics: parameter counts, FLOPs,
remaining channels/neurons, GPU memory, and simple CSV-writing helpers.
"""

import csv
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from bayesian_layers import BayesianConv2d, BayesianLinear, collect_bayesian_layers


def count_parameters(model: nn.Module, exclude_gates: bool = True) -> int:
    """
    Count the model's "real" (architectural) parameters.

    By default excludes `log_alpha` gate parameters, since those are an
    artifact of the Bayesian training procedure, not part of the deployed
    network's weights -- reporting them as part of "model size" would be
    misleading once the model has been physically pruned into a plain
    (non-Bayesian) network by pruning.py.
    """
    total = 0
    for name, p in model.named_parameters():
        if exclude_gates and "log_alpha" in name:
            continue
        total += p.numel()
    return total


def count_remaining_structures(model: nn.Module, threshold: float) -> Dict[str, Dict[str, int]]:
    """
    For every Bayesian layer in `model`, report (total, remaining) neuron
    or channel counts under the given pruning threshold. Does not modify
    the model -- this is a read-only diagnostic used both before and after
    physical pruning.
    """
    report: Dict[str, Dict[str, int]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            prunable = module.prunable_mask(threshold)
            total = prunable.numel()
            remaining = int((~prunable).sum().item())
            report[name] = {"total": total, "remaining": remaining, "pruned": total - remaining}
    return report


def compression_ratio(original_params: int, remaining_params: int) -> float:
    """Ratio of original to remaining parameters (>= 1.0 for a compressed model)."""
    if remaining_params == 0:
        return float("inf")
    return original_params / remaining_params


def pruning_percentage(original_params: int, remaining_params: int) -> float:
    """Percentage of parameters removed."""
    if original_params == 0:
        return 0.0
    return 100.0 * (1.0 - remaining_params / original_params)


class _FlopCounter:
    """Accumulates multiply-accumulate FLOPs via forward hooks on Conv2d/Linear."""

    def __init__(self) -> None:
        self.total_flops = 0
        self._handles: List[Any] = []

    def _conv_hook(self, module: nn.Conv2d, inputs: Any, output: torch.Tensor) -> None:
        batch_size, out_channels, out_h, out_w = output.shape
        in_channels_per_group = module.in_channels // module.groups
        kernel_flops = (
            in_channels_per_group * module.kernel_size[0] * module.kernel_size[1]
        )
        flops_per_instance = kernel_flops * out_channels * out_h * out_w
        self.total_flops += flops_per_instance * batch_size

    def _linear_hook(self, module: nn.Linear, inputs: Any, output: torch.Tensor) -> None:
        batch_size = output.shape[0]
        self.total_flops += module.in_features * module.out_features * batch_size

    def register(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self._handles.append(module.register_forward_hook(self._conv_hook))
            elif isinstance(module, nn.Linear):
                self._handles.append(module.register_forward_hook(self._linear_hook))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def estimate_flops(model: nn.Module, input_shape: "tuple[int, int, int, int]", device: torch.device) -> int:
    """
    Estimate total multiply-accumulate FLOPs for one full forward pass
    (i.e. across all `num_steps` simulated timesteps, since the model's
    own forward() internally loops over time).

    A lightweight hand-rolled hook-based counter is used (rather than an
    external profiling library) so it works transparently with the custom
    BayesianConv2d / BayesianLinear wrapper modules without needing custom
    op registrations.
    """
    was_training = model.training
    model.eval()
    counter = _FlopCounter()
    counter.register(model)

    dummy_input = torch.randn(*input_shape, device=device)
    model(dummy_input)

    flops = counter.total_flops
    counter.remove()
    model.train(was_training)
    return flops


def compute_and_set_unit_costs(
    model: nn.Module, input_shape: "tuple[int, int, int, int]", device: torch.device
) -> None:
    """
    Compute each Bayesian layer's static per-unit (per-neuron/per-channel)
    FLOPs cost -- the same multiply-accumulate formula _FlopCounter uses,
    disaggregated to one value per output unit instead of a single layer
    total -- and write it into that layer's `unit_cost` buffer via
    `set_unit_cost()`. This is the static `cost_j` term consumed by
    BayesianLinear/BayesianConv2d.expected_cost() in the FLOPs-aware loss
    (see losses.bayesian_snn_loss).

    Cost only depends on fixed architecture (kernel size, channel counts,
    spatial dimensions), never on learned weights, so this must be called
    exactly once, right after the model is built -- not per-epoch or
    per-batch. Hooks are registered on each Bayesian layer's inner
    `conv`/`linear` submodule and accumulate with `+=` across the one dummy
    forward pass, since that submodule is called once per simulated
    timestep inside the model's own forward loop (matching how
    estimate_flops already totals FLOPs "per full forward pass" the same
    way).
    """
    layers = collect_bayesian_layers(model)
    for layer in layers:
        layer.unit_cost.zero_()

    handles: List[Any] = []

    def _make_conv_hook(layer: "BayesianConv2d"):
        def _hook(module: nn.Conv2d, inputs: Any, output: torch.Tensor) -> None:
            _, _, out_h, out_w = output.shape
            in_channels_per_group = module.in_channels // module.groups
            kernel_flops = (
                in_channels_per_group * module.kernel_size[0] * module.kernel_size[1]
            )
            per_unit = kernel_flops * out_h * out_w
            layer.unit_cost += per_unit

        return _hook

    def _make_linear_hook(layer: "BayesianLinear"):
        def _hook(module: nn.Linear, inputs: Any, output: torch.Tensor) -> None:
            layer.unit_cost += module.in_features

        return _hook

    for layer in layers:
        if isinstance(layer, BayesianConv2d):
            handles.append(layer.conv.register_forward_hook(_make_conv_hook(layer)))
        else:
            handles.append(layer.linear.register_forward_hook(_make_linear_hook(layer)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape, device=device)
        model(dummy_input)
    model.train(was_training)

    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# SynOps: the SNN compute-cost metric
#
# A synapse only does work when the neuron feeding it fires, so an SNN's
# cost is the number of spike-triggered synaptic operations, not its dense
# MAC count. estimate_flops prices every connection as if it fired every
# timestep -- correct for an ANN, and blind to the one thing that makes an
# SNN efficient. SynOps is the metric DPAP, SCA and the wider neuromorphic
# literature report. Claim discipline (HANDOFF.md): SynOps is an energy
# proxy for event-driven hardware; it is NOT a GPU speedup -- a GPU
# multiplies zeros at full price, and measure_latency_ms already captures
# the real GPU-side gain from physically smaller tensors. Report the two
# separately.
# ---------------------------------------------------------------------------


class _LayerInputStats:
    """Per-layer accumulator: nonzero counts of a module's input, both as a
    per-channel/per-feature vector and as scalar totals."""

    def __init__(self) -> None:
        self.per_unit: "torch.Tensor | None" = None
        self.nonzero = 0.0
        self.elements = 0.0
        self.dense_macs = 0.0
        self.synops = 0.0
        # Input/output spatial sizes (H*W) of a conv's forward calls; used
        # to price MACs-per-event correctly for valid-padding convs.
        self.in_spatial = 0.0
        self.out_spatial = 0.0


def _dense_macs_for_call(module: nn.Module, output: torch.Tensor) -> float:
    """Multiply-accumulates one forward call of `module` would cost densely
    -- the same formulas _FlopCounter uses."""
    if isinstance(module, nn.Conv2d):
        batch, out_channels, out_h, out_w = output.shape
        in_per_group = module.in_channels // module.groups
        kernel = module.kernel_size[0] * module.kernel_size[1]
        return float(in_per_group * kernel * out_channels * out_h * out_w * batch)
    batch = output.shape[0]
    return float(module.in_features * module.out_features * batch)


@torch.no_grad()
def measure_synops(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    max_batches: int = 8,
) -> Dict[str, Any]:
    """
    Measure the model's SynOps per inference (one sample, all timesteps) on
    real data, alongside its dense MAC count for reference.

    Per Conv2d/Linear call, event-driven MACs = (nonzero fraction of the
    input) * (dense MACs of the call): only a nonzero input element
    triggers its synapses. For stride-1 same-padded convs this equals
    nonzero_inputs * k^2 * C_out exactly. Since every model's forward()
    loops over `num_steps` internally, hooks fire once per timestep and
    the totals come out already summed over time. The first conv's input
    is the analog image (nonzero fraction ~1 under direct encoding), so
    it is automatically priced densely -- the standard convention for the
    input layer in the SNN literature.

    Works on any model containing Conv2d/Linear submodules, gated or
    physically pruned, so before/after comparisons use one code path.
    Runs in eval mode (deterministic gates, dropout off) and restores the
    previous training state.

    Returns {"synops_per_sample", "dense_macs_per_sample", "per_layer"}
    where per_layer maps module name -> {"synops", "dense_macs",
    "input_event_frac"} (all per sample).
    """
    name_of = {m: name for name, m in model.named_modules()}
    stats: Dict[nn.Module, _LayerInputStats] = {}
    handles: List[Any] = []

    def _hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        x = inputs[0]
        s = stats.setdefault(module, _LayerInputStats())
        nonzero = float((x != 0).sum())
        elements = float(x.numel())
        dense = _dense_macs_for_call(module, output)
        s.nonzero += nonzero
        s.elements += elements
        s.dense_macs += dense
        s.synops += dense * (nonzero / elements) if elements else 0.0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    n_samples = 0
    try:
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            model(images)
            n_samples += images.shape[0]
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    if n_samples == 0:
        raise ValueError("measure_synops saw no data; loader was empty or max_batches=0")

    per_layer: Dict[str, Dict[str, float]] = {}
    total_synops = 0.0
    total_dense = 0.0
    for module, s in stats.items():
        per_layer[name_of.get(module, repr(module))] = {
            "synops": s.synops / n_samples,
            "dense_macs": s.dense_macs / n_samples,
            "input_event_frac": s.nonzero / s.elements if s.elements else 0.0,
        }
        total_synops += s.synops
        total_dense += s.dense_macs

    return {
        "synops_per_sample": total_synops / n_samples,
        "dense_macs_per_sample": total_dense / n_samples,
        "per_layer": per_layer,
    }


def _synops_chain(model: nn.Module, model_name: str) -> List["tuple[Any, nn.Module, nn.Module]"]:
    """(gated_layer, its inner Conv2d/Linear, consumer inner Conv2d/Linear)
    for every *structurally prunable* gated layer, in depth order. The
    consumer is the module that reads the gated layer's output, which is
    where that layer's spikes cost downstream work.

    SpikingResNet18 is covered by its prunable set only: each BasicBlock's
    conv1, whose output (bn1 -> gate -> lif1) is read by exactly one module,
    that same block's conv2 -- the residual addition happens on conv2's
    *output*, after the spikes conv1 pays for have already been consumed.
    The genuinely non-attributable outputs (the stem's and every conv2's,
    which feed both the next block and its identity/downsample path) belong
    to gates that are structurally_prunable=False, are excluded from every
    cost sum (see bayesian_layers.collect_prunable_bayesian_layers), and so
    never need a unit cost. The sequential families list every gated layer
    because every gated layer is prunable there; the invariant in both
    cases is chain coverage == prunable set.
    """
    from models import LeNetSNN, SpikingResNet18, VGG9SNN, VGGStyleSNN  # local: avoid an import cycle

    if isinstance(model, VGGStyleSNN):
        convs = list(model.conv_layers)
        fcs = list(model.fc_layers)
        chain = []
        for i, conv in enumerate(convs):
            if i + 1 < len(convs):
                consumer = convs[i + 1].conv
            elif fcs:
                consumer = fcs[0].linear
            else:
                consumer = model.fc_out
            chain.append((conv, conv.conv, consumer))
        for i, fc in enumerate(fcs):
            consumer = fcs[i + 1].linear if i + 1 < len(fcs) else model.fc_out
            chain.append((fc, fc.linear, consumer))
        return chain

    if isinstance(model, LeNetSNN):
        return [
            (model.conv1, model.conv1.conv, model.conv2.conv),
            (model.conv2, model.conv2.conv, model.fc1.linear),
            (model.fc1, model.fc1.linear, model.fc2.linear),
            (model.fc2, model.fc2.linear, model.fc_out),
        ]

    if isinstance(model, VGG9SNN):
        convs = list(model.conv_layers)
        chain = []
        for i, conv in enumerate(convs):
            consumer = convs[i + 1].conv if i + 1 < len(convs) else model.fc1.linear
            chain.append((conv, conv.conv, consumer))
        chain.append((model.fc1, model.fc1.linear, model.fc_out))
        return chain

    if isinstance(model, SpikingResNet18):
        return [
            (block.conv1, block.conv1.conv, block.conv2.conv)
            for stage in model._all_stages()
            for block in stage
        ]

    raise ValueError(
        f"per-unit SynOps costs are not defined for '{model_name}' "
        f"({type(model).__name__}); supported: LeNetSNN, VGG9SNN, "
        "SpikingResNet18, and any VGGStyleSNN."
    )


def _macs_per_input_event(consumer: nn.Module, out_over_in_spatial: float = 1.0) -> float:
    """MACs one nonzero input element triggers in `consumer`: every synapse
    that reads it. Linear: one weight per output feature. Conv: an average
    input position lies inside k^2 * (out_hw / in_hw) kernel windows of
    each of the C_out filters. The spatial ratio is exactly 1 for the
    same-padded stride-1 convs of the VGG families, and below 1 for
    valid-padding convs (LeNet's 5x5s: a 14x14 input maps to a 10x10
    output, so an average event sits in 25 * 100/196 ~ 12.8 windows, not
    25 -- border positions fall inside fewer windows). Using the measured
    ratio keeps this pricing identical to measure_synops's
    frac-times-dense accounting, so kept-cost fractions reconcile with the
    network-level SynOps figure."""
    if isinstance(consumer, nn.Conv2d):
        return float(
            consumer.kernel_size[0] * consumer.kernel_size[1]
            * consumer.out_channels // consumer.groups
        ) * out_over_in_spatial
    return float(consumer.out_features)


@torch.no_grad()
def measure_synops_unit_costs(
    model: nn.Module,
    model_name: str,
    loader: Any,
    device: torch.device,
    max_batches: int = 8,
) -> Dict[nn.Module, torch.Tensor]:
    """
    Measure, per prunable unit j, the SynOps saved by removing it:

        cost_j = r_in(l) * static_unit_cost_j   (computing j: only nonzero
                                                 inputs trigger its synapses)
               + events_j * macs_per_event(l+1) (j's own spikes driving the
                                                 consumer layer)

    where r_in(l) is the measured nonzero fraction of layer l's input,
    events_j is unit j's measured per-sample output event count as seen at
    the consumer's input (post-pooling, summed over timesteps), and the
    static per-unit dense cost comes from compute_and_set_unit_costs (which
    this function calls first, so `unit_cost` buffers are consistent
    intermediates). The second term is the point of using SynOps at all: a
    channel that rarely fires is nearly free to keep regardless of its
    parameter count.

    First-order approximation, stated rather than hidden: removing units
    changes the surviving units' firing rates, which this snapshot cannot
    see. That is why gate training re-measures every
    `synops_recount_every` epochs rather than trusting one snapshot.

    Returns {gated_layer_module: per-unit cost tensor (cpu, float)} for
    every gated layer in the chain -- keyed by module, matching how
    KeepPlan is keyed. Write into the model's buffers with
    set_synops_unit_costs when the costs are for the loss term rather than
    for selection.
    """
    chain = _synops_chain(model, model_name)

    first_images, _ = next(iter(loader))
    compute_and_set_unit_costs(model, (1, *first_images.shape[1:]), device)

    hooked = {inner for _, inner, _ in chain} | {consumer for _, _, consumer in chain}
    stats: Dict[nn.Module, _LayerInputStats] = {m: _LayerInputStats() for m in hooked}
    handles: List[Any] = []

    def _make_hook(module: nn.Module):
        def _hook(_module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            x = inputs[0]
            s = stats[module]
            nz = x != 0
            if x.dim() == 4:
                per_unit = nz.sum(dim=(0, 2, 3))
                s.in_spatial = float(x.shape[2] * x.shape[3])
                s.out_spatial = float(output.shape[2] * output.shape[3])
            else:
                per_unit = nz.sum(dim=0)
            per_unit = per_unit.float().cpu()
            s.per_unit = per_unit if s.per_unit is None else s.per_unit + per_unit
            s.nonzero += float(nz.sum())
            s.elements += float(x.numel())

        return _hook

    for module in hooked:
        handles.append(module.register_forward_hook(_make_hook(module)))

    was_training = model.training
    model.eval()
    n_samples = 0
    try:
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            model(images)
            n_samples += images.shape[0]
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    if n_samples == 0:
        raise ValueError("measure_synops_unit_costs saw no data; loader was empty or max_batches=0")

    costs: Dict[nn.Module, torch.Tensor] = {}
    for layer, inner, consumer in chain:
        own = stats[inner]
        in_frac = own.nonzero / own.elements if own.elements else 0.0
        term1 = in_frac * layer.unit_cost.detach().float().cpu()

        events = stats[consumer].per_unit / n_samples
        n_units = int(layer.unit_cost.numel())
        if events.numel() != n_units:
            # A conv feeding a Linear through flatten: the consumer sees
            # C * spatial features; fold each channel's spatial block.
            block = events.numel() // n_units
            if block * n_units != events.numel():
                raise RuntimeError(
                    f"cannot map {events.numel()} consumer input features onto "
                    f"{n_units} producer units; the flatten geometry is not an "
                    "integer block per channel"
                )
            events = events.view(n_units, block).sum(dim=1)
        cs = stats[consumer]
        ratio = (
            cs.out_spatial / cs.in_spatial
            if isinstance(consumer, nn.Conv2d) and cs.in_spatial
            else 1.0
        )
        term2 = events * _macs_per_input_event(consumer, ratio)

        costs[layer] = (term1 + term2).float()
    return costs


def set_synops_unit_costs(model: nn.Module, costs: Dict[nn.Module, torch.Tensor]) -> None:
    """Write measured per-unit SynOps costs into each layer's `unit_cost`
    buffer, for consumption by bayesian_layers.total_expected_synops. After
    this, the buffers no longer hold static dense FLOPs -- which is why the
    SynOps budget term and the legacy gamma term are mutually exclusive."""
    for layer, cost in costs.items():
        layer.set_unit_cost(cost)


def measure_gpu_memory_mb(device: torch.device) -> float:
    """Peak CUDA memory allocated, in megabytes. Returns 0.0 on CPU."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


@torch.no_grad()
def measure_latency_ms(
    model: nn.Module,
    input_shape: "tuple[int, int, int, int]",
    device: torch.device,
    num_warmup: int,
    num_runs: int,
) -> float:
    """
    Average wall-clock inference latency in milliseconds, over `num_runs`
    forward passes following `num_warmup` untimed warmup passes.
    """
    was_training = model.training
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    for _ in range(num_warmup):
        model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model.train(was_training)
    return (elapsed / num_runs) * 1000.0


def gather_log_alpha_values(model: nn.Module) -> Dict[str, "torch.Tensor"]:
    """Return a dict of {layer_name: log_alpha tensor} for every Bayesian layer.

    Used by run_all.py to build the uncertainty-histogram / posterior-
    variance-distribution plots requested for the dissertation figures.
    """
    values: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            values[name] = module.log_alpha.detach().cpu().clone()
    return values


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of flat dicts to a CSV file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(row: Dict[str, Any], path: str) -> None:
    """Append a single row to a CSV file, writing a header if the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
