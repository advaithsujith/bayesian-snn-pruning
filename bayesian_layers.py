"""
Structured Bayesian layers: BayesianLinear and BayesianConv2d.

Methodology
-----------
Each output neuron (BayesianLinear) or output channel (BayesianConv2d) is
given one multiplicative stochastic gate z_j. During training, the layer's
ordinary deterministic pre-activation h_j is scaled by a sampled gate:

    z_j = 1 + sqrt(alpha_j) * eps_j,      eps_j ~ N(0, 1)
    output_j = h_j * z_j

This is the standard "Gaussian multiplicative noise" parameterisation of
variational dropout (Kingma, Salimans & Welling, 2015, "Variational
Dropout and the Local Reparameterization Trick"), extended to a structured
(one-gate-per-neuron/channel) setting following Neklyudov, Molchanov,
Ashukha & Vetrov (2017, "Structured Bayesian Pruning via Log-Normal
Multiplicative Noise"). Each gate's only free parameter is
log_alpha_j = log(sigma_j^2 / mu_j^2), the noise-to-signal ratio of the
underlying log-normal gate posterior.

A log-uniform prior is placed over each gate. Because a log-uniform prior
has no preferred scale, the KL divergence between the learned posterior
and this prior is minimised precisely when a gate's noise dominates its
mean (log_alpha -> +infinity), which is what drives redundant
neurons/channels toward the "always noisy, effectively off" regime during
training. The closed-form KL approximation used below is from Molchanov,
Ashukha & Vetrov (2017, "Variational Dropout Sparsifies Deep Neural
Networks", eq. 14), and is the same approximation used by essentially all
follow-up structured/unstructured variational-dropout-pruning work.

After training, any gate with log_alpha above `prune_threshold` (3.0 by
default, following Molchanov et al., corresponding to an effective binary
dropout rate > 95%) is treated as redundant and physically removed by
pruning.py -- this file only implements the probabilistic gating
mechanism, not the physical removal.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants for the closed-form KL(q(z|mu,alpha) || log-uniform prior)
# approximation, Molchanov, Ashukha & Vetrov (2017), eq. 14.
_KL_K1 = 0.63576
_KL_K2 = 1.87320
_KL_K3 = 1.48695


def kl_divergence_from_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """
    Closed-form approximation of KL(q(z | log_alpha) || p(z)) where p(z) is
    the (improper) log-uniform prior, summed over every element of
    `log_alpha`.

    Returns a scalar tensor. This function is shared by BayesianLinear and
    BayesianConv2d so that the exact same approximation is used everywhere
    in the codebase.
    """
    neg_kl = (
        _KL_K1 * torch.sigmoid(_KL_K2 + _KL_K3 * log_alpha)
        - 0.5 * F.softplus(-log_alpha)
        - _KL_K1
    )
    return (-neg_kl).sum()


class BayesianLinear(nn.Module):
    """
    A fully-connected layer with one structured Bayesian gate per output
    neuron.

    The underlying deterministic weights (`self.linear`) are ordinary,
    individually-learned weights -- exactly like a normal `nn.Linear`.
    What is Bayesian and structured is the single extra gate per output
    neuron layered on top of the whole neuron's output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        log_alpha_init: float = -3.0,
        log_alpha_clamp_min: float = -8.0,
        log_alpha_clamp_max: float = 8.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_alpha_clamp_min = log_alpha_clamp_min
        self.log_alpha_clamp_max = log_alpha_clamp_max

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.log_alpha = nn.Parameter(torch.full((out_features,), log_alpha_init))

        # Non-learned buffer used by pruning.py to zero out already-decided
        # dead neurons at evaluation time, before the model is physically
        # rebuilt into a smaller architecture.
        self.register_buffer("hard_mask", torch.ones(out_features))

        # Static per-neuron FLOPs cost, populated once (post-construction,
        # pre-training) by metrics.compute_and_set_unit_costs. Zero until
        # then, so the expected-cost loss term is inert if this is ever
        # accidentally left unset, rather than injecting a bogus flat cost.
        self.register_buffer("unit_cost", torch.zeros(out_features))

        # Whether pruning.py is permitted to physically remove this layer's
        # gated units. Set False by architectures (e.g. ResNet BasicBlock's
        # conv2) whose output dimension is tied to a residual addition and
        # therefore cannot be independently resized. True by default.
        self.structurally_prunable = True

        # When False, this layer behaves as a plain deterministic
        # nn.Linear (no gate sampled, log_alpha receives no gradient) --
        # used for the "Train" pipeline stage, which trains a conventional
        # SNN baseline *before* "Converting to Bayesian" flips this flag
        # on for the Bayesian-pruning training stage. See
        # bayesian_layers.set_bayesian_mode().
        self.enable_gate_noise = True

    def _clamped_log_alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(self.log_alpha_clamp_min, self.log_alpha_clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. During training with gate noise enabled, injects the
        reparameterised multiplicative gate noise. Otherwise (evaluation,
        or gate noise disabled for deterministic pretraining), applies the
        (learned) hard_mask deterministically with no injected noise,
        since the expected value of the gate under its posterior is 1.
        """
        h = self.linear(x)
        if self.training and self.enable_gate_noise:
            alpha = torch.exp(self._clamped_log_alpha())
            eps = torch.randn_like(h)
            z = 1.0 + torch.sqrt(alpha) * eps
            return h * z
        return h * self.hard_mask

    def kl(self) -> torch.Tensor:
        """Sum of the per-neuron KL divergence to the log-uniform prior."""
        return kl_divergence_from_log_alpha(self._clamped_log_alpha())

    def expected_cost(self, threshold: float) -> torch.Tensor:
        """
        Expected FLOPs cost of this layer's surviving neurons under the
        current posterior: sum_j(unit_cost_j * p_keep_j), where
        p_keep_j = 1 - sigmoid(log_alpha_j - threshold) is a smooth,
        differentiable surrogate for "will neuron j survive pruning"
        (unlike prunable_mask's hard boolean, this must stay differentiable
        so gradient can reach log_alpha -- same reasoning as kl()).
        """
        p_keep = 1.0 - torch.sigmoid(self._clamped_log_alpha() - threshold)
        return (p_keep * self.unit_cost).sum()

    def prunable_mask(self, threshold: float) -> torch.Tensor:
        """Boolean tensor, True where a neuron's log_alpha exceeds `threshold`."""
        return self._clamped_log_alpha().detach() > threshold

    def set_hard_mask(self, keep_mask: torch.Tensor) -> None:
        """Freeze a keep/drop decision into `hard_mask` for eval-time use."""
        self.hard_mask.copy_(keep_mask.float())

    def set_unit_cost(self, cost: torch.Tensor) -> None:
        """Set the static per-neuron FLOPs cost used by expected_cost()."""
        self.unit_cost.copy_(cost.float())


class BayesianConv2d(nn.Module):
    """
    A 2D convolution with one structured Bayesian gate per output channel.

    Mechanically identical to BayesianLinear except the gate is broadcast
    over the spatial (H, W) dimensions of the conv output, since an output
    "structure" here is a whole output feature map (channel), not a single
    scalar. One `eps` is drawn per (example, channel) and reused at every
    spatial position -- see apply_gate for why that is not a detail.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        log_alpha_init: float = -3.0,
        log_alpha_clamp_min: float = -8.0,
        log_alpha_clamp_max: float = 8.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.log_alpha_clamp_min = log_alpha_clamp_min
        self.log_alpha_clamp_max = log_alpha_clamp_max

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.log_alpha = nn.Parameter(torch.full((out_channels,), log_alpha_init))
        self.register_buffer("hard_mask", torch.ones(out_channels))

        # See BayesianLinear.unit_cost.
        self.register_buffer("unit_cost", torch.zeros(out_channels))

        # See BayesianLinear.structurally_prunable.
        self.structurally_prunable = True

        # See BayesianLinear.enable_gate_noise.
        self.enable_gate_noise = True

        # When True, forward() returns the raw convolution output and the
        # caller must apply the gate itself via apply_gate(). This exists so
        # a gate can sit *after* a BatchNorm rather than before it.
        #
        # Ordering matters more than it looks. With the gate inside forward()
        # the sequence is conv -> gate -> BatchNorm, and BatchNorm then
        # divides out precisely the variance the gate just injected: at
        # log_alpha = 8 the gate multiplies by noise of standard deviation
        # ~55, which BatchNorm immediately renormalises away. Raising
        # log_alpha therefore barely changes what the next layer sees, so the
        # task loss exerts almost no gradient pressure against it while the
        # KL term's pull is undiminished -- and every gate runs away to the
        # clamp ceiling regardless of beta_max. This was observed on every
        # batch-normalised architecture in this project and on none of the
        # others. Deferring the gate to after the normalisation restores the
        # task loss's ability to feel it.
        self.defer_gate = False

    def _clamped_log_alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(self.log_alpha_clamp_min, self.log_alpha_clamp_max)

    def apply_gate(self, h: torch.Tensor) -> torch.Tensor:
        """Apply this layer's gate to `h`, which must have this layer's
        output-channel count. Used with `defer_gate` so the gate can act on
        the post-normalisation activation.

        The noise is drawn once per (example, channel) and broadcast across
        the feature map, which is what makes this a *structured* gate: the
        whole channel is scaled by one common random factor, exactly as in
        Neklyudov et al. (2017). Drawing an independent eps per spatial
        position instead (`torch.randn_like(h)`) would still share alpha
        per channel, but the resulting perturbation largely cancels when
        the next layer sums over space -- so the task loss barely feels
        alpha, the KL faces almost no opposition, and conv gates rise for
        reasons that have nothing to do with the channel being redundant.
        It is also H*W times more random numbers to generate per timestep.
        """
        if self.training and self.enable_gate_noise:
            alpha = torch.exp(self._clamped_log_alpha())
            eps = torch.randn(
                h.shape[0], h.shape[1], 1, 1, device=h.device, dtype=h.dtype
            )
            z = 1.0 + torch.sqrt(alpha).view(1, -1, 1, 1) * eps
            return h * z
        return h * self.hard_mask.view(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; see BayesianLinear.forward for the training/eval logic."""
        h = self.conv(x)
        if self.defer_gate:
            return h
        return self.apply_gate(h)

    def kl(self) -> torch.Tensor:
        """Sum of the per-channel KL divergence to the log-uniform prior."""
        return kl_divergence_from_log_alpha(self._clamped_log_alpha())

    def expected_cost(self, threshold: float) -> torch.Tensor:
        """See BayesianLinear.expected_cost -- identical formula, per-channel."""
        p_keep = 1.0 - torch.sigmoid(self._clamped_log_alpha() - threshold)
        return (p_keep * self.unit_cost).sum()

    def prunable_mask(self, threshold: float) -> torch.Tensor:
        """Boolean tensor, True where a channel's log_alpha exceeds `threshold`."""
        return self._clamped_log_alpha().detach() > threshold

    def set_hard_mask(self, keep_mask: torch.Tensor) -> None:
        """Freeze a keep/drop decision into `hard_mask` for eval-time use."""
        self.hard_mask.copy_(keep_mask.float())

    def set_unit_cost(self, cost: torch.Tensor) -> None:
        """Set the static per-channel FLOPs cost used by expected_cost()."""
        self.unit_cost.copy_(cost.float())


def collect_bayesian_layers(model: nn.Module) -> list:
    """Return every BayesianLinear / BayesianConv2d submodule of `model`, in
    the order PyTorch's module iteration discovers them (which for models
    built the standard sequential way corresponds to network depth order).
    """
    return [m for m in model.modules() if isinstance(m, (BayesianLinear, BayesianConv2d))]


def collect_prunable_bayesian_layers(model: nn.Module) -> list:
    """Return only those Bayesian layers pruning.py is allowed to physically
    shrink, i.e. those with `structurally_prunable=True`.

    Every loss term and every gate-noise decision is scoped to this subset
    rather than to `collect_bayesian_layers`. A gate on a layer that can
    never be removed buys no compression, so any pressure applied to it is
    pure cost: the KL drives its log_alpha to the clamp ceiling, and the
    resulting multiplicative noise (std ~55 at log_alpha=8) is injected
    into a layer that will still be there, at full width, after pruning.
    On SpikingResNet18 that describes 1984 of 3904 gates -- the stem and
    every block's residual-tied conv2 -- so slightly over half the KL
    pressure was being spent damaging the network for zero compression.
    """
    return [m for m in collect_bayesian_layers(model) if m.structurally_prunable]


def _zero_like_model(model: nn.Module) -> torch.Tensor:
    """A scalar zero on the same device as `model`, so an empty sum can be
    added to a device-resident loss without a cross-device error."""
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return torch.zeros((), device=device)


def total_kl(model: nn.Module) -> torch.Tensor:
    """Sum of `.kl()` over every *structurally prunable* Bayesian layer.

    See collect_prunable_bayesian_layers for why non-prunable layers are
    excluded rather than merely tolerated.
    """
    layers = collect_prunable_bayesian_layers(model)
    if not layers:
        return _zero_like_model(model)
    return sum(layer.kl() for layer in layers)


def total_expected_cost(model: nn.Module, threshold: float) -> torch.Tensor:
    """Sum of `.expected_cost(threshold)` over every structurally prunable
    Bayesian layer -- scoped identically to total_kl, and for the same
    reason: a layer that cannot shrink cannot save any FLOPs either."""
    layers = collect_prunable_bayesian_layers(model)
    if not layers:
        return _zero_like_model(model)
    return sum(layer.expected_cost(threshold) for layer in layers)


def set_bayesian_mode(model: nn.Module, active: bool) -> None:
    """
    Toggle `enable_gate_noise` on every Bayesian layer in `model`.

    `active=False` (used for the initial deterministic "Train" pipeline
    stage) makes every BayesianLinear/BayesianConv2d behave as a plain,
    noise-free layer -- log_alpha never enters the forward computation and
    therefore receives no gradient. `active=True` ("Converting to
    Bayesian...") switches on gate sampling so the subsequent Bayesian
    training stage can actually learn meaningful posterior uncertainty.

    Structurally non-prunable layers stay noise-free either way. Their
    gates receive no KL pressure (see total_kl), so sampling them would
    only add variance to a layer whose width is fixed -- and the pruned
    network they are rebuilt into is deterministic regardless.
    """
    for layer in collect_bayesian_layers(model):
        layer.enable_gate_noise = active and layer.structurally_prunable
