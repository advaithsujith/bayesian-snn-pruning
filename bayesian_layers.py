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
    scalar.
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

    def _clamped_log_alpha(self) -> torch.Tensor:
        return self.log_alpha.clamp(self.log_alpha_clamp_min, self.log_alpha_clamp_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass; see BayesianLinear.forward for the training/eval logic."""
        h = self.conv(x)
        if self.training and self.enable_gate_noise:
            alpha = torch.exp(self._clamped_log_alpha())
            eps = torch.randn_like(h)
            z = 1.0 + torch.sqrt(alpha).view(1, -1, 1, 1) * eps
            return h * z
        return h * self.hard_mask.view(1, -1, 1, 1)

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


def total_kl(model: nn.Module) -> torch.Tensor:
    """Sum of `.kl()` over every Bayesian layer in `model`."""
    layers = collect_bayesian_layers(model)
    if not layers:
        return torch.tensor(0.0)
    return sum(layer.kl() for layer in layers)


def total_expected_cost(model: nn.Module, threshold: float) -> torch.Tensor:
    """Sum of `.expected_cost(threshold)` over every Bayesian layer in `model`."""
    layers = collect_bayesian_layers(model)
    if not layers:
        return torch.tensor(0.0)
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
    """
    for layer in collect_bayesian_layers(model):
        layer.enable_gate_noise = active
