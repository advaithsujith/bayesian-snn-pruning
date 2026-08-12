"""
Three spiking-neural-network architectures, each built from the
structured Bayesian layers in bayesian_layers.py: LeNet-SNN (~62K params),
VGG9-SNN (~9M params), and Spiking ResNet-18 (~11.7M params).

Encoding scheme
----------------
All three models use "direct encoding": the same static CIFAR-10 image is
fed into the network at every simulated timestep, and temporal dynamics
arise entirely from the leaky-integrate-and-fire (LIF) membrane potentials
accumulating and firing over time. This is the standard simplification
used throughout the modern surrogate-gradient SNN literature (e.g. Fang et
al., 2021, "Deep Residual Learning in Spiking Neural Networks") and avoids
needing a separate Poisson/rate input-encoding stage.

Residual pruning caveat (Spiking ResNet-18 only)
-------------------------------------------------
Channels that participate in a residual addition must keep matching
dimensions on both branches, so not every BayesianConv2d in a
BasicBlock can be safely, independently channel-pruned. Each BasicBlock
exposes exactly one "internal" BayesianConv2d (`conv1`) whose output
channel count is free to shrink, and one "boundary" BayesianConv2d
(`conv2`) whose output channel count is tied to the block's residual
addition and is therefore excluded from physical structured pruning
(`structurally_prunable=False`) even though it still carries a Bayesian
gate and still contributes to the KL regularisation term during training.
This is a standard, well-documented simplification in the structured
pruning literature for residual architectures.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from bayesian_layers import BayesianConv2d, BayesianLinear
from config import ArchConfig, SNNConfig, BayesianConfig
from encoding import encode_timestep


# Normalisation layers that renormalise away a gate applied before them.
# Used by assert_gate_after_norm.
_NORM_TYPES = (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)


def read_output(module: nn.Module, cur_out: torch.Tensor, mem_out: torch.Tensor):
    """
    One timestep of the output layer, honouring `SNNConfig.output_readout`.

    Returns `(value_to_record, mem_out)`, where the value is either the output
    neuron's spikes (default, every original experiment) or `fc_out`'s
    pre-synaptic current. Shared by all six forward passes -- the four models
    here and the pruned rebuilds in pruning.py -- so a model and the network
    rebuilt from it cannot disagree about how their output is read. They must
    not: a fine-tuned accuracy read one way is not on the same scale as a
    baseline read the other.

    Under "current" the output neuron is deliberately constructed but never
    called. It holds no trainable parameter (SNNConfig.__post_init__ rejects
    learn_beta with this readout), so skipping it changes no state_dict, while
    calling it would burn compute on a spike train nothing reads. `mem_out` is
    then returned unchanged and stays at its initial value.

    See SNNConfig.output_readout for why the analog readout exists: TET
    defines its per-timestep output as the output layer's pre-synaptic input,
    and at T=4 a summed spike count takes only five values per class.
    """
    if module.output_readout == "current":
        return cur_out, mem_out
    return module.lif_out(cur_out, mem_out)


def _get_surrogate_grad(name: str):
    """Map a config string to an snnTorch surrogate-gradient function."""
    mapping = {
        "atan": surrogate.atan(),
        "fast_sigmoid": surrogate.fast_sigmoid(),
        "sigmoid": surrogate.sigmoid(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown surrogate gradient '{name}'. Options: {list(mapping)}")
    return mapping[name]


def _make_leaky(
    snn_cfg: SNNConfig, output: bool = False, threshold_override: "float | None" = None
) -> snn.Leaky:
    """Construct an snnTorch Leaky neuron layer from the shared SNN config.

    `threshold_override` replaces the config's global firing threshold for
    this one neuron layer, which the ANN-to-SNN conversion path needs: there
    each layer's threshold is calibrated separately from that layer's own
    activation distribution instead of being shared network-wide (see
    docs/replication_targets.md). All pre-existing callers omit it and are
    unaffected.

    `learn_beta` makes the membrane decay a trained parameter rather than a
    fixed constant -- snnTorch's nearest equivalent to the parametric LIF
    (PLIFNode) that DPAP uses.
    """
    return snn.Leaky(
        beta=snn_cfg.beta,
        threshold=snn_cfg.threshold if threshold_override is None else threshold_override,
        spike_grad=_get_surrogate_grad(snn_cfg.spike_grad),
        init_hidden=False,
        output=output,
        learn_beta=snn_cfg.learn_beta,
        reset_mechanism=snn_cfg.reset_mechanism,
    )


class LeNetSNN(nn.Module):
    """
    Spiking LeNet-5, adapted for 3x32x32 CIFAR-10 input.

    Architecture: conv(3->6,5) - pool - conv(6->16,5) - pool -
    fc(400->120) - fc(120->84) - fc(84->10). This is the standard
    LeNet-5-for-CIFAR configuration used throughout the SNN pruning
    literature (~62K parameters), with every layer except the final
    classifier gated by a structured Bayesian gate.
    """

    def __init__(self, snn_cfg: SNNConfig, bayesian_cfg: BayesianConfig, num_classes: int = 10):
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.output_readout = snn_cfg.output_readout
        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )

        self.conv1 = BayesianConv2d(3, 6, kernel_size=5, **gate_kwargs)
        self.lif1 = _make_leaky(snn_cfg)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = BayesianConv2d(6, 16, kernel_size=5, **gate_kwargs)
        self.lif2 = _make_leaky(snn_cfg)
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = BayesianLinear(16 * 5 * 5, 120, **gate_kwargs)
        self.lif3 = _make_leaky(snn_cfg)

        self.fc2 = BayesianLinear(120, 84, **gate_kwargs)
        self.lif4 = _make_leaky(snn_cfg)

        self.fc_out = nn.Linear(84, num_classes)  # classifier: never pruned, fixed width
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate `num_steps` timesteps under direct (static) encoding.

        Returns spk_rec of shape [num_steps, batch_size, num_classes].
        """
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
            out_t, mem_out = read_output(self, cur_out, mem_out)
            spk_out_rec.append(out_t)

        return torch.stack(spk_out_rec, dim=0)


_VGG9_CONV_CHANNELS = [64, 64, "M", 128, 128, "M", 256, 256, 512, "M"]
_VGG9_FC_HIDDEN = 800


class VGG9SNN(nn.Module):
    """
    Spiking VGG-9 (7 convolutional + 2 fully-connected weight layers,
    following the "VGGx = x weight layers" naming convention), adapted for
    3x32x32 CIFAR-10 input with ~9M parameters. Channel configuration is a
    standard VGG-style pattern used across the SNN literature (e.g. the
    VGG variants used in Sengupta et al., 2019, "Going Deeper in Spiking
    Neural Networks", scaled down to 9 weight layers).
    """

    def __init__(self, snn_cfg: SNNConfig, bayesian_cfg: BayesianConfig, num_classes: int = 10):
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.snn_cfg = snn_cfg
        self.output_readout = snn_cfg.output_readout
        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )

        conv_layers: List[nn.Module] = []
        lif_layers: List[snn.Leaky] = []
        pool_flags: List[bool] = []

        in_channels = 3
        for entry in _VGG9_CONV_CHANNELS:
            if entry == "M":
                pool_flags[-1] = True
                continue
            conv_layers.append(
                BayesianConv2d(in_channels, entry, kernel_size=3, padding=1, **gate_kwargs)
            )
            lif_layers.append(_make_leaky(snn_cfg))
            pool_flags.append(False)
            in_channels = entry

        self.conv_layers = nn.ModuleList(conv_layers)
        self.lif_layers = nn.ModuleList(lif_layers)
        self.pool_flags = pool_flags  # python list of bool, not learnable state
        self.pool = nn.MaxPool2d(2)

        # 3 max-pools halve 32 -> 16 -> 8 -> 4
        self.flatten_dim = in_channels * 4 * 4

        self.fc1 = BayesianLinear(self.flatten_dim, _VGG9_FC_HIDDEN, **gate_kwargs)
        self.lif_fc1 = _make_leaky(snn_cfg)

        self.fc_out = nn.Linear(_VGG9_FC_HIDDEN, num_classes)  # classifier: never pruned
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate `num_steps` timesteps under direct (static) encoding.

        Returns spk_rec of shape [num_steps, batch_size, num_classes].
        """
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
            out_t, mem_out = read_output(self, cur_out, mem_out)
            spk_out_rec.append(out_t)

        return torch.stack(spk_out_rec, dim=0)


class SpikingBasicBlock(nn.Module):
    """
    A spiking version of the ResNet BasicBlock.

    `conv1` is the internal, freely structurally-prunable layer. `conv2`'s
    output channel count is tied to the block's residual addition and to
    the input of the next block, so it is marked non-prunable
    (structurally_prunable=False) -- see the module-level docstring for
    why. Both still carry Bayesian gates, but only `conv1`'s is sampled
    and only `conv1`'s enters the KL (see
    bayesian_layers.collect_prunable_bayesian_layers).

    Both convs feed a BatchNorm, so both defer their gate to after it --
    see BayesianConv2d.defer_gate. Applying the gate first lets BatchNorm
    divide the injected noise straight back out, which leaves the task
    loss unable to feel log_alpha at all.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        snn_cfg: SNNConfig,
        bayesian_cfg: BayesianConfig,
    ) -> None:
        super().__init__()
        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )

        self.conv1 = BayesianConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, **gate_kwargs
        )
        self.conv1.structurally_prunable = True
        self.conv1.defer_gate = True  # conv -> bn -> gate -> lif
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = _make_leaky(snn_cfg)

        self.conv2 = BayesianConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, **gate_kwargs
        )
        self.conv2.structurally_prunable = False
        self.conv2.defer_gate = True
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = _make_leaky(snn_cfg)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def init_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lif1.init_leaky(), self.lif2.init_leaky()

    def forward(
        self, x: torch.Tensor, mem1: torch.Tensor, mem2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        identity = x if self.downsample is None else self.downsample(x)

        # `conv(x)` returns the ungated convolution because defer_gate is
        # set, so the gate is applied to the normalised activation instead.
        out = self.conv1.apply_gate(self.bn1(self.conv1(x)))
        spk1, mem1 = self.lif1(out, mem1)

        # conv2's gate acts on its own output, before the residual addition
        # -- the identity path is not this layer's to scale.
        out = self.conv2.apply_gate(self.bn2(self.conv2(spk1)))
        out = out + identity
        spk2, mem2 = self.lif2(out, mem2)

        return spk2, mem1, mem2


class SpikingResNet18(nn.Module):
    """
    Spiking ResNet-18 for CIFAR-10 (~11.7M parameters), following the
    standard CIFAR ResNet stem (3x3 conv, no initial maxpool, unlike the
    ImageNet 7x7-conv+maxpool stem) with 4 stages of 2 BasicBlocks each
    ([2, 2, 2, 2]) and channel widths [64, 128, 256, 512], matching the
    standard ResNet-18 configuration (He et al., 2016) adapted with
    spiking neurons (e.g. following the spiking-ResNet design pattern of
    Fang et al., 2021, "Deep Residual Learning in Spiking Neural
    Networks").
    """

    def __init__(self, snn_cfg: SNNConfig, bayesian_cfg: BayesianConfig, num_classes: int = 10):
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.output_readout = snn_cfg.output_readout
        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )

        self.stem_conv = BayesianConv2d(3, 64, kernel_size=3, stride=1, padding=1, **gate_kwargs)
        # The stem's output channel count is the fixed input width every
        # stage-1 block expects; kept non-prunable so no block boundary
        # needs input-side reindexing (see module docstring).
        self.stem_conv.structurally_prunable = False
        self.stem_conv.defer_gate = True  # conv -> bn -> gate -> lif
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_lif = _make_leaky(snn_cfg)

        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1, snn_cfg=snn_cfg, bayesian_cfg=bayesian_cfg)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, snn_cfg=snn_cfg, bayesian_cfg=bayesian_cfg)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, snn_cfg=snn_cfg, bayesian_cfg=bayesian_cfg)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2, snn_cfg=snn_cfg, bayesian_cfg=bayesian_cfg)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512, num_classes)  # classifier: never pruned
        self.lif_out = _make_leaky(snn_cfg, output=True)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        snn_cfg: SNNConfig,
        bayesian_cfg: BayesianConfig,
    ) -> nn.ModuleList:
        blocks = [SpikingBasicBlock(in_channels, out_channels, stride, snn_cfg, bayesian_cfg)]
        for _ in range(1, num_blocks):
            blocks.append(SpikingBasicBlock(out_channels, out_channels, 1, snn_cfg, bayesian_cfg))
        return nn.ModuleList(blocks)

    def _all_stages(self) -> List[nn.ModuleList]:
        return [self.stage1, self.stage2, self.stage3, self.stage4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate `num_steps` timesteps under direct (static) encoding.

        Returns spk_rec of shape [num_steps, batch_size, num_classes].
        """
        stages = self._all_stages()
        block_mem_states = [[block.init_state() for block in stage] for stage in stages]
        stem_mem = self.stem_lif.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec: List[torch.Tensor] = []
        for _ in range(self.num_steps):
            cur = self.stem_conv.apply_gate(self.stem_bn(self.stem_conv(x)))
            spk, stem_mem = self.stem_lif(cur, stem_mem)

            for stage_idx, stage in enumerate(stages):
                for block_idx, block in enumerate(stage):
                    mem1, mem2 = block_mem_states[stage_idx][block_idx]
                    spk, mem1, mem2 = block(spk, mem1, mem2)
                    block_mem_states[stage_idx][block_idx] = (mem1, mem2)

            pooled = self.global_pool(spk).flatten(1)
            cur_out = self.fc_out(pooled)
            out_t, mem_out = read_output(self, cur_out, mem_out)
            spk_out_rec.append(out_t)

        return torch.stack(spk_out_rec, dim=0)


class VGGStyleSNN(nn.Module):
    """
    A configurable plain-feedforward conv-stack SNN, driven entirely by an
    ArchConfig.

    Every architecture this project needs beyond the residual ResNet is a
    conv stack followed by fully-connected layers: the original VGG9, and
    the three published setups being replicated (Chowdhury's 8-conv VGG9,
    SCA's VGG16, DPAP's 6Conv2FC -- see docs/replication_targets.md). Rather
    than four near-duplicate classes (and, worse, four near-duplicate
    physical-rebuild routines in pruning.py, the most error-prone code in
    the repo), this one class expresses all of them via configuration.

    `VGG9SNN` above is deliberately left in place and untouched: existing
    VGG9 results and checkpoints were produced by it and must stay
    reproducible. `ArchConfig()`'s defaults describe that same architecture,
    which `tests/test_vggstyle.py` asserts by comparing parameter count,
    per-layer weight shapes, pooling depths and output shape.

    Note that the two classes are *architecturally* equivalent but not
    state_dict-compatible: this class names its fully-connected stack
    `fc_layers.0` / `lif_fc_layers.0` where VGG9SNN uses `fc1` / `lif_fc1`,
    so an existing VGG9 checkpoint needs key remapping before it will load
    here. That is deliberate -- the names generalise to any number of
    hidden fc layers, including none.

    Layer naming (`conv_layers`, `lif_layers`, `fc_layers`, `lif_fc_layers`,
    `fc_out`) is what pruning.py and activity_pruning.py key off, so both
    pruning paths work for any config this class accepts.
    """

    def __init__(
        self,
        arch_cfg: ArchConfig,
        snn_cfg: SNNConfig,
        bayesian_cfg: BayesianConfig,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.arch_cfg = arch_cfg
        self.snn_cfg = snn_cfg
        self.encoding = arch_cfg.encoding
        self.dropout_p = arch_cfg.dropout_p
        self.output_readout = snn_cfg.output_readout

        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )

        # Per-layer thresholds, if calibrated, are consumed in depth order:
        # every conv layer first, then every hidden fc layer.
        thresholds = arch_cfg.layer_thresholds
        n_gated = len(arch_cfg.conv_channels()) + len(arch_cfg.fc_hidden)
        if thresholds is not None and len(thresholds) != n_gated:
            raise ValueError(
                f"layer_thresholds has {len(thresholds)} entries but this "
                f"architecture has {n_gated} gated layers"
            )

        def threshold_for(idx: int) -> "float | None":
            return None if thresholds is None else thresholds[idx]

        conv_layers: List[nn.Module] = []
        norm_layers: List[nn.Module] = []
        lif_layers: List[snn.Leaky] = []
        pool_flags: List[bool] = []

        in_channels = arch_cfg.in_channels
        gated_idx = 0
        for entry in arch_cfg.conv_spec:
            if entry == "M":
                if not pool_flags:
                    raise ValueError("conv_spec cannot start with a pooling entry 'M'")
                pool_flags[-1] = True
                continue
            conv_layers.append(
                BayesianConv2d(
                    in_channels,
                    entry,
                    kernel_size=arch_cfg.kernel_size,
                    padding=arch_cfg.padding,
                    bias=arch_cfg.conv_bias,
                    **gate_kwargs,
                )
            )
            use_norm = arch_cfg.norm_type == "batch"
            norm_layers.append(nn.BatchNorm2d(entry) if use_norm else nn.Identity())
            # With a BatchNorm between the conv and its neuron, the gate must
            # act *after* the normalisation -- see BayesianConv2d.defer_gate
            # for why placing it before makes the gate invisible to the task
            # loss. Without a norm layer the two orderings are identical, so
            # the existing in-forward path is left untouched.
            conv_layers[-1].defer_gate = use_norm
            lif_layers.append(_make_leaky(snn_cfg, threshold_override=threshold_for(gated_idx)))
            pool_flags.append(False)
            in_channels = entry
            gated_idx += 1

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layers = nn.ModuleList(norm_layers)
        self.lif_layers = nn.ModuleList(lif_layers)
        self.pool_flags = pool_flags  # python list of bool, not learnable state
        self.pool = (
            nn.AvgPool2d(arch_cfg.pool_size)
            if arch_cfg.pool_type == "avg"
            else nn.MaxPool2d(arch_cfg.pool_size)
        )

        self.flatten_dim = arch_cfg.flatten_dim()

        fc_layers: List[nn.Module] = []
        lif_fc_layers: List[snn.Leaky] = []
        width = self.flatten_dim
        for hidden in arch_cfg.fc_hidden:
            fc_layers.append(BayesianLinear(width, hidden, **gate_kwargs))
            lif_fc_layers.append(
                _make_leaky(snn_cfg, threshold_override=threshold_for(gated_idx))
            )
            width = hidden
            gated_idx += 1

        self.fc_layers = nn.ModuleList(fc_layers)
        self.lif_fc_layers = nn.ModuleList(lif_fc_layers)

        self.fc_out = nn.Linear(width, num_classes)  # classifier: never pruned
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def _dropout(
        self, spk: torch.Tensor, cache: List["torch.Tensor | None"], idx: int
    ) -> torch.Tensor:
        """
        Apply dropout with a mask held *constant across all timesteps*.

        Chowdhury et al. specify that the dropout mask is kept fixed for the
        whole simulation of a given sample, rather than resampled per
        timestep as `nn.Dropout` would do. Resampling every timestep would
        average the noise away over the 100-timestep window and stop it
        regularising at all, so the mask is drawn on the first timestep and
        cached for the rest of the forward pass.
        """
        if self.dropout_p <= 0.0 or not self.training:
            return spk
        if cache[idx] is None:
            keep_prob = 1.0 - self.dropout_p
            cache[idx] = torch.bernoulli(torch.full_like(spk, keep_prob)) / keep_prob
        return spk * cache[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate `num_steps` timesteps under this config's encoding.

        Returns the output record, shape [num_steps, batch_size, num_classes]
        -- output-layer spikes, or fc_out's pre-synaptic current when
        `output_readout="current"` (see ArchConfig.output_readout).
        """
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
                # conv -> norm -> gate when a BatchNorm is present, so the
                # normalisation cannot divide the gate's noise back out; the
                # plain conv -> gate path is used otherwise. `conv(spk)`
                # returns ungated output exactly when defer_gate is set.
                cur = norm(conv(spk))
                if conv.defer_gate:
                    cur = conv.apply_gate(cur)
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
            out_t, mem_out = read_output(self, cur_out, mem_out)
            spk_out_rec.append(out_t)

        return torch.stack(spk_out_rec, dim=0)


def assert_gate_after_norm(model: nn.Module) -> None:
    """
    Raise if any gated convolution whose output feeds a normalisation layer
    still applies its gate *before* that normalisation.

    This is the single most expensive class of bug this project has hit: a
    gate placed before a BatchNorm has its injected variance renormalised
    straight back out, so the task loss cannot feel log_alpha, the KL term
    meets no opposition, and every gate runs to the clamp ceiling no matter
    what beta_max is set to. It cost three collapsed DPAP runs and silently
    degraded ResNet18's stage4 before being found, and neither failure
    looked like a placement bug from the outside.

    Ordering cannot be recovered from the module tree (it lives in each
    architecture's `forward`), so each family is checked explicitly. Adding
    a new normalised architecture means adding it here -- an unrecognised
    model that mixes BatchNorm with gated convs is rejected rather than
    waved through, because "no check exists" and "the check passed" must
    not look the same.
    """
    if isinstance(model, SpikingResNet18):
        offenders = [
            name
            for name, module in model.named_modules()
            if isinstance(module, BayesianConv2d) and not module.defer_gate
        ]
    elif isinstance(model, VGGStyleSNN):
        offenders = [
            f"conv_layers.{i}"
            for i, (conv, norm) in enumerate(zip(model.conv_layers, model.norm_layers))
            if isinstance(norm, _NORM_TYPES) and not conv.defer_gate
        ]
    elif isinstance(model, (LeNetSNN, VGG9SNN)):
        # Neither places a normalisation layer between a gated layer and its
        # neuron, so both orderings are identical here.
        offenders = []
    else:
        has_norm = any(isinstance(m, _NORM_TYPES) for m in model.modules())
        has_gated_conv = any(isinstance(m, BayesianConv2d) for m in model.modules())
        if has_norm and has_gated_conv:
            raise NotImplementedError(
                f"{type(model).__name__} mixes normalisation layers with gated "
                "convolutions but has no gate-placement check. Add one to "
                "models.assert_gate_after_norm before training it -- see that "
                "function's docstring for what goes wrong silently otherwise."
            )
        offenders = []

    if offenders:
        raise ValueError(
            f"{type(model).__name__}: gated convolution(s) {offenders} feed a "
            "normalisation layer but have defer_gate=False, so the gate is "
            "applied before the normalisation and the task loss cannot see it. "
            "Set defer_gate=True and apply the gate via apply_gate() after the "
            "norm in forward()."
        )


def build_model(
    name: str,
    snn_cfg: SNNConfig,
    bayesian_cfg: BayesianConfig,
    num_classes: int = 10,
    arch_cfg: "ArchConfig | None" = None,
) -> nn.Module:
    """Factory: construct a model by name.

    The three original names ('lenet', 'vgg9', 'resnet18') build their
    hard-coded classes and ignore `arch_cfg`. Any other name builds a
    configurable VGGStyleSNN from `arch_cfg`, which is how the paper
    replications are expressed -- see docs/replication_targets.md.

    Every model is checked for correct gate/normalisation ordering before
    being returned (assert_gate_after_norm), so a misplacement cannot reach
    a GPU queue.
    """
    builders = {
        "lenet": LeNetSNN,
        "vgg9": VGG9SNN,
        "resnet18": SpikingResNet18,
        # A replication config that reuses one of the hard-coded architectures
        # must be listed here, or it falls through to the VGGStyleSNN branch
        # and is built from whatever ArchConfig its config happens to carry --
        # which for a ResNet config is the *default*, i.e. VGG9. That is a
        # silently wrong architecture that trains perfectly happily, the exact
        # failure ArchConfig.validate exists to prevent elsewhere.
        "spear_repl_resnet18": SpikingResNet18,
    }
    if name in builders:
        model = builders[name](snn_cfg, bayesian_cfg, num_classes=num_classes)
    elif arch_cfg is None:
        raise ValueError(
            f"Unknown model name '{name}' and no arch_cfg given. Built-in options: "
            f"{list(builders)}; any other name requires an ArchConfig."
        )
    else:
        model = VGGStyleSNN(arch_cfg, snn_cfg, bayesian_cfg, num_classes=num_classes)

    assert_gate_after_norm(model)
    return model
