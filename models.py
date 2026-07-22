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
from config import SNNConfig, BayesianConfig


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


def _make_leaky(snn_cfg: SNNConfig, output: bool = False) -> snn.Leaky:
    """Construct an snnTorch Leaky neuron layer from the shared SNN config."""
    return snn.Leaky(
        beta=snn_cfg.beta,
        threshold=snn_cfg.threshold,
        spike_grad=_get_surrogate_grad(snn_cfg.spike_grad),
        init_hidden=False,
        output=output,
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
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

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
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)


class SpikingBasicBlock(nn.Module):
    """
    A spiking version of the ResNet BasicBlock.

    `conv1` is the internal, freely structurally-prunable layer. `conv2`'s
    output channel count is tied to the block's residual addition and to
    the input of the next block, so it is marked non-prunable
    (structurally_prunable=False) -- see the module-level docstring for
    why. Both still carry Bayesian gates and contribute to the KL loss.
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = _make_leaky(snn_cfg)

        self.conv2 = BayesianConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, **gate_kwargs
        )
        self.conv2.structurally_prunable = False
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

        out = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(out, mem1)

        out = self.bn2(self.conv2(spk1))
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


def build_model(name: str, snn_cfg: SNNConfig, bayesian_cfg: BayesianConfig, num_classes: int = 10) -> nn.Module:
    """Factory: construct a model by name ('lenet', 'vgg9', 'resnet18')."""
    builders = {
        "lenet": LeNetSNN,
        "vgg9": VGG9SNN,
        "resnet18": SpikingResNet18,
    }
    if name not in builders:
        raise ValueError(f"Unknown model name '{name}'. Options: {list(builders)}")
    return builders[name](snn_cfg, bayesian_cfg, num_classes=num_classes)
