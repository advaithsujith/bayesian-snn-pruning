"""
Posterior-uncertainty-based structured pruning.

Pruning criterion
------------------
For every structurally-prunable BayesianConv2d / BayesianLinear layer, a
channel/neuron is marked for removal if its learned posterior gate
satisfies log_alpha > threshold (default 3.0, following Molchanov,
Ashukha & Vetrov, 2017 -- corresponding to an effective binary dropout
rate above 95%, i.e. the network learned that this structure's output is
indistinguishable from noise). This is a posterior-uncertainty criterion:
it is derived purely from each gate's learned noise-to-signal ratio, not
from weight magnitude, activation statistics, or any hand-designed
importance heuristic.

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

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from bayesian_layers import BayesianConv2d, BayesianLinear, collect_bayesian_layers
from config import SNNConfig
from models import (
    LeNetSNN,
    VGG9SNN,
    SpikingResNet18,
    SpikingBasicBlock,
    _make_leaky,
    _VGG9_CONV_CHANNELS,
)


# ---------------------------------------------------------------------------
# Mask / index utilities shared across architectures
# ---------------------------------------------------------------------------


def compute_keep_indices(layer: "BayesianConv2d | BayesianLinear", threshold: float) -> torch.Tensor:
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
    min / median / max log_alpha and the fraction of units past the
    pruning threshold. Used for the "Computing posterior uncertainty..."
    logging step and for the uncertainty-histogram plots.
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, (BayesianConv2d, BayesianLinear)):
            log_alpha = module.log_alpha.detach()
            report[name] = {
                "min": float(log_alpha.min()),
                "median": float(log_alpha.median()),
                "max": float(log_alpha.max()),
                "frac_prunable": float((log_alpha > threshold).float().mean()),
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


def prune_lenet(model: LeNetSNN, threshold: float, snn_cfg: SNNConfig) -> PrunedLeNetSNN:
    """Physically prune a trained LeNetSNN into a smaller PrunedLeNetSNN."""
    keep1 = compute_keep_indices(model.conv1, threshold)
    keep2 = compute_keep_indices(model.conv2, threshold)
    keep_fc1 = compute_keep_indices(model.fc1, threshold)
    keep_fc2 = compute_keep_indices(model.fc2, threshold)

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


def prune_vgg9(model: VGG9SNN, threshold: float, snn_cfg: SNNConfig) -> PrunedVGG9SNN:
    """Physically prune a trained VGG9SNN into a smaller PrunedVGG9SNN.

    Channel keep-masks are propagated sequentially through the conv chain:
    layer i's kept input channels are exactly layer (i-1)'s kept output
    channels (the very first layer's input is always the full 3 RGB
    channels, which are never pruned).
    """
    new_convs: List[nn.Conv2d] = []
    prev_keep = torch.arange(3)

    for conv in model.conv_layers:
        keep_out = compute_keep_indices(conv, threshold)
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

    keep_fc1 = compute_keep_indices(model.fc1, threshold)
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


def _prune_basic_block(block: SpikingBasicBlock, threshold: float, snn_cfg: SNNConfig) -> PrunedBasicBlock:
    """Prune one BasicBlock's internal (conv1) channels only."""
    keep_mid = compute_keep_indices(block.conv1, threshold)

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


def prune_resnet18(model: SpikingResNet18, threshold: float, snn_cfg: SNNConfig) -> PrunedSpikingResNet18:
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
        new_stages.append([_prune_basic_block(block, threshold, snn_cfg) for block in stage])

    new_fc_out = nn.Linear(model.fc_out.in_features, model.fc_out.out_features)
    with torch.no_grad():
        new_fc_out.weight.copy_(model.fc_out.weight)
        new_fc_out.bias.copy_(model.fc_out.bias)

    return PrunedSpikingResNet18(new_stem, new_stem_bn, new_stages, new_fc_out, snn_cfg)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def prune_model(model: nn.Module, model_name: str, threshold: float, snn_cfg: SNNConfig) -> nn.Module:
    """Dispatch to the correct architecture-specific physical pruning routine."""
    dispatch = {
        "lenet": prune_lenet,
        "vgg9": prune_vgg9,
        "resnet18": prune_resnet18,
    }
    if model_name not in dispatch:
        raise ValueError(f"Unknown model name '{model_name}'. Options: {list(dispatch)}")
    model.eval()
    with torch.no_grad():
        return dispatch[model_name](model, threshold, snn_cfg)
