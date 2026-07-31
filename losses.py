"""
Loss functions: the SNN task loss (spike-rate cross-entropy) and its
combination with the Bayesian KL term, plus the KL annealing schedule.
"""

import torch
import torch.nn.functional as F

from bayesian_layers import total_expected_cost, total_kl


def spike_rate_cross_entropy(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Standard rate-coded SNN classification loss.

    `spk_rec` has shape [num_steps, batch_size, num_classes] -- the binary
    spike output of the final layer at every simulated timestep. Spikes
    are summed over time to obtain a per-class spike count per example,
    which is then used as logits for ordinary cross-entropy. This is the
    standard approach used throughout the snnTorch tutorials and the wider
    surrogate-gradient SNN training literature for rate-coded outputs.
    """
    spike_counts = spk_rec.sum(dim=0)  # [batch_size, num_classes]
    return F.cross_entropy(spike_counts, targets)


def linear_warmup_schedule(epoch: int, warmup_epochs: int, max_value: float) -> float:
    """
    Linear warmup of a loss weight from 0 to `max_value` over the first
    `warmup_epochs` epochs, then held constant.

    Used for both the KL weight `beta` and the expected-FLOPs-cost weight
    `gamma`. Starting a pressure term at 0 and ramping it up (rather than
    applying the full penalty from epoch 0) prevents every gate from
    collapsing immediately, before the network has had a chance to learn
    which structures are actually useful -- standard practice for the KL
    term in the variational-dropout pruning literature (Molchanov et al.,
    2017; Neklyudov et al., 2017), and the same collapse risk applies to
    any other term that pushes gates toward pruning (see HANDOFF.md bug #6
    on LeNet's beta_max collapse).
    """
    if warmup_epochs <= 0:
        return max_value
    return max_value * min(1.0, epoch / warmup_epochs)


def bayesian_snn_loss(
    spk_rec: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module,
    beta: float,
    gamma: float,
    prune_threshold: float,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Full training loss for a Bayesian SNN: task loss + beta * KL +
    gamma * expected FLOPs cost.

    Returns (total_loss, task_loss, kl_term, cost_term) so callers can log
    each component separately.
    """
    task_loss = spike_rate_cross_entropy(spk_rec, targets)
    kl_term = total_kl(model)
    cost_term = total_expected_cost(model, prune_threshold)
    total_loss = task_loss + beta * kl_term + gamma * cost_term
    return total_loss, task_loss, kl_term, cost_term
