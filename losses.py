"""
Loss functions: the SNN task loss (spike-rate cross-entropy) and its
combination with the Bayesian KL term, plus the KL annealing schedule.
"""

import torch
import torch.nn.functional as F

from bayesian_layers import total_expected_cost, total_expected_synops, total_kl


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


def unilateral_mse(spk_rec: torch.Tensor, targets: torch.Tensor, thresh: float = 1.0) -> torch.Tensor:
    """
    DPAP's task loss: MSE between mean firing rate and a scaled one-hot
    target. Needed to replicate Han et al. (TPAMI 2024), whose released
    code trains with `UnilateralMse(1.)` rather than cross-entropy -- see
    docs/replication_targets.md.

    `spk_rec` is [num_steps, batch, num_classes]; the mean over time gives
    each output neuron's firing rate in [0, 1], which is regressed onto a
    target of `thresh` for the correct class and 0 elsewhere. So the
    objective is "make the right neuron fire at rate `thresh` and the rest
    stay silent", rather than "make the right neuron out-fire the others"
    as cross-entropy does.

    Faithfulness note -- the "unilateral" part is inert in the original.
    BrainCog's implementation reads:

        def forward(self, x, target):
            torch.clip(x, max=self.thresh)      # result discarded
            ...
            return self.loss(x, one_hot_scaled)

    `torch.clip` returns a new tensor and the return value is never
    assigned, so the intended one-sided clamp (don't penalise a neuron for
    over-firing) never actually takes effect. This function reproduces the
    *effective* behaviour, because that is what produced the published
    94.54% baseline we are comparing against -- replicating the intent
    instead would be a different loss than the one they trained with. The
    clamp is therefore deliberately omitted rather than overlooked.
    """
    rates = spk_rec.mean(dim=0)  # [batch, num_classes], each in [0, 1]
    target_rates = torch.zeros_like(rates).scatter_(1, targets.view(-1, 1), thresh)
    return F.mse_loss(rates, target_rates)


def tet_loss(
    out_rec: torch.Tensor,
    targets: torch.Tensor,
    lam: float = 0.05,
    phi: float = 1.0,
) -> torch.Tensor:
    """
    Temporal Efficient Training (Deng et al., ICLR 2022, arXiv 2202.11946).

    SPEAR's task loss, needed to replicate Xie et al. (arXiv 2507.02945) --
    see docs/replication_targets.md. Where the standard rate-coded loss
    (spike_rate_cross_entropy above) averages the output over time and takes
    cross-entropy once, TET takes cross-entropy at *every* timestep and
    averages the losses:

        L_TET  = (1/T) * sum_t CE(O(t), y)                        [their Eq. 9]
        L_MSE  = (1/T) * sum_t MSE(O(t), phi)                     [their Eq. 12]
        L_total = (1 - lam) * L_TET + lam * L_MSE                 [their Eq. 13]

    The point of the per-timestep form is optimisation, not accuracy: under a
    surrogate gradient the summed-output loss lets momentum accumulate around
    sharp minima, and constraining every moment's output instead drives the
    network toward a flatter one (their Sec. 4.2). The MSE term regularises
    each timestep's output toward a constant `phi` so that a single outlier
    timestep cannot dominate the integrated output.

    `out_rec` is [num_steps, batch, num_classes]. Note that TET defines O(t)
    as the *pre-synaptic input* of the output layer -- an analog value -- not
    the output layer's spikes, which is why the SPEAR replication config sets
    `output_readout="current"`. Passing binary spikes here would put
    per-timestep cross-entropy on a {0,1} logit vector, which carries almost
    no gradient signal at SPEAR's T=4.

    Defaults are the values the TET paper states for CIFAR: lam = 0.05
    ("The lambda is set to 0.05", their Sec. 5.2) and phi = V_th, the firing
    threshold ("we set phi = V_th in our experiments", their Sec. 4.2), which
    is 1.0 under SPEAR's LIF settings. SPEAR itself states only "TET is used
    as loss function" and gives neither value, so both are recorded as
    assumptions in docs/replication_targets.md rather than as replication.
    """
    num_steps = out_rec.shape[0]
    # Flattening to [T*B, C] and repeating the targets averages over both
    # time and batch in one call, which is exactly (1/T) sum_t CE(O(t), y).
    flat_ce = F.cross_entropy(out_rec.flatten(0, 1), targets.repeat(num_steps))
    reg = F.mse_loss(out_rec, torch.full_like(out_rec, phi))
    return (1.0 - lam) * flat_ce + lam * reg


TASK_LOSSES = {
    "spike_rate_ce": spike_rate_cross_entropy,
    "unilateral_mse": unilateral_mse,
    "tet": tet_loss,
}


def get_task_loss(name: str):
    """Look up a task-loss function by config name."""
    if name not in TASK_LOSSES:
        raise ValueError(f"Unknown task loss '{name}'. Options: {list(TASK_LOSSES)}")
    return TASK_LOSSES[name]


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
    loss_type: str = "spike_rate_ce",
    lam: float = 0.0,
    synops_budget: float = 0.0,
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
    """
    Full training loss for a Bayesian SNN: task loss + beta * KL +
    gamma * expected FLOPs cost + lambda * SynOps budget violation.

    `loss_type` selects the task-loss term; it defaults to the spike-rate
    cross-entropy every existing experiment uses, and only the DPAP
    replication overrides it.

    The SynOps term is a Lagrangian, not a weighted penalty:
    `lam * (E[SynOps] / synops_budget - 1)`, active only when
    `synops_budget > 0`. Dividing by the budget makes the term (and
    therefore `lam` and its dual-ascent step size) dimensionless, so the
    same `synops_lambda_lr` transfers across architectures whose absolute
    SynOps differ by orders of magnitude. `lam` itself is owned and
    updated by train.run_training via dual ascent; this function only
    consumes the current value. The plain linear form (rather than a
    relu hinge) is standard dual ascent: while `lam > 0` an under-budget
    model earns a small bonus, and the multiplier update then decays
    `lam` back toward zero.

    Returns (total_loss, task_loss, kl_term, cost_term, synops_term_raw)
    where synops_term_raw is E[SynOps] itself (not multiplied by lam), so
    callers can log the interpretable quantity.
    """
    task_loss = get_task_loss(loss_type)(spk_rec, targets)
    kl_term = total_kl(model)
    cost_term = total_expected_cost(model, prune_threshold)
    total_loss = task_loss + beta * kl_term + gamma * cost_term
    if synops_budget > 0.0:
        expected_synops = total_expected_synops(model)
        total_loss = total_loss + lam * (expected_synops / synops_budget - 1.0)
    else:
        expected_synops = torch.zeros_like(task_loss)
    return total_loss, task_loss, kl_term, cost_term, expected_synops
