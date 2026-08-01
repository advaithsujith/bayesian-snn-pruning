"""
Input encoding schemes for spiking networks.

Two schemes are supported:

**Direct encoding** (the default everywhere in this project, and what
LeNetSNN / VGG9SNN / SpikingResNet18 use): the same static, real-valued
image is presented to the network at every simulated timestep, and all
temporal structure comes from the LIF membrane dynamics rather than from
the input. This is standard in the modern surrogate-gradient literature
(e.g. Fang et al., 2021) and needs no code -- the tensor is passed through
unchanged.

**Poisson rate encoding**: at each timestep, every input pixel independently
emits a spike with probability proportional to its intensity, so the
network sees a stochastic binary spike train whose *rate* carries the
image. This is the classical encoding used by ANN-to-SNN conversion work
and is required to replicate Chowdhury, Garg & Roy (IJCNN 2021) -- see
docs/replication_targets.md.
"""

import torch


def poisson_encode(x: torch.Tensor, generator: "torch.Generator | None" = None) -> torch.Tensor:
    """
    Draw one timestep of Poisson (Bernoulli-per-timestep) spikes from `x`.

    Each element of the returned tensor is 1 with probability equal to the
    corresponding element of `x` clamped to [0, 1], and 0 otherwise. Calling
    this once per simulated timestep produces a spike train whose long-run
    firing rate approximates the input intensity.

    Note on normalisation: CIFAR-10 inputs are mean/std normalised before
    reaching the network, which pushes values outside [0, 1] (and negative).
    Rates are probabilities, so the clamp below is what makes this
    well-defined; negative-valued pixels therefore encode as silence. This
    matches the standard practice in rate-coded SNN pipelines, where the
    normalisation for a Poisson-encoded network is chosen to keep inputs
    predominantly non-negative (Chowdhury et al. use mean=std=0.5, mapping
    [0, 1] pixels to [-1, 1]).
    """
    rates = x.clamp(0.0, 1.0)
    if generator is None:
        return torch.bernoulli(rates)
    return torch.bernoulli(rates, generator=generator)


def encode_timestep(x: torch.Tensor, scheme: str) -> torch.Tensor:
    """
    Produce the network input for a single timestep under `scheme`.

    "direct" returns `x` unchanged (the same static image every timestep);
    "poisson" draws a fresh Bernoulli sample. Dispatching through one
    function keeps every model's per-timestep loop identical regardless of
    which encoding its architecture config selected.
    """
    if scheme == "direct":
        return x
    if scheme == "poisson":
        return poisson_encode(x)
    raise ValueError(f"Unknown encoding scheme '{scheme}'. Options: ['direct', 'poisson']")
