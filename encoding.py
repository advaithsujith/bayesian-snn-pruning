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

    Firing *rate* is set by |x| and the spike carries x's sign, so a returned
    element is +1 with probability min(|x|, 1) when x > 0, -1 with the same
    probability when x < 0, and 0 otherwise. Calling this once per simulated
    timestep produces a signed spike train whose long-run mean approximates
    the input.

    Why signed rather than clamped to [0, 1]: CIFAR-10 inputs are mean/std
    normalised before reaching the network, and Chowdhury et al. normalise
    with mean = std = 0.5, mapping [0, 1] pixels onto [-1, 1] (see
    docs/replication_targets.md). Rates are probabilities, so a naive
    `clamp(0, 1)` would encode every negative pixel -- about half the input
    distribution -- as complete silence, discarding half the image before
    the network ever sees it. Preserving the sign keeps the full dynamic
    range while still emitting genuine spike events, and is the standard
    treatment of signed inputs in rate-coded pipelines.
    """
    rates = x.abs().clamp(0.0, 1.0)
    if generator is None:
        spikes = torch.bernoulli(rates)
    else:
        spikes = torch.bernoulli(rates, generator=generator)
    return spikes * torch.sign(x)


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
