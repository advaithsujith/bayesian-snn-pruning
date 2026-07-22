# Bayesian Structured Pruning for Spiking Neural Networks

Structured, posterior-uncertainty-driven pruning of Spiking Neural
Networks (SNNs), implemented and benchmarked across three architectures
of increasing scale: LeNet-SNN (~62K parameters), VGG9-SNN (~9M
parameters), and Spiking ResNet-18 (~11.7M parameters), trained on
CIFAR-10.

## What this is

Each output neuron / convolutional channel is given a learned stochastic
gate. The gate's posterior is trained via the reparameterization trick
under a KL-divergence penalty against a log-uniform prior (structured
variational dropout). Once trained, any gate whose posterior noise
dominates its signal (`log_alpha > threshold`) is judged redundant on the
grounds of learned **posterior uncertainty** — not weight magnitude, not
activation statistics, not a hand-designed heuristic — and is physically
removed, producing a smaller, ordinary (non-Bayesian) network.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ and, for realistic training times, a CUDA GPU (the
code runs on CPU too, but VGG9/ResNet18 training will be very slow).

## Running locally

```bash
python run_all.py
```

This runs the entire pipeline for all three architectures sequentially,
with no arguments and no source modification required. CIFAR-10 is
downloaded automatically into `./data` on first run.

To experiment with a single architecture's hyperparameters, edit the
corresponding `get_*_config()` function in `config.py` — every
hyperparameter used anywhere in the codebase is defined there.

## Running on an HPC cluster (SLURM)

```bash
sbatch slurm.sh
```

Edit `slurm.sh`'s `#SBATCH` header (partition name, GPU type, walltime,
memory) to match your cluster's queue configuration first — these vary
between institutions and cannot be guessed correctly in advance. The
script creates a virtual environment, installs `requirements.txt`, prints
GPU/hardware diagnostics via `nvidia-smi`, then runs `python run_all.py`,
with stdout/stderr captured to `logs/slurm_<jobid>.out` / `.err`.

## Pipeline

For each architecture, in order:

1. **Build** the model with Bayesian gates present but inactive.
2. **Train** a conventional (deterministic, gate-noise-disabled) SNN
   baseline.
3. **Evaluate** the baseline on the held-out test set (`Accuracy Before`).
4. **Convert to Bayesian**: activate every structurally-prunable gate.
5. **Train the Bayesian gates** under the KL-regularised loss (with
   linear beta warmup) until posteriors converge.
6. **Compute posterior uncertainty**: read off each gate's `log_alpha`.
7. **Structured pruning**: physically remove every channel/neuron whose
   `log_alpha` exceeds the threshold (default 3.0), rebuilding smaller,
   ordinary dense/conv layers — not masks.
8. **Fine-tune** the smaller network.
9. **Final evaluation** (`Accuracy After`, FLOPs, latency, GPU memory).
10. **Save** every checkpoint, log, plot, and summary; move to the next
    architecture.

## Directory structure

```
bayesian_snn_pruning/
├── run_all.py            # top-level orchestrator; entry point
├── train.py               # training / fine-tuning loops
├── evaluate.py             # full test-set + resource evaluation
├── pruning.py              # uncertainty-based masks + physical rebuilding
├── bayesian_layers.py      # BayesianLinear / BayesianConv2d, KL divergence
├── losses.py                # SNN spike-rate loss + KL combination
├── models.py                 # LeNet-SNN, VGG9-SNN, Spiking ResNet-18
├── datasets.py                # CIFAR-10 loaders + augmentation
├── metrics.py                  # params, FLOPs, memory, latency, CSV I/O
├── utils.py                     # seeding, logging, checkpoints
├── config.py                     # every hyperparameter, per architecture
├── requirements.txt
├── slurm.sh
├── checkpoints/    # raw model .pt files written during training
├── outputs/
│   ├── lenet/       # trained_model.pt, bayesian_model.pt, pruned_model.pt,
│   ├── vgg9/         # metrics.csv, training_log.csv, plots.png,
│   ├── resnet18/      # config.json, summary.txt, remaining_structures.csv
│   └── final_results.csv   # cross-architecture comparison table
├── plots/           # cross-architecture comparison figures
└── logs/             # per-architecture .log files + SLURM stdout/stderr
```

## Bayesian pruning methodology

The gating mechanism is the Gaussian multiplicative-noise parameterisation
of variational dropout, extended to a structured (one-gate-per-neuron /
per-channel) setting:

```
z_j = 1 + sqrt(alpha_j) * eps_j,   eps_j ~ N(0, 1)
output_j = h_j * z_j
```

Each gate's only free parameter is `log_alpha_j`, its noise-to-signal
ratio. Training minimises `task_loss + beta * KL(q(z) || p(z))`, where
`p(z)` is an (improper) log-uniform prior. Because this prior has no
preferred scale, the KL term is minimised precisely when a gate's noise
swamps its mean — this is what drives redundant structures toward
`log_alpha -> infinity` during training, and is the criterion pruning.py
thresholds on (default 3.0, corresponding to an effective binary dropout
rate above 95%, following Molchanov et al., 2017).

**Residual pruning caveat.** In Spiking ResNet-18, only each
`BasicBlock`'s internal `conv1` output channels are physically prunable.
`conv2`'s output channels are tied to the block's residual addition (and
to the next block's input), and the stem's output feeds every stage-1
block, so both are excluded from physical channel removal to keep every
residual addition dimensionally valid without extra projection logic —
see the docstrings in `models.py` and `pruning.py` for the full rationale.
This is a standard, documented simplification in the structured-pruning
literature for residual architectures, not an oversight.

## References

- Molchanov, D., Ashukha, A., & Vetrov, D. (2017). *Variational Dropout
  Sparsifies Deep Neural Networks.* ICML. (Closed-form KL approximation;
  the `log_alpha > 3` pruning threshold.)
- Neklyudov, K., Molchanov, D., Ashukha, A., & Vetrov, D. (2017).
  *Structured Bayesian Pruning via Log-Normal Multiplicative Noise.*
  NeurIPS. (Structured, per-neuron/channel gate formulation.)
- Kingma, D. P., Salimans, T., & Welling, M. (2015). *Variational Dropout
  and the Local Reparameterization Trick.* NeurIPS.
- Louizos, C., Welling, M., & Kingma, D. P. (2018). *Learning Sparse
  Neural Networks through L0 Regularization.* ICLR.
- Fang, W. et al. (2021). *Deep Residual Learning in Spiking Neural
  Networks.* NeurIPS. (Spiking-ResNet design pattern, direct encoding.)
- Rathi, N. & Roy, K. et al. — LeNet-5 / VGG-style SNN configurations
  standard in the SNN-conversion and SNN-pruning literature.

## A note on scope

The combination of stochastic Bayesian gates with surrogate-gradient SNN
training (used throughout this codebase) does not have one single
universally agreed-upon reference implementation in the published
literature at the time of writing — this codebase implements a principled
synthesis of the structured-variational-dropout literature (ANN-domain)
and standard surrogate-gradient SNN training practice. Treat the exact
convergence behaviour, prune-threshold sensitivity, and per-architecture
hyperparameters in `config.py` as a starting point to validate and tune
empirically on your own HPC runs, not as pre-verified, literature-matched
numbers.
