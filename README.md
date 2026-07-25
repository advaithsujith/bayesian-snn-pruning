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
├── run_all.py            # top-level orchestrator (Bayesian pipeline); entry point
├── run_bio_pruning.py     # top-level orchestrator (bio-inspired baselines)
├── train.py                # training / fine-tuning loops
├── evaluate.py               # full test-set + resource evaluation
├── pruning.py                 # uncertainty-based masks + physical rebuilding
├── activity_pruning.py         # bio-inspired criteria (naive/SCA/DPAP) + physical rebuilding
├── bayesian_layers.py            # BayesianLinear / BayesianConv2d, KL divergence
├── losses.py                       # SNN spike-rate loss + KL combination
├── models.py                        # LeNet-SNN, VGG9-SNN, Spiking ResNet-18
├── datasets.py                       # CIFAR-10 loaders + augmentation
├── metrics.py                         # params, FLOPs, memory, latency, CSV I/O
├── utils.py                            # seeding, logging, checkpoints
├── config.py                            # every hyperparameter, per architecture
├── requirements.txt
├── slurm.sh
├── checkpoints/    # raw model .pt files written during training
├── outputs/
│   ├── lenet/       # trained_model.pt, bayesian_model.pt, pruned_model.pt,
│   │                # bio/<criterion>/keep_<fraction>/, ...
│   ├── vgg9/         # metrics.csv, training_log.csv, plots.png,
│   ├── resnet18/      # config.json, summary.txt, remaining_structures.csv
│   ├── final_results.csv   # cross-architecture Bayesian comparison table
│   └── bio_results.csv       # cross-architecture bio-inspired comparison table
├── plots/           # cross-architecture comparison figures
└── logs/             # per-architecture .log files + SLURM stdout/stderr
```

## Bio-inspired pruning baselines

```bash
python run_bio_pruning.py
```

Run after (or before — it will train and cache its own baseline if needed)
`run_all.py`. For each architecture, forks from the exact same
deterministic pretrained checkpoint the Bayesian pipeline uses
(`outputs/<model>/trained_model.pt`), then structurally prunes it under
three alternative, non-Bayesian criteria, sweeping every sparsity level in
`cfg.bio.keep_fractions` to produce accuracy-vs-sparsity curves comparable
to the Bayesian side at matched sparsity:

- **Naive static firing-rate**: one forward pass, rank by mean spike
  rate, keep the top `keep_fraction`. No training-time dynamics — the
  cheap, static baseline.
- **SCA** (Li et al., 2024): cyclic dynamic pruning driven by mean
  |membrane potential| per channel, recomputed every cycle as the target
  sparsity ramps up.
- **DPAP-structured** (developmental-plasticity-inspired): per-epoch EMA
  "survival score" of spike-rate activity with a constant decay each
  epoch ("use it or lose it").

All three reuse `pruning.py`'s physical-rebuild container classes and
`train.py`'s training loop unchanged; `pruning.py`, `bayesian_layers.py`,
`models.py`, and `train.py` are never modified by this module. See
`activity_pruning.py`'s module docstring for full literature citations and
the deliberate simplifications made relative to each source paper
(monotonic prune-only schedules rather than true prune+regrow; DPAP's
structured/neuron branch only; explicit target-sparsity selection instead
of DPAP's emergent score-crossing rule) — each is chosen so results stay
directly comparable to the Bayesian side, not to overclaim fidelity to the
original papers.

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

## Hyperparameter search

```bash
python hpo_search.py                       # all three methods
python hpo_search.py --methods bayesian     # split across sbatch submissions if needed
```

Bayesian pruning's hyperparameters were tuned through real, iterative CSF3 runs;
`activity_pruning.py`'s bio-inspired criteria originally shipped with untuned
defaults, and LeNet's Bayesian `beta_max` was fixed from a collapse (`fc2` pruned to
1 surviving neuron) with a one-off manual guess. Both are a threat to a fair
comparison — if only one method gets real tuning effort, "method A beats method B"
may just mean "A got more tuning." `hpo_search.py` replaces both with an actual
search: three independent Optuna samplers (random search, TPE, CMA-ES), 15 trials
each, against all three methods' key hyperparameters (Bayesian's `beta_max`; SCA's
epoch budget/cycle count/learning rate; DPAP's epoch budget/EMA decay/survival
decay/learning rate), run entirely on LeNet (cheapest architecture) at a shortened
(~1/3) epoch budget per trial, followed by one full-length confirmation run per
method on the held-out test set. Independent samplers converging on similar values
is the actual fairness claim — see `outputs/hpo/convergence_summary.txt` after a run.

Only LeNet's Bayesian `beta_max` gets replaced by the search result — VGG9's and
ResNet18's Bayesian `beta_max` already produced validated, working results and are
left untouched. SCA's and DPAP's searched hyperparameters transfer as-is into
`BioPruningConfig` for all three architectures (matching the existing convention
that `bayesian_train_epochs` is already shared identically across architectures).

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
- Li, Y., Xu, Q., Shen, J., Xu, H., Chen, L., & Pan, G. (2024). *Towards
  Efficient Deep Spiking Neural Networks Construction with Spiking
  Activity based Pruning.* ICML. (SCA baseline in `activity_pruning.py`.)
- Han, B., Zhao, F., Zeng, Y., & Shen, G. (2022, IEEE TPAMI 2024).
  *Developmental Plasticity-inspired Adaptive Pruning for Deep Spiking and
  Artificial Neural Networks.* arXiv 2211.12714. (DPAP-structured
  baseline; only the structured/neuron branch is reproduced here.)

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
