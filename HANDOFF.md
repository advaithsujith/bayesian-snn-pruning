# Handoff: Bayesian SNN Pruning — MSc Dissertation Project

This document summarizes a long prior conversation building and debugging a
structured Bayesian pruning codebase for Spiking Neural Networks, then
building a bio-inspired comparison side, a fairness-driven hyperparameter
search, and scoping a new efficiency-aware loss extension. Read this fully
before doing anything — it captures hard-won debugging context that took
many hours of real HPC time to discover.

---

# READ FIRST: state as of 2026-08-03, and the open decision

## UPDATE, 2026-08-03 evening: the blocker is BROKEN. First positive core result.

The `beta_max=0.01` curve run on `dpap_repl` finished at 19:03. Gate ranking
USABLE (std 0.308, saturation 0.000, every row). Test accuracies, against
the random control and DPAP's published numbers:

| pruned % | Bayesian | random control | gap | DPAP published |
|---|---|---|---|---|
| 19.98 | 93.28 | 91.74 | +1.54 | -- |
| 33.35 | 93.38 | 91.13 | +2.25 | 94.27 |
| 50.80 | 93.34 | 90.62 | +2.72 | 93.83 |
| 70.08 | 92.46 | 88.76 | +3.70 | -- |
| 89.98 | 89.39 | 84.14 | +5.25 | -- |

Beats random everywhere with the gap *widening* in sparsity -- the
signature of an informative ranking, and the dissertation's core figure.
Under DPAP's own operating points by <1pp (drops from own baseline:
ours -0.97/-1.01 vs their -0.27/-0.71). Note the flat plateau from 20% to
50.8% (93.3 +- 0.05): the limit there is recovery from the gate phase, not
capacity, so a longer fine-tune (30 -> 60 epochs) on one point is a cheap
probe that might lift the whole plateau. n=1 seed -- the easiest examiner
attack; seeds at the two DPAP points are the next GPU priority after the
SynOps pair. Results in `outputs/dpap_repl/sparsity_curve_beta0.01/`
(push from CSF3 before they get lost).

## HEADLINE RESULT (2026-08-04): the SynOps budget in the loss wins on both axes

Paired experiment, same budgets, same cost-blind selection rule, the only
difference being whether the gate-training loss carried the SynOps budget
Lagrangian (`outputs/dpap_repl/sparsity_curve_synops` vs
`sparsity_curve_lagrangian0.5`):

| budget | gates | accuracy | SynOps/sample | params left | layers at min width |
|---|---|---|---|---|---|
| 0.5 | baseline | 87.97% | 234.1M | 559,272 | 0 |
| 0.5 | **budget-trained** | **90.27%** | **212.0M** | 986,763 | 0 |
| 0.3 | baseline | 67.92% | 101.0M | 114,897 | 1 |
| 0.3 | **budget-trained** | **80.95%** | **88.4M** | 207,931 | 0 |

**+2.3pp accuracy with 9% fewer SynOps at 0.5; +13.0pp with 12% fewer at
0.3.** A strict improvement on both axes, not a trade, and the margin
widens as the budget tightens. Gate health held (`log_alpha` std 0.279,
zero saturation). Note the baseline's 0.3 point drives a layer to minimum
width while neither budget-trained point does: training under the
constraint lets the network *make* units cheap rather than having a
post-hoc rule sever a layer to afford the budget.

This is the dissertation's novelty claim, evidenced. Caveat: n=1 seed on
both arms; the comparison is paired and therefore clean, but seeds are
still the top remaining GPU priority.

## Negative result, same evening: cost-aware *selection* fails, and why

The first SynOps budget run (`sparsity_curve_synops`, reusing the usable
gates) shows the density heuristic -- keep units by importance-per-SynOps
-- performing far worse than cost-blind selection at the same budget. At
budget 0.5: cost-blind kept 610/2304 units and reached **87.97%** test
accuracy; cost-aware kept **1836/2304** units and still began fine-tuning
at 19% train accuracy (vs 50%), recovering far more slowly.

**Mechanism, and it is the interesting part: per-unit cost and posterior
importance are positively correlated across layers here.** Ranking the
layers by cost and by median `log_alpha` gives the same top three
(conv1, conv3, conv2). conv1 is both the most expensive (~9.4M per unit,
128 input channels at 32x32) and the most important (median log_alpha
0.73, and cost-blind selection keeps 127/128 of it); `fc_layers.0` is
both the cheapest (~0.26M) and the least important (1.49). Dividing
importance by cost therefore does not rebalance the ranking, it roughly
**inverts** it -- buying cheap units in the layers that matter least
while starving the wide mid-network convs. `min_keep=1` is no protection.

This is a property of the heuristic, not a bug: summed importance is not
a proxy for network function, and a greedy knapsack exploits that
whenever cost correlates with importance. A fractional per-layer floor
was considered and **deliberately not run** -- given the correlation it
would likely lose too, and the GPU hours are better spent on the
loss-side formulation.

**It motivates the Lagrangian.** Post-hoc selection imposes a cost
trade-off on a ranking learned without any knowledge of cost; the
budget-in-the-loss version lets gradient descent negotiate accuracy
against compute during gate training, so the network can *make* units
cheap instead of having cheapness forced on it afterwards. That run
(`--tag lagrangian0.5 --synops-loss-budget 0.5 --synops-budgets 0.5 0.3
--synops-rank-by importance`) is queued and is now the primary novelty
vehicle. It is directly paired with this run's cost-blind arms: same
budgets, same selection rule, the only difference being whether the loss
knew about SynOps.

`--synops-rank-by importance` skips the density arm in future jobs, since
the result is established.

Also this session (2026-08-03, local commits `990b0e5`, `397344a`,
independent-reviewed): the SynOps-aware machinery. SynOps measurement
(`metrics.measure_synops`, `measure_synops_unit_costs`), budgeted
cost-aware-vs-cost-blind selection (`pruning.synops_budget_plan`), a
dual-ascent SynOps-budget Lagrangian in the loss (replaces the disproven
fixed-gamma approach; see `BayesianConfig.synops_budget_fraction`), and a
split SGD gate optimizer (`--gate-optimizer sgd`, untested on GPU, held in
reserve since Adam at beta 0.01 worked). Next submission, reusing the
usable gates (no new gate training):
`sbatch slurm_curve.sh --model dpap_repl --tag beta0.01 --reuse-gates --synops-budgets 0.5 0.3`
after `git pull` and the three test suites (`tests/test_synops.py` is new).
Deadline, finally answered: **early September 2026.** Experiments must
effectively end by ~mid-August to leave writing time; the dissertation
template is in `dissertation/` (untracked; fix `meng` -> `msc` in the
documentclass).

**Everything from "The research goal" down to "Session 3" is historical and
parts of it are stale**, including several quoted results and the entire
"QUESTION FOR THE FRESH-CONTEXT READER" section, which has already been
answered twice. This block supersedes the stale framing. Technical detail
for the current state is in **Session 3** at the end of the file, plus
`docs/review_response_2026-08-03.md` and
`docs/fresh_review_2026-08-03.md`.

Reading order if short on time: this block, then Session 3, then Session 2's
"THE BIG BUG" and "The pivot" sections. The rest is background.

## What the user is asking

The user is deciding whether to **reframe the project around
computation-aware (SynOps) Bayesian pruning** instead of the
Bayesian-vs-bio-inspired comparison, and wants a fresh, independent
judgement. They are 13 days in (first commit 2026-07-22), frustrated by
repeated failed runs, and one thing is genuinely blocking everything else.
**The submission deadline is unknown; ask, because it is the deciding
variable and the previous assistant asked three times without an answer.**

## Status of every result in this repo

| | status |
|---|---|
| DPAP baseline replication, **94.35%** vs their published 94.54% | **Solid.** Pretraining runs with gates inert, so no gate bug touches it. `outputs/dpap_repl/trained_model.pt`. This is the strongest asset in the project. |
| Infrastructure: `VGGStyleSNN`, shared physical rebuild, 132 CPU tests | **Solid.** |
| Random-pruning control curve (see Session 3) | **Solid**, obtained accidentally. A control the project needed. |
| Three bio-inspired criteria in `activity_pruning.py` | **Code is correct.** Every run is at `keep_fraction=0.1` (~98.5% pruned), so no run is at a sparsity comparable to the Bayesian side. |
| VGG9 Bayesian, the +19pp headline result | **Needs redoing.** Produced under a bug that flattered it, n=1, and the exact run was overwritten (`outputs/vgg9/summary.txt` now reads 98.83%/85.24%, not the 98.71%/86.08% quoted below). |
| LeNet and ResNet18 Bayesian | **Invalid.** ResNet18 doubly so: gate misplaced *and* half its KL wasted. |
| FLOPs-aware loss | **Never worked.** Four runs. See Session 2. |
| Optuna HPO search | **Abandoned by the user.** `hpo_search.py` is dead weight. |
| Bayesian pruning of `dpap_repl` | **Never once produced a usable ranking.** This is the blocker. |

**Do not trust any number in this file without checking the CSV.** Several
quoted figures no longer match `outputs/`.

## The blocker

Gate training on `dpap_repl` has failed three times by collapse and once by
producing a ranking that was entirely ties. Until it yields gates that
**differentiate from each other while the network stays intact**, neither
the comparison study nor the SynOps idea can produce a result, because both
read the same `log_alpha` values.

A run at `beta_max=0.01` (down from 0.4, set from measurement, see Session
3) was the next action. Its outcome decides a lot:

- **Stops early at the gate check**: the mechanism does not work in *DPAP's*
  setup specifically (T=8, threshold 0.5, learned decay). It gave 98.7%
  pruning on VGG9. Read this as "change platform to VGG9", not "the method
  is broken".
- **Ranking real, beats the random control**: this is the result. Continue.
- **Ranking real, does not beat random**: the criterion does not work. This
  is the only outcome that genuinely justifies changing topic.

## The open decision

The user proposed dropping the comparison study and starting a new project
on SynOps-aware Bayesian pruning, comparing only against published numbers.

The previous assistant recommended **reframe, do not restart**, on these
grounds. Challenge them rather than inheriting them:

1. It is not a new project. Gates, rebuild, replication, ranked pruning,
   metrics, tests are all shared. New work is a spike counter plus a budget
   constraint, most of it zero GPU. `ActivityAccumulator` already hooks
   every LIF neuron in all four architectures.
2. It does not escape the blocker. The cost term multiplies `p_keep_j`,
   a function of the same `log_alpha` that is currently failing.
3. Comparing only against published numbers is weaker than it sounds. The
   contribution is that the SynOps term *adds* something, which needs an
   ablation against `gamma=0` at matched sparsity. That is the user's own
   baseline, and it is load-bearing.
4. The bio-inspired code is already written and passing. Removing it from
   the narrative saves no implementation time and only removes evidence.
   Demote it to a baseline family rather than deleting it.
5. A reframe is reversible; a restart bets everything on the component that
   has already failed four times.

**Honest risk of the reframe:** it promotes the part of the project with the
worst track record. The failures are diagnosed (wrong cost metric, wrong
control lever, suppressed surrogate gradient) but none of the fixes has been
verified to work.

The working title discussed already covers both framings: *"Bayesian
Structured Pruning for Spiking Neural Networks: A Computationally-Aware
Approach."*

## Proposed plan at the time of writing

1. The `beta_max=0.01` run. Needed under either framing.
2. SynOps counting in `metrics.py`. Zero GPU.
3. Budget-constrained cost term replacing the `gamma` multiplier. Zero GPU.
4. One run: `gamma=0` versus SynOps-budget at matched sparsity. **This
   single comparison is the contribution** under the reframe.
5. Bio criteria at matching keep-fractions (~30 min on LeNet).
6. Three LeNet seeds; n=1 is the easiest thing for an examiner to attack.
7. Demote ResNet18 to a documented negative result unless time appears.
8. Cut SCA and Chowdhury replications entirely. One replication is enough.

---

## The research goal

MSc dissertation comparing **Bayesian uncertainty-based structured pruning**
against **biologically-inspired / activity-based pruning** in Spiking Neural
Networks (SNNs), on CIFAR-10, across three architectures of increasing
scale. **Both sides are now built and have produced results** (see below).

> **STALE.** The original "current focus" named here was (1) a multi-sampler
> hyperparameter search and (2) a FLOPs-aware loss term. The search was
> abandoned by the user and the FLOPs term was built and did not work. See
> the READ FIRST block at the top of this file for the actual current state.
> The research goal itself, below, is unchanged and still correct.

**Critical framing for the dissertation** (worked out at length with the
user): absolute compression percentages are not the interesting claim by
themselves — they depend heavily on how overparameterized a given
architecture is for CIFAR-10 specifically (e.g. VGG9 has huge slack, LeNet
has little). The actual scientific contribution is the **comparison at
matched sparsity levels between pruning criteria on the same
architecture/dataset** — this controls for the "how much redundancy exists"
confound. Report accuracy-vs-sparsity curves, not single numbers. Use the
three architectures as different "redundancy regimes" to test whether the
comparative ranking between methods holds consistently.

## Where everything lives

- **Local repo**: `c:\Users\Advaith\OneDrive\Desktop\Projects\bayesian_snn_pruning`
- **GitHub**: https://github.com/advaithsujith/bayesian-snn-pruning (public repo, owned by user `advaithsujith`)
- **HPC cluster**: University of Manchester CSF3, repo cloned at
  `/net/scratch/w36160as/bayesian-snn-pruning`. User connects via
  `login2`/`login3.csf3.man.alces.network`. Git push from CSF3 now works via
  **SSH** (an SSH key was set up this session specifically because HTTPS
  token auth kept failing with 403s — likely an account-mismatch issue that
  was never fully root-caused, just routed around). If push auth breaks
  again, go straight to SSH rather than re-debugging HTTPS tokens.

## Codebase structure (all fully implemented, no placeholders)

```
config.py          # ALL hyperparameters, per-architecture, via get_lenet_config() etc.
                    # + BioPruningConfig (bio-inspired hyperparams) + ExperimentConfig.bio
utils.py            # seeding, logging, checkpoint save/load
datasets.py          # CIFAR-10 loaders + augmentation, train/val/test split
bayesian_layers.py    # BayesianLinear/BayesianConv2d, KL divergence, gate mechanism
losses.py               # spike-rate cross-entropy + KL combination, beta annealing schedule
models.py                 # LeNetSNN, VGG9SNN, SpikingResNet18 (snnTorch LIF neurons)
metrics.py                  # param counts, FLOPs, latency, GPU memory, CSV I/O
pruning.py                    # posterior-uncertainty pruning + physical model rebuilding
train.py                        # training loops (run_training, train_one_epoch)
evaluate.py                       # full test-set evaluation (accuracy, FLOPs, latency, memory)
run_all.py                          # top-level orchestrator (Bayesian pipeline), entry point, plots
activity_pruning.py                  # bio-inspired criteria: naive firing-rate, SCA, DPAP
run_bio_pruning.py                    # top-level orchestrator (bio-inspired pipeline)
hpo_search.py                          # multi-sampler (Optuna) hyperparameter search, all 3 methods
requirements.txt, slurm.sh, slurm_bio.sh, slurm_hpo.sh, README.md
```

**Nothing in `pruning.py`, `bayesian_layers.py`, `models.py`, or `train.py`
was ever modified while building the bio-inspired/HPO additions** — those
four files are the original, verified Bayesian pipeline and all new work
deliberately only calls into them, never edits them. Keep it that way for
any future addition too (see the FLOPs/latency idea below, which *will*
need to touch `bayesian_layers.py`/`losses.py` — that'll be the first time
those files are touched since the original build).

## The Bayesian pruning mechanism (built, working)

Structured variational-dropout-style gating (Neklyudov et al. 2017 /
Molchanov et al. 2017 lineage): one stochastic gate per output neuron/channel,
parameterized by a single learned `log_alpha`. During training:
```
z_j = 1 + sqrt(exp(log_alpha_j)) * eps_j,  eps_j ~ N(0,1)
output_j = h_j * z_j
```
KL divergence to a log-uniform prior (closed-form Molchanov approximation,
constants k1=0.63576, k2=1.87320, k3=1.48695) provides the pruning pressure.
Threshold for pruning: `log_alpha > 3.0` (from Molchanov et al.). Gates are
clamped to `[-8, 8]`.

**The core mechanism, conceptually** (useful if re-explaining to the user —
they've asked for this from-scratch explanation twice and it took real
back-and-forth to land): the gate does NOT learn a 0-1 importance score. It
learns a *noise level* (`alpha_j`, via `log_alpha_j`). `E[z_j]=1` always,
regardless of `alpha_j` — only the *variance* is learned. The KL term to a
scale-free (log-uniform) prior is a strictly *decreasing* function of
`log_alpha` — i.e. it always wants noise to increase. Task loss pushes back
only where noise genuinely hurts predictions. The equilibrium each gate
settles at (not a direct label) is what determines pruning. `beta` controls
how strongly the "increase noise" side gets to pull — this is why it's the
single most consequential, most fragile hyperparameter (see LeNet collapse
below).

**Pipeline per architecture**: Build model (gates present but inert) →
**Train** deterministic baseline (`set_bayesian_mode(model, False)`) →
Evaluate baseline → **Convert to Bayesian** (`set_bayesian_mode(model,
True)`) → **Train Bayesian gates** under KL-annealed loss → Compute
posterior uncertainty → **Structured pruning** (physical rebuild, not
masking) → **Fine-tune** the smaller network → Final evaluation → Save.

**Residual pruning caveat (ResNet18 only)**: only each `BasicBlock`'s
internal `conv1` is physically prunable. `conv2`'s output channels are tied
to the residual addition and are NOT pruned. The stem conv is also
non-prunable. See `models.py`/`pruning.py` docstrings for full rationale.

## Critical bugs found and fixed (do not reintroduce these)

1. **Weight decay was corrupting gates** (`train.py`, commit `b969427`).
   Fixed via `_param_groups_excluding_gates` — `log_alpha` gets
   `weight_decay=0`.

2. **`restore_best_checkpoint` was silently discarding ALL pruning
   progress** (`train.py`, commit `bd99179` — the single most consequential
   bug in the original build). Fixed via the `restore_best_checkpoint` flag,
   `False` specifically for the `bayesian_train` phase.

3. **`beta_max` needed to be much larger than intuition suggests** — the
   closed-form KL gradient decays ~20x between `log_alpha=-3` and
   `log_alpha=3`. Working values as of the original build: LeNet `1.0`,
   VGG9 `0.4`, ResNet18 `0.2`, `bayesian_train_epochs=75`.

4. **CSF3-specific SLURM fixes** (commit `291e459`): `--partition=gpuA`/
   `gpuL`, `-G 1` not `--gres=gpu:1`. `gpuV` retired. 96h max wallclock.

5. **CIFAR-10 download hangs on GPU compute nodes** (no internet there) —
   pre-download on the login node before submitting jobs.

6. **LeNet's `beta_max=1.0` caused a real collapse, found and fixed this
   session.** `fc2` (120→84 neurons) hit `frac_prunable=1.000` — literally
   every gate crossed threshold, leaving only the "keep at least 1 unit"
   fallback. Physically pruned network became `fc1(87)→fc2(1)→fc_out(10)`,
   a hard information bottleneck fine-tuning couldn't recover from (crashed
   to ~27% accuracy from a 68.95% baseline, despite only 38.9% of params
   being pruned overall — nowhere near a proportionate cost). Root cause:
   LeNet has far less redundant capacity than VGG9/ResNet18 (62K vs.
   millions of params) and no residual bypass to cushion a collapsed layer,
   so `beta_max=1.0` (2.5-5x VGG9/ResNet18's) was too aggressive
   specifically for it. **Fix applied**: `beta_max: 1.0 → 0.4` (commit
   `9b7ebeb`). Rerun succeeded: **27.7% pruned, accuracy 68.95% → 70.09%**
   (confirmed via console output the user pasted; the underlying
   `outputs/lenet/summary.txt`/`remaining_structures.csv`/
   `training_log.csv`/`logs/lenet.log` were **not yet pushed to git as of
   this handoff** — push and re-verify per-layer breakdown before fully
   trusting this number, same diagnostic process as below).

7. **ResNet18's dynamic bio-inspired training (SCA/DPAP) had a BatchNorm
   bias-leakage bug**, found by a spawned review agent and fixed this
   session (`activity_pruning.py`). Masking a channel via `hard_mask=0`
   zeroes it *before* BatchNorm, but `BatchNorm2d` in training mode
   normalizes an all-zero-input channel to exactly `bn.bias[c]` — a
   nonzero, still-training constant that leaked through `lif1` into
   `conv2`'s input (never pruned, so it would train against this leakage).
   Fixed via `_register_resnet_bn_remask_hooks()`, which re-applies the
   live `hard_mask` after BatchNorm too. Verified experimentally (dead
   channels confirmed exactly zero downstream despite a large nonzero
   `bn.bias`). LeNet/VGG9 have no BatchNorm between a prunable layer and its
   LIF neuron, so this was ResNet18-specific.

8. **SCA's activity accumulator was picking up validation-set forward
   passes**, also found by the review agent — the hook stayed registered
   across both `train_one_epoch` and `evaluate_loader` calls each epoch,
   contaminating the membrane-potential statistics used to pick the next
   alive set. Fixed by registering/deregistering strictly around
   `train_one_epoch` only (matching how DPAP already did it correctly).

## Results obtained so far (all on CIFAR-10)

**Bayesian** (`outputs/final_results.csv`):

| Model | Params: orig → remaining | Pruned % | Accuracy: before → after |
|---|---|---|---|
| LeNet-SNN | 62,006 → ~44,830 (approx, see bug #6) | 27.7% | 68.95% → 70.09% |
| VGG9-SNN | 8,887,978 → 114,347 | 98.71% | 85.00% → **86.08%** |
| Spiking ResNet18 | 11,177,866 → 1,050,757 | 90.60% | 89.26% → 88.93% |

VGG9 is the clean, strong result (`fc1` alone collapsed 800→~21 neurons,
accuracy *improved*). ResNet18 showed a ~11pp train/val gap during
fine-tuning, likely from `stage4` blocks hitting `frac_prunable=1.000` —
report this as a real finding (contrast with VGG9), not something to hide.
LeNet's number above is post-fix (bug #6); **push the updated output files
to git and re-verify the per-layer breakdown before citing it**.

**Bio-inspired** (`outputs/bio_results.csv`, all at `keep_fraction=0.1`) —
**caveat: these used untuned hyperparameters for SCA/DPAP; see the fairness
section below before trusting any comparison drawn from this table**:

| Model | Method | Pruned % | Accuracy |
|---|---|---|---|
| LeNet | naive firing-rate | 98.49% | 22.35% |
| LeNet | SCA | 98.49% | 24.73% |
| LeNet | DPAP | 98.49% | 22.97% |
| VGG9 | naive firing-rate | 98.99% | 66.74% |
| VGG9 | SCA | 98.99% | 51.78% |
| VGG9 | DPAP | 98.99% | 65.92% |
| ResNet18 | naive firing-rate | 88.50% | 83.43% |
| ResNet18 | SCA | 88.50% | 82.57% |
| ResNet18 | DPAP | 88.50% | 85.10% |

At matched-or-tighter sparsity, Bayesian beats every bio-inspired criterion
on VGG9 (+19pp over the best bio result) and ResNet18 (+3.8pp) — the result
the dissertation's hypothesis predicts. LeNet's bio numbers are stale
relative to the beta_max-fixed pretrained checkpoint and should be rerun.

## The three bio-inspired criteria (`activity_pruning.py`)

All three fork from the exact same pretrained checkpoint the Bayesian
pipeline uses (`outputs/<model>/trained_model.pt`), reuse `pruning.py`'s
physical-rebuild classes and `train.py`'s training loop unchanged, so only
the criterion differs:

1. **Naive static firing-rate** — one forward pass, rank by mean spike
   rate, keep top `keep_fraction`. No training-time dynamics.
2. **SCA** (Li et al., ICML 2024, `arxiv.org/abs/2406.01072`) — cyclic
   dynamic pruning by mean |membrane potential| (their own biological
   framing: depolarization/hyperpolarization level, verified via direct
   paper fetch). Structured, prune-and-regrow in the original paper;
   implemented here as monotonic-per-cycle (see module docstring for why
   true regrow isn't supported — masking a unit also zeroes its gradient).
3. **DPAP-structured** (Han et al., arXiv 2211.12714) — per-epoch EMA
   "survival score" of spike-rate activity with constant decay ("use it or
   lose it"). Structured/neuron branch only; the paper's unstructured
   synapse branch was deliberately excluded (see README for why unstructured
   criteria don't fit this project's physical-rebuild comparison).

Deliberate simplifications relative to the source papers (documented in
`activity_pruning.py`'s module docstring, don't relitigate as bugs): no true
prune+regrow, DPAP's threshold-crossing rule replaced with explicit
target-fraction selection for matched-sparsity comparability.

Two other papers were investigated and explicitly scoped OUT: STDP-based
pruning and Grad R (Chen et al., IJCAI 2021) are both unstructured
(connection-level), which would require a second, structurally different
pruning pipeline (weight masks instead of physical rebuild) — a real
engineering cost, deliberately not taken on. "ST-PBT" (originally proposed
by the user as a candidate) could not be verified to exist in the
literature after searching — dropped, not implemented.

## The fairness problem, and the multi-sampler HPO search (`hpo_search.py`) — IN PROGRESS

**The problem, self-diagnosed by the user from the results table**: Bayesian
pruning's hyperparameters were tuned through real, iterative CSF3 runs.
SCA/DPAP's hyperparameters (epoch budgets, learning rates, decay constants)
were untuned first-guess defaults written from scratch this session. Smoking
gun: **SCA loses to the naive static baseline** on both VGG9 (51.8% vs.
66.7%) and ResNet18 (82.6% vs. 83.4%) — a sophisticated dynamic method
losing to a one-shot heuristic is a classic undertuning symptom, not a real
finding about the criterion.

**The fix, agreed with the user after discussing literature-transplant vs.
manual tuning vs. systematic search**: `hpo_search.py` runs three
independent Optuna samplers (random search, TPE, CMA-ES), 15 trials each,
against all three methods (Bayesian `beta_max`; SCA's epoch
budget/cycles/LR; DPAP's epoch budget/EMA decay/survival decay/LR) —
entirely on **LeNet** (cheapest architecture) at a shortened (~1/3) epoch
budget per trial, with one full-length confirmation run per method
afterward. Independent samplers converging on similar values is the actual
fairness claim. Full design rationale is in the plan file this was built
from and in `hpo_search.py`'s module docstring; see README's "Hyperparameter
search" section for the run instructions and scope (only LeNet's Bayesian
`beta_max` gets replaced — VGG9/ResNet18's Bayesian `beta_max` already
work and are left untouched; SCA/DPAP's searched values transfer to all
three architectures).

**Known bug in the current running search, found by the user pasting a log
snippet, NOT YET FIXED**: the Bayesian objective's constraint is a *flat*
penalty (`accuracy - 0.5` if `pruning_pct < 20%`), which gives the optimizer
zero gradient toward actually satisfying the constraint — it only has
incentive to maximize accuracy *within* the penalized region, so it drifts
toward the lowest `beta_max` (best accuracy, worst pruning) rather than
toward compliance. Confirmed from real CSF3 log output: every CMA-ES trial
for the `bayesian` method hit `pruning_pct=0.0%`, and the sampler's "best"
trial was just whichever preserved accuracy best among failed trials. Likely
compounded by the shortened 25-epoch trial budget being too short for
KL-driven pruning to manifest at low-to-moderate `beta_max` (LeNet needed
dozens of epochs at full pressure to cross the threshold even in the
original, non-shortened runs). **Fix needed before trusting Bayesian's HPO
results**: use a graded penalty (proportional to shortfall from the 20%
target, not a fixed step), and likely lengthen the shortened-epoch budget
specifically for Bayesian trials. SCA and DPAP's objectives don't have this
flaw (plain accuracy at a fixed `keep_fraction`, no constraint) — their
search results should be trustworthy as-is.

**Status as of this handoff**: search was mid-run on CSF3 — Bayesian's 45
trials finished (all 3 samplers), SCA was running/finishing, DPAP not yet
started. `--methods bayesian` can be rerun standalone once the penalty is
fixed, without redoing SCA/DPAP. Also needed: a `cmaes` package dependency
that isn't bundled with `optuna` — already added to `requirements.txt` and
verified via a CPU smoke test with tiny synthetic data (same pattern used
for `activity_pruning.py`'s smoke test) before this was ever run for real.

**Once complete**: review `outputs/hpo/convergence_summary.txt` for genuine
per-sampler agreement (not just picking the numerically-best trial), update
`config.py` accordingly, rerun VGG9/ResNet18 for SCA/DPAP with the
transferred hyperparameters (Bayesian VGG9/ResNet18 stay as-is, only LeNet's
`beta_max` gets replaced).

## What's next: FLOPs/latency-aware Bayesian pruning loss — NOT YET STARTED

User's idea: extend the Bayesian loss so pruning explicitly optimizes for
inference speed, not just accuracy vs. sparsity. Discussed at length,
nothing implemented yet. Key points to preserve:

- **You cannot add real measured latency (`measure_latency_ms()` in
  `metrics.py`) directly to the loss** — it's a stopwatch reading, not a
  differentiable function of `log_alpha`. Any hardware-aware loss term needs
  a differentiable *proxy* instead.
- **Two proxy options, discussed and compared**:
  1. **Expected-FLOPs regularizer**: precompute static per-unit FLOPs cost
     (same formula `metrics.py`'s `_FlopCounter` uses, `×num_steps` since
     SNN ops run once per simulated timestep), build a smooth
     `p_keep_j = 1 - sigmoid(log_alpha_j - threshold)` per gate, add
     `gamma * sum_j(cost_j * p_keep_j)` to the loss. Simple, zero setup
     cost, but **`_FlopCounter` only hooks `Conv2d`/`Linear`** — blind to
     LIF neuron updates, per-timestep Python loop overhead, and memory ops,
     which is a bigger gap for an SNN (25 timesteps of neuron bookkeeping
     per forward pass) than it would be for a plain ANN.
  2. **Calibrated latency lookup table** (recommended by both the
     discussion and the user's own conclusion): measure real latency (via
     the existing `measure_latency_ms()`) across ~15-20 different pruned
     configurations offline, fit per-layer ms-coefficients via
     least-squares regression, use those *fixed, real-hardware-calibrated*
     constants as `cost_j` in the same `sum_j(cost_j * p_keep_j)` formula.
     Actually anchored to true speed (captures everything FLOPs misses),
     at the cost of a calibration step and being tied to specific hardware
     (recalibrate if switching between CSF3's `gpuA`/`gpuL` — different
     GPUs).
- **The "expected" in "expected cost" is literally expected value** (probability-weighted average, same concept as `E[z_j]=1` in the gate mechanism) — `p_keep_j` isn't a certainty, it's a probability, and `expected_cost` sums each unit's cost weighted by its survival probability. This mirrors the reparameterization trick conceptually: replace a hard, non-differentiable decision with a smooth probability of that decision, which *is* differentiable.
- **Novelty check performed via real web search** (not assumed): the
  general pattern (stochastic gates + expected-cost regularization) is
  established for ANNs — Louizos, Welling & Kingma's L0 regularization
  (already in this project's README references) does essentially this for
  raw parameter count; Lemaire et al., CVPR 2019, *"Structured Pruning of
  Neural Networks with Budget-Aware Regularization"* looks like a closer
  match (budget/FLOPs-specific), though full text couldn't be fetched to
  confirm the exact mechanism. For SNNs, differentiable gradient-based
  energy/spike-cost regularization also has precedent (BPSR,
  `pmc.ncbi.nlm.nih.gov/articles/PMC9047717`, but via plain L1/L2 magnitude
  regularization, not a Bayesian gate). **Could not find** the specific
  combination this project would be doing — Molchanov/Neklyudov-style
  log-normal multiplicative-noise Bayesian gating + a differentiable
  FLOPs/latency cost term, for SNNs. SPEAR (SynOps-based SNN pruning) uses
  reinforcement learning instead of gradient descent, which is weak
  circumstantial evidence the field may not have had an easy differentiable
  path to this for SNNs specifically. **Flag to the user**: this is
  promising as a genuine novelty angle but not proven absent from the
  literature by a few searches — recommend a more thorough lit-review pass
  before claiming novelty in the dissertation itself.
- **Sketched integration plan (not built)**: `bayesian_layers.py` gets a
  `unit_cost` buffer per gated layer (same pattern as the existing
  `hard_mask` buffer); `losses.py` gets an `expected_cost(model)` function
  walking `collect_bayesian_layers()`; `bayesian_snn_loss()` becomes
  `task_loss + beta*KL + gamma*expected_cost`; `config.py` gets a new
  `gamma` hyperparameter on `BayesianConfig`. This is the first planned
  change to touch `bayesian_layers.py`/`losses.py` since the original
  build — treat with the same care/plan-mode process used for the HPO
  search before implementing.
- **Immediate next decision needed from the user**: which cost source to
  implement first — the FLOPs+LIF-extended proxy, or the calibrated latency
  table (their own stated preference by the end of the discussion).

## Working style notes for this user

- Prefers direct, technically precise answers; pushes back on premature
  "this is amazing!" framing until results are actually verified — match
  that skepticism, don't over-celebrate results without checking for
  overfitting/bugs first.
- Actively runs jobs on CSF3 and pastes raw SLURM/log output — read it
  carefully for anomalies. Several real bugs this session (LeNet's
  collapse, the HPO Bayesian objective flaw) were caught specifically by
  the user reading pasted log lines closely and asking "why," not by either
  party assuming results were correct.
- Wants claims grounded in real reference implementations/literature via
  web search rather than assumed from training knowledge, especially for
  hyperparameter choices and any novelty claim — this was reinforced hard
  this session (asked directly "have people tried this before" for the
  FLOPs idea, expected an actual search, not a guess).
- When something isn't understood, wants genuinely ground-up, first-
  principles explanations (asked twice for the Bayesian gate mechanism to
  be re-explained "from the beginning," each time going deeper) — willing
  to iterate through several rounds of follow-up questions until it
  actually lands; don't compress or skip steps to save space.
- For non-trivial implementation work that touches core methodology
  (the HPO search, and the upcoming FLOPs/latency loss extension), this
  user responds well to an explicit plan-mode pass (design + tradeoffs +
  AskUserQuestion for genuine forks) before code gets written, rather than
  jumping straight to implementation.
- Cares about experimental fairness as a first-class methodological
  concern, not an afterthought — noticed the tuning-effort asymmetry
  between Bayesian and bio-inspired methods unprompted and pushed for a
  systematic fix rather than accepting a quick patch.
- Git/GitHub: pushes happen from CSF3 (not the local machine) via SSH (see
  "Where everything lives" above for the auth history). Commits should have
  descriptive messages explaining *why*, not just *what* changed.

---

# Session 2 (2026-08-01 -> 08-03)

Two workstreams: the FLOPs-aware loss extension (built, **paused,
inconclusive**), and a pivot into **replicating three published setups** so
the comparison has an external reference. Read the FLOPs section before
resuming that idea -- the naive version has been tried and does not work.

## HPO search: abandoned

The multi-sampler Optuna search was dropped entirely ("hpo search was a
flop it didnt work"). All hyperparameters are now chosen manually.
`hpo_search.py` and its outputs still exist but are dead weight; the
fairness problem it was meant to solve is now addressed differently, by
replicating each paper's own setup rather than tuning our
re-implementations of their methods.

## FLOPs-aware pruning loss -- BUILT, PAUSED, DID NOT WORK

Implemented exactly as scoped in Session 1: `unit_cost` buffer per gated
layer (`bayesian_layers.py`), `compute_and_set_unit_costs` (`metrics.py`,
hooks the inner conv/linear and accumulates across timesteps so the
x`num_steps` factor falls out automatically), `expected_cost` /
`total_expected_cost`, `gamma * expected_cost` added to
`bayesian_snn_loss`, `gamma` warmed up via the renamed generic
`linear_warmup_schedule`. `gamma_max` defaults to 0.0 everywhere, so the
term is opt-in and inert unless a config enables it.

**Results: the mechanism runs correctly but never changed which units get
pruned.**

| model | gamma_max | outcome |
|---|---|---|
| LeNet | 1e-5 (magnitude-calibrated) | 27.7% pruned, 70.5% acc -- *identical* to gamma=0 baseline; `conv1`/`conv2` untouched, all pruning in `fc1`/`fc2` |
| LeNet | 3e-4 (30x) | conv layers *still* `frac_prunable=0.000`; `fc2` over-pruned to ~2 units, accuracy collapsed |
| VGG9 | 3e-7 (calibrated) | per-layer survivors byte-identical to the gamma=0 baseline for conv0-conv4; accuracy 86.08% -> 85.24% |
| VGG9 | 1e-5 (30x) | total collapse: 100% pruned, 10.01% accuracy (chance) |

So: calibrated values do nothing, 30x values collapse, nothing useful in
between. The "match the magnitude of `beta_max * KL` at init" calibration
heuristic is **not** a valid way to pick `gamma_max` -- disproven twice.

**Literature check (real, PDFs read, not assumed).** Current best practice
does *not* use a fixed cost-weight multiplier at all:
- **FALCON** (Meng et al., MIT, AISTATS 2024, arXiv 2403.07094) -- specifies
  hard FLOP *and* sparsity budgets and solves an ILP. Their ablations show
  budget-constrained formulations beat "mere FLOP minimization", i.e. they
  directly tested our approach and found it worse.
- **HALP** (arXiv 2110.10811) -- target latency budget via a Lagrangian;
  pruning intensity adjusts based on whether it is currently over budget.
- **Lemaire et al.** (CVPR 2019, arXiv 1811.09332) -- a *shrinking* target
  budget with a barrier function; lambda is fixed at 1e-5 and is not the
  control knob.
- **Louizos, Welling & Kingma** (ICLR 2018) -- lambda always expressed as
  `constant/N`; and for LeNet-5 they used a **20x larger lambda for conv
  layers than FC**, noting that sparsity-prior methods (our lineage)
  "sparsify parameters irrespective of that extra cost ... achieve similar
  sparsity on all layers". That directly predicts the LeNet result above.

**If resuming: do not tune `gamma_max`.** Switch to a target-budget
formulation (specify the desired FLOPs reduction, let the loss supply
whatever pressure is needed), and/or a per-layer cost weight a la Louizos.

## The pivot: replicating three published setups

**Motivation.** Our baselines sit well below the papers we compare against
(VGG9 85.00% vs Chowdhury 90.10%, SCA 91.14%, DPAP 94.54%), so "Bayesian
beat our re-implementation of DPAP on our architecture" is a weak claim.
Instead: replicate each paper's *architecture and training recipe* so our
baseline matches theirs, then run *our* Bayesian pruning in their setup and
compare against their *published* pruned numbers. Their pruning methods do
**not** need reimplementing -- `activity_pruning.py` already has them.

**`docs/replication_targets.md` is the source of truth.** Every setup value
is recorded with provenance: `[paper]`, `[code]`, or `[UNKNOWN]`. Do not
"fix" a discrepancy between it and `config.py` without reading it first --
values that look wrong are usually deliberate replication fidelity.

Key findings:
- **Chowdhury (IJCNN 2021)** is the *only* one that documents its setup
  fully, and is also the most expensive to replicate: hybrid ANN->SNN
  conversion with per-layer 99.9-percentile threshold balancing, Poisson
  rate coding, average pooling, dropout instead of BN, no bias, 100
  timesteps. **Not yet built.**
- **SCA (ICML 2024)** does *not* state optimizer, LR, loss or LIF params,
  and no public code could be found (OpenReview is behind a bot wall). A
  faithful replication is **not possible**; any attempt must be labelled an
  assumption, not a replication. **Not yet built.**
- **DPAP (TPAMI 2024)** states nothing about its CIFAR-10 setup in the
  paper either, but its code exists. Real values: **8 timesteps**,
  PLIFNode (**learned** membrane decay) at tau=2.0 => beta 0.5, **threshold
  0.5**, **AdamW** wd=0.01, lr 5e-3 scaled by batch/1024, cosine+warmup,
  300 epochs, batch 50, and a **one-sided MSE** loss, not cross-entropy.
  **Built -- see below.**

### Fidelity note that will look like a bug: DPAP's `UnilateralMse`

Their released `forward` calls `torch.clip(x, max=thresh)` and **discards
the result** (`torch.clip` is not in-place), so the one-sided clamp the
class is named after never runs and over-firing *is* penalised.
`losses.unilateral_mse` reproduces the **effective** behaviour, because
that is what produced the 94.54% we compare against. Do not "fix" it.

## Infrastructure added (Phase 1)

All three target architectures are plain conv stacks, so rather than three
near-copies of the most error-prone code in the repo:
- **`config.ArchConfig`** -- conv spec (with `"M"` pooling markers), fc
  hidden sizes, pool type, norm type, bias, dropout, encoding, optional
  per-layer thresholds. Validates its own geometry (rejects consecutive
  `"M"`, size-changing convs, over-pooling) because a wrong `flatten_dim`
  otherwise trains happily on the wrong architecture.
- **`models.VGGStyleSNN`** -- one class expressing VGG9, DPAP's 6Conv2FC,
  SCA's VGG16 and Chowdhury's 8-conv VGG9. `ArchConfig()` defaults
  reproduce the original VGG9 exactly (asserted on param count and layer
  shapes -- *not* state_dict keys; the fc stack is named `fc_layers.0` to
  generalise, so old VGG9 checkpoints need key remapping).
- **`pruning._rebuild_vgg_style`** -- one physical-rebuild routine shared by
  **both** criteria, so Bayesian and bio-inspired cannot silently disagree
  about how a network is rebuilt given the same keep-set.
- **`encoding.py`** -- direct (default no-op) and Poisson. Poisson is
  sign-preserving: clamping to [0,1] would silence ~half the input under
  Chowdhury's mean=std=0.5 normalisation.
- **`tests/test_vggstyle.py`** -- 97 CPU checks, no GPU/CIFAR needed. Run it
  before any CSF3 submission; it catches snntorch version differences.

`VGG9SNN`, `prune_vgg9`, `prune_vgg9_activity`, `LeNetSNN` and
`SpikingResNet18` are **untouched**; param counts verified identical
(62,006 / 8,887,978 / 11,177,866).

## THE BIG BUG: the gate was applied before BatchNorm

**Most consequential finding of the session, and it retroactively affects
ResNet18's results recorded in Session 1.**

Symptom: the DPAP replication collapsed to `frac_prunable=1.000` with every
`log_alpha` pinned at the clamp ceiling -- three times, across `beta_max`
from 0.005 to 0.4 and under two different task losses.

Two wrong diagnoses were pursued first (recorded so they are not repeated):
1. *"MSE's gradient is too weak"* -- plausible, but switching the pruning
   phases to cross-entropy changed nothing.
2. *"BatchNorm neutralises the gate"* -- correct, but a first test compared
   gated vs clean signals by **correlation** and showed no difference,
   which briefly killed the theory. Correlation is **scale-invariant** and
   BatchNorm's effect *is* a scale change, so that measure was structurally
   incapable of seeing it. **Measure gradients, not outputs.**

Root cause: the gate lived inside `BayesianConv2d.forward`, giving
`conv -> gate -> BatchNorm`. BatchNorm renormalises to unit variance, so
growing the gate's noise is divided straight back out and the output barely
moves. Measured `d(task_loss)/d(log_alpha)`:

| log_alpha | gate **before** BN | gate **after** BN | **no** BN | KL grad |
|---|---|---|---|---|
| -3.0 | 0.00016 | 0.0062 | 0.0019 | 0.538 |
| 3.0 | 0.00016 | 2.50 | 0.76 | 0.025 |
| 8.0 | **0.00002** | 377 | 119 | 0.0002 |

The task gradient is ~3400x below the KL's pull *and falls as things get
worse* -- no restoring force at any `beta_max`, which is why lowering it 80x
achieved nothing. After the norm it climbs 0.006 -> 377, giving an
equilibrium. The no-BN column climbs the same way, which is exactly why
LeNet and VGG9 never had this problem.

The pattern is 4-for-4 across the project: **every** BatchNorm architecture
collapsed (ResNet18 partially -- its `stage4` `frac_prunable=1.000`,
previously written off as a residual-architecture quirk; DPAP totally),
**every** non-BatchNorm one worked.

**Fix:** `BayesianConv2d.defer_gate` + `apply_gate()`. `VGGStyleSNN` sets
it when `norm_type="batch"` and does `conv -> BN -> gate -> LIF`. Only the
VGG-style family is affected. It also makes Session 1's BatchNorm mask-leak
fix (`_register_bn_remask_hooks`) structural rather than a hook, so those
hooks are no longer registered for these models.

**WARNING: ResNet18's numbers (90.60% pruned, 88.93%) were produced with
this bug and are probably understating the method.** It has *not* been
re-run. `SpikingResNet18` still applies its gate before `bn1`. Decide
whether to port the fix and re-run before citing those figures.

## DPAP replication status

**Baseline: PASSED the go/no-go gate.** 91.96% on the first attempt (2.58pp
short) -> **94.35% vs their 94.54%** after fixing the data pipeline. The
curve ruled out undertraining (val acc 0.9124/0.9184/0.9224/0.9230 at
epochs 150/200/250/300 -- flat). Reading BrainCog's loader found three
undocumented differences:
1. **no validation split** -- they train on all 50k; we held out 10%;
2. **much heavier augmentation** -- RandAugment + ColorJitter 0.4 +
   RandomErasing 0.25;
3. a different **normalisation std**.

WARNING: `val_fraction=0.0` means validation *is* the test set, matching
their protocol but making checkpoint selection test-set-informed. Documented
in `datasets.get_cifar10_loaders` and **must be stated in the
dissertation**, not buried.

**Pruning stage: still unresolved.** After the gate fix, `log_alpha` now
moves *smoothly* (-3.05 -> 3.54 over 17 epochs) and gates differentiate from
each other, instead of slamming to the ceiling -- the mechanism works. But
`beta_max=0.4` is still too much pressure for this network and accuracy
dies by epoch 8, before any gate crosses the threshold.

There is a derivable reason it needs less: gate noise is resampled every
timestep and the output sums over `T`, so signal grows as `T` and noise as
`sqrt(T)` => tolerable `alpha` scales with `T`. DPAP's **T=8** absorbs only
8/25 of VGG9's **T=25** -- about **1.14 lower in `log_alpha`** for equal
damage -- and its threshold of 0.5 compounds it.

**Next action: `sbatch slurm_sweep.sh`** -- `sweep_beta.py` runs
`beta_max` in {0.4, 0.05, 0.02, 0.005} back-to-back against the frozen
baseline (~1.2h each, ~5h total). Read the ranked table for high pruning %,
accuracy near 94%, and **zero dead layers** -- a nonzero "dead layers" count
is the collapse signature regardless of headline accuracy. Targets to beat:
DPAP's 94.27% @ 33.5% pruned and 93.83% @ 50.8%. If all four still collapse,
stop sweeping: that points at something beyond `beta_max`.

## Workflow improvement worth keeping

`ExperimentConfig.reuse_pretrained` (on for `dpap_repl`) skips the pretrain
phase and loads `outputs/<model>/trained_model.pt`. Pretraining dominates
runtime (5.4h of a 7h DPAP run) and is independent of every pruning
hyperparameter, so tuning runs now cost ~1.6h and all fork from one
identical baseline, making them comparable to each other. A mismatched
checkpoint raises with a message naming the cause. Turn it **off** after any
architecture or data-pipeline change.

## Additions to the working-style notes

- Prefers being told plainly when a hypothesis was wrong, and asks "how come
  before it worked and now its not" -- wants the *mechanism*, and responds
  well to derivations (the `sqrt(T)` noise-averaging argument landed).
- Sets and respects pre-agreed go/no-go gates; hold the line on them rather
  than talking past a missed threshold.
- Gets understandably impatient with repeated failed runs ("sigh"). Prefer
  one discriminating experiment (or a sweep) over a sequence of single
  guesses, and say up front what each run will and will not settle.
- Wants claims grounded in what the code *actually does*, not what the paper
  says -- reading BrainCog's source is what unblocked both the DPAP
  hyperparameters and the baseline gap.
- Titles/framing discussed: dissertation title should reflect *both* novelty
  layers (Bayesian structured pruning for SNNs at all, plus the
  computationally-aware extension), e.g. "Bayesian Structured Pruning for
  Spiking Neural Networks: A Computationally-Aware Approach". The novelty
  claim rests on a handful of searches, **not** a systematic review -- it
  needs a proper lit-review pass before being stated confidently in the
  dissertation text.

---

---

# QUESTION FOR THE FRESH-CONTEXT READER

**Everything above was written by an assistant that had been inside one long
conversation for its entire duration. It made at least three wrong calls in
that session (blaming the MSE loss for the collapses; briefly abandoning the
correct BatchNorm theory after using a scale-invariant metric that could not
detect a scale change; calibrating `gamma_max` by loss magnitude twice).
Assume it is anchored on its own decisions and treat its plan as a proposal,
not a conclusion.**

You have no stake in those choices. Please answer, bluntly:

1. **Is the stated next action right?** The plan is `sbatch slurm_sweep.sh`
   (a 4-point `beta_max` sweep), then replicate Chowdhury and SCA. Given
   limited remaining time and GPU budget — and that SCA's recipe is
   documented here as *unrecoverable* — is that the right use of it? What
   would you cut?

2. **What obvious thing was missed?** Look hard at: whether the
   gate-before-BatchNorm diagnosis is actually complete; whether leaving
   `SpikingResNet18` unfixed invalidates the cross-architecture comparison
   that is the dissertation's stated core contribution; whether the FLOPs
   term's failure has a simpler mechanical cause than "the calibration
   heuristic is wrong"; and whether the optimiser (Adam) changes how
   `beta_max` should be reasoned about at all.

3. **Methodology an examiner would attack.** Especially `val_fraction=0.0`,
   the "matched sparsity" claim versus what `BioPruningConfig.keep_fractions`
   actually runs, single-seed results, and whether the numbers quoted in this
   file still match the CSVs in `outputs/`.

4. **Is there a cheaper path to a defensible dissertation** than replicating
   three papers?

Read `docs/replication_targets.md`, the code, and the actual CSVs under
`outputs/` — do not take this document's numbers on trust.

## Fresh-context review: already run TWICE (2026-08-03)

**The four questions above have been answered and largely acted on. Do not
re-answer them from scratch.**

Round 1: a no-context reviewer, findings in
`docs/fresh_review_2026-08-03.md`. Headline: do not submit the beta sweep,
fix the zero-GPU issues first, because they change results that already
exist.

Round 2: a second fresh reader implemented all seven of its zero-GPU items,
confirmed most of its claims by direct measurement, and corrected three of
them. See `docs/review_response_2026-08-03.md` and Session 3 below.

Treat both as prior art, and challenge them rather than inheriting them, but
the useful question now is the one in the READ FIRST block at the top of
this file, not the four below.

---

# Session 3 (2026-08-03)

A fresh reader worked through the review's zero-GPU list, then two curve
runs happened. `docs/review_response_2026-08-03.md` records which of the
reviewer's claims were confirmed by direct measurement and the three it got
factually wrong. Commits: `1a703e9`, `db2ccf8`.

## Three gate bugs, all present since the first commit

Verified by `git show 93d5ac0`. **Nothing introduced them.** They were
latent from 2026-07-22 and only surfaced now because of which architecture
they were pointed at.

1. **The conv gate was never structured.** `eps = torch.randn_like(h)` on
   `[B,C,H,W]` drew independent noise per spatial position with only alpha
   shared per channel. The perturbation largely cancels when the next layer
   sums over space, so the task loss barely felt `log_alpha`. Measured
   within-channel std 2.876 where a structured gate must give 0. Fixed to
   one draw per (example, channel).

2. **Half of ResNet18's KL was spent on gates that can never be removed:**
   the stem plus every residual-tied `conv2`, **1984 of 3904 gate units**.
   They ran to the clamp and injected noise into layers that survive at full
   width. Non-prunable layers are now excluded from the KL, from
   `total_expected_cost`, and from gate sampling entirely.

3. **ResNet18 still gated before `bn1`.** Isolating the mechanism gives a
   downstream gradient on `log_alpha` of **2.4e-7 before the norm versus 2.4
   after**, a factor of ten million. `models.assert_gate_after_norm` now runs
   from `build_model` and rejects unrecognised normalised architectures
   rather than waving them through.

**Why they hid for two weeks.** Bugs 2 and 3 cannot fire on LeNet or VGG9
(no BatchNorm, no non-prunable layers). Bug 1 produced no symptom at all
because it made results look *better*: a weaker task gradient means less
resistance to the KL, which is part of why VGG9 pruned 98.7% with accuracy
*rising*. Nobody audits a bug that inflates their headline number. The DPAP
replication was the first architecture combining BatchNorm with a plain conv
stack, so everything surfaced at once and looked new.

Worth stating in the write-up: bug 3 was *suppressing* ResNet18 while bug 1
was *inflating* VGG9. Those pull in opposite directions across the three
architectures the cross-architecture comparison rests on.

## Sparsity is now an input, not an outcome

The largest design change. Under the threshold rule (`log_alpha > 3.0`) the
only way to reach a given sparsity was to re-run gate training at a new
`beta_max` and hope, which is why one value gave 27.7% on LeNet and 98.8% on
VGG9. `pruning.KeepPlan` ranks gates by `log_alpha` and cuts at an explicit
target instead. The criterion itself is unchanged.

This buys: one gate-training run yields a whole accuracy-vs-sparsity curve
(`run_sparsity_curve.py`); matched sparsity against the bio criteria holds
by construction, using the same rounding as
`activity_pruning.select_keep_mask`; and published operating points are
directly reachable (DPAP's 33.46% and 50.80% land at 33.35% and 50.80%).
`BayesianConfig.prune_mode` defaults to `"threshold"`, so prior experiments
still reproduce.

**Units are not parameters.** A uniform keep_fraction of 0.72 on LeNet
removes 46.4% of parameters, not 28%. Use
`pruning.keep_fraction_for_param_target`. On small networks an arbitrary
parameter target is unreachable: LeNet's closest point to 27.74% is 26.53%,
because `conv2` going 13 to 14 channels moves `fc1`'s input by 350 columns.
State matched sparsity at a shared **keep_fraction** and report the
parameter percentage alongside.

## The silent failure mode ranked pruning introduces

Because widths are an input, **an unusable ranking still produces a clean,
monotone accuracy-vs-sparsity curve.** It just measures random structured
pruning. Two ways a ranking dies, needing separate checks:

- **Saturation.** Gates pinned at the clamp are tied; ties break by index
  order.
- **Dispersion.** Gates can be undifferentiated *without* saturating, if the
  KL drags them upward in lockstep. Observed: `std(log_alpha) = 0.12` across
  2304 gates for twenty epochs at `frac_saturated = 0.000`.

`KeepPlan.ranking_is_usable` checks both. `run_sparsity_curve.py` calls it
straight after gate training and **stops before the fine-tunes** unless
`--force`, so a dead gate phase costs ~1.5h rather than ~4h.

## Run 1: the random-pruning control (`beta_max=0.4`)

Finished cleanly and is **not evidence about the criterion**:

| pruned % | accuracy | saturation |
|---|---|---|
| 19.98 | 0.9174 | 0.889 |
| 33.35 | 0.9113 | 0.889 |
| 50.80 | 0.9062 | 0.889 |
| 70.08 | 0.8876 | 0.889 |
| 89.98 | 0.8414 | 0.889 |

88.9% of gates sat at the clamp, so keep-sets were index order among ties.
**Keep this as the random-pruning control** (`mv` it to
`outputs/dpap_repl/sparsity_curve_RANDOM_CONTROL`). It is a baseline the
project needed and the bar any real ranking must clear: **90.62% at 50.80%
pruned, 84.14% at 90%.**

Note the 91.74% at only 20% pruned, against a 94.35% baseline. That gap is
not the pruning. The gate phase drove the network to 10% accuracy, so every
rebuild started from wrecked weights.

## Why `beta_max` went 0.4 to 0.01

0.4 was borrowed from VGG9 because the KL's *value* at the start of gate
training matched (4874 vs 4671). Wrong comparison: what opposes the KL is
the task loss's **gradient** on `log_alpha`.
`train.gate_pressure_diagnostic` read that ratio at **1.4e-4 to 3.2e-3** on
this baseline at `beta_max=0.4`, and printed it before epoch 1. It was
ignored.

The gates then marched at a near-constant **0.27/epoch**, which is Adam
sign-following at `lr=5e-4` over ~900 steps, straight through `log_alpha=0`
(where every channel is scaled by `1+N(0,1)`) and on into the clamp.
Accuracy hit chance by epoch 15, which killed the task gradient, which
removed the last thing opposing the KL. A feedback loop, not a threshold
event.

**Lowering `beta_max` does not slow that march.** Under Adam the step is
about `lr * sign(grad)` while one term dominates, so `beta_max` moves where
the sign flips, not how fast `log_alpha` travels. Hence the gate LR also
dropped, 5e-4 to 2e-4; sign-following at 5e-4 covers 0.45/epoch and steps
over any equilibrium narrower than that. Warmup went 45 to 10 epochs so the
outcome is visible early.

**Success now looks different.** Gates no longer need to cross a threshold,
they need to separate. Target signature: `log_alpha` **stops rising** around
-1 to 0, `std` climbs past ~0.25, `val_acc` stays near 0.98. A run that
prunes nothing by the old threshold is still a good ranking. If `log_alpha`
is still climbing linearly at epoch 25, try 0.002.

## The SynOps proposal, in enough detail to evaluate

Rationale: **FLOPs is the wrong cost metric for an SNN.** `_FlopCounter`
counts dense MACs on `Conv2d`/`Linear` and never counts a spike, so it is
blind to the one thing that makes SNNs efficient. That is a candidate
explanation for why the cost term never once reached a conv layer across
four runs.

Per unit `j` in layer `l`:

```
cost_j  =  r_{l-1} * unit_cost_j     # computing j: only input spikes trigger MACs
         + r_j     * fanout_j        # j's own spikes driving layer l+1
```

The second term is the point: a channel that rarely fires is nearly free to
keep regardless of its parameter count.

Implementation is close to drop-in. `bayesian_snn_loss` and `expected_cost`
are unchanged; `unit_cost` is already a buffer. Only
`metrics.compute_and_set_unit_costs` changes, from once-at-build to
once-per-epoch with firing rates measured by hooks.
`activity_pruning.ActivityAccumulator(pairs, signal="spike_rate")` already
does exactly that accumulation for every architecture.

**Three things the metric swap does not fix:**

1. **Surrogate gradient suppression.** `p_keep = 1 - sigmoid(log_alpha -
   threshold)`; at `log_alpha=-3`, `threshold=3` the derivative is 0.0025,
   so the term is ~400x weaker by gradient than by value during exactly the
   period the KL decides everything. A property of the surrogate, inherited
   unchanged. Fix by widening the temperature or via the budget formulation.
2. **`gamma` is a disproven lever.** Four runs. FALCON and HALP use budgets;
   Lemaire fixes lambda at 1e-5 and moves the target; Louizos needed a 20x
   larger weight for conv than FC on LeNet-5 and predicted this exact
   failure.
3. **Gate training still has to work.**

**Claim discipline.** SynOps is a *count*, fully measurable on an A100, and
is the accepted neuromorphic energy proxy (DPAP, SCA and the rest all ran on
GPUs and report it). It is **not** a GPU speedup: a GPU multiplies zeros at
full price. Structured pruning *does* give real GPU latency gains because
the tensors genuinely shrink, and `measure_latency_ms` already captures
that. Report two separate claims, never one fudged one.

## Also changed this session

- **Validation protocol split.** `dpap_repl` keeps `val_fraction=0.0` for
  pretraining because that is the protocol whose baseline it reproduces, but
  gate training, fine-tuning and any validation-selected checkpoint now use a
  held-out 10% (`DataConfig.pruning_val_fraction`). Previously `val_loader`
  *was* the test set, so checkpoints and `beta_max` were both selected on it.
  Caveat to state rather than bury: those held-out images were seen during
  pretraining, so this is clean with respect to the test set but is not a
  pure generalisation estimate. `val_acc` near 0.98 during gate training is
  inflated for this reason; read it as a relative signal.
- **`slurm_sweep.sh` now exits 1.** Submit `slurm_curve.sh`. Under Adam a
  `beta_max` sweep shows a cliff, not a curve, at ~2h per point.
- **Reports cannot disagree with rebuilds.** `remaining_structures_report`
  reads the same plan the rebuild uses. It previously showed ResNet18's
  `conv2` as 0/512 when it is never pruned, and 0 remaining where the
  rebuild keeps 1.
- **`final_results.csv` merges instead of overwriting.** Narrowing
  `MODEL_ORDER` was deleting other architectures' rows. That is how the
  quoted VGG9 figures stopped existing anywhere in the repo.
- **`run_sparsity_curve.py --tag`** keeps successive runs from deleting each
  other.
- **Tests.** `tests/test_ranked_pruning.py` (34 CPU checks) plus
  `tests/test_vggstyle.py` (98). **Run both before every CSF3 submission**,
  and activate the venv first, or they fail with `ModuleNotFoundError` while
  `sbatch` proceeds regardless. That happened once already.

## Additions to the working-style notes

- Wants to know *why* a problem appeared now rather than earlier, and the
  answer ("it was always there, the architecture changed") landed better than
  a fix would have. Run `git show` on the first commit before claiming
  anything was introduced recently.
- Asks for reassurance ("so it won't collapse yeah") at exactly the points
  where the honest answer is qualified. Give the qualified answer.
  Confident predictions here have been wrong twice.
- Gets lost across long multi-session work and asks for a full rundown.
  Worth volunteering periodically rather than waiting to be asked.
- Reaches for changing the research topic when a run fails. Check first
  whether the proposed new direction actually escapes the current blocker;
  twice now it would have inherited it.
