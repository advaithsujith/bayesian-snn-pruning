# Handoff: Bayesian SNN Pruning — MSc Dissertation Project

This document summarizes a long prior conversation building and debugging a
structured Bayesian pruning codebase for Spiking Neural Networks, then
building a bio-inspired comparison side, a fairness-driven hyperparameter
search, and scoping a new efficiency-aware loss extension. Read this fully
before doing anything — it captures hard-won debugging context that took
many hours of real HPC time to discover.

---

# READ FIRST: state as of 2026-08-13

## SCOPE DECISION (user, 2026-08-13). The comparison is SPEAR only.

Narrowed for time, and it is a better scope than the wide one.

**Run:** the Lagrangian SynOps-budget method on **VGG16 and ResNet18**.
**Compare against:** SPEAR's **published rows**. No reimplementation of SPEAR.
**Related work only:** Chen et al. 2023 and Sorbaro et al. 2020.
**Future work:** the bio criteria, Network Slimming, the other platforms.

| platform | SynOps % | Params % | Acc % | run as |
|---|---|---|---|---|
| VGG16 | 52.5 | 14.4 | **91.77** | `--synops-budgets 0.525 --synops-loss-budget 0.525` |
| ResNet18 | 39.2 | 30.3 | **92.78** | `--synops-budgets 0.392 --synops-loss-budget 0.392` |

Note these are **SynOps-budget runs, not parameter-target curves.**

### What SPEAR actually claims, i.e. the bar

Their full CIFAR-10 / VGG16 table (Table 1 + Appendix B Table 7):

| method | SynOps % | Params % | Acc % |
|---|---|---|---|
| NetworkSlimming | 87.3 | 40.3 | 91.22 |
| NetworkSlimming | 87.3 | 14.3 | 91.16 |
| SCA-based | 67.8 | 28.4 | 91.67 |
| SCA-based | 63.0 | 9.3 | 90.26 |
| **SPEAR** | 62.5 | 33.1 | **92.49** |
| **SPEAR** | **52.5** | **14.4** | **91.77** |
| **SPEAR** | 46.4 | 11.9 | 91.62 |

They claim wins on **both axes at every operating point**: at essentially the
same parameter count as NetworkSlimming (14.4 vs 14.3%) they use 52.5% SynOps
against 87.3% and score higher; against SCA-based they are cheaper and more
accurate twice over.

**A fair criticism to raise in related work.** The accuracy and parameter
figures in their SCA rows match SCA's *own published* numbers exactly (91.67%
@ 28.39% connectivity, 90.26% @ 9.31%), but SCA never reports SynOps -- so the
SynOps column must have been measured on SPEAR's own implementation. If so,
the two columns of that row do not describe the same network: SCA trained 300
epochs, SPEAR trains 210 with TET and SGD at 0.1. Not confirmed from the text,
and the "SCA-based" label may mean they reimplemented the criterion. Worth
raising, and it makes this project's approach look better by contrast:
**comparing against their published row with no reimplementation on either
side has no such ambiguity.**

**Why this scope is stronger, not just cheaper:** SPEAR's numbers come from
their paper, so the comparison contains **no reimplementations of anyone
else's method**. The undertuning objection that killed the SCA comparison and
motivated the abandoned HPO search cannot be raised at all.

**Caveat to state for ResNet18:** only each BasicBlock's `conv1` is prunable
here (conv2 is residual-tied, stem fixed), so their 30.3% of parameters is
probably unreachable. The **SynOps axis and accuracy remain comparable**; the
parameter axis does not.

### SynOps denominators: MEASURED 2026-08-13

`outputs/<model>/baseline_synops.json`, both committed. Without these "52.5%
SynOps" has no meaning, since it is a fraction *of the unpruned model's
measured SynOps*. Two earlier attempts died on the stale-registry bug.

| platform | unpruned SynOps/sample | dense MACs | event-driven fraction | **their target in absolute terms** |
|---|---|---|---|---|
| VGG16 | **136,362,269** | 1.25G | 10.9% | 52.5% = **71.6M** |
| ResNet18 | **268,573,701** | 2.22G | 12.1% | 39.2% = **105.3M** |

The event-driven fraction is worth noticing on its own: these networks perform
only ~11-12% of their dense MACs as actual synaptic operations. That is the
whole reason FLOPs is the wrong cost metric for an SNN, quantified on your own
models, and it belongs in the write-up.

Per-layer breakdowns are in the JSON. On VGG16 the input-event fraction falls
from 1.00 at `conv_layers.0` to 0.049 at `conv_layers.9`, so deep layers fire
rarely and are nearly free to keep -- exactly the effect a parameter-count
budget cannot see and a SynOps budget can.

**Related-work note.** The user named Chen et al. as the Lagrangian precedent.
Include **Sorbaro et al. 2020 in the same paragraph** -- it is the more
dangerous omission, being the one that puts a differentiable SynOps target
directly in a training loss. Chen has this project's *mechanism* with a
spike-blind cost; Sorbaro has its *cost* with a fixed weight rather than a
dual variable. Between them they bracket the formulation.

**Consequences for work already queued:** Network Slimming is no longer
needed for the comparison (it was the harness check, which this scope makes
unnecessary since nothing is reimplemented). The bio criteria are not needed
on the SPEAR platforms. Both stay useful as motivation from the `dpap_repl`
data that already exists.

**Everything here is blocked on gate training working on at least one SPEAR
platform, which it currently does not.** See the blocker section below.

## What you can and cannot claim. Read this before writing anything.

**CLAIM THESE. All verified against committed CSVs.**

1. **The compute budget belongs inside the criterion, during training.**
   +2.30pp with 9.4% fewer SynOps at budget 0.5, +13.03pp with 12.5% fewer at
   0.3. Paired: same criterion, same budgets, same selection rule, differing
   only in whether the loss carried the Lagrangian. 5-50x the noise floor.
   **This is the dissertation.**
2. **Cost-aware *selection* fails, and why.** Cost-blind kept 610/2304 units
   and reached 87.97%; cost-aware kept 1836/2304 and reached 68.32%. Cost and
   posterior importance correlate across layers, so dividing one by the other
   roughly inverts the ranking. This is the negative result that motivates (1).
3. **The criterion is informative.** Beats random structured pruning by
   +1.54 / +2.25 / +2.72 / +3.70 / +5.25pp, widening with sparsity.
4. **The DPAP replication is faithful**: 94.35% against their 94.54%.

**DO NOT CLAIM: that any structured criterion beats any other.** See below.

## The noise floor is 0.2-0.5pp, measured, and it kills the crossover

`sparsity_curve_compare` reran the *same* config on the *same* baseline,
differing only by the cosine_warmup fix. Against the archived `beta0.01` run:

| pruned % | beta0.01 | compare | delta |
|---|---|---|---|
| 33.35 | 93.38 | 93.63 | +0.25 |
| 50.80 | 93.34 | 92.85 | **-0.49** |
| 70.08 | 92.46 | 92.25 | -0.21 |
| 89.98 | 89.39 | 89.06 | -0.33 |

**Run-to-run variability is 0.2 to 0.5pp on this platform.** Not a statistical
argument -- a measurement. It is also consistent with theory: the binomial
standard error at 93% on a 10,000-image test set is 0.26pp.

The consequence: the criteria differ by *less than this*, and the crossover
**reverses** between the two runs.

| | archived | rerun |
|---|---|---|
| 33.35%, Bayesian vs SCA | -0.23 (SCA ahead) | +0.02 (tied) |
| 50.80%, Bayesian vs SCA | +0.26 (Bayesian ahead) | -0.23 (SCA ahead) |

Resolving a 0.25pp difference at 2 sigma with a 0.3pp SD needs ~12 seeds per
criterion per sparsity point. Not feasible. **Report the criteria as
statistically indistinguishable at matched sparsity, and say so as a
finding** -- it is more useful and more defensible than a quarter-point claim
nobody can reproduce.

## Four criteria at four sparsities (2026-08-13, `dpap_repl`)

| criterion | 33.35% | 50.80% | 70.08% | 89.98% |
|---|---|---|---|---|
| Bayesian (rerun) | 93.63 | 92.85 | 92.25 | 89.06 |
| SCA | 93.61 | 93.08 | 91.78 | 89.42 |
| DPAP | 93.09 | 92.95 | **92.43** | **90.01** |
| naive firing rate | 92.83 | 92.83 | 92.14 | 88.58 |
| random control | 91.13 | 90.62 | 88.76 | 84.14 |

All within ~0.6pp of each other; all 2-5pp above random. DPAP is nominally
best at the two heaviest sparsities. This completes the "extend the bio side
to 70% and 90%" run that Session 3 called the highest-value remaining work.
Per-point files under `outputs/dpap_repl/bio/<criterion>/keep_*/summary.txt`.

**Network Slimming is missing from this table** -- see the bug below.

## Baselines: all three trained and saved

| platform | ours | reference | note |
|---|---|---|---|
| `dpap_repl` 6Conv2FC T=8 | **94.35%** | DPAP published 94.54% | -0.19pp |
| `spear_repl` VGG16 T=4 | **90.62%** | SCA's 91.14% | SPEAR publishes none |
| `spear_repl_resnet18` T=4 | **93.05%** | SPEAR's *pruned* 92.78% | **above their pruned result** |

**ResNet18 is the better SPEAR platform.** On VGG16 our dense baseline sits
1.15pp *below* SPEAR's pruned row, so any comparison there is confounded by a
baseline gap. On ResNet18 we start 0.27pp *above* it, so a result there is
attributable to the pruning. Prefer ResNet18 for the head-to-head if only one
works.

Weights backed up at `~/snn_checkpoints/` on CSF3 home (snapshotted,
replicated) -- not in git, and not on scratch.

## BLOCKER: gate training has never worked on either SPEAR platform

Three attempts, all the *same* failure, and it is the opposite of every
previous failure in this project. No collapse, no saturation, network healthy
at ~99% val accuracy throughout -- and the gates never differentiate.

| run | optimizer | beta_max | gate_lr | log_alpha reached | std | network | usable |
|---|---|---|---|---|---|---|---|
| VGG16 (18500857) | Adam | 0.01 | 2e-4 | -3.0 -> -2.87 | 0.002 | fine | no |
| VGG16 (18506598) | Adam | 0.05 | 2e-4 | -3.0 -> -2.34 | 0.002 | fine | no |
| ResNet18 (18506392) | Adam | 0.01 | 2e-4 | -3.0 -> -2.87 | 0.002 | fine | no |
| VGG16 (18556206) | **SGD** | 0.05 | **0.05** | -3.0 -> **+3.06** | 0.028 | **DEAD** | no |
| `dpap_repl` (works) | Adam | 0.01 | 2e-4 | -- | **0.297** | fine | yes |

### BOTH failure modes are now bracketed (2026-08-13)

The SGD run behaved completely differently and is the more informative
failure. It fixed the *movement* problem entirely -- the gates marched from
-3.0 all the way through the pruning threshold to +3.06 -- but they marched
**together**, and on the way the network died:

```
epoch  1  train_acc=0.872  log_alpha -3.0
epoch 11  train_acc=0.783  log_alpha -1.4
epoch 21  train_acc=0.194  log_alpha +0.9   <- collapsed
epoch 31  train_acc=0.104  task_loss 2.56 (= ln 10, chance)
epoch 71  train_acc=0.100  log_alpha +3.06
```

**Accuracy collapsed exactly as log_alpha crossed 0**, where every channel is
multiplied by 1+N(0,1). After that the task loss is pinned at chance, its
gradient is gone, and nothing opposes the KL, so the gates ran to the ceiling
unopposed. This is precisely the runaway feedback loop documented for the
dpap_repl beta=0.4 collapse -- "accuracy hit chance by epoch 15, which killed
the task gradient, which removed the last thing opposing the KL."

So: **Adam undershoots (gates stall), SGD at gate_lr=0.05 overshoots (network
dies first).** The usable setting is between them. SGD is still the right
mechanism -- it is the only thing that made the gates move at all. `gate_lr`
0.05 came from an untested example command in this file and was never sanity
checked against the observed march rate (~0.16/epoch); it needs to be roughly
5x lower.

`run_sparsity_curve.py --beta-max` was added so the (beta_max, gate_lr) search
can happen from the command line. **beta_max sets WHERE the equilibrium sits;
gate_lr sets how fast the gates get there and whether they overshoot it.**

**Attempts cost ~32 minutes each**, so this is searchable. Abort rule while
watching a run: if `train_acc` falls below ~0.5 it is the runaway mode, cancel
rather than waiting for the full 75 epochs.

### The Lagrangian may be what breaks the tie (untested hypothesis)

Worth considering before more blind search. Look at what actually acts on
each gate:

- **The KL** pushes every gate by almost exactly the same amount -- it is a
  function of `log_alpha` alone, so 4224 gates at the same `log_alpha` feel an
  identical push. It carries **no per-gate information at all**.
- **The task loss** does differ per gate, but on VGG16 it is ~200x weaker than
  the KL. That is the signal being drowned.
- **The SynOps Lagrangian** pushes each gate in proportion to *that unit's own
  SynOps cost*, which varies ~36x across layers on dpap_repl -- far more than
  the 12x variation in the task gradient.

**The Lagrangian is the only term in the loss that varies strongly per gate**,
so it may be exactly what separates gates the KL alone cannot. Risk attached:
differentiating by *cost* rather than *importance* is what made cost-aware
selection fail (68.32% vs 87.97%); the difference here is that in the loss the
network can adapt during training rather than having cheapness imposed
afterwards, which is the argument behind the headline result anyway.

### Run BOTH arms, they are control and treatment

Do not cancel the plain-gate run in favour of the Lagrangian one. The headline
claim is a *paired* comparison -- identical everything except whether the loss
carried the SynOps term -- so VGG16 needs both:

| tag | role |
|---|---|
| `sgd2` (no `--synops-loss-budget`) | **control**: gates without the budget |
| `lagr` (`--synops-loss-budget 0.525`) | **treatment**: gates trained under it |

Four outcomes and what each means:

| control | treatment | reading |
|---|---|---|
| differentiates | differentiates | ideal, the paired comparison exists |
| **stalls** | **differentiates** | **the cost term is what makes the criterion identifiable -- a finding, not just a fix** |
| differentiates | stalls | the budget is destabilising; back the budget off |
| both stall | settings still wrong, or VGG16 is not workable |

### Hyperparameter-search integrity, decided 2026-08-13

The search is over `(beta_max, gate_lr)` and is judged by **`log_alpha std`
and `ranking_is_usable`, which never look at accuracy**. That is "tuning until
the mechanism functions", not "tuning until the score is good", and the
distinction is worth stating explicitly in the write-up.

**The line not to cross:** if several settings work, do **not** pick the one
with the best *test* accuracy. Select on **`Fine Tune Best Val`** (the summary
CSVs already carry it separately from `Accuracy After`), or simply take the
first setting that passes `ranking_is_usable`. Report test accuracy once, for
the chosen setting. `pruning_val_fraction=0.1` gives a genuine held-out split
for exactly this. Caveat to state: those held-out images were seen during
pretraining, so validation is clean with respect to the test set but is not a
pure generalisation estimate.

For context, SPEAR's method **is** a search -- DDPG reinforcement learning
over pruning configurations, guided by a reward containing accuracy and a
SynOps penalty. Whatever search is needed here to make the gates function is
far less accuracy-driven than their method performs by design, and they report
no seeds, no variance, and no unpruned baseline.

**The ratio-matching hypothesis was WRONG and is now disproven.** beta was
raised 5x specifically to move VGG16's gate-pressure ratio onto `dpap_repl`'s
known-good 5.46e-3. It landed at 5.24e-3, essentially exactly on target, and
the gates *still* did not differentiate. So the task-vs-KL gradient ratio is
**not** the sufficient statistic for a usable ranking. Do not re-try that.

What did change: the gates travelled 5x further (-2.87 -> -2.34), linearly
with beta, and stayed uniform the whole way. `train_kl` went byte-identical
for the last four epochs, so they reached equilibrium rather than running out
of time.

**Next hypothesis: Adam.** It normalises each update by that parameter's own
gradient history, so a gate whose task gradient is 12x larger still takes the
same `lr * sign(grad)` step -- exactly the magnitude information that should
separate the gates gets normalised away. The per-layer task gradients *do*
vary 12x (5.5e-4 at conv_layers.3 down to 4.5e-5 at conv_layers.12), so the
ranking signal exists and is being discarded.

`BayesianConfig.gate_optimizer="sgd"` was built for precisely this and has
never been run on a GPU ("held in reserve since Adam at beta 0.01 worked").
It did not work here. **This is the 30-minute test that decides whether the
SPEAR comparison is possible at all:**

```
sbatch slurm_curve.sh --model spear_repl --tag sgd --targets 33.46 50.80 \
    --gate-optimizer sgd --gate-lr 0.05 --finetune-epochs 30
```

Watch `log_alpha std` over the first 20 epochs. Past ~0.1 and it is working.
Still 0.002 and Adam was not the cause -- stop and rethink rather than
spending more hours.

**If it cannot be made to work**, the dissertation stands on `dpap_repl`:
claims 1-4 above are complete and verified there. The SPEAR head-to-head was
always the ambitious extra, not the contribution.

## Two bugs of mine that cost GPU. Both fixed, both now tested.

1. **`run_bio_pruning` never imported `run_network_slimming_pruning`** while
   its dispatch called it. The import was dropped when an edit pair was
   interrupted; the call went in. Because network_slimming runs *last*, a
   13-hour job completed all four sparsity points for the other three criteria
   and then died on `NameError`. `import run_bio_pruning` does not catch this
   (NameError fires at call time) and neither did the tests, which imported
   the function straight from `activity_pruning`. `tests/test_spear.py` now
   asserts every criterion in `ALL_CRITERIA` resolves inside the *runner's*
   namespace; verified to fail without the fix. **Network Slimming still needs
   its ~4.8h run on `dpap_repl`.**
2. **`measure_baseline_synops.py` kept a private four-config registry**, so
   `--model spear_repl` and `--model spear_repl_resnet18` failed argparse with
   exit code 2. Both SynOps jobs died in two minutes and neither
   `baseline_synops.json` was ever written. Now uses `config.ALL_EXPERIMENTS`.
   **Both SynOps measurements still need running** (20 min each) -- without
   them every SynOps percentage on those platforms has no denominator.

## Tooling added this session

- `slurm_compare.sh <model>` -- one platform end to end: gate diagnostic
  (aborting on a bad ratio), Bayesian curve, derived keep_fractions, then the
  other four criteria. Model argument is **required**; a lock file stops two
  concurrent runs sharing an output directory. Note its default 4 targets x 4
  criteria is ~16-19h on `dpap_repl`, not the 3-4h first estimated.
- `--emit-keep-fractions` -- derives the bio side's keep_fractions from the
  same geometry the Bayesian plan uses, instead of transcribing them by eye
  (which is what once compared a 27.7%-pruned network against a 98.5% one).
  Reproduces the recorded 0.8164 / 0.7012 exactly.
- `--fail-below` / `--fail-above` on `--diagnose-only`. The usable region is a
  **band**, not a floor: 1.4e-4 was too low (collapse), 5.46e-3 works, 2.62e-2
  was too high (no differentiation). A floor alone passed that last case.
- `--seed` on both runners, with seed-qualified output paths so a repeat
  cannot overwrite the run it is meant to be compared against.
- `--finetune-epochs` on both runners. The SPEAR configs' 210 is only needed
  for the headline matched-SynOps number; 30 for internal comparison points.
- Dispatch by `isinstance` rather than config-name string in five places.
  `spear_repl_resnet18` reuses `SpikingResNet18` under a new name, and
  `build_model` would have silently built a **VGG9-shaped** network from the
  default `ArchConfig`, while `_register_bn_remask_hooks` would have skipped
  the BatchNorm remask hooks entirely.
- `SNNConfig.output_readout` moved off `ArchConfig`, because `LeNetSNN` and
  `SpikingResNet18` never receive an `ArchConfig`.

## Priority order from here

1. **The SGD gate test** (~30 min). Highest information per GPU-hour; decides
   whether the SPEAR arm exists.
2. **Network Slimming on `dpap_repl`** (~4.8h). Completes the comparison table
   and, since SPEAR publishes NetworkSlimming on the same setup, doubles as a
   check on whether the SCA/DPAP reimplementations are trustworthy.
3. **SynOps denominators** for both SPEAR platforms (20 min each).
4. **Seeds.** Not to rescue the criterion ranking -- 12 seeds would be needed
   for that -- but to put error bars on the SynOps result (which will look
   very strong against a 0.3pp SD) and to license stating the null honestly.

---

# State as of 2026-08-12

## DATA RISK: RESOLVED. Nothing was ever lost.

The files were on disk the whole time. **`.gitignore` was swallowing them**:
it listed `outputs/**/*.csv`, `*.json`, `*.png`, `summary.txt`,
`final_results.csv` and `logs/*.log`, so `git add outputs/...` reported
nothing and added nothing. The commit titled "Add the beta_max=0.01 sparsity
curve results" contained only two log files that happened to be forced in.
Fixed 2026-08-12 (commit 158e7c3): only model weights are ignored now.
Everything is committed and pushed (0018f8f).

Weights are backed up outside git at `~/snn_checkpoints/` on CSF3 home, which
is snapshotted and replicated, unlike scratch.

## EVERY HEADLINE NUMBER IS NOW VERIFIED AGAINST ITS CSV

Checked 2026-08-12, cell by cell. The prose in this file was accurate all
along. Ignore the old "do not trust any number in this file" warning for the
results below; they are now backed by committed files.

**Bayesian vs random control** (`sparsity_curve_beta0.01/summary.csv` vs
`sparsity_curve_RANDOM_CONTROL/summary.csv`): 93.28/93.38/93.34/92.46/89.39
against 91.74/91.13/90.62/88.76/84.14, i.e. **+1.54 / +2.25 / +2.72 / +3.70 /
+5.25**. Gate health confirmed: std 0.3078, saturation 0.000, ranking usable,
zero layers at min width. Control saturation 0.8889 as described.

**SynOps budget in the loss** (`sparsity_curve_lagrangian0.5` vs
`sparsity_curve_synops`): **+2.30pp with 9.4% fewer SynOps at budget 0.5,
+13.03pp with 12.5% fewer at 0.3.** The mechanism claim also holds in the
data: the baseline drives a layer to minimum width at 0.3 while the
budget-trained arm does not. Cost-blind kept 610/2304 units, cost-aware
1836/2304, exactly as recorded.

**Four-way comparison** (`outputs/bio_results_dpap_repl.csv`): Bayesian
93.38/93.34, SCA 93.61/93.08, DPAP 93.09/92.95, naive 92.83/92.83. Crossover
confirmed.

## Two open items from this block: both CLOSED

**naive_firing_rate's identical 0.9283 was not a keep-set bug.** Full
precision reads 0.9282999780774116 and 0.9282999774813652 -- computed
independently, and both genuinely 9283/10000 on the test set. Coincidence.

**The "Bayesian criterion may be a wash" bug is explained.** The bug was
Session 3's unstructured conv gate, which inflated VGG9's old "+19pp over the
best bio criterion" headline (already listed below as "produced under a bug
that flattered it"). The `dpap_repl` four-way table was run *after* that fix,
at matched sparsity, and is the honest replacement. Nothing in it is invalid.

## THE REAL CONSEQUENCE: the bio comparison is inconclusive at n=1

A 0.23-0.26pp gap on a 10,000-image test set is inside seed noise. **Do not
claim the Bayesian criterion beats SCA or DPAP on this evidence.** An examiner
will not accept it and they will be right.

What survives and should carry the dissertation:

1. **The SynOps budget result.** +2.30pp and +13.03pp are *paired* comparisons
   of the same criterion with and without the budget term, at margins 10-50x
   the noise floor. This is the novelty claim and it is solid.
2. **The degradation rate**, Bayesian -0.04 against SCA's -0.53 over the same
   sparsity increase. A within-method paired change, less exposed to baseline
   noise than the absolute gap.
3. **Beating random by +1.54 to +5.25pp**, far outside noise, which
   establishes the criterion is informative at all.

Report the bio comparison as inconclusive with a suggestive crossover. Seeds
are what would settle it, and they remain the top GPU priority.

**Observation worth using:** at *identical* parameter sparsity the criteria
produce quite different SynOps (SCA 458M, DPAP 508M, naive 563M at 33.35%
pruned). Activity-based criteria naturally select lower-firing units. That
strengthens the motivation for making SynOps an explicit objective rather than
leaving it a byproduct.

---

# Superseded: state as of 2026-08-11

## DATA RISK, unresolved and blocking everything

**Every headline number below is unverifiable from the local repo.** Checked
2026-08-11:

- `outputs/dpap_repl/sparsity_curve_synops/` — **empty directory**
- `outputs/dpap_repl/sparsity_curve_x/` — **empty directory**
- `outputs/dpap_repl/sparsity_curve_lagrangian0.5/` — **does not exist**
- `outputs/dpap_repl/sparsity_curve_beta0.01/` — **does not exist**
- `outputs/bio_results_dpap_repl.csv` — **does not exist** (only the older
  `bio_results.csv`, which is the `keep_fraction=0.1` sweep at ~98% pruning on
  lenet/vgg9/resnet18, not the matched-sparsity four-way comparison)

What *is* local contradicts the story: `outputs/final_results.csv` holds a single
`dpap_repl` row reading 91.96% before, **9.99% after**, 99.999% pruned, 183
parameters left. That is a collapsed run, not the 94.35% replication.

So the SynOps table, the four-way comparison, the random-control curve and the
94.35% baseline exist only as prose in this file and on CSF3 scratch. **Pull
`outputs/` off CSF3 before doing anything else.** Session 3 already flagged this
("push from CSF3 before they get lost") and it was never done.

## Reported but undocumented: the Bayesian criterion may be a wash

User reports (2026-08-11, verbally, details not yet captured) that the earlier
finding of Bayesian beating the activity-based criteria **was a bug**, and that
corrected numbers put them within ~0.25pp of each other. That is consistent in
magnitude with the four-way table below (SCA ahead by 0.23pp at light sparsity,
Bayesian ahead by 0.26pp at 50.8%), and at n=1 seed a gap that size is inside
the noise either way.

**Outstanding and needed before this can be written up**: what the bug was, when
it entered, which runs it invalidates, and which survive. The DPAP baseline
replication is believed unaffected because pretraining runs with gates inert.

If this holds, the project's thesis moves off "Bayesian criterion is better" and
onto "cost-awareness must live inside the criterion and must enter during
training, not selection" — which the SynOps result and the cost-aware-selection
negative result already support jointly.

## SPEAR: the field already prunes SNNs against SynOps

Found 2026-08-11. **SPEAR**, *"Structured Pruning for Spiking Neural Networks via
Synaptic Operation Estimation and Reinforcement Learning"*, arXiv 2507.02945.
This was flagged in Session 2 as circumstantial evidence only; the full text has
now been read.

What it does:

- Structured pruning of SNNs against an **explicit SynOps target**. So "nobody
  has made SNN pruning SynOps-aware" is **dead as a novelty claim.**
- Search is **DDPG reinforcement learning** over an already-trained network, not
  gradient descent. Pruning is treated as a sequential decision problem.
- SynOps enters as a **soft penalty in the reward** (Target-Aware Reward), not
  as a hard constraint.
- Post-fine-tune SynOps is **predicted by linear regression** (their LRE), not
  measured, because SynOps shifts during fine-tuning. Note this is the same
  drift problem `synops_recount_every` exists to handle here.
- **No Bayesian or variational gating.** The search is not differentiable, so
  the budget cannot influence what the network *becomes* during training.
- Baselines: SCA-based, NetworkSlimming.

SPEAR's published numbers (static datasets at **T=4** via image copying;
CIFAR10-DVS at T=10):

| Dataset | Arch | SynOps(%) | Params(%) | Acc(%) |
|---|---|---|---|---|
| CIFAR-10 | VGG16 | 52.5 | 14.4 | 91.77 |
| CIFAR-10 | ResNet18 | 39.2 | 30.3 | 92.78 |
| CIFAR-100 | VGG16 | 69.0 | 35.0 | 70.50 |
| CIFAR-100 | ResNet18 | 48.2 | 20.4 | 68.86 |
| Tiny-ImageNet | VGG16 | 69.5 | 39.0 | 59.47 |
| Tiny-ImageNet | ResNet18 | 37.8 | 23.3 | 56.62 |
| ImageNet | ResNet18 | 72.9 | 57.2 | 54.66 |
| CIFAR10-DVS | 5Conv+1FC | 39.3 | 17.1 | 80.05 |

**Surviving novelty claim**, and it should be stated this narrowly: a
*differentiable* budget carried *inside* a *Bayesian* criterion, as a hard
constraint via Lagrangian relaxation with dual ascent, so the same posterior
that decides redundancy is shaped by cost. SPEAR choosing RL is weak
circumstantial evidence the field lacked an easy differentiable path here.

## Decision (user, 2026-08-11): replicate SPEAR and compare head to head

Chosen over the cheaper alternative of citing SPEAR's numbers as context only.

- **Target row: CIFAR-10 / VGG16, 52.5% SynOps, 91.77%.** Closest to existing
  work, and 52.5% SynOps sits almost exactly on the existing 0.5 budget, so it
  is a near-matched operating point.
- **Needs a new config**: VGG16 architecture, **T=4**, their training recipe.
  `dpap_repl` cannot be reused — it is a 6-conv VGG-style stack
  (`128,128,M,256,256,M,512,512` + 512FC) at **T=8**. Different architecture and
  different timestep count, and SynOps scales with timesteps, so nothing is
  comparable without a fresh setup.
- **Fresh pretrain required.** Pretraining dominated 5.4h of the ~7h DPAP run.
  This competes directly with the outstanding seed runs for GPU.
- **Expect imperfect replication.** Same problem class as SCA (see Session 2's
  replication-targets work): the paper may not state its full training setup.
  Accepted by the user, to be acknowledged explicitly in the write-up alongside
  the existing DPAP replication caveat rather than glossed over. Follow
  `docs/replication_targets.md` conventions: record provenance for every value,
  and label anything assumed as an assumption rather than a replication.

## COMPARISON PLAN (2026-08-12): three architectures, Network Slimming added

Decided with the user. **Compare against published numbers wherever possible
and reimplement as little as possible**, because every reimplementation is a
fairness risk this project has already been burned by once (SCA losing to the
naive static baseline was an undertuning artifact, and it is what killed the
HPO search).

**Three platforms, all with BatchNorm so every criterion runs on all of them:**

| config | anchor | status |
|---|---|---|
| `spear_repl` (VGG16, T=4) | SPEAR 91.77% @ 52.5% SynOps | baseline done, 90.62% |
| `spear_repl_resnet18` (T=4) | SPEAR 92.78% @ 39.2% SynOps | **built, not run** |
| `dpap_repl` (6Conv2FC, T=8) | DPAP 94.27% / 93.83% | baseline done, 94.35% |

`dpap_repl` is the cheap third platform: already trained, has BatchNorm, and
the existing four-way matched-sparsity comparison already lives there, so
Network Slimming slots in with no new pretrain.

**Rejected: reimplementing SPEAR itself.** DDPG + the LRE regression predictor
+ the TAR reward, with no public code. A fortnight minimum and RL is fragile
to tune. Under-tuning a competitor's agent and then reporting a win would be
worse for the dissertation than not comparing at all. Their published numbers
are used instead, which cannot be accused of sandbagging.

**Rejected: LeNet as a third architecture.** Two independent reasons. SPEAR
publishes nothing for LeNet, so there is no anchor; and LeNet has no
BatchNorm, so Network Slimming cannot run on it at all.

### Network Slimming added (Liu et al., ICCV 2017)

`activity_pruning.run_network_slimming_pruning`. L1 penalty on the BatchNorm
gammas during a sparsity-training phase, then rank channels by |gamma|. It is
the one competitor cheap enough to reimplement *and* independently published
on the setup we replicate (SPEAR reports it at 91.16% @ 87.3% SynOps, 14.3%
params), which makes it **a check on the harness itself**: if ours lands near
their row, the SCA and DPAP reimplementations are more credible.

Full provenance and both deliberate deviations are in
`docs/replication_targets.md` section 5. Two things to carry into the write-up:

- **It cannot run on lenet or vgg9 at all**, because neither has a BatchNorm
  between a prunable layer and its neuron. That is a limitation of the method,
  and the contrast is favourable: the Bayesian criterion attaches its own gate
  so it applies unchanged to all five architectures. State it.
- **Watch `gamma_std`.** All gammas start at 1.0 and the L1 subgradient is
  identical for every positive channel, so the penalty alone pushes them down
  uniformly and produces no ranking at all; differentiation comes only from
  the task gradient pushing back. `gamma_std` near zero means an arbitrary
  keep-set, the same silent failure as an undifferentiated `log_alpha`, and it
  still yields a clean-looking curve.

### Structural fix that came out of this: name-based dispatch was unsafe

`spear_repl_resnet18` reuses `SpikingResNet18` under a new config name, and
**five dispatch points keyed on the config name string** would have broken on
it. `build_model` was the dangerous one: it would have fallen through to the
VGGStyleSNN branch and silently built a **VGG9-shaped network** from the
default `ArchConfig`, training happily on the wrong architecture. Worse,
`_register_bn_remask_hooks` checked `model_name != "resnet18"` and would have
silently skipped the BatchNorm remask hooks, reintroducing bug #7 for that
config only, with nothing to notice.

All five now dispatch on `isinstance`, which is name-independent.
`SNNConfig.output_readout` also moved off `ArchConfig` for the same underlying
reason: `LeNetSNN` and `SpikingResNet18` hard-code their structure and never
receive an `ArchConfig`, so a setting that lives there cannot reach them.

## NOVELTY, final position after a proper literature pass (2026-08-12)

The claim has now narrowed **four times**. Anything of the form "first to make
SNN pruning SynOps-aware" or "first budget-constrained SNN pruning" is
**demonstrably false** and an examiner who knows this literature will catch it.

**Sorbaro, Liu, Bortone & Sheik 2020**, "Optimizing the Energy Consumption of
Spiking Neural Networks for Neuromorphic Applications", Frontiers in
Neuroscience 10.3389/fnins.2020.00662. Full text read. They add a **SynOp loss
term targeting a specified SynOps value S0**, made differentiable through a
quantised ReLU with a surrogate gradient, normalised by alpha = S0^2. CIFAR-10
All-ConvNet: 90.37% at 127M SynOps against 2179M unconstrained.

So **"put SynOps in the training loss as a differentiable target" is prior art
from 2020.** What they do *not* do: any pruning at all (weights merely drift
toward zero, >90% null, which they describe as a side effect); an adaptive
multiplier (fixed weight and target, set up front); direct SNN training (they
train a quantised analog CNN and convert); anything Bayesian.

### Where the claim actually stands

Each prior work has two of the four ingredients. None has all four.

| | SynOps cost | differentiable in training | structured pruning | Bayesian criterion |
|---|---|---|---|---|
| Sorbaro 2020 | yes | yes | **no pruning** | no |
| Chen 2023 | **no** (weight-count) | yes | mostly unstructured | no |
| SPEAR 2025 | yes | **no** (RL search) | yes | no |
| **This project** | yes | yes | yes | yes |

**Write it as an intersection claim, never as a first.** "The combination of a
Bayesian posterior criterion with a differentiable, activity-dependent SynOps
budget has not been reported" is defensible. "SynOps-aware SNN pruning is
novel" is not.

### The gap this exposes, and it is a fair examiner question

Sorbaro et al. used a **fixed-weight SynOps penalty toward a target** and cut
SynOps 17x with it. This project rejected fixed-weight on the basis of four
failed `gamma_max` runs -- **but those used FLOPs, not SynOps.** Two causes are
confounded in the justification for the Lagrangian: fixed-weight being the
wrong lever, versus FLOPs being the wrong metric. Sorbaro is evidence for the
second. **Fixed-gamma-on-SynOps has never been run here.** Expect to be asked
"did you compare your Lagrangian against a simple weighted SynOps penalty?"
and the honest answer today is no. One gate run at matched budget would settle
it.

## NOVELTY NARROWED (2026-08-12): Chen et al. already do the Lagrangian

Found in a proper literature pass, full text read, not inferred from an
abstract. **This is the closest prior work to the project and it was not in
this file before.**

**Chen, Yuan, Tan, Chen, Song & Zhang, "Resource Constrained Model Compression
via Minimax Optimization for Spiking Neural Networks", ACM MM 2023,
arXiv 2308.04672.** Code: `github.com/chenjallen/Resource-Constrained-Compression-on-SNN`.

Their Eq. 7 is a minimax (Lagrangian) reformulation with dual variables y, z
and a hard resource budget; Algorithm 1 line 7 is
`z <- max(0, z + eta*(R(s) - R_budget))`, i.e. **dual ascent on budget
violation**, structurally identical to `BayesianConfig.synops_budget_fraction`'s
`lambda <- max(0, lambda + lr*(E[SynOps]-budget)/budget)`. Solved end-to-end by
gradient descent-ascent with a straight-through estimator, on SNNs, in
SpikingJelly, on CIFAR-10 / VGG16 / ResNet19 / VGGSNN. Their CIFAR-10 6Conv2FC
baseline is 92.88% and they report **+0.84%** accuracy at 75% sparsity.

**So "a differentiable budget as a hard constraint via Lagrangian dual ascent
for SNN compression" is dead as a novelty claim.** Third narrowing, after
SPEAR killed "SynOps-aware SNN pruning" and Bayesian Bits killed
"cost-weighted variational gates". Do not state it in the dissertation.

**What survives, and it is sharper than the previous claim.** Two differences:

1. **Their criterion is not Bayesian.** Difference-of-convex sparsity
   reformulation plus STE on weight magnitudes. No stochastic gate, no
   posterior.
2. **Their resource function is spike-blind, and this is the better
   argument.** Their Sec. 3.1: *"R(s) evaluates a general resource consumption
   (e.g., Flops or latency) based on the number of (nonzero) weights for each
   layer."* The budget is a function of surviving weight count only and cannot
   see firing rates. SynOps = spikes x connections, so this project's budget
   tracks a quantity that **moves as the network learns to fire less**, which
   is why `synops_recount_every` re-measures during training. Chen et al. are
   a concrete instance of exactly the failure Session 3 already argued
   ("FLOPs is the wrong cost metric for an SNN ... never counts a spike").

**Claim to make, stated this narrowly:** a Bayesian posterior-uncertainty
criterion trained under a differentiable, *activity-dependent* SynOps budget.
Chen et al. have the constrained-optimisation machinery with a static cost
model; SPEAR has SynOps but a non-differentiable RL search over an
already-trained network. The combination was not found.

**Treat Chen et al. as a cited precedent and baseline**, the way
`docs/replication_targets.md` treats Bayesian Bits, not as something to work
around. They have public code, so their numbers are checkable.

Also found, none of them threats:
- **Criticality-Constrained Iterative Pruning** (arXiv 2606.30676, 2026):
  unstructured, 3-layer FC on MNIST/FMNIST, importance = magnitude x
  surrogate-gradient criticality, energy measured post-hoc only. Worth citing
  because they **tried** a Lagrangian soft-penalty for sparsity and report it
  failing (their "continuous-relaxation trap") -- useful contrast for why the
  dual-ascent formulation here has to be argued rather than assumed.
- **SLAMP** (arXiv 2603.14946): layer-adaptive magnitude, temporal
  distortion-constrained. Different lineage.
- **HAPQ** (MDPI Sensors 2026): hardware-aware pruning + quantisation,
  event-based detection.
- **Towards Energy Efficient SNNs** (ICLR 2024): unstructured framework.

## SPEAR baseline DONE (2026-08-12): 90.62%, and the gap that now matters

Attempt 2, with standard crop+flip restored: **90.62% test**, train_acc 0.9973
against val_acc 0.9003, val_loss 0.708 and flat. Saved at
`outputs/spear_repl/trained_model.pt`; `reuse_pretrained` is now True so
nothing retrains it. 210 epochs, 26 s/epoch, ~1.5h. The augmentation diagnosis
below is confirmed: +4.53pp over the literal reading.

**0.52pp under SCA's 91.14% reference**, inside the ~1% band DPAP used as its
go/no-go. Good enough to prune on.

**But the real problem is now visible, and it is not the baseline gate.**
SPEAR's *pruned* row is **91.77%**, which is **higher than our unpruned
90.62%**. They never published a dense baseline, but landing at 91.77-92.49%
after pruning puts theirs at roughly 92-93%. So we start 1.5-2.5pp behind
before the criterion does anything, and any accuracy cost from pruning drops
us to ~89-90% against their 91.77%. The headline would read "SPEAR beats
ours" when the truth is "their baseline was better than ours" -- exactly the
confound the whole replication pivot exists to remove.

Options, in the order they should be tried:
1. **Test `pool_type="avg"`** (assumption 6). TET's own VGGSNN uses average
   pooling throughout and this is a spiking net where max-pooling binary
   spike trains loses information -- the argument Chowdhury makes and this
   repo's `ArchConfig` docstring already records. One line, ~1.5h, and it
   must set `reuse_pretrained=False` since it invalidates the checkpoint.
2. **Report the drop from our own baseline as the primary metric**, noting
   theirs is uncomparable because unpublished. Isolates the criterion, weaker
   headline, and an examiner will ask why the drops cannot be compared.
3. **Compare absolutes and declare the baseline gap** as a limitation.

Do not start the ~5h curve before deciding this: every pruned number inherits
whichever baseline is chosen.

## SPEAR pretrain attempt 1 (2026-08-12): 86.09%, augmentation line disproven

210 epochs, **25.3 s/epoch, ~1.5h on an A100**. Use that number for planning:
it makes the 5-point curve's 210-epoch fine-tunes roughly 3-5h, not the ~40
min that 30 epochs would have cost.

**Result: 86.09% test, with train_acc = 1.0000 against val_acc = 0.8565 and
validation loss flat-to-rising.** Memorisation of all 50k images, not
undertraining. Do not "train longer".

**Diagnosis, and it is a real finding.** SPEAR says "For static datasets, no
data augmentation is applied", which was implemented literally. 86% is exactly
where VGG16 lands on CIFAR-10 with no augmentation, and their own table
(91.2-92.5%, SCA baseline 91.14%) is not reachable from an 86% baseline by
pruning. So the sentence must mean no augmentation *beyond* the standard crop
and flip. `get_spear_repl_config` now sets `random_crop_padding=4` and
`horizontal_flip_prob=0.5`; RandAugment / jitter / erasing stay off. Recorded
as an assumption with the 86.09% run as its evidence, in
`docs/replication_targets.md` section 4.

Same class of failure as DPAP's first attempt (91.96% vs 94.54%, also
augmentation) and the same lesson: these papers under-describe what was run.

**Next: re-run the pretrain.** ~1.5h. Expect ~90-92%. If it lands there the
replication is good enough to prune on; if it is still near 86%, the next
suspect is assumption 6 (max vs average pooling: TET's own VGGSNN uses
average throughout).

## SPEAR replication: BUILT 2026-08-11, not yet run

Config, loss, tests and submission script are in. Zero GPU spent so far.
Full provenance is in **`docs/replication_targets.md` section 4**, which is
the source of truth; this is the summary.

**Recipe, all stated in the SPEAR paper text** (unlike SCA): VGG16, T=4 via
image copying, LIF with **hard reset**, threshold **1.0**, tau **2.0**, no
input-current decay, arctan surrogate, **TET** loss, **SGD** momentum 0.9,
wd 5e-5, max lr **0.1**, **210 epochs** (10 linear warm-up + 200 cosine),
**no augmentation**, fine-tune 210 epochs in the same configuration.

**What was added:**
- `config.get_spear_repl_config()` (+ `"spear_repl"` in `ALL_EXPERIMENTS`).
- `losses.tet_loss`, registered as `"tet"`. TET is Deng et al., ICLR 2022
  (arXiv 2202.11946): per-timestep CE averaged over T, plus an MSE term
  toward a constant, mixed by lambda. Their CIFAR values are lambda=0.05 and
  phi=V_th=1.0; SPEAR states neither, so both are assumptions.
- `ArchConfig.output_readout` ("spikes" default | "current"). See below.
- `FineTuneConfig.lr_warmup_epochs` / `min_lr`, plumbed through `run_all.py`,
  `run_sparsity_curve.py` and `run_bio_pruning.py`. Defaults 0 / 0.0, so every
  existing fine-tune is bit-identical.
- `run_all.py --model NAME --pretrain-only`. `MODEL_ORDER` previously had to
  be edited in source to switch architecture.
- `tests/test_spear.py` (41 CPU checks) and `slurm_spear.sh`.

**The one real design decision: `output_readout="current"`.** TET defines its
O(t) as the output layer's *pre-synaptic current*, not its spikes, and at T=4
a summed spike count takes only five values per class, so argmax ties
constantly and breaks toward class 0. The test measures this: on an untrained
net the spike readout ties on 4/4 examples, the current readout on 0/4. So
the SPEAR arm reads `fc_out`'s analog output and skips `lif_out`. Contained
change: the tensor keeps its [T,B,C] shape and both `_spike_accuracy` and
`spike_rate_cross_entropy` reduce with `sum(dim=0)`, which is argmax-identical
on currents, so nothing downstream changed. SynOps counting is untouched
(`measure_synops` hooks Conv2d/Linear inputs).

**Three things to know before running it:**

1. **There is no go/no-go baseline gate.** SPEAR never publishes an unpruned
   VGG16 accuracy. The nearest published reference is SCA's **91.14%**, and
   the link is solid: SPEAR's table quotes SCA's pruned rows *verbatim*
   (91.67% @ 28.4% params, 90.26% @ 9.3% params match SCA's published rows
   exactly) and both use VGG16 at T=4. Treat it as a sanity check, not a
   target. Report our own baseline and our drop from it in the write-up
   rather than claiming to have reproduced a number they never printed.
2. **`beta_max=0.01` is a placeholder carried over from DPAP, not a measured
   value.** It does not transfer: it balances the KL against the task loss's
   *gradient* on log_alpha, and that ratio moves with architecture, T, and
   now the analog readout, which removes the quantisation that shaped the
   gradient on every previous run. Run `train.gate_pressure_diagnostic` on
   the new baseline and read the ratio first. This is why `slurm_spear.sh`
   uses `--pretrain-only`.
3. **Eight values had to be assumed** because neither SPEAR nor SCA states
   them: the VGG16 spec (assumed standard CIFAR VGG16, 13 conv / 5 pools /
   no hidden FC / 512->10, verified at 14,728,266 params), BatchNorm
   presence, TET's lambda and phi, the normalisation constants, batch size,
   max pooling, conv bias, and the train/validation protocol. All eight are
   numbered in `get_spear_repl_config()` and listed in the doc. The last
   three were initially silent dataclass defaults and are now pinned
   explicitly so each reads as a decision. Note assumption 6 cuts both ways:
   torchvision's VGG16 uses max pooling, but TET's own VGGSNN uses average,
   so revisit it if the baseline lands well under 91.14%.

**Independent review, 2026-08-11.** A no-context agent re-read both papers
against the code. It caught a citation error (the paper is **Xie et al.**, not
Zhang; corrected in five places, author list now recorded in full in the doc)
and a silent scheduler bug, below. It confirmed by direct derivation that the
tau->beta mapping is exact *including reset ordering* (snnTorch's
`reset_delay=True` default makes `mem_t = beta*(mem_{t-1}*(1-S_{t-1})) + I_t`,
which is exactly SpikingJelly's hard reset), that `tet_loss` matches Eqs.
9/12/13 with no target mis-broadcast, that no downstream consumer assumes a
binary output tensor, and that the parameter count is right.

## BEHAVIOUR CHANGE 2026-08-11: cosine_warmup no longer degenerates silently

Found by the review above, and it affects **dpap_repl**, not just SPEAR.

`train.build_scheduler` accepted `scheduler_name="cosine_warmup"` with
`warmup_epochs=0` and quietly returned a **plain cosine starting at the full
learning rate**. Any call site that forgot to forward `lr_warmup_epochs`
therefore ran with no warm-up while looking, from the config, as though it
had. **Five call sites had forgotten**: `run_bio_pruning.py` (a full pretrain
from scratch), `run_sparsity_curve.py` (gate training), `sweep_beta.py` x2 and
`hpo_search.py` x2. All five now forward the arguments, and `build_scheduler`
**raises** on `cosine_warmup` with no warm-up rather than serving a cosine,
because "no warm-up was configured" and "warm-up ran" must not look the same.

**What this changes for existing results.** `dpap_repl` sets
`lr_scheduler="cosine_warmup"` with 5 warm-up epochs, so its **gate-training
phase was silently running a plain cosine**. It now warms up over 5 epochs.
The archived `beta0.01` curve was produced under the old path and a re-run
will not reproduce it exactly. The pretrained 94.35% baseline is unaffected
(`run_all.py` always forwarded the arguments for the pretrain phase). If the
beta0.01 curve is ever regenerated, say so rather than presenting the two as
the same run.

**Next action:** `sbatch slurm_spear.sh` after `git pull` and the four test
suites. Estimated ~4-6h for the 210-epoch pretrain, never measured for VGG16
at T=4 on CSF3, so the script asks for 24h. Note the fine-tune decision was
**210 epochs at every sparsity point**, matching their recipe rather than
this project's 30, which is roughly 7x the fine-tune hours of every other
experiment here and contends directly with the outstanding seed runs.

## Added 2026-08-11: measure_baseline_synops.py (commit aae6beb)

SynOps budgets are defined as a fraction of the unpruned network's measured
SynOps (`pruning.synops_budget_plan` docstring), but **that denominator was
never recorded**, which leaves every reported SynOps figure uninterpretable in
absolute terms.

It cannot be back-solved from the results table: the 0.5 row implies an unpruned
total of ~468M and the 0.3 row implies ~337M, for the same network. Three
reasons they disagree:

1. Cost-blind selection (`rank_by="importance"`) stops at the largest importance
   prefix that fits and does **not** skip-and-continue, so it under-spends the
   budget by an unknown margin.
2. Layer floors are satisfied first and **win over the budget**, so a run can
   also land over.
3. Reported SynOps is measured post-fine-tune while the budget is enforced at
   selection time, and firing rates drift in between.

The script loads the saved baseline with gates inert, runs `measure_synops` over
N test batches, and writes `outputs/<model>/baseline_synops.json` with SynOps per
sample, dense MACs, the event-driven fraction and the absolute ceilings the 0.5
and 0.3 budgets correspond to. `slurm_synops.sh` submits it: 20-minute
wallclock, no training, no gradients.

---

# State as of 2026-08-03, and the open decision

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

## The four-way comparison (2026-08-04), matched sparsity on the validated platform

`outputs/bio_results_dpap_repl.csv`. All four criteria fork from the same
94.35% baseline, same 30-epoch fine-tune, same held-out split, and
identical layer widths at each keep-fraction (0.8164 / 0.7012 from
`--plan-only`), so only the choice of units differs.

| method | 33.35% pruned | 50.80% pruned | change |
|---|---|---|---|
| **Bayesian** | 93.38 | **93.34** | **−0.04** |
| SCA | **93.61** | 93.08 | −0.53 |
| DPAP | 93.09 | 92.95 | −0.14 |
| naive firing-rate | 92.83 | 92.83 | 0.00 |

**There is a crossover, and it is the result.** SCA leads by 0.23pp at
light sparsity; Bayesian leads by 0.26pp at 50.8% and degrades least of
all four (−0.04 against SCA's −0.53). Light pruning leaves slack so any
sensible criterion survives and the comparison is uninformative; the
ranking only starts to matter as compression tightens. Same shape as the
random-pruning comparison, whose margin widened from +1.5 to +5.3pp over
the same direction of travel -- two independent comparisons agreeing.

**Highest-value next run: extend the bio side to 70% and 90%**, where the
Bayesian curve already sits (92.46 / 89.39). If the bio degradation rate
holds, that is the strongest figure available and it completes the
crossover story. Six hours.

**Open check:** naive_firing_rate reports exactly 0.9283 at *both*
sparsities. Verify full precision in the CSV before quoting; identical to
four decimals across two different networks is possible but worth ruling
out as a keep-set bug.

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

Reading order if short on time: the **2026-08-13 READ FIRST block at the top**
(what you can claim, the noise floor, the gate blocker), then the 2026-08-12
block, then Session 3, then Session 2's "THE BIG BUG" and "The pivot"
sections. The rest is background.

**Note for anyone reading the old blocks below**: the repeated warning "do not
trust any number in this file without checking the CSV" has been discharged.
Every headline figure was checked cell by cell against committed CSVs on
2026-08-12 and matched. The files were never missing; `.gitignore` was hiding
them.

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
