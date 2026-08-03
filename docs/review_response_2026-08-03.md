# Response to `fresh_review_2026-08-03.md`, what was implemented, and where I disagree

Written by a second fresh-context reader, working from HANDOFF.md, the
reviewer's document, the code and the CSVs. Every factual claim below was
re-checked against the code rather than taken from either document.

## Verdict on the reviewer

Its seven zero-GPU items are all worth doing and are all now done. Three of
its factual claims needed correcting, and two of its conclusions are
overstated in ways that matter for how the results get used.

### Confirmed by direct measurement

| claim | verified |
|---|---|
| Conv gate noise is per-element, not per-channel | Yes. `randn_like` on `[B,C,H,W]`; within-channel std was 2.876 where a structured gate must give 0. The docstring claimed the opposite. |
| Half of ResNet18's KL is spent on non-prunable gates | Yes, though the arithmetic was slightly off: **1984 of 3904** gate units, not 3968. 50.8%. |
| ResNet18 still gated before `bn1` | Yes, including the stem and every `conv2`. |
| `sweep_beta.py` ranks trials by test accuracy | Yes. `dpap_repl` has `val_fraction=0.0`, so `val_loader is test_set`, `restore_best_checkpoint` defaults True, and the printed ranking sorts on it. |
| Bio runs are all at `keep_fractions=[0.1]` | Yes, every bio result sits at ~98.5% pruned. |
| `remaining_structures.csv` misreports | Yes, two ways: ResNet18's `conv2` shown as 0/512 when it is never pruned, *and* a fully-thresholded layer shown as 0 remaining when the rebuild silently keeps 1. |
| HANDOFF's VGG9 numbers no longer exist in the repo | Worse than stated. `outputs/vgg9/summary.txt` says 98.83%/85.24% against HANDOFF's 98.71%/86.08%, **and** `outputs/final_results.csv` now contains a single `dpap_repl` row, `MODEL_ORDER` was narrowed and the overwrite deleted every other architecture. |

The gate-before-BatchNorm diagnosis is also confirmed, and more strongly
than the original measurement: isolating the mechanism gives a downstream
gradient on `log_alpha` of **2.4e-7 before the norm versus 2.4 after**, a
factor of ten million (`tests/test_ranked_pruning.py`).

### Corrected

1. **`keep_fraction ≈ 0.72` is wrong.** That gives **46.4%** parameter
   pruning on LeNet, not the 27.7% it was meant to match. The right value
   is **≈0.844**. Estimating this by eye is the same error that produced
   the mismatch in the first place, so it is now computed:
   `pruning.keep_fraction_for_param_target`.

2. **A uniform keep_fraction cannot hit an arbitrary parameter target on a
   small network.** LeNet's closest achievable point to 27.74% is 26.53%:
   `conv2` going 13 → 14 channels moves `fc1`'s input by 14·5·5 columns, so
   parameter percentage jumps 30.8% → 26.5% with nothing in between. State
   matched-sparsity comparisons at a shared **keep_fraction** (exact by
   construction) and report the parameter percentage alongside, not the
   other way round. DPAP's architecture is large enough that its published
   points are reachable almost exactly: 33.46% → 33.35%, 50.80% → 50.80%.

3. **The ResNet18 gate count** is 3904 total / 1920 prunable, not 3968.

### Where the reviewer is overstated

1. **"`beta_max` leaves the critical path entirely" is false.** It no
   longer has to *land* a sparsity, which is the real win. It still has to
   make gates **differentiate from each other**, and it still governs how
   much the network is damaged in transit, the DPAP run died by epoch 8,
   and ranked pruning rebuilds from those damaged weights. What changed is
   that `beta_max` needs to be *sane*, not *correct*.

2. **Ranked pruning introduces a silent failure mode the plan did not
   mention.** Gates pinned at the clamp ceiling are tied, and ties break by
   index order. A run whose gates all saturated will still hit its sparsity
   target exactly and report a plausible accuracy, it looks like a result.
   The three collapsed DPAP runs would have produced exactly this. So
   `frac_saturated` is now logged per epoch and per layer, written into
   `summary.txt`, and warned on above 0.5. **Check it before believing any
   ranked result.**

3. **"Free to fix" undersells the cost of items 3 and 4.** Per-channel gate
   noise changes the gate dynamics of *every* conv architecture, so LeNet,
   VGG9, ResNet18 and `dpap_repl` all need re-running before their numbers
   mean anything. Excluding non-prunable layers from the KL changes
   ResNet18. The code changes are free; the results they invalidate are
   not. This is acceptable only because the pivot to ranked pruning means
   re-running anyway.

4. **The gradient-ratio diagnostic is not a replacement for the sweep.** It
   diagnoses *balance*; it does not choose `beta_max`. Under ranked pruning
   the more directly useful signal is gate **dispersion**, `std` of
   `log_alpha` and `frac_saturated`. Both are now logged; the gradient
   ratio is logged too, every 5 epochs, because it is what identifies a
   placement bug.

## What was implemented

All seven zero-GPU items, plus the consumer that makes item 2 usable.

| # | item | where |
|---|---|---|
| 1 | Do not submit the sweep | `slurm_sweep.sh` now exits 1 with an explanation; `sweep_beta.py` carries a superseded banner |
| 2 | Target-sparsity ranked pruning | `pruning.KeepPlan` + `uniform_ratio_plan` / `global_ratio_plan` / `param_target_plan` / `threshold_plan`; `BayesianConfig.prune_mode` |
| 3 | Non-prunable layers out of the KL | `bayesian_layers.collect_prunable_bayesian_layers`; also excluded from gate *noise* and from `total_expected_cost` |
| 4 | Per-channel conv gate noise | `BayesianConv2d.apply_gate` draws `[B,C,1,1]` |
| 5 | `defer_gate` on ResNet18 + assertion | `SpikingBasicBlock`, `SpikingResNet18`, `models.assert_gate_after_norm`, called from `build_model` |
| 6 | Split validation protocol | `DataConfig.pruning_val_fraction`, `datasets.get_pruning_phase_loaders`; `dpap_repl` pretrains on their protocol, prunes on a held-out 10% |
| 7 | Gradient-ratio diagnostic | `train.gate_pressure_diagnostic`, logged every 5 epochs |
| + | The curve script itself | `run_sparsity_curve.py`, `slurm_curve.sh` |
| + | Report/rebuild can no longer disagree | `pruning.remaining_structures_report` reads the same plan the rebuild uses |
| + | Stop losing results | `run_all.merge_final_results` keeps rows for architectures not in this run's `MODEL_ORDER` |

Defaults are unchanged: `prune_mode="threshold"` and
`pruning_val_fraction=None` reproduce every pre-existing experiment.

Verified by `tests/test_ranked_pruning.py` (34 CPU checks, no GPU or
CIFAR-10) and the pre-existing `tests/test_vggstyle.py`, which still
passes in full.

## Still open, in priority order

1. **`dpap_repl` gate training has never produced a usable ranking.** The
   last run saturated. Ranked pruning does not fix that, it makes it
   *invisible*. Read `frac_saturated` from the first curve run before
   trusting any point on it.
2. **ResNet18 must be re-run or demoted.** It was produced with the gate
   misplaced, with half its KL wasted, and with per-element gate noise.
3. **n=1.** Differences of 0.27pp are being reported against CIFAR-10 seed
   noise of ±0.3–0.5pp.
4. **VGG9's quoted numbers are not recoverable from the repo.** Re-run or
   stop citing them.
5. The FLOPs term: the reviewer's gradient-suppression diagnosis
   (`sigmoid'(-6) ≈ 0.0025`, so magnitude-matching by *value* leaves the
   term ~400x too weak by *gradient*) is mechanically correct and was not
   implemented here, it is below the cut for remaining GPU budget. Write
   it up as a negative result with the FALCON/HALP/Lemaire analysis unless
   time appears.
