# Fresh-context review, 2026-08-03

A reviewer with **no context on the project** was given HANDOFF.md, the code
and the output CSVs, and asked whether the plan was right and what had been
missed. Its verdict, recorded verbatim in substance because several items
contradict decisions made in-session.

> **Headline: do not submit `slurm_sweep.sh` yet.** Three or four cheap
> code-level fixes would change the results that already exist, so any GPU
> hour spent before they land is at risk of being spent twice.

## Findings that contradict the in-session plan

### 1. Adam makes `beta_max` reasoning wrong (the biggest miss)

`train.py` uses Adam, which normalises each parameter's update by its own
gradient's running second moment. Where one term dominates, the update on
`log_alpha` is approximately `lr * sign` -- **independent of gradient
magnitude and therefore of `beta_max`**. `beta_max` sets *where the sign
flips* (the equilibrium), not the speed of approach.

Checked against the log: gates moved -3.05 -> 3.54 in 17 epochs. At batch 50
(1000 steps/epoch) and `bayesian_train_lr=5e-4`, pure sign-following gives
0.5/epoch => 6.6 units in 13 epochs. That is the observed trajectory almost
exactly. **It is Adam marching at constant velocity, not a KL/task balance.**

Consequence: the 4-point sweep will show a **cliff, not a curve**, and costs
~8h (not the ~5h estimated: `outputs/dpap_repl/training_log.csv` gives
75s/epoch x 75 + 51s x 30 ~= 2h per trial). The discriminating experiment is
instead to log, once per epoch on one batch, the per-layer ratio
`|d task/d log_alpha| / (beta * |d KL/d log_alpha|)`. Seconds, not hours.

### 2. Half of ResNet18's KL pressure is wasted

`total_kl` sums over all gated layers regardless of `structurally_prunable`.
For `SpikingResNet18` that is `stem_conv` (64) + every block's `conv2` (1920)
= 1984 of 3968 gates that **can never be removed**. They run to the clamp
ceiling and inject std~55 multiplicative noise into layers that are never
pruned -- destroying the network during gate training for zero compression.
An independent contributor to `stage4`'s collapse, on top of the BatchNorm
ordering. Free to fix.

### 3. The conv gate is not actually a structured gate

`bayesian_layers.py:243`: `eps = torch.randn_like(h)` on `[B,C,H,W]` -- only
`sqrt(alpha)` is broadcast per channel, the **noise is independent per
spatial position**. The docstring claims the opposite, and Neklyudov et al.'s
SBP (the cited method) shares one draw per channel. Fix is one line
(`torch.randn(B, C, 1, 1)`). Also likely explains why gate training is 2x
slower per epoch than pretraining on ResNet18 (566s vs 289s).

### 4. Leaving `SpikingResNet18` unfixed is not defensible

It still gates before `bn1`. ResNet18 is one of the three "redundancy
regimes" the central comparison rests on, so publishing a cross-architecture
ranking where one architecture ran a known-broken gate placement is the most
attackable thing in the project. Note the deterministic pretrain is
**unaffected** by gate placement, so `outputs/resnet18/trained_model.pt`
stays valid and `reuse_pretrained` applies -- re-running is gate-training +
fine-tune only. Alternative: demote ResNet18 to a documented negative result
(free, honest).

### 5. The FLOPs failure has a specific mechanical cause

`p_keep = 1 - sigmoid(log_alpha - threshold)`. At `log_alpha_init=-3`,
`threshold=3`, the sigmoid derivative is `s(1-s)` with `s=0.0025` -- so the
cost term's **gradient** is suppressed ~100x relative to where it eventually
acts. Magnitude-matching by *value* therefore leaves it ~100x too weak *by
gradient* during exactly the period when the KL decides the outcome. This is
the same error the session correctly diagnosed for MSE, repeated. Fixable by
widening the surrogate's temperature, or by the target-budget formulation.
Pausing was right; abandoning is not.

### 6. Test-set contamination reaches the *pruning* result, not just the baseline

With `val_fraction=0.0`, `val_set = test_set`. In `sweep_beta.run_one_trial`
the fine-tune phase gets `val_loader` (= test) with `restore_best_checkpoint`
defaulting to True, the reported accuracy is test accuracy, and the script
**ranks trials by test accuracy for `beta_max` selection**. That is selecting
a pruning hyperparameter on the test set, on a checkpoint also selected on
the test set. Stronger criticism than the one currently documented. Fix costs
zero GPU: paper protocol for pretrain only, held-out split for gate training,
fine-tuning and beta selection.

### 7. The "matched sparsity" claim is not implemented

`BioPruningConfig.keep_fractions = [0.1]`, so every bio run sits at ~98.5%
pruned while Bayesian lands wherever the threshold puts it (LeNet 27.7%,
VGG9 98.8%, ResNet18 90.6%). The LeNet row "Bayesian 70.5% vs SCA 24.7%"
compares a 27.7%-pruned network against a 98.5%-pruned one. Since
matched-sparsity comparison is stated as *the* scientific contribution, this
must be fixed -- three LeNet bio runs at `keep_fraction ~= 0.72`, ~10 min each.

### 8. Other examiner-bait

- **n=1 everywhere** (`seed=42`), while reporting differences as small as
  0.27pp against CIFAR-10 seed noise of +/-0.3-0.5pp.
- `remaining_structures.csv` reports `stage4.0.conv2` as 0/512 remaining, but
  `_prune_basic_block` never removes conv2 outputs -- misleading as printed.
- **HANDOFF's VGG9 numbers no longer match the repo**: it cites 98.71% /
  86.08%, but `outputs/vgg9/summary.txt` now says 98.83% / 85.24%. The gamma
  experiments overwrote the quoted run. Recover or re-run before citing.

## The cheaper path the reviewer recommends

**Replace threshold pruning (`log_alpha > 3`) with ranking gates by
`log_alpha` and pruning to explicit target sparsities.** ~20 lines;
`activity_pruning.select_keep_mask` is already the needed top-k function.

This buys: one gate-training run yields a whole accuracy-vs-sparsity curve;
matched-sparsity becomes true by construction; `beta_max` leaves the critical
path entirely (it only needs to *differentiate* gates, not land on a target);
and DPAP's exact published points (33.46%, 50.80%) become directly hittable.
The "failed" DPAP run -- where gates moved smoothly and differentiated --
becomes usable.

### Ordered plan

Zero GPU cost: (1) do not submit the sweep; (2) add target-sparsity ranked
pruning; (3) exclude non-prunable layers from `total_kl`; (4) per-channel
conv gate noise; (5) port `defer_gate` to ResNet18 + assert it wherever a
gated conv feeds a norm; (6) split the validation protocol; (7) replace the
sweep with the gradient-ratio diagnostic.

Then by cost: (8) LeNet bio runs at matched sparsity (~30 min); (9) three
LeNet seeds (~4.5h); (10) one `dpap_repl` gate-training run + rebuild at
DPAP's two published sparsities (~2h + ~1h each); (11) ResNet18 re-run
(~8-14h) **or** demote it; (12) FLOPs as budget-constrained, only if time
remains, else write up as a negative result with the FALCON/HALP/Lemaire
analysis; (13) **cut SCA and Chowdhury**.

### Suggested reframing

> "Bayesian structured pruning has not previously been applied to SNNs. We
> evaluate it against three activity-based criteria at matched sparsity
> across three architectures spanning a 180x parameter range, and replicate
> one published setup (DPAP, 94.35% vs their 94.54%) to show the result is
> not an artefact of a weak baseline."

Reachable, complete, and externally anchored by one replication instead of
three.

## Caveat on the pivot's own logic

Replicating the baseline does **not** make our pruning comparable to their
pruning -- our pruning hyperparameters are still ours (`config.py`
deliberately uses cross-entropy and `weight_decay=5e-5` for the gate phases
rather than DPAP's MSE/0.01, which is the right call). The replication
anchors the **baseline only**. State that in the write-up rather than letting
an examiner state it.
