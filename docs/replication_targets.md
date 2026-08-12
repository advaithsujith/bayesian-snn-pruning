# Replication targets: published SNN pruning setups

Verified setup details for the three papers this project replicates, so that
our reproduced baselines are directly comparable to their published numbers.
Every row is marked with its provenance: **[paper]** = stated in the paper
text, **[code]** = read from the authors' released implementation,
**[UNKNOWN]** = not stated in either and must be assumed (assumption
recorded explicitly).

Do not "fix" a discrepancy between this table and `config.py` without
checking here first -- a value that looks wrong (e.g. DPAP's threshold of
0.5, or its MSE loss) is usually deliberate replication fidelity.

## Current project baseline, for contrast

| Setting | This project |
|---|---|
| Architecture (VGG9) | 7 conv `64,64,M,128,128,M,256,256,512,M` + 800FC + 10FC |
| Timesteps | 25 |
| Encoding | direct (static image re-fed each timestep) |
| Neuron | `snn.Leaky`, fixed `beta=0.95`, `threshold=1.0`, atan surrogate |
| Pooling | max |
| Loss | spike-rate cross-entropy (`spike_rate_cross_entropy`) |
| Optimizer | Adam, lr 1e-3, weight decay 5e-5, cosine |
| Reported CIFAR-10 baseline | **85.00%** |

---

## 1. Chowdhury, Garg & Roy -- IJCNN 2021

"Spatio-Temporal Pruning and Quantization for Low-latency Spiking Neural
Networks", arXiv:2104.12528. The only one of the three that documents its
setup fully in the paper.

| Setting | Value | Source |
|---|---|---|
| Architecture | VGG9: 8 conv `64,128,128,256,256,512,512,512`, then **a single FC layer** (the usual 3 VGG FC layers are collapsed to 1) | [paper] Table I |
| CIFAR-10 baseline | **90.10%** (`VGG9o`, 100 timesteps) | [paper] Table II |
| Pruned result | 89.04% @ ~93% params removed (`VGG9s`, 0.07x params) | [paper] Table II |
| Timesteps | **100** | [paper] Table II |
| Encoding | **Poisson rate coding** | [paper] Alg. 1 |
| Training | **hybrid ANN->SNN**: train ANN, convert, threshold-balance, then surrogate fine-tune | [paper] Sec. IV.A |
| Threshold balancing | per-layer threshold = **99.9th percentile** of that layer's pre-activation distribution | [paper] Sec. III |
| Leak | lambda = **0.9901** | [paper] Sec. IV.B |
| Reset | soft reset (subtract threshold) | [paper] Eq. 2 |
| Pooling | **average** (explicitly, to avoid information loss in SNNs) | [paper] Sec. IV.A |
| Norm | **dropout, no batch-norm**; dropout mask held **constant across all timesteps** | [paper] Sec. IV.A |
| Bias | **none** | [paper] Sec. IV.A |
| ANN pretrain | SGD, momentum 0.9, weight decay 1e-4, lr **0.1**, /10 every 100 epochs | [paper] Sec. IV.A |
| SNN fine-tune | Adam, weight decay 5e-4, lr **1e-4**, halved every 5 epochs, **20-30 epochs** | [paper] Sec. IV.A |
| Augmentation | pad 4px, random 32x32 crop, horizontal flip p=0.5 | [paper] Sec. IV.A |
| Normalization | mean 0.5, std 0.5 (all channels) | [paper] Sec. IV.A |

**Note**: this is by far the largest departure from the current codebase --
Poisson encoding, average pooling, dropout-not-BN, no bias, per-layer
thresholds, and an entire ANN->SNN conversion stage none of which exist here.

---

## 2. Li et al. (SCA) -- ICML 2024

"Towards Efficient Deep Spiking Neural Networks Construction with Spiking
Activity based Pruning", arXiv:2406.01072.

| Setting | Value | Source |
|---|---|---|
| Architecture | VGG16 (also Pre-ResNet18; we replicate VGG16 only) | [paper] Sec. 5.1 |
| CIFAR-10 baseline | **91.14%** | [paper] Table 1 |
| Pruned results | 91.67% @ 28.39% connectivity; **90.26% @ 9.31% connectivity** (~91% pruned) | [paper] Table 1 |
| Timesteps | **4** | [paper] Sec. 5.1 |
| Epochs | **300** | [paper] Sec. 5.1 |
| Framework | SpikingJelly | [paper] Sec. 5.1 |
| Norm | BN, placed between conv and spiking neuron (post-activation variant) | [paper] Sec. 4.4, Fig. 1b |
| Encoding | direct | [paper] Sec. 5.1 |
| Surrogate | sigmoid, `g(x) = 1/(1+e^{-ax})` | [paper] Eq. 3 |
| Optimizer / lr / schedule | **[UNKNOWN]** | not in paper; no public code found |
| LIF tau / threshold | **[UNKNOWN]** (paper states both IF and LIF are used, without values) | [paper] Sec. 3 |
| Loss function | **[UNKNOWN]** | not stated |
| Augmentation | **[UNKNOWN]** | not stated |

**Gaps**: no code repository could be located (searched; the OpenReview page
is behind a bot-verification wall). Assumption to use if these stay unknown:
SpikingJelly's standard CIFAR-10 recipe (SGD momentum 0.9 + cosine over 300
epochs, LIF tau=2.0, threshold=1.0, cross-entropy on mean spike count) --
**record in the dissertation as an assumption, not as replication**.

---

## 3. Han et al. (DPAP) -- IEEE TPAMI 2024

"Developmental Plasticity-inspired Adaptive Pruning", arXiv:2211.12714.
Paper text omits the CIFAR-10 training setup entirely; values below are read
from the authors' released code at
`BrainCog-X/Brain-Cog/examples/Structural_Development/DPAP/prun_main.py`.

| Setting | Value | Source |
|---|---|---|
| Architecture | `128C3-BN-128C3-BN-MaxPool2-256C3-BN-256C3-BN-MaxPool2-512C3-BN-512C3-BN-512FC-10FC` = 6 conv + 2 FC, **only 2 max-pools** | [paper] Table I |
| Flatten dim | 2 pools => 32->16->8, so `512 x 8 x 8 = 32768` into the 512-unit FC | derived |
| CIFAR-10 baseline | **94.54%** (implied: 94.27% at 33.46% sparsity, AccLoss -0.27%) | [paper] Table III |
| Pruned results | 94.27% @ 33.46% sparsity; 93.83% @ 50.80% sparsity | [paper] Table III |
| Timesteps | **8** | [code] `step=8` |
| Neuron | **PLIFNode** (parametric LIF -- membrane time constant is *learned*) | [code] `node_type='PLIFNode'` |
| tau | **2.0** (=> decay factor ~0.5, vs. this project's 0.95) | [code] `tau=2.0` |
| Threshold | **0.5** (vs. this project's 1.0) | [code] `thresh=0.5` |
| Loss | **UnilateralMse(1.0)** -- MSE of mean firing rate against a one-hot target scaled by 1.0, *not* cross-entropy. See the fidelity note below. | [code] `train_loss_fn = UnilateralMse(1.)` |
| Optimizer | **AdamW** | [code] `cfg.opt='adamw'` |
| Learning rate | 5e-3, linearly scaled: `lr * batch_size / 1024` | [code] |
| Schedule | cosine, warmup 5 epochs, cooldown 10 epochs, min_lr 1e-5 | [code] |
| Epochs | **300** | [code] `epochs=300` |
| Batch size | **50** | [code] `batch_size=50` |
| Weight decay | **0.01** | [code] `cfg.weight_decay=0.01` |
| Augmentation | RandomCrop(32, pad=4) + hflip + **RandAugment `rand-m9-mstd0.5-inc1`** + **ColorJitter 0.4** + **RandomErasing p=0.25** (timm `create_transform`) | [code] `braincog/datasets/datasets.py` |
| Normalisation | mean (0.4914, 0.4822, 0.4465), std **(0.2023, 0.1994, 0.2010)** | [code] `CIFAR10_DEFAULT_STD` |
| Validation split | **none** -- trains on all 50k, evaluates on the 10k test set | [code] |

**Notes on porting to snnTorch**: `PLIFNode` has a *learnable* membrane time
constant -- snnTorch's nearest equivalent is `snn.Leaky(..., learn_beta=True)`.
The tau->beta mapping is `beta = 1 - 1/tau = 0.5` for tau=2.0 under the
standard parameterisation, but BrainCog's exact update rule should be checked
before relying on this.

**Fidelity note -- `UnilateralMse`'s clamp is inert.** The released source
(`braincog/base/utils/criterions.py`) reads:

```python
def forward(self, x, target):
    torch.clip(x, max=self.thresh)          # <- return value discarded
    if x.shape == target.shape:
        return self.loss(x, target)
    return self.loss(x, torch.zeros_like(x).scatter_(1, target.view(-1, 1), self.thresh))
```

`torch.clip` is not in-place and its result is never assigned, so the
"unilateral" one-sided clamp -- the thing the class is named after -- never
takes effect. Effectively this is a plain MSE between the mean firing rate
and a scaled one-hot target, and over-firing *is* penalised.

`losses.unilateral_mse` deliberately reproduces the **effective** behaviour
rather than the apparent intent, because the published 94.54% baseline was
produced by the code as written. Implementing the intended clamp would be a
different loss than the one that produced the number we are comparing
against. Flagged here so it is not later "fixed" into a mismatch.

**First replication attempt: 91.96% vs. their 94.54%** (2.58pp short, outside
the ~1% go/no-go gate). Diagnosis from the training curve: validation accuracy
was 0.9124 / 0.9184 / 0.9224 / 0.9230 at epochs 150 / 200 / 250 / 300 -- gaining
~0.005 per 50 epochs and essentially flat, so **not** undertrained. That pointed
at recipe differences, and reading BrainCog's data loader found three, all now
fixed in `get_dpap_repl_config`:

1. **No validation split.** Their pipeline trains on all 50,000 images; ours
   held out 10% and trained on 45,000. Worth roughly half a point.
2. **Much heavier augmentation.** RandAugment + colour jitter + random erasing,
   versus our crop + flip alone. Typically worth 1-2pp on CIFAR-10.
3. **A different normalisation std** (0.2023, 0.1994, 0.2010 vs. ours).

**Known deviations that remain** (record these in the dissertation):
- snnTorch's `learn_beta` learns a single scalar decay per neuron layer;
  BrainCog's PLIFNode parameterisation may differ in detail.
- Their cosine schedule includes a 10-epoch *cooldown* that
  `cosine_warmup` does not model; warmup (5 epochs) and `min_lr` (1e-5) are
  replicated.
- timm's RandAugment spec `rand-m9-mstd0.5-inc1` carries a magnitude-std and
  an "increasing severity" flag that torchvision's `RandAugment` does not
  expose; only the magnitude (9) is reproduced.
- Running with `val_fraction = 0.0` means best-checkpoint selection happens
  on the test set, matching their protocol but making the figure a
  reproduction of that protocol rather than a clean generalisation estimate.
  See `datasets.get_cifar10_loaders`.

---

## 4. Xie et al. (SPEAR) -- arXiv 2507.02945

"SPEAR: Structured Pruning for Spiking Neural Networks via Synaptic Operation
Estimation and Reinforcement Learning", Hui Xie, Yuhe Liu, Shaoqi Yang,
Jinyang Guo, Yufei Guo, Yuqing Ma, Jiaxin Chen, Jiaheng Liu, Xianglong Liu.
Author list recorded in full because the first draft of this section
misattributed it to "Zhang et al.", which would have propagated straight into
the dissertation bibliography. The head-to-head target for this
project's SynOps-budget work, because it is the only prior method that also
prunes SNNs against an explicit SynOps target. No public code could be
located (searched 2026-08-11; the OpenReview page lists no repository), so
every value below is from the paper text.

**Why this paper and not the others as the comparison target.** SPEAR
searches with DDPG reinforcement learning over an already-trained network and
enters SynOps as a *soft penalty in the reward*, with post-fine-tune SynOps
*predicted by linear regression* rather than measured. Ours is a
differentiable hard constraint carried inside the criterion during gate
training. Same objective, opposite mechanism, so the comparison is meaningful
rather than incidental.

### Target operating point

| Dataset | Arch | SynOps(%) | Params(%) | Top-1 Acc(%) |
|---|---|---|---|---|
| **CIFAR-10** | **VGG16** | **52.5** | **14.4** | **91.77** |

Chosen because 52.5% SynOps sits almost exactly on this project's existing
0.5 SynOps budget, making it a near-matched operating point. Their full
CIFAR-10 / VGG16 table (Table 1 and Appendix B Table 7), for context:

| Method | SynOps(%) | Params(%) | Acc(%) |
|---|---|---|---|
| NetworkSlimming | 87.3 | 40.3 | 91.22 |
| NetworkSlimming | 87.3 | 14.3 | 91.16 |
| SCA-based | 67.8 | 28.4 | 91.67 |
| SCA-based | 63.0 | 9.3 | 90.26 |
| SPEAR | 62.5 | 33.1 | 92.49 |
| **SPEAR** | **52.5** | **14.4** | **91.77** |
| SPEAR | 46.4 | 11.9 | 91.62 |

### Setup

| Setting | Value | Source |
|---|---|---|
| Architecture | VGG16 (also ResNet18; we replicate VGG16 only) | [paper] Sec. 5 |
| Conv/FC specification | **[UNKNOWN]** -- "VGG16" is named but never specified | see assumption below |
| Timesteps | **4** | [paper] "We copy the images 4 times along the timeline to obtain input for 4 time steps." |
| Encoding | **direct** (static image copied per timestep) | [paper] same sentence |
| Neuron | LIF, **hard reset** | [paper] "LIF neurons with a hard reset mechanism" |
| Threshold | **1.0** | [paper] "We set the fire threshold as 1.0" |
| Membrane time constant | **tau = 2.0** | [paper] "set membrane potential time constant as 2.0" |
| Input current decay | **none** | [paper] "No decay for input currents is used." |
| Surrogate | **arctan** | [paper] "We use arctan function as the surrogate function." |
| Loss | **TET** (Deng et al., ICLR 2022) | [paper] "TET is used as loss function." |
| TET lambda / phi | **[UNKNOWN]** in SPEAR; lam = 0.05, phi = V_th from the TET paper | [paper, TET] Secs. 5.2 / 4.2 |
| Optimizer | **SGD, momentum 0.9** | [paper] |
| Weight decay | **5e-5** | [paper] |
| Epochs | **210** = 10 linear warm-up + 200 cosine annealing | [paper] |
| Max learning rate | **0.1** | [paper] |
| Augmentation | **none** -- "For static datasets, no data augmentation is applied" | [paper] |
| Normalisation | **[UNKNOWN]** | not stated |
| Batch size | **[UNKNOWN]** | not stated |
| Framework | SpikingJelly | [paper] |
| **Unpruned baseline accuracy** | **[UNKNOWN] -- never reported** | see below |
| Fine-tune after pruning | 210 epochs, "the same configuration as training" | [paper] |
| SynOps | `SynOps = sum_k s_k * c_k` (spikes fired x synaptic connections), per sample, averaged over the test set; timesteps are inside the spike counts | [paper] |
| SynOps(%) denominator | "the ratio of SynOps and #parameters over those from pre-trained model" -- i.e. relative to the *unpruned* model's measured SynOps | [paper] |

### The missing baseline, and what to compare against instead

**SPEAR never reports an unpruned VGG16 CIFAR-10 accuracy**, in the text or
any table. This is the single biggest difference from the DPAP replication,
which had a published 94.54% to act as a go/no-go gate. There is no
equivalent gate here.

The usable substitute: SPEAR's table quotes the SCA-based rows **verbatim
from SCA's own paper** -- 91.67% @ 28.4% params and 90.26% @ 9.3% params
match SCA's published "91.67% @ 28.39% connectivity" and "90.26% @ 9.31%
connectivity" exactly (see section 2 above). Both papers also use VGG16 at
T=4 on CIFAR-10. So the two setups are the same family, and **SCA's stated
baseline of 91.14% is the closest thing to a published unpruned reference for
SPEAR's table**. Treat it as context, not as SPEAR's own number.

Practical consequence for the write-up: report the pruned comparison against
SPEAR's 91.77% as the primary claim, and report *our* replicated baseline and
our accuracy drop from it alongside, rather than claiming to have reproduced
a baseline they never published.

### Assumptions (record as assumptions, not as replication)

1. **VGG16 specification.** Neither SPEAR nor SCA states one. Assume the
   standard CIFAR VGG16: conv `64,64,M,128,128,M,256,256,256,M,512,512,512,M,
   512,512,512,M` (13 conv layers, 5 pooling stages), no hidden FC layers, a
   single `512 -> 10` classifier. Five pools take 32px to 1px, so the
   flattened width is 512. This is the configuration used almost universally
   for VGG16 on CIFAR-10 and the one SpikingJelly-based SNN work adopts.
2. **BatchNorm.** Not mentioned by SPEAR. Assumed present (`norm_type="batch"`),
   as in SCA, which places BN between conv and spiking neuron. Note this makes
   the gate-placement fix load-bearing: the gate must be applied *after* the
   norm (`models.assert_gate_after_norm`), the bug that cost three collapsed
   DPAP runs.
3. **TET lambda = 0.05, phi = V_th = 1.0**, from the TET paper's CIFAR
   settings. SPEAR states neither.
4. **Normalisation** mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994,
   0.2010) -- the standard CIFAR-10 values, and the same std DPAP's code uses.
5. **Batch size 128**, this project's default. Note their lr of 0.1 is not
   batch-size-scaled in the paper, unlike DPAP's, so batch size and lr are not
   coupled here as they were there.
6. **Max pooling.** SPEAR states no pooling type. Max is what torchvision's
   VGG16 uses and SPEAR names VGG16 unqualified, so max is the reading. Worth
   knowing it cuts the other way too: TET's own VGGSNN uses **average**
   pooling throughout (their Sec. 5, `64C3-128C3-AP2-...`), and this repo's
   `ArchConfig` docstring argues avg pooling suits binary spike trains, which
   is why Chowdhury uses it. Revisit if the baseline lands well below SCA's
   91.14% reference.
7. **Conv bias present.** Also unstated. torchvision's `vgg16_bn` keeps conv
   bias, and it is required to reach the 14,728,266 parameter count.
8. **Train/validation protocol** (`val_fraction=0.0`). Unstated. Training on
   all 50k and evaluating on test is what SCA and DPAP both do.

Assumptions 6, 7 and 8 were initially left as silent `ArchConfig`/`DataConfig`
defaults; they are now pinned explicitly in `get_spear_repl_config()` so each
reads as a decision rather than an oversight.

### Porting notes

- **tau = 2.0 with no input-current decay maps exactly onto snnTorch.**
  SpikingJelly's `LIFNode(decay_input=False)` updates
  `v <- v - (v - v_reset)/tau + x`, i.e. `v <- (1 - 1/tau) * v + x`. snnTorch's
  `Leaky` is `mem <- beta * mem + input`. So `beta = 1 - 1/tau = 0.5`, an
  exact correspondence rather than the approximate one DPAP's PLIFNode needed.
- **Hard reset** is `reset_mechanism="zero"` (DPAP and this project's own
  experiments use `"subtract"`).
- **`beta` is fixed, not learned** (`learn_beta=False`), unlike DPAP's PLIF.
- **TET reads the output layer's pre-synaptic current, not its spikes.** Their
  Eq. 7-9 define `O(t)` as "pre-synaptic input `I(t)` of the output layer".
  This project's models classify by summed output *spikes*, which at T=4 gives
  only five distinguishable levels per class and would both cripple the TET
  cross-entropy and depress accuracy through ties. See the `output_readout`
  setting on `ArchConfig`. Provenance: this is **[paper] for TET**, which is
  sufficient since TET is the loss SPEAR trains with, but **[UNKNOWN] for
  SPEAR**, which never states its readout. Do not write it up as a documented
  choice of SPEAR's.
- **No augmentation** is reached with `random_crop_padding=0` and
  `horizontal_flip_prob=0.0`; both transforms then become no-ops rather than
  needing a code path of their own.

### First attempt: 86.09%, and what it settled

Run 2026-08-12, 210 epochs, 25.3 s/epoch, ~1.5h on an A100. Implemented the
augmentation line **literally** (`random_crop_padding=0`,
`horizontal_flip_prob=0.0`). Result:

| | |
|---|---|
| Final train accuracy | **1.0000** |
| Final validation accuracy | 0.8565 |
| Validation loss, epochs 208-210 | 1.0591 / 1.0508 / 1.0728 (flat, rising) |
| **Test accuracy** | **86.09%** |

Train accuracy pinned at 1.0 with a rising validation loss is memorisation of
all 50k images, not undertraining, so more epochs would not have helped. 86%
is also precisely where a 14.7M-parameter VGG16 lands on CIFAR-10 with no
augmentation.

**This is inconsistent with their own table.** SPEAR reports 91.2-92.5% for
pruned VGG16 and SCA's baseline is 91.14%; none of those is reachable by
pruning an 86% baseline. So `"For static datasets, no data augmentation is
applied"` cannot mean literally none. Taken as the field's usual convention:
no augmentation *beyond* `RandomCrop(32, padding=4)` and horizontal flip,
with the term reserved for RandAugment / Cutout / Mixup. Only that pair is
restored; RandAugment, colour jitter and random erasing stay off.

Same class of problem as DPAP's first attempt (91.96% vs 94.54%, also an
augmentation mismatch), and the same resolution: the paper's words
under-describe what was run.

**Write-up note:** report the 86.09% run as the evidence for this reading
rather than silently using crop+flip. It is a real finding about the paper's
reproducibility, and it is the kind of thing an examiner will ask about.

### Known deviations that remain

Record these in the dissertation, as with DPAP's list in section 3.

- **Standard crop and flip are used despite the paper saying no augmentation.**
  Evidence and reasoning immediately above. This is an assumption, not a
  replication.

- **The pruning stages do not use TET.** `pruning_loss_type="spike_rate_ce"`,
  so gate training *and the fine-tune* run under cross-entropy while only the
  pretrain uses TET. SPEAR fine-tunes "in the same configuration as training",
  so this is a real departure. Rationale, same as DPAP's: a replication's job
  is to reproduce the paper's baseline, the pruning criterion under test is
  ours, and TET is a training-dynamics loss aimed at flatter minima whose
  interaction with the gate mechanism has never been characterised. Accuracy
  is loss-independent, so the comparison against their published pruned row
  still holds.
- **The fine-tune trains on 45k, the pretrain on 50k.** `val_fraction=0.0`
  gives the pretrain their protocol; `pruning_val_fraction=0.1` then holds out
  a genuine validation split for the pruning stages so no decision of ours is
  made on the test set. SPEAR's fine-tune presumably saw all 50k.
- **`val_fraction=0.0` makes best-checkpoint selection test-set-informed** for
  the pretrain, exactly as for DPAP. This matches their protocol but makes the
  baseline a reproduction of that protocol rather than a clean generalisation
  estimate. **Must be stated in the dissertation, not buried.**
- **No unpruned baseline to match**, unlike DPAP's 94.54%. See above.
- **The VGG16 specification is assumed**, so a parameter-count mismatch
  against theirs cannot be detected: they publish params only as a percentage
  of their own unpruned model.
- **`losses.tet_loss`'s `lam` and `phi` are not reachable from the config.**
  Task losses are looked up as `fn(out_rec, targets)`, so the defaults
  (0.05 / 1.0) are the only values that can be used. `phi` is meant to track
  `SNNConfig.threshold`; it is an independent literal, so changing the
  threshold would silently desync it. Fine at SPEAR's threshold of 1.0, which
  `tests/test_spear.py` now asserts.

---

## Summary of what each replication costs us

| | Chowdhury | SCA | DPAP | SPEAR |
|---|---|---|---|---|
| New architecture | yes (8-conv VGG9) | yes (VGG16) | yes (6Conv2FC) | yes (VGG16) |
| Reuses existing direct-surrogate training | no | yes | yes | yes |
| New encoding | Poisson | no | no | no |
| New pooling | average | no | no | no |
| New loss | no | [UNKNOWN] | yes (UnilateralMse) | yes (TET) |
| New neuron behaviour | per-layer thresholds | no | learnable tau (PLIF) | hard reset |
| Whole new training stage | yes (ANN pretrain + threshold balance + convert) | no | no | no |
| Setup fully documented? | **yes** | **no** (optimizer/lr/loss unknown) | yes, via code | **mostly** (no arch spec, no baseline) |

Ironically the paper whose architecture name matched ours (Chowdhury's
"VGG9") is the most expensive to replicate, and the one with the cleanest
published numbers (SCA) is the one whose training recipe we cannot fully
recover.
