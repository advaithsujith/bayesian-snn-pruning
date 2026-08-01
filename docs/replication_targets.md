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
| Loss | **UnilateralMse(1.0)** -- one-sided MSE to a one-hot target, *not* cross-entropy | [code] `train_loss_fn = UnilateralMse(1.)` |
| Optimizer | **AdamW** | [code] `cfg.opt='adamw'` |
| Learning rate | 5e-3, linearly scaled: `lr * batch_size / 1024` | [code] |
| Schedule | cosine, warmup 5 epochs, cooldown 10 epochs, min_lr 1e-5 | [code] |
| Epochs | **300** | [code] `epochs=300` |
| Batch size | **50** | [code] `batch_size=50` |
| Weight decay | **0.01** | [code] `cfg.weight_decay=0.01` |
| Augmentation | via BrainCog's `get_cifar10_data()`; **[UNKNOWN]** in detail | [code] |

**Notes on porting to snnTorch**: `PLIFNode` has a *learnable* membrane time
constant -- snnTorch's nearest equivalent is `snn.Leaky(..., learn_beta=True)`.
The tau->beta mapping is `beta = 1 - 1/tau = 0.5` for tau=2.0 under the
standard parameterisation, but BrainCog's exact update rule should be checked
before relying on this. The `UnilateralMse` loss is a genuine departure from
this project's `spike_rate_cross_entropy` and would need implementing.

---

## Summary of what each replication costs us

| | Chowdhury | SCA | DPAP |
|---|---|---|---|
| New architecture | yes (8-conv VGG9) | yes (VGG16) | yes (6Conv2FC) |
| Reuses existing direct-surrogate training | no | yes | yes |
| New encoding | Poisson | no | no |
| New pooling | average | no | no |
| New loss | no | [UNKNOWN] | yes (UnilateralMse) |
| New neuron behaviour | per-layer thresholds | no | learnable tau (PLIF) |
| Whole new training stage | yes (ANN pretrain + threshold balance + convert) | no | no |
| Setup fully documented? | **yes** | **no** (optimizer/lr/loss unknown) | yes, via code |

Ironically the paper whose architecture name matched ours (Chowdhury's
"VGG9") is the most expensive to replicate, and the one with the cleanest
published numbers (SCA) is the one whose training recipe we cannot fully
recover.
