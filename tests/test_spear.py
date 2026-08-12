"""
CPU smoke test for the SPEAR replication (Xie et al., arXiv 2507.02945):
the TET task loss, the analog output readout it requires, and the
get_spear_repl_config() transcription. No CIFAR-10 or GPU needed.

Run before any CSF3 submission, alongside tests/test_vggstyle.py,
tests/test_ranked_pruning.py and tests/test_synops.py. Activate the venv
first -- these fail with ModuleNotFoundError otherwise while sbatch proceeds
regardless, which has happened once already.

Checks:
1. ArchConfig.output_readout validates, and defaults to "spikes" so every
   pre-existing experiment is untouched.
2. "current" readout returns fc_out's analog pre-synaptic output, of the same
   shape as the spike readout, and is genuinely non-binary.
3. The readout survives the physical rebuild -- a pruned SPEAR network reads
   its output the same way the network it was pruned from did.
4. tet_loss implements Deng et al.'s Eqs. 9/12/13 exactly, verified against a
   direct per-timestep reference implementation.
5. get_spear_repl_config() matches docs/replication_targets.md section 4 row
   for row, and the model it describes actually builds and forwards.
6. The T=4 tie problem the "current" readout exists to avoid is real, and is
   in fact absent under "current".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from bayesian_layers import set_bayesian_mode
from config import ArchConfig, BayesianConfig, SNNConfig, get_spear_repl_config
from losses import get_task_loss, tet_loss
from metrics import count_parameters
from models import VGGStyleSNN, build_model
from pruning import prune_vgg_style, uniform_ratio_plan


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(f"FAILED: {name}")


BAY = BayesianConfig()


def main():
    torch.manual_seed(0)

    # --- 1. output_readout validation and default ---
    check("output_readout defaults to 'spikes'", ArchConfig().output_readout == "spikes")
    rejected = False
    try:
        ArchConfig(output_readout="membrane")
    except ValueError:
        rejected = True
    check("invalid output_readout is rejected", rejected)

    # A small stand-in for the real VGG16 so the test stays fast: same
    # structural features that matter here (BatchNorm, no hidden fc, T=4).
    small = dict(conv_spec=[8, "M", 16, "M"], fc_hidden=[], norm_type="batch", input_size=8)
    snn4 = SNNConfig(num_steps=4, beta=0.5, threshold=1.0, reset_mechanism="zero")
    x = torch.randn(4, 3, 8, 8)

    spike_model = VGGStyleSNN(ArchConfig(**small), snn4, BAY)
    cur_model = VGGStyleSNN(ArchConfig(**small, output_readout="current"), snn4, BAY)
    cur_model.load_state_dict(spike_model.state_dict())

    # --- 2. "current" returns analog values of the same shape ---
    with torch.no_grad():
        spk_out, cur_out = spike_model(x), cur_model(x)
    check("current readout keeps [T, B, C] shape", spk_out.shape == cur_out.shape == (4, 4, 10))
    check("spike readout is binary", bool(((spk_out == 0) | (spk_out == 1)).all()))
    check("current readout is not binary", not bool(((cur_out == 0) | (cur_out == 1)).all()))
    # The readout must not change the weights: same state_dict, so skipping
    # lif_out is the only difference between the two forwards.
    check("readout adds no parameters",
          count_parameters(spike_model, exclude_gates=True)
          == count_parameters(cur_model, exclude_gates=True))

    # --- 3. the readout survives the physical rebuild ---
    set_bayesian_mode(cur_model, True)
    plan = uniform_ratio_plan(cur_model, keep_fraction=0.5, min_keep=1)
    pruned = prune_vgg_style(cur_model, plan, snn4)
    with torch.no_grad():
        pruned_out = pruned(x)
    check("pruned model keeps [T, B, C] shape", pruned_out.shape == (4, 4, 10))
    check("pruned model inherits the current readout",
          pruned.output_readout == "current"
          and not bool(((pruned_out == 0) | (pruned_out == 1)).all()))

    # --- 4. tet_loss matches Deng et al. Eqs. 9 / 12 / 13 ---
    out = torch.randn(4, 6, 10)
    targets = torch.randint(0, 10, (6,))
    lam, phi = 0.05, 1.0
    # Reference: the equations written out literally, one timestep at a time.
    ref_ce = sum(F.cross_entropy(out[t], targets) for t in range(4)) / 4
    ref_mse = sum(F.mse_loss(out[t], torch.full_like(out[t], phi)) for t in range(4)) / 4
    ref = (1 - lam) * ref_ce + lam * ref_mse
    check("tet_loss == per-timestep reference",
          torch.allclose(tet_loss(out, targets), ref, atol=1e-6))
    check("tet_loss registered as 'tet'", get_task_loss("tet") is tet_loss)
    # lam=0 must collapse to the plain per-timestep cross-entropy, and lam=1
    # to the pure MSE regulariser -- guards the (1-lam)/lam wiring.
    check("tet_loss at lam=0 is per-timestep CE",
          torch.allclose(tet_loss(out, targets, lam=0.0), ref_ce, atol=1e-6))
    check("tet_loss at lam=1 is the MSE term",
          torch.allclose(tet_loss(out, targets, lam=1.0), ref_mse, atol=1e-6))
    # TET is not the same objective as averaging first, then taking CE once;
    # if these ever coincide the per-timestep form has been lost.
    check("tet_loss differs from mean-then-CE",
          not torch.allclose(tet_loss(out, targets, lam=0.0),
                             F.cross_entropy(out.mean(0), targets), atol=1e-4))

    # --- 5. get_spear_repl_config() vs docs/replication_targets.md section 4 ---
    cfg = get_spear_repl_config()
    check("T = 4", cfg.snn.num_steps == 4)
    check("beta = 0.5 (tau=2.0, no input decay)", cfg.snn.beta == 0.5)
    check("threshold = 1.0", cfg.snn.threshold == 1.0)
    check("hard reset", cfg.snn.reset_mechanism == "zero")
    check("fixed tau, not learned", cfg.snn.learn_beta is False)
    check("arctan surrogate", cfg.snn.spike_grad == "atan")
    check("TET task loss", cfg.loss_type == "tet")
    check("pruning phases use cross-entropy", cfg.pruning_loss() == "spike_rate_ce")
    check("SGD optimizer", cfg.train.optimizer == "sgd")
    check("max lr 0.1", cfg.train.lr == 0.1)
    check("weight decay 5e-5", cfg.train.weight_decay == 5e-5)
    check("210 epochs", cfg.train.epochs == 210)
    check("10 warm-up epochs then cosine",
          cfg.train.lr_warmup_epochs == 10 and cfg.train.lr_scheduler == "cosine_warmup")
    # Standard crop+flip only. The literal "no data augmentation" reading was
    # tried and produced train_acc 1.0 / test 86.09%, 5pp under the published
    # reference; see the deviation note in get_spear_repl_config.
    check("standard crop and flip restored",
          cfg.data.random_crop_padding == 4 and cfg.data.horizontal_flip_prob == 0.5)
    check("no augmentation beyond crop and flip",
          cfg.data.rand_augment == "" and cfg.data.color_jitter == 0.0
          and cfg.data.random_erasing_prob == 0.0)
    check("finetune matches the training recipe (210 ep, warm-up, lr 0.1)",
          cfg.finetune.epochs == 210 and cfg.finetune.lr_warmup_epochs == 10
          and cfg.finetune.lr == 0.1 and cfg.finetune.optimizer == "sgd")
    check("reuses the saved 90.62% baseline", cfg.reuse_pretrained is True)

    # Standard CIFAR VGG16: 13 convs, 5 pools, 32px -> 1px, flatten 512.
    check("13 conv layers", len(cfg.arch.conv_channels()) == 13)
    check("5 pooling stages", cfg.arch.num_pools() == 5)
    check("no hidden fc layers", cfg.arch.fc_hidden == [])
    check("flatten_dim = 512", cfg.arch.flatten_dim() == 512)
    check("BatchNorm present", cfg.arch.norm_type == "batch")
    check("current readout", cfg.arch.output_readout == "current")

    # The real thing must build and forward. build_model also runs
    # assert_gate_after_norm, so this is the check that every conv gate is
    # deferred to after its BatchNorm.
    model = build_model("spear_repl", cfg.snn, cfg.bayesian, arch_cfg=cfg.arch)
    check("every conv gate deferred past its BatchNorm",
          all(c.defer_gate for c in model.conv_layers))
    with torch.no_grad():
        real_out = model(torch.randn(2, 3, 32, 32))
    check("VGG16 forwards to [4, 2, 10]", real_out.shape == (4, 2, 10))
    check("VGG16 output is analog", not bool(((real_out == 0) | (real_out == 1)).all()))

    # --- 6. the tie problem the current readout exists to avoid ---
    # This is a property of binary spikes at T=4, not of any one network, so
    # it is demonstrated on synthetic spike trains at a realistic batch size.
    # Doing it on the untrained toy model above would pass vacuously: an
    # untrained net frequently fires nothing at all, making every summed count
    # identically zero, which satisfies both "few levels" and "ties" for the
    # wrong reason and would pass at any T.
    torch.manual_seed(1)
    synth_spikes = (torch.rand(4, 512, 10) < 0.3).float()
    sums = synth_spikes.sum(dim=0)
    n_levels = len(torch.unique(sums))
    top2 = sums.topk(2, dim=1).values
    tie_rate = (top2[:, 0] == top2[:, 1]).float().mean().item()
    check(f"summed spikes at T=4 take at most 5 levels (saw {n_levels})",
          n_levels <= 5 and n_levels > 1)
    check(f"spike argmax ties on most examples (saw {tie_rate:.0%})", tie_rate > 0.5)

    synth_cur = torch.randn(4, 512, 10)
    cur_top2 = synth_cur.sum(dim=0).topk(2, dim=1).values
    cur_tie_rate = (cur_top2[:, 0] == cur_top2[:, 1]).float().mean().item()
    check(f"current argmax does not tie (saw {cur_tie_rate:.0%})", cur_tie_rate == 0.0)
    # The real model's spike readout must actually exhibit the quantisation.
    check("model spike readout is integer-valued when summed",
          bool((spk_out.sum(dim=0) == spk_out.sum(dim=0).round()).all()))

    # --- 7. guards that must fail loudly ---
    # lif_out is skipped under "current", so a learnable decay there would be
    # a parameter that never receives a gradient.
    orphaned = False
    try:
        VGGStyleSNN(ArchConfig(**small, output_readout="current"),
                    SNNConfig(num_steps=4, learn_beta=True), BAY)
    except ValueError:
        orphaned = True
    check("learn_beta + current readout is rejected", orphaned)
    check("learn_beta + spikes readout still allowed",
          VGGStyleSNN(ArchConfig(**small), SNNConfig(num_steps=4, learn_beta=True),
                      BAY) is not None)

    # 'cosine_warmup' with no warm-up silently degenerated to a plain cosine,
    # so a call site that forgot to forward the argument started cold at the
    # full LR. Five call sites had. It must now raise.
    from train import build_optimizer, build_scheduler
    tiny = torch.nn.Linear(2, 2)
    degenerated = False
    try:
        build_scheduler(build_optimizer(tiny, "sgd", 0.1, 0.0), "cosine_warmup",
                        epochs=210, warmup_epochs=0)
    except ValueError:
        degenerated = True
    check("cosine_warmup without a warm-up is rejected", degenerated)

    # phi is meant to be V_th but is an independent literal in tet_loss, so
    # pin the coupling: if SPEAR's threshold ever moves, this trips.
    check("tet phi default still equals SPEAR's threshold", cfg.snn.threshold == 1.0)

    # --- 8. config rows the earlier checks did not cover ---
    check("normalisation std assumption",
          cfg.data.normalize_std == [0.2023, 0.1994, 0.2010])
    check("normalisation mean assumption",
          cfg.data.normalize_mean == [0.4914, 0.4822, 0.4465])
    check("batch size assumption", cfg.train.batch_size == 128)
    check("trains on all 50k, pruning stages hold out 10%",
          cfg.data.val_fraction == 0.0 and cfg.data.pruning_val_fraction == 0.1)
    check("cosine floor 0.0", cfg.train.min_lr == 0.0)
    check("direct encoding", cfg.arch.encoding == "direct")
    check("max pooling assumption pinned explicitly", cfg.arch.pool_type == "max")
    check("conv bias assumption pinned explicitly", cfg.arch.conv_bias is True)
    check("14,728,266 parameters", count_parameters(model, exclude_gates=True) == 14_728_266)

    print("\nAll SPEAR replication checks passed.")


if __name__ == "__main__":
    main()
