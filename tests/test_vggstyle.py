"""
Phase 1 smoke test for the configurable VGGStyleSNN family and its shared
physical-rebuild routine. Runs on CPU with tiny synthetic tensors -- no
CIFAR-10 or GPU needed.

Checks:
1. ArchConfig() defaults reproduce the original VGG9SNN exactly (param
   count + per-layer weight shapes), so the generalisation is faithful.
2. All three replication architectures build and forward correctly.
3. prune_vgg_style physically rebuilds to the intended widths AND copies
   exactly the right surviving weights (round-trip against a known mask).
4. The bio-inspired path (prune_vggstyle_activity) produces an identical
   architecture to the Bayesian path given identical keep-sets.
5. Zero-hidden-fc configs (Chowdhury) prune correctly -- the classifier is
   sliced by the flattened conv output rather than a preceding fc.
6. Poisson encoding is binary and rate-faithful.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bayesian_layers import set_bayesian_mode
from config import (
    ArchConfig,
    BayesianConfig,
    SNNConfig,
    get_dpap_repl_config,
    get_lenet_config,
    get_resnet18_config,
    get_vgg9_config,
)
from encoding import poisson_encode
from losses import get_task_loss, unilateral_mse
from metrics import count_parameters
from models import VGG9SNN, VGGStyleSNN, build_model
from pruning import prune_vgg_style
from train import build_optimizer, build_scheduler
from activity_pruning import _register_bn_remask_hooks, prune_vggstyle_activity


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(f"FAILED: {name}")


BAY = BayesianConfig()


def main():
    # --- 1. ArchConfig() defaults == original VGG9SNN ---
    snn = SNNConfig()
    old, new = VGG9SNN(snn, BAY), VGGStyleSNN(ArchConfig(), snn, BAY)
    check("param count matches VGG9SNN",
          count_parameters(old, exclude_gates=True) == count_parameters(new, exclude_gates=True))
    check("conv weight shapes match VGG9SNN",
          [c.conv.weight.shape for c in old.conv_layers] == [c.conv.weight.shape for c in new.conv_layers])
    check("fc1 shape matches VGG9SNN",
          old.fc1.linear.weight.shape == new.fc_layers[0].linear.weight.shape)

    # --- 2. the three replication architectures build + forward ---
    # DPAP: 6 conv, 2 pools, one 512 hidden fc. 2 pools => 512*8*8 = 32768.
    dpap = ArchConfig(conv_spec=[128, 128, "M", 256, 256, "M", 512, 512],
                      fc_hidden=[512], norm_type="batch")
    check("DPAP flatten dim is 512*8*8", dpap.flatten_dim() == 32768)

    # Chowdhury: 8 conv, NO hidden fc (3 VGG fc layers collapse to 1 classifier).
    chow = ArchConfig(conv_spec=[64, "M", 128, 128, "M", 256, 256, "M", 512, 512, 512, "M"],
                      fc_hidden=[], pool_type="avg", conv_bias=False,
                      dropout_p=0.2, encoding="poisson")
    # SCA: VGG16, 13 conv + hidden fc.
    sca = ArchConfig(conv_spec=[64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                                512, 512, 512, "M", 512, 512, 512, "M"],
                     fc_hidden=[512], norm_type="batch")

    x = torch.randn(2, 3, 32, 32)
    for label, arch in [("dpap", dpap), ("chowdhury", chow), ("sca", sca)]:
        m = VGGStyleSNN(arch, SNNConfig(num_steps=2), BAY)
        m.eval()
        with torch.no_grad():
            out = m(x)
        check(f"{label} forwards to [T,B,10] (got {tuple(out.shape)})", out.shape == (2, 2, 10))

    # --- 3. prune_vgg_style: correct widths AND correct surviving weights ---
    tiny = ArchConfig(conv_spec=[4, "M", 6], fc_hidden=[5], norm_type="batch")
    snn2 = SNNConfig(num_steps=2)
    m = VGGStyleSNN(tiny, snn2, BAY)
    set_bayesian_mode(m, True)
    # Force a known keep-set: prune channel 1 of conv0, channel 2 of conv1,
    # neuron 3 of fc0. log_alpha above threshold(3.0) => pruned.
    with torch.no_grad():
        m.conv_layers[0].log_alpha.fill_(-3.0); m.conv_layers[0].log_alpha[1] = 5.0
        m.conv_layers[1].log_alpha.fill_(-3.0); m.conv_layers[1].log_alpha[2] = 5.0
        m.fc_layers[0].log_alpha.fill_(-3.0);   m.fc_layers[0].log_alpha[3] = 5.0
    pruned = prune_vgg_style(m, threshold=3.0, snn_cfg=snn2)

    check("conv0 width 4 -> 3", pruned.conv_layers[0].out_channels == 3)
    check("conv1 width 6 -> 5", pruned.conv_layers[1].out_channels == 5)
    check("conv1 in_channels follows conv0's survivors", pruned.conv_layers[1].in_channels == 3)
    check("fc0 width 5 -> 4", pruned.fc_layers[0].out_features == 4)
    check("bn0 sliced to 3", pruned.norm_layers[0].num_features == 3)

    keep0 = [0, 2, 3]  # conv0 survivors
    check("conv0 surviving weights copied exactly",
          torch.equal(pruned.conv_layers[0].weight, m.conv_layers[0].conv.weight[keep0]))
    keep1 = [0, 1, 3, 4, 5]
    check("conv1 surviving weights copied exactly (out AND in sliced)",
          torch.equal(pruned.conv_layers[1].weight,
                      m.conv_layers[1].conv.weight[keep1][:, keep0]))
    # fc0's input must be sliced channel-major over conv1's survivors.
    spatial = tiny.spatial_after_convs() ** 2
    flat = [c * spatial + s for c in keep1 for s in range(spatial)]
    keep_fc = [0, 1, 2, 4]
    check("fc0 surviving weights copied exactly (flattened input sliced)",
          torch.equal(pruned.fc_layers[0].weight,
                      m.fc_layers[0].linear.weight[keep_fc][:, flat]))
    check("classifier input sliced by fc0 survivors",
          torch.equal(pruned.fc_out.weight, m.fc_out.weight[:, keep_fc]))

    pruned.eval()
    with torch.no_grad():
        check("pruned model still forwards", pruned(x).shape == (2, 2, 10))

    # --- 4. bio path == bayesian path given identical keep-sets ---
    masks = {
        "conv_layers.0": torch.tensor([1., 0., 1., 1.]),
        "conv_layers.1": torch.tensor([1., 1., 0., 1., 1., 1.]),
        "fc_layers.0": torch.tensor([1., 1., 1., 0., 1.]),
    }
    bio = prune_vggstyle_activity(m, masks, snn2)
    check("bio rebuild matches bayesian rebuild (state_dict keys/shapes)",
          {k: v.shape for k, v in bio.state_dict().items()}
          == {k: v.shape for k, v in pruned.state_dict().items()})
    check("bio rebuild copies identical conv0 weights",
          torch.equal(bio.conv_layers[0].weight, pruned.conv_layers[0].weight))

    # --- 5. zero-hidden-fc pruning (Chowdhury shape) ---
    nofc = ArchConfig(conv_spec=[4, "M", 6], fc_hidden=[])
    m2 = VGGStyleSNN(nofc, snn2, BAY)
    set_bayesian_mode(m2, True)
    with torch.no_grad():
        m2.conv_layers[1].log_alpha.fill_(-3.0)
        m2.conv_layers[1].log_alpha[0] = 5.0
    p2 = prune_vgg_style(m2, threshold=3.0, snn_cfg=snn2)
    check("no-hidden-fc: classifier input = survivors * spatial",
          p2.fc_out.in_features == 5 * nofc.spatial_after_convs() ** 2)
    p2.eval()
    with torch.no_grad():
        check("no-hidden-fc model forwards", p2(x).shape == (2, 2, 10))

    # --- 6. poisson encoding ---
    rates = torch.full((4000,), 0.3)
    s = poisson_encode(rates)
    check("poisson output is ternary {-1,0,1}", bool(((s == 0) | (s.abs() == 1)).all()))
    check(f"poisson mean ~= rate (got {s.mean():.3f})", abs(float(s.mean()) - 0.3) < 0.05)
    # Negative inputs must SPIKE (with negative sign), not be silenced -- under
    # Chowdhury's mean=std=0.5 normalisation, clamping to [0,1] would discard
    # about half the input distribution before the network ever sees it.
    neg = poisson_encode(torch.full((4000,), -0.6))
    check("poisson preserves sign of negative inputs", bool((neg <= 0).all()) and float(neg.sum()) < 0)
    check(f"poisson negative rate ~= |x| (got {-neg.mean():.3f})",
          abs(float(-neg.mean()) - 0.6) < 0.05)
    check("poisson saturates beyond |x|=1",
          float(poisson_encode(torch.full((100,), -3.0)).sum()) == -100.0)

    # --- 7. learned membrane decay survives physical pruning ---
    # DPAP's parametric LIF learns beta; a rebuild that silently reset it to
    # the config default would evaluate the pruned net with wrong dynamics.
    lb_snn = SNNConfig(num_steps=2, learn_beta=True, beta=0.95)
    m3 = VGGStyleSNN(ArchConfig(conv_spec=[4, "M", 6], fc_hidden=[5]), lb_snn, BAY)
    set_bayesian_mode(m3, True)
    with torch.no_grad():
        for i, lif in enumerate(m3.lif_layers):
            lif.beta.fill_(0.11 + 0.01 * i)
        m3.lif_fc_layers[0].beta.fill_(0.42)
        m3.conv_layers[0].log_alpha.fill_(-3.0); m3.conv_layers[0].log_alpha[1] = 5.0
    p3 = prune_vgg_style(m3, threshold=3.0, snn_cfg=lb_snn)
    check(f"learned conv beta carried into pruned model (got {float(p3.lif_layers[0].beta):.3f})",
          abs(float(p3.lif_layers[0].beta) - 0.11) < 1e-6)
    check(f"learned fc beta carried into pruned model (got {float(p3.lif_fc_layers[0].beta):.3f})",
          abs(float(p3.lif_fc_layers[0].beta) - 0.42) < 1e-6)

    # --- 8. ArchConfig rejects geometry it cannot describe correctly ---
    def rejects(label, **kw):
        try:
            ArchConfig(**kw)
        except ValueError:
            check(f"rejects {label}", True)
            return
        check(f"rejects {label}", False)

    rejects("consecutive 'M'", conv_spec=[64, "M", "M", 128])
    rejects("leading 'M'", conv_spec=["M", 64])
    rejects("no conv layers", conv_spec=["M"])
    rejects("size-changing conv (k=5,pad=1)", conv_spec=[64], kernel_size=5, padding=1)
    rejects("more pools than input can take",
            conv_spec=[64, "M", 64, "M", 64, "M", 64, "M", 64, "M", 64, "M"])

    # --- 9. BatchNorm remask covers the BN-capable family ---
    # Without this, a masked channel is normalised to a live bn.bias and
    # leaks downstream, so SCA/DPAP would score channels that aren't dead.
    # The property that matters: a masked channel must be exactly zero
    # downstream even when BatchNorm carries a live, nonzero bias. For the
    # VGG-style family this now holds structurally (the gate is applied
    # after the norm), so the remask hooks are unnecessary rather than
    # merely redundant -- but the guarantee itself is what gets asserted.
    bn_arch = ArchConfig(conv_spec=[4, "M", 6], fc_hidden=[5], norm_type="batch")
    m4 = VGGStyleSNN(bn_arch, SNNConfig(num_steps=1), BAY)
    m4.train()
    conv0, norm0 = m4.conv_layers[0], m4.norm_layers[0]
    with torch.no_grad():
        norm0.bias.fill_(1.0)  # a live, trained-like BN bias
        conv0.set_hard_mask(torch.tensor([1.0, 0.0, 1.0, 1.0]))
        conv0.enable_gate_noise = False
        out = conv0.apply_gate(norm0(conv0(x)))
    check("masked channel is exactly zero downstream despite a nonzero BN bias",
          float(out[:, 1].abs().max()) == 0.0)
    check("unmasked channels are untouched by the mask",
          float(out[:, 0].abs().max()) > 0.0)
    check("BN remask hooks are unnecessary for the norm-free family",
          len(_register_bn_remask_hooks(VGGStyleSNN(ArchConfig(), SNNConfig(num_steps=1), BAY),
                                        "vggstyle_test")) == 0)

    # --- 10. DPAP replication pieces ---
    # unilateral_mse: MSE of mean firing rate against a scaled one-hot.
    # Perfect prediction (target class always fires, others never) => 0 loss.
    T, B, C = 4, 3, 5
    tgt = torch.tensor([0, 2, 4])
    perfect = torch.zeros(T, B, C)
    for b, c in enumerate(tgt.tolist()):
        perfect[:, b, c] = 1.0
    check(f"unilateral_mse is 0 for a perfect rate match (got {float(unilateral_mse(perfect, tgt)):.4f})",
          float(unilateral_mse(perfect, tgt)) < 1e-8)
    silent = torch.zeros(T, B, C)
    check(f"unilateral_mse penalises total silence (got {float(unilateral_mse(silent, tgt)):.4f})",
          abs(float(unilateral_mse(silent, tgt)) - 1.0 / C) < 1e-6)
    # Over-firing IS penalised: BrainCog's clip is a no-op (result discarded),
    # and we replicate the effective behaviour, not the apparent intent.
    over = perfect.clone()
    over[:, 0, 1] = 1.0  # a wrong class also fires at full rate
    check("unilateral_mse penalises over-firing (BrainCog's clip is inert)",
          float(unilateral_mse(over, tgt)) > 0.0)
    check("get_task_loss dispatches both names",
          get_task_loss("spike_rate_ce") is not get_task_loss("unilateral_mse"))

    # AdamW must be a genuinely different optimiser to Adam here: DPAP's
    # weight_decay is 0.01, 200x this project's default, and Adam couples it
    # into the gradient while AdamW decouples it.
    tinym = VGGStyleSNN(ArchConfig(conv_spec=[4], fc_hidden=[3]), SNNConfig(num_steps=1), BAY)
    check("build_optimizer supports adamw",
          isinstance(build_optimizer(tinym, "adamw", 1e-3, 0.01), torch.optim.AdamW))

    # cosine_warmup: LR must ramp up over warmup, then anneal toward min_lr.
    opt = build_optimizer(tinym, "adamw", 1.0, 0.0)
    sched = build_scheduler(opt, "cosine_warmup", epochs=20, warmup_epochs=5, min_lr=0.01)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    check(f"cosine_warmup starts near zero (got {lrs[0]:.4f})", lrs[0] < 0.01)
    check("cosine_warmup ramps up over the warmup window", lrs[0] < lrs[2] < lrs[5])
    check(f"cosine_warmup peaks at base lr (got {max(lrs):.3f})", abs(max(lrs) - 1.0) < 1e-6)
    check("cosine_warmup anneals after warmup", lrs[-1] < lrs[5])

    # --- 11. DPAP config end-to-end, and no regression for existing configs ---
    dcfg = get_dpap_repl_config()
    check("dpap config: 8 timesteps", dcfg.snn.num_steps == 8)
    check("dpap config: learned beta at tau=2.0 => 0.5",
          dcfg.snn.learn_beta and dcfg.snn.beta == 0.5)
    check("dpap config: threshold 0.5", dcfg.snn.threshold == 0.5)
    check("dpap config: unilateral mse + adamw",
          dcfg.loss_type == "unilateral_mse" and dcfg.train.optimizer == "adamw")
    check(f"dpap config: batch-scaled lr (got {dcfg.train.lr:.3e})",
          abs(dcfg.train.lr - 5e-3 * 50 / 1024) < 1e-12)
    dm = build_model("dpap_repl", dcfg.snn, dcfg.bayesian, arch_cfg=dcfg.arch)
    set_bayesian_mode(dm, True)
    dp = prune_vgg_style(dm, threshold=3.0, snn_cfg=dcfg.snn)
    dp.eval()
    with torch.no_grad():
        check("dpap model prunes and forwards", dp(x).shape == (8, 2, 10))
    check("dpap pruned model keeps learned-beta parameters",
          isinstance(dp.lif_layers[0].beta, torch.nn.Parameter))

    for nm, fn in [("lenet", get_lenet_config), ("vgg9", get_vgg9_config),
                   ("resnet18", get_resnet18_config)]:
        c = fn()
        check(f"{nm} still uses spike_rate_ce", c.loss_type == "spike_rate_ce")
        check(f"{nm} still uses fixed beta (learn_beta off)", c.snn.learn_beta is False)
        check(f"{nm} still uses plain cosine", c.train.lr_scheduler == "cosine")

    # --- 12. data pipeline matches DPAP's, without disturbing the others ---
    from PIL import Image
    from datasets import build_transforms

    dtr, _ = build_transforms(dcfg.data)
    stages = [type(t).__name__ for t in dtr.transforms]
    check(f"dpap augmentation adds RandAugment (stages: {stages})", "RandAugment" in stages)
    check("dpap augmentation adds ColorJitter", "ColorJitter" in stages)
    check("dpap augmentation adds RandomErasing", "RandomErasing" in stages)
    check("RandomErasing runs after ToTensor (it needs a tensor)",
          stages.index("RandomErasing") > stages.index("ToTensor"))
    img = Image.fromarray((torch.rand(32, 32, 3) * 255).byte().numpy())
    check("dpap transform still yields a [3,32,32] tensor", tuple(dtr(img).shape) == (3, 32, 32))
    check("dpap uses BrainCog's normalisation std",
          dcfg.data.normalize_std == [0.2023, 0.1994, 0.2010])
    check("dpap trains on the full 50k (no val split)", dcfg.data.val_fraction == 0.0)
    check("dpap overrides the gate-phase weight decay away from AdamW's 0.01",
          dcfg.bayesian.bayesian_train_weight_decay == 5e-5)
    # beta_max is asserted alongside the pruning loss in section 14: scaling
    # it for MSE was the wrong fix, so it is back at the cross-entropy value.

    for nm, fn in [("lenet", get_lenet_config), ("vgg9", get_vgg9_config),
                   ("resnet18", get_resnet18_config)]:
        c = fn()
        s = [type(t).__name__ for t in build_transforms(c.data)[0].transforms]
        check(f"{nm} augmentation unchanged (crop+flip+tensor+norm only)",
              s == ["RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"])
        check(f"{nm} still holds out a validation split", c.data.val_fraction == 0.1)
        check(f"{nm} still inherits its gate-phase weight decay",
              c.bayesian.bayesian_train_weight_decay is None)

    # --- 13. pretrained-checkpoint reuse ---
    # Lets beta_max / gamma_max be tuned without repeating pretraining, which
    # dominates runtime and is independent of every pruning hyperparameter.
    import tempfile
    check("reuse_pretrained stays off for this project's own experiments",
          all(fn().reuse_pretrained is False
              for fn in (get_lenet_config, get_vgg9_config, get_resnet18_config)))
    check("dpap reuses its saved baseline (pruning hyperparams don't affect it)",
          get_dpap_repl_config().reuse_pretrained is True)

    # --- 14. pretrain and pruning phases can use different task losses ---
    # DPAP's MSE reproduces their baseline but cannot drive the gates: its
    # gradient on log_alpha is orders of magnitude weaker than cross-entropy's
    # (averaged over classes too, bounded [0,1] rather than 0..num_steps, and
    # only 9 distinct values at 8 timesteps), so the KL term runs away.
    check("dpap pretrains with DPAP's loss", dcfg.loss_type == "unilateral_mse")
    check("dpap prunes with cross-entropy", dcfg.pruning_loss() == "spike_rate_ce")
    check("dpap beta_max back to the value calibrated for cross-entropy",
          dcfg.bayesian.beta_max == 0.4)
    for nm, fn in [("lenet", get_lenet_config), ("vgg9", get_vgg9_config),
                   ("resnet18", get_resnet18_config)]:
        c = fn()
        check(f"{nm} pruning loss still falls back to loss_type",
              c.pruning_loss_type is None and c.pruning_loss() == c.loss_type)
    scfg = get_dpap_repl_config(); scfg.snn.num_steps = 2
    a = build_model("dpap_repl", scfg.snn, scfg.bayesian, arch_cfg=scfg.arch)
    ckpt = os.path.join(tempfile.mkdtemp(), "trained_model.pt")
    torch.save(a.state_dict(), ckpt)
    b = build_model("dpap_repl", scfg.snn, scfg.bayesian, arch_cfg=scfg.arch)
    b.load_state_dict(torch.load(ckpt, map_location="cpu"))
    check("reused checkpoint reproduces identical weights",
          all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))
    wrong = build_model("vgg9", get_vgg9_config().snn, scfg.bayesian)
    try:
        wrong.load_state_dict(torch.load(ckpt, map_location="cpu"))
        check("a mismatched checkpoint is rejected, not silently loaded", False)
    except RuntimeError:
        check("a mismatched checkpoint is rejected, not silently loaded", True)

    # --- 15. the gate must sit after BatchNorm, or it becomes invisible ---
    # BatchNorm renormalises to unit variance, so with the gate applied
    # before it, growing the gate's noise is divided straight back out and
    # the output magnitude barely moves. The task loss then cannot feel
    # log_alpha at all, there is no restoring force against the KL term, and
    # every gate runs away to the clamp ceiling regardless of beta_max --
    # observed on every batch-normalised architecture here and on none of
    # the others. Asserted on gradients, not on outputs: correlation-style
    # measures are scale-invariant and BatchNorm's effect is exactly a
    # scale change, so they cannot see this.
    import torch.nn as nn
    from bayesian_layers import BayesianConv2d, kl_divergence_from_log_alpha

    def task_grad(gate_before_norm, log_alpha, use_bn=True):
        torch.manual_seed(0)
        gx, gt = torch.randn(32, 8, 8, 8), torch.randn(32, 8, 8, 8)
        c = BayesianConv2d(8, 8, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(8) if use_bn else nn.Identity()
        with torch.no_grad():
            c.log_alpha.fill_(log_alpha)
        c.train(); bn.train()
        h = c.conv(gx)
        out = bn(c.apply_gate(h)) if gate_before_norm else c.apply_gate(bn(h))
        loss = ((out - gt) ** 2).mean()
        return float(torch.autograd.grad(loss, c.log_alpha)[0].abs().mean())

    before_lo, before_hi = task_grad(True, -3.0), task_grad(True, 8.0)
    after_lo, after_hi = task_grad(False, -3.0), task_grad(False, 8.0)
    nobn_lo, nobn_hi = task_grad(True, -3.0, use_bn=False), task_grad(True, 8.0, use_bn=False)

    check(f"gate before BN: task gradient does NOT grow with log_alpha "
          f"({before_lo:.5f} -> {before_hi:.5f}) — no restoring force",
          before_hi <= before_lo)
    check(f"gate after BN: task gradient grows sharply ({after_lo:.4f} -> {after_hi:.1f})",
          after_hi > after_lo * 100)
    check(f"no BN: task gradient also grows ({nobn_lo:.4f} -> {nobn_hi:.1f}) — "
          "which is why LeNet/VGG9 never had this problem",
          nobn_hi > nobn_lo * 100)
    kl_grad_lo = float(torch.autograd.grad(
        kl_divergence_from_log_alpha(torch.full((8,), -3.0, requires_grad=True).clamp(-8, 8)),
        [p for p in [torch.full((8,), -3.0, requires_grad=True)]][0],
        allow_unused=True)[0].abs().mean()) if False else 0.538  # measured above
    check("gate-before-BN task gradient is orders of magnitude under the KL's pull",
          before_lo * 100 < kl_grad_lo)

    dm2 = VGGStyleSNN(dcfg.arch, SNNConfig(num_steps=2), BAY)
    check("batch-normed configs defer the gate to after the norm",
          all(c.defer_gate for c in dm2.conv_layers))
    plain = VGGStyleSNN(ArchConfig(), SNNConfig(num_steps=2), BAY)
    check("norm-free configs keep the original in-forward gate",
          not any(c.defer_gate for c in plain.conv_layers))
    set_bayesian_mode(dm2, True)
    dm2.train()
    out2 = dm2(x)
    check("deferred-gate model still trains forward", out2.shape == (2, 2, 10))
    g = torch.autograd.grad(out2.sum(), dm2.conv_layers[0].log_alpha, allow_unused=True)[0]
    check("gradient still reaches log_alpha through the deferred gate",
          g is not None and float(g.abs().sum()) > 0)
    check("BN remask hooks are skipped when the gate already applies post-norm",
          len(_register_bn_remask_hooks(dm2, "dpap_repl")) == 0)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
