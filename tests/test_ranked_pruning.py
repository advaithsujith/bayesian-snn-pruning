"""
CPU checks for the changes made on 2026-08-03 in response to
docs/fresh_review_2026-08-03.md. No GPU, no CIFAR-10, seconds to run.

Covers:
1.  Ranked target-sparsity pruning: exact widths, matched to the bio side's
    rounding, min_keep floor, parameter-target bisection.
2.  KeepPlan reporting agrees with what the rebuild actually builds.
3.  The conv gate is per-channel, not per-spatial-position.
4.  total_kl / gate noise exclude structurally non-prunable layers.
5.  ResNet18 applies its gate after BatchNorm, and the assertion catches it
    if that ever regresses.
6.  The gate-pressure diagnostic measures what it claims to.

Run before any CSF3 submission, alongside tests/test_vggstyle.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from activity_pruning import select_keep_mask
from bayesian_layers import (
    BayesianConv2d,
    collect_prunable_bayesian_layers,
    set_bayesian_mode,
    total_kl,
)
from config import ArchConfig, BayesianConfig, SNNConfig, get_lenet_config
from metrics import count_parameters
from models import VGGStyleSNN, assert_gate_after_norm, build_model
from pruning import (
    global_ratio_plan,
    keep_fraction_for_param_target,
    param_target_plan,
    prune_model,
    remaining_structures_report,
    threshold_plan,
    uniform_ratio_plan,
)
from train import gate_pressure_diagnostic


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(f"FAILED: {name}")


BAY = BayesianConfig()
SNN2 = SNNConfig(num_steps=2)


def main():
    # ------------------------------------------------------------------
    # 1. The conv gate is structured: one draw per (example, channel).
    # ------------------------------------------------------------------
    conv = BayesianConv2d(3, 4, 3, padding=1)
    conv.train()
    with torch.no_grad():
        conv.log_alpha.fill_(2.0)
        conv.conv.weight.zero_()
        conv.conv.bias.fill_(1.0)  # pre-gate output is a constant 1 everywhere
    out = conv(torch.zeros(2, 3, 8, 8))
    out = out.detach()
    per_channel_spread = max(
        float(out[b, c].std()) for b in range(2) for c in range(4)
    )
    check("conv gate is constant within a channel", per_channel_spread < 1e-6)
    check(
        "conv gate still differs between channels",
        float(out[0, :, 0, 0].std()) > 1e-3,
    )
    check(
        "conv gate still differs between examples",
        float((out[0, :, 0, 0] - out[1, :, 0, 0]).abs().max()) > 1e-3,
    )

    # ------------------------------------------------------------------
    # 2. Ranked plans keep exactly the lowest-log_alpha units.
    # ------------------------------------------------------------------
    arch = ArchConfig(conv_spec=[8, "M", 10], fc_hidden=[6])
    m = VGGStyleSNN(arch, SNN2, BAY)
    set_bayesian_mode(m, True)
    with torch.no_grad():
        # Distinct, deliberately unsorted values so index order cannot pass
        # for a ranking by accident.
        m.conv_layers[0].log_alpha.copy_(torch.tensor([3.0, -1.0, 5.0, 0.0, -4.0, 2.0, 7.0, 1.0]))
        m.conv_layers[1].log_alpha.copy_(torch.arange(10, dtype=torch.float).flip(0))
        m.fc_layers[0].log_alpha.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0]))

    plan = uniform_ratio_plan(m, 0.5)
    check(
        "uniform_ratio keeps the 4 lowest log_alpha of conv0",
        plan.indices(m.conv_layers[0]).tolist() == [1, 3, 4, 7],
    )
    check(
        "uniform_ratio keeps the 5 lowest log_alpha of conv1",
        plan.indices(m.conv_layers[1]).tolist() == [5, 6, 7, 8, 9],
    )
    check(
        "uniform_ratio keeps the 3 lowest log_alpha of fc0",
        plan.indices(m.fc_layers[0]).tolist() == [1, 2, 4],
    )

    # ------------------------------------------------------------------
    # 3. Widths agree exactly with the bio side's rounding, so a matched
    #    keep_fraction really does give matched architectures.
    # ------------------------------------------------------------------
    ok = True
    for f in [0.1, 0.15, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1.0]:
        p = uniform_ratio_plan(m, f)
        for layer in collect_prunable_bayesian_layers(m):
            n = layer.log_alpha.numel()
            bio_count = int(select_keep_mask(torch.randn(n), f).sum())
            if int(p.indices(layer).numel()) != bio_count:
                ok = False
    check("ranked widths match activity_pruning.select_keep_mask at every fraction", ok)

    # ------------------------------------------------------------------
    # 4. The rebuild really is that wide, and the report agrees with it.
    # ------------------------------------------------------------------
    plan = uniform_ratio_plan(m, 0.5)
    pruned = prune_model(m, "vggstyle_test", plan, SNN2)
    check("rebuilt conv0 width == plan", pruned.conv_layers[0].out_channels == 4)
    check("rebuilt conv1 width == plan", pruned.conv_layers[1].out_channels == 5)
    check("rebuilt conv1 in_channels follows conv0", pruned.conv_layers[1].in_channels == 4)
    check("rebuilt fc0 width == plan", pruned.fc_layers[0].out_features == 3)
    check(
        "rebuilt conv0 weights are exactly the kept rows",
        torch.equal(
            pruned.conv_layers[0].weight, m.conv_layers[0].conv.weight[[1, 3, 4, 7]]
        ),
    )
    report = remaining_structures_report(m, plan)
    check(
        "report agrees with the rebuild",
        report["conv_layers.0"]["remaining"] == 4
        and report["conv_layers.1"]["remaining"] == 5
        and report["fc_layers.0"]["remaining"] == 3,
    )

    # ------------------------------------------------------------------
    # 5. min_keep is a hard floor even when a global ranking would empty a
    #    layer, and the threshold path's "never zero a layer" rule is kept.
    # ------------------------------------------------------------------
    with torch.no_grad():
        m.conv_layers[0].log_alpha.fill_(9.0)  # worst layer in the network
        m.conv_layers[1].log_alpha.fill_(-9.0)
        m.fc_layers[0].log_alpha.fill_(-9.0)
    gp = global_ratio_plan(m, 0.5, min_keep=2)
    check(
        "global_ratio floors a would-be-emptied layer at min_keep",
        int(gp.indices(m.conv_layers[0]).numel()) == 2,
    )
    tp = threshold_plan(m, 3.0)
    check(
        "threshold path still keeps one unit rather than emptying a layer",
        int(tp.indices(m.conv_layers[0]).numel()) == 1,
    )
    check(
        "threshold path keeps every unit below the threshold",
        int(tp.indices(m.conv_layers[1]).numel()) == 10,
    )

    # ------------------------------------------------------------------
    # 6. Saturation is reported, because ranked pruning cannot detect it.
    # ------------------------------------------------------------------
    with torch.no_grad():
        for layer in collect_prunable_bayesian_layers(m):
            layer.log_alpha.fill_(8.0)
    sat = uniform_ratio_plan(m, 0.5).saturation_report(m)
    check("all-saturated gates are reported as such", sat["frac_saturated_high"] == 1.0)

    # ------------------------------------------------------------------
    # 7. Parameter targets are hit, and keep_fraction != parameter fraction.
    # ------------------------------------------------------------------
    cfg = get_lenet_config()
    lenet = build_model("lenet", cfg.snn, cfg.bayesian, num_classes=10)
    total = count_parameters(lenet, exclude_gates=True)
    for target in [25.0, 50.0, 90.0]:
        plan, achieved = param_target_plan(lenet, "lenet", cfg.snn, target)
        rebuilt = count_parameters(prune_model(lenet, "lenet", plan, cfg.snn), exclude_gates=True)
        realised = 100.0 * (1 - rebuilt / total)
        check(
            f"param_target {target:g}% lands within 1pp (got {realised:.2f}%)",
            abs(realised - target) < 1.0,
        )
        check(
            f"param_target {target:g}% report matches the rebuild",
            abs(realised - achieved) < 1e-6,
        )

    # LeNet's own Bayesian result sits at 27.74% pruned. The keep_fraction a
    # bio run needs to match it is ~0.84 -- not the ~0.72 estimated by eye,
    # which would have put the "matched" bio runs at 46% pruned.
    f27, pct27 = keep_fraction_for_param_target(lenet, "lenet", cfg.snn, 27.74)
    check(
        f"keep_fraction for LeNet's 27.74% is ~0.84, not ~0.72 (got {f27:.3f})",
        0.80 < f27 < 0.88,
    )
    # ...and it lands on 26.53%, not 27.74%, because a uniform keep_fraction
    # is quantised: LeNet's conv2 going 13 -> 14 channels moves fc1's input
    # by 14 * 5 * 5 columns, so parameter percentage jumps 30.8 -> 26.5 with
    # nothing available in between. On a 62K-parameter network an arbitrary
    # parameter target is simply not reachable this way, which is why the
    # matched-sparsity comparison should be stated at a shared
    # *keep_fraction* (exact by construction) with the parameter percentage
    # reported alongside, rather than the other way round.
    check(
        f"the closest achievable point is reported honestly (got {pct27:.2f}%)",
        abs(pct27 - 27.74) < 1.5,
    )
    neighbours = []
    for f in [f27 - 0.01, f27, f27 + 0.01]:
        p = uniform_ratio_plan(lenet, f)
        rebuilt = count_parameters(prune_model(lenet, "lenet", p, cfg.snn), exclude_gates=True)
        neighbours.append(abs(100.0 * (1 - rebuilt / total) - 27.74))
    check("bisection returned the closest achievable fraction", min(neighbours) == neighbours[1])

    # ------------------------------------------------------------------
    # 8. Non-prunable layers are excluded from the KL and from gate noise.
    # ------------------------------------------------------------------
    resnet = build_model("resnet18", SNN2, BAY, num_classes=10)
    set_bayesian_mode(resnet, True)
    noisy = [l for l in resnet.modules() if hasattr(l, "enable_gate_noise") and l.enable_gate_noise]
    prunable = collect_prunable_bayesian_layers(resnet)
    # 8 BasicBlocks: their conv1 is prunable, their conv2 and the stem are
    # not, so 8 of 17 gated layers -- but 1920 of 3904 gate *units*, i.e.
    # 50.8% of the KL was previously being spent on gates that can never be
    # removed.
    check(
        "gate noise is enabled on exactly the prunable layers",
        len(noisy) == len(prunable) and len(prunable) == 8,
    )
    gate_units = sum(l.log_alpha.numel() for l in prunable)
    all_units = sum(
        l.log_alpha.numel() for l in resnet.modules() if hasattr(l, "log_alpha")
    )
    check(f"prunable gate units are 1920 of 3904 (got {gate_units}/{all_units})",
          gate_units == 1920 and all_units == 3904)

    with torch.no_grad():
        for layer in resnet.modules():
            if hasattr(layer, "log_alpha"):
                layer.log_alpha.fill_(-3.0)
    kl_all_prunable = float(total_kl(resnet))
    with torch.no_grad():
        # Only a non-prunable layer changes: the KL must not notice.
        resnet.stem_conv.log_alpha.fill_(7.0)
    check(
        "total_kl ignores structurally non-prunable layers",
        abs(float(total_kl(resnet)) - kl_all_prunable) < 1e-6,
    )

    # ------------------------------------------------------------------
    # 9. ResNet18 gates after BatchNorm, and regressions are caught.
    # ------------------------------------------------------------------
    check(
        "every ResNet18 gated conv defers its gate past its BatchNorm",
        all(
            mod.defer_gate
            for mod in resnet.modules()
            if isinstance(mod, BayesianConv2d)
        ),
    )
    resnet.stage1[0].conv1.defer_gate = False
    caught = False
    try:
        assert_gate_after_norm(resnet)
    except ValueError:
        caught = True
    check("assert_gate_after_norm catches a gate placed before a norm", caught)
    resnet.stage1[0].conv1.defer_gate = True

    # The mechanism itself, isolated: a gate applied *before* a BatchNorm has
    # the variance it injects divided straight back out, so a downstream loss
    # barely feels log_alpha. Measured as a gradient, not a correlation --
    # correlation is scale-invariant and BatchNorm's effect *is* a scale
    # change, which is why an earlier correlation-based test wrongly cleared
    # this and the diagnosis was briefly abandoned.
    def downstream_grad(gate_first: bool) -> float:
        torch.manual_seed(0)
        gated = BayesianConv2d(3, 8, 3, padding=1)
        bn = torch.nn.BatchNorm2d(8)
        gated.train()
        bn.train()
        with torch.no_grad():
            gated.log_alpha.fill_(3.0)
        x = torch.randn(16, 3, 8, 8)
        torch.manual_seed(1)  # identical eps draw in both orderings
        h = gated.conv(x)
        out = bn(gated.apply_gate(h)) if gate_first else gated.apply_gate(bn(h))
        grad = torch.autograd.grad(out.square().mean(), gated.log_alpha)[0]
        return float(grad.abs().mean())

    gate_before_bn = downstream_grad(gate_first=True)
    gate_after_bn = downstream_grad(gate_first=False)
    check(
        f"a gate before BatchNorm is near-invisible to a downstream loss "
        f"(before={gate_before_bn:.3e}, after={gate_after_bn:.3e})",
        gate_after_bn > 10 * gate_before_bn,
    )

    # ------------------------------------------------------------------
    # 10. The gate-pressure diagnostic reports both sides per layer.
    # ------------------------------------------------------------------
    small = build_model("lenet", SNN2, BAY, num_classes=10)
    set_bayesian_mode(small, True)
    pressure = gate_pressure_diagnostic(
        small, torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)), beta=0.4
    )
    check("diagnostic covers every gated layer", len(pressure) == 4)
    check(
        "diagnostic reports a finite KL pressure everywhere",
        all(s["kl_grad_scaled"] > 0 for s in pressure.values()),
    )

    print("\nAll ranked-pruning checks passed.")


if __name__ == "__main__":
    main()
