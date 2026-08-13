"""
CPU checks for the SynOps-aware additions (2026-08-03): event-driven SynOps
measurement, per-unit SynOps costs, budgeted cost-aware selection, the
SynOps-budget Lagrangian loss term, and the split gate optimizer. No GPU,
no CIFAR-10, seconds to run.

Run before any CSF3 submission, alongside tests/test_ranked_pruning.py and
tests/test_vggstyle.py (activate the venv first).

The measurement checks use a network forced to fire every neuron on every
timestep (positive weights, near-zero threshold, all-ones input), because
that makes every quantity hand-computable: with all rates at 1, SynOps
must equal dense MACs exactly, and every per-unit cost reduces to closed
geometry. The numbers in section 3 are derived by hand in the comments --
they are NOT re-derived from the implementation's own formulas, which is
the point.
"""

import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bayesian_layers import (
    collect_prunable_bayesian_layers,
    set_bayesian_mode,
    total_expected_synops,
    total_kl,
    total_unit_cost,
)
from config import ArchConfig, BayesianConfig, SNNConfig
from losses import bayesian_snn_loss
from metrics import measure_synops, measure_synops_unit_costs, set_synops_unit_costs
from models import VGGStyleSNN, build_model
from pruning import plan_synops_fraction, synops_budget_plan
from train import build_gate_split_optimizers, run_training, train_one_epoch


def check(name, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(f"FAILED: {name}")


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


BAY = BayesianConfig()
CPU = torch.device("cpu")


def loader_of(images, batch_size=2):
    labels = torch.zeros(images.shape[0], dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def make_all_fire_model():
    """A tiny VGGStyleSNN in which every neuron fires on every timestep:
    all-positive weights and biases, threshold 0.01, all-ones input."""
    arch = ArchConfig(conv_spec=[2, "M", 3], fc_hidden=[4], input_size=8)
    snn = SNNConfig(num_steps=2, threshold=0.01)
    m = VGGStyleSNN(arch, snn, BAY)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(mod.weight.abs() + 0.05)
                if mod.bias is not None:
                    mod.bias.fill_(0.1)
    return m


def main():
    # ------------------------------------------------------------------
    # 1. measure_synops: exact on a single Linear with known input sparsity.
    # ------------------------------------------------------------------
    class OneLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return self.fc(x)

    one = OneLinear()
    # 3 nonzero of 8 input elements; dense = 4*3 = 12 MACs per sample.
    images = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    r = measure_synops(one, loader_of(images, batch_size=2), CPU, max_batches=1)
    check("linear dense MACs per sample = 12", close(r["dense_macs_per_sample"], 12.0))
    check("linear SynOps = frac * dense = (3/8)*24/2 = 4.5", close(r["synops_per_sample"], 4.5))
    check("linear input event frac = 3/8", close(r["per_layer"]["fc"]["input_event_frac"], 0.375))

    # ------------------------------------------------------------------
    # 2. measure_synops: exact on a conv, and totals sum over timesteps.
    # ------------------------------------------------------------------
    class OneConvTwice(nn.Module):
        """Calls its conv twice per forward, standing in for a 2-timestep
        SNN loop -- totals must come out summed over both calls."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x) + self.conv(x)

    twice = OneConvTwice()
    img = torch.zeros(1, 1, 4, 4)
    img[0, 0, :2, :] = 1.0  # 8 nonzero of 16 -> frac 0.5
    # dense per call: 1*9*2*16 = 288; two calls = 576/sample; synops = 288.
    r = measure_synops(twice, loader_of(img, batch_size=1), CPU, max_batches=1)
    check("conv dense MACs sum over calls = 576", close(r["dense_macs_per_sample"], 576.0))
    check("conv SynOps = 0.5 * dense = 288", close(r["synops_per_sample"], 288.0))

    # ------------------------------------------------------------------
    # 3. Per-unit SynOps costs, hand-derived on the all-fire network.
    #    Geometry: conv0 (2ch, 8x8, pool to 4x4) -> conv1 (3ch, 4x4)
    #    -> fc0 (4 units) -> fc_out (10), T=2, kernel 3x3, RGB input.
    #    With every rate at 1:
    #      conv0: compute 9*3*64*2 = 3456; downstream 4*4*2 events * 9*3 = 864 -> 4320
    #      conv1: compute 9*2*16*2 =  576; downstream 4*4*2 events * 4   = 128 ->  704
    #      fc0:   compute 48*2     =   96; downstream 2 events * 10      =  20 ->  116
    # ------------------------------------------------------------------
    m = make_all_fire_model()
    data = loader_of(torch.ones(4, 3, 8, 8), batch_size=2)
    costs = measure_synops_unit_costs(m, "tiny", data, CPU, max_batches=2)

    c0 = costs[m.conv_layers[0]]
    c1 = costs[m.conv_layers[1]]
    cf = costs[m.fc_layers[0]]
    check("conv0 cost vector has one entry per channel", c0.numel() == 2)
    check("conv0 unit cost = 4320", all(close(float(v), 4320.0) for v in c0))
    check("conv1 unit cost = 704", all(close(float(v), 704.0) for v in c1))
    check("fc0 unit cost = 116", all(close(float(v), 116.0) for v in cf))

    # With every rate at 1, SynOps must equal dense MACs exactly:
    # 2*3456 + 3*576 + 4*96 + 4*10*2 = 6912 + 1728 + 384 + 80 = 9104.
    r = measure_synops(m, data, CPU, max_batches=2)
    check("all-fire SynOps = dense MACs", close(r["synops_per_sample"], r["dense_macs_per_sample"]))
    check("all-fire SynOps = 9104", close(r["synops_per_sample"], 9104.0))

    # set_synops_unit_costs writes through to the buffers total_unit_cost reads.
    set_synops_unit_costs(m, costs)
    expected_total = 2 * 4320.0 + 3 * 704.0 + 4 * 116.0
    check("total_unit_cost sums the written buffers", close(total_unit_cost(m), expected_total))

    # ------------------------------------------------------------------
    # 3b. Valid-padding geometry (LeNet's 5x5 convs, padding 0): an average
    #     input event lands in k^2 * out_hw/in_hw kernel windows, NOT k^2 --
    #     found by the fresh-context review; the naive k^2 pricing overpriced
    #     conv1's downstream term by in_hw/out_hw = 196/100 ~ 2x. Hand
    #     numbers, all-fire, T=2, 32x32 RGB input:
    #       conv1 (6ch, 28x28, pool to 14x14):
    #         compute 25*3*784*2 = 117600
    #         downstream 14*14*2 events * (25*16*100/196) = 80000 -> 197600
    #       conv2 (16ch, 10x10, pool to 5x5):
    #         compute 25*6*100*2 = 30000; downstream 25*2 * 120 = 6000 -> 36000
    #       fc1 (120): compute 400*2 = 800; downstream 2*84 = 168 -> 968
    #       fc2 (84):  compute 120*2 = 240; downstream 2*10 =  20 -> 260
    # ------------------------------------------------------------------
    lenet = build_model("lenet", SNNConfig(num_steps=2, threshold=0.01), BAY)
    with torch.no_grad():
        for mod in lenet.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(mod.weight.abs() + 0.05)
                if mod.bias is not None:
                    mod.bias.fill_(0.1)
    ldata = loader_of(torch.ones(2, 3, 32, 32), batch_size=2)
    lcosts = measure_synops_unit_costs(lenet, "lenet", ldata, CPU, max_batches=1)
    check("lenet conv1 unit cost = 197600", all(close(float(v), 197600.0) for v in lcosts[lenet.conv1]))
    check("lenet conv2 unit cost = 36000", all(close(float(v), 36000.0) for v in lcosts[lenet.conv2]))
    check("lenet fc1 unit cost = 968", all(close(float(v), 968.0) for v in lcosts[lenet.fc1]))
    check("lenet fc2 unit cost = 260", all(close(float(v), 260.0) for v in lcosts[lenet.fc2]))
    lr_ = measure_synops(lenet, ldata, CPU, max_batches=1)
    check(
        "lenet all-fire SynOps = dense MACs (pricing consistency)",
        close(lr_["synops_per_sample"], lr_["dense_macs_per_sample"]),
    )

    # ------------------------------------------------------------------
    # 3c. SpikingResNet18: unit costs exist for exactly the prunable set
    #     (each block's conv1, whose sole consumer is that block's conv2;
    #     the residual addition happens on conv2's output, so it never
    #     touches the spikes conv1's units are billed for). Hand numbers,
    #     all-fire, T=2, 32x32 RGB input; stride-1 blocks keep their
    #     stage's spatial size, stride-2 stage openers halve it:
    #       stage1 conv1 (64 in, 32x32 out):
    #         compute 9*64*1024*2 = 1179648
    #         downstream 32*32*2 events * 9*64 = 1179648   -> 2359296
    #       stage2.0 conv1 (64 in, 16x16 out, stride 2):
    #         compute 9*64*256*2 = 294912
    #         downstream 16*16*2 events * 9*128 = 589824   ->  884736
    #       stage2.1 conv1 (128 in, 16x16 out):
    #         compute 9*128*256*2 = 589824; downstream 589824 -> 1179648
    #       stage4.1 conv1 (512 in, 4x4 out):
    #         compute 9*512*16*2 = 147456
    #         downstream 4*4*2 events * 9*512 = 147456     ->  294912
    # ------------------------------------------------------------------
    rn = build_model("resnet18", SNNConfig(num_steps=2, threshold=0.01), BAY)
    with torch.no_grad():
        for mod in rn.modules():
            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                mod.weight.copy_(mod.weight.abs() + 0.05)
                if mod.bias is not None:
                    mod.bias.fill_(0.1)
    rdata = loader_of(torch.ones(2, 3, 32, 32), batch_size=2)
    rcosts = measure_synops_unit_costs(rn, "resnet18", rdata, CPU, max_batches=1)
    check(
        "resnet18 costs cover exactly the prunable set",
        set(rcosts.keys()) == set(collect_prunable_bayesian_layers(rn)),
    )
    check(
        "resnet18 stage1 conv1 unit cost = 2359296",
        all(close(float(v), 2359296.0) for v in rcosts[rn.stage1[0].conv1]),
    )
    check(
        "resnet18 stage2.0 conv1 unit cost = 884736",
        all(close(float(v), 884736.0) for v in rcosts[rn.stage2[0].conv1]),
    )
    check(
        "resnet18 stage2.1 conv1 unit cost = 1179648",
        all(close(float(v), 1179648.0) for v in rcosts[rn.stage2[1].conv1]),
    )
    check(
        "resnet18 stage4.1 conv1 unit cost = 294912",
        all(close(float(v), 294912.0) for v in rcosts[rn.stage4[1].conv1]),
    )
    rr = measure_synops(rn, rdata, CPU, max_batches=1)
    check(
        "resnet18 all-fire SynOps = dense MACs (pricing consistency)",
        close(rr["synops_per_sample"], rr["dense_macs_per_sample"]),
    )

    # ------------------------------------------------------------------
    # 4. synops_budget_plan: the two ranking rules differ exactly as
    #    designed. Layer A: log_alpha [-5,-4,0,1], costs [1,1000,1,1];
    #    layers B/C: log_alpha -3, cost 1 each (4 units each).
    #    Budget 15 of 1011 total. Floors (1 unit/layer, by importance)
    #    spend 3. Density then takes every cheap unit and SKIPS the
    #    expensive log_alpha=-4 unit; importance stops dead at it.
    # ------------------------------------------------------------------
    arch = ArchConfig(conv_spec=[4, "M", 4], fc_hidden=[4], input_size=8)
    m2 = VGGStyleSNN(arch, SNNConfig(num_steps=2), BAY)
    set_bayesian_mode(m2, True)
    with torch.no_grad():
        m2.conv_layers[0].log_alpha.copy_(torch.tensor([-5.0, -4.0, 0.0, 1.0]))
        m2.conv_layers[1].log_alpha.fill_(-3.0)
        m2.fc_layers[0].log_alpha.fill_(-3.0)
    unit_costs = {
        m2.conv_layers[0]: torch.tensor([1.0, 1000.0, 1.0, 1.0]),
        m2.conv_layers[1]: torch.ones(4),
        m2.fc_layers[0]: torch.ones(4),
    }
    budget = 15.0 / 1011.0

    dense_plan = synops_budget_plan(m2, unit_costs, budget, rank_by="density")
    check(
        "density keeps the cheap units and skips the expensive one",
        dense_plan.indices(m2.conv_layers[0]).tolist() == [0, 2, 3],
    )
    check(
        "density fills the rest of the budget with cheap B/C units",
        dense_plan.indices(m2.conv_layers[1]).numel() == 4
        and dense_plan.indices(m2.fc_layers[0]).numel() == 4,
    )
    check(
        "density plan retains 11/1011 of SynOps",
        close(plan_synops_fraction(m2, dense_plan, unit_costs), 11.0 / 1011.0),
    )

    imp_plan = synops_budget_plan(m2, unit_costs, budget, rank_by="importance")
    check(
        "importance is a strict prefix: stops at the unaffordable unit",
        imp_plan.indices(m2.conv_layers[0]).tolist() == [0]
        and imp_plan.indices(m2.conv_layers[1]).numel() == 1
        and imp_plan.indices(m2.fc_layers[0]).numel() == 1,
    )
    check(
        "importance plan spends only the floors",
        close(plan_synops_fraction(m2, imp_plan, unit_costs), 3.0 / 1011.0),
    )

    # A fractional floor stops the budget buying its way out of a whole
    # layer. Reproduces the dpap_repl failure in miniature: layer A's units
    # cost 1000 each except one, so density strips A to its floor while
    # keeping the cheap layers whole. With min_keep_fraction=0.5, A must
    # retain 2 of 4 no matter how expensive it is.
    with torch.no_grad():
        m2.conv_layers[0].log_alpha.copy_(torch.tensor([-5.0, -4.0, -3.5, -3.0]))
    pricey = {
        m2.conv_layers[0]: torch.tensor([1000.0, 1000.0, 1000.0, 1000.0]),
        m2.conv_layers[1]: torch.ones(4),
        m2.fc_layers[0]: torch.ones(4),
    }
    starved = synops_budget_plan(m2, pricey, 15.0 / 4008.0, rank_by="density")
    check(
        "without a fractional floor an expensive layer collapses to 1 unit",
        starved.indices(m2.conv_layers[0]).numel() == 1,
    )
    floored = synops_budget_plan(
        m2, pricey, 15.0 / 4008.0, rank_by="density", min_keep_fraction=0.5
    )
    check(
        "min_keep_fraction holds the expensive layer at half width",
        floored.indices(m2.conv_layers[0]).tolist() == [0, 1],
    )
    check(
        "the floor keeps the most important units, not the cheapest",
        floored.indices(m2.conv_layers[0]).tolist()
        == torch.argsort(m2.conv_layers[0].log_alpha.detach())[:2].sort().values.tolist(),
    )
    check("fractional floor is recorded in the detail", "min_keep_fraction" in floored.detail)

    tiny = synops_budget_plan(m2, unit_costs, 1e-9, rank_by="density")
    check(
        "min_keep floors survive an impossible budget",
        all(tiny.indices(l).numel() == 1 for l in collect_prunable_bayesian_layers(m2)),
    )
    check("floor overage is recorded in the detail", "exceed" in tiny.detail)

    # ------------------------------------------------------------------
    # 5. Split gate optimizer: gates in SGD, weights in Adam, and one
    #    training step moves both.
    # ------------------------------------------------------------------
    m3 = VGGStyleSNN(ArchConfig(conv_spec=[2, "M", 2], fc_hidden=[3], input_size=8), SNNConfig(num_steps=2), BAY)
    set_bayesian_mode(m3, True)

    main_opt, gate_opt = build_gate_split_optimizers(m3, "adam", 1e-3, 5e-5, "inherit")
    check("inherit mode returns a single optimizer", gate_opt is None)

    main_opt, gate_opt = build_gate_split_optimizers(m3, "adam", 1e-3, 5e-5, "sgd", gate_lr=0.05)
    check("sgd mode returns a plain SGD for the gates", isinstance(gate_opt, torch.optim.SGD))
    check("gate SGD uses gate_lr", close(gate_opt.param_groups[0]["lr"], 0.05))
    check("gate SGD has no momentum", gate_opt.param_groups[0]["momentum"] == 0.0)
    check("gate SGD has no weight decay", gate_opt.param_groups[0]["weight_decay"] == 0.0)

    gate_ids = {id(p) for g in gate_opt.param_groups for p in g["params"]}
    main_ids = {id(p) for g in main_opt.param_groups for p in g["params"]}
    la_ids = {id(p) for n, p in m3.named_parameters() if "log_alpha" in n}
    check("every log_alpha is in the gate optimizer", la_ids == gate_ids)
    check("no log_alpha is in the main optimizer", not (la_ids & main_ids))
    n_params = sum(1 for _ in m3.parameters())
    check("no parameter is orphaned", len(gate_ids) + len(main_ids) == n_params)

    main_opt_a, gate_opt_a = build_gate_split_optimizers(m3, "adamw", 1e-3, 5e-5, "adam", gate_lr=2e-4)
    check("adam mode returns a plain Adam for the gates", type(gate_opt_a) is torch.optim.Adam)
    check("gate Adam uses gate_lr", close(gate_opt_a.param_groups[0]["lr"], 2e-4))
    check("gate Adam has no weight decay", gate_opt_a.param_groups[0]["weight_decay"] == 0.0)
    check("weights keep the requested optimizer", isinstance(main_opt_a, torch.optim.AdamW))
    gate_ids_a = {id(p) for g in gate_opt_a.param_groups for p in g["params"]}
    check("adam split holds exactly the log_alphas", gate_ids_a == la_ids)
    err = None
    try:
        build_gate_split_optimizers(m3, "adam", 1e-3, 5e-5, "rmsprop")
    except ValueError as e:
        err = e
    check("an unknown gate optimizer is rejected loudly", err is not None)

    la_before = m3.conv_layers[0].log_alpha.detach().clone()
    w_before = m3.conv_layers[0].conv.weight.detach().clone()
    stats = train_one_epoch(
        m3, loader_of(torch.ones(2, 3, 8, 8)), main_opt, CPU,
        beta=0.5, grad_clip_norm=5.0, use_amp=False, scaler=None,
        gate_optimizer=gate_opt,
    )
    check("one split step moves the gates", not torch.allclose(la_before, m3.conv_layers[0].log_alpha))
    check("one split step moves the weights", not torch.allclose(w_before, m3.conv_layers[0].conv.weight))
    check("train_one_epoch reports a synops field", "synops" in stats)

    # ------------------------------------------------------------------
    # 6. The Lagrangian term: value, wiring, and dual ascent.
    # ------------------------------------------------------------------
    m4 = make_all_fire_model()
    set_bayesian_mode(m4, True)
    for layer in collect_prunable_bayesian_layers(m4):
        layer.set_unit_cost(torch.ones_like(layer.unit_cost))

    expected = sum(
        float((torch.sigmoid(-l.log_alpha.detach()) * l.unit_cost).sum())
        for l in collect_prunable_bayesian_layers(m4)
    )
    check("total_expected_synops matches the by-hand sum", close(float(total_expected_synops(m4)), expected))

    spk = torch.rand(2, 3, 10)
    targets = torch.tensor([1, 2, 0])
    lam, budget_abs = 2.0, 4.0
    total, task, kl, cost, syn = bayesian_snn_loss(
        spk, targets, m4, beta=0.1, gamma=0.0, prune_threshold=3.0,
        lam=lam, synops_budget=budget_abs,
    )
    manual = float(task) + 0.1 * float(kl) + lam * (float(syn) / budget_abs - 1.0)
    check("loss wires the Lagrangian term in", close(float(total), manual))
    check("returned synops equals total_expected_synops", close(float(syn), expected))
    total_off, *_ = bayesian_snn_loss(spk, targets, m4, beta=0.1, gamma=0.0, prune_threshold=3.0)
    check("term is inert when no budget is given", close(float(total_off), float(task) + 0.1 * float(kl)))

    def refresh(model):
        for layer in collect_prunable_bayesian_layers(model):
            layer.set_unit_cost(torch.ones_like(layer.unit_cost))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        result = run_training(
            m4, loader_of(torch.ones(2, 3, 8, 8)), loader_of(torch.ones(2, 3, 8, 8)),
            epochs=2, lr=1e-3, weight_decay=0.0,
            optimizer_name="adam", scheduler_name="none",
            grad_clip_norm=5.0, use_amp=False, device=CPU,
            beta_max=0.01, kl_warmup_epochs=1,
            logger=logging.getLogger("test_synops"),
            checkpoint_path=os.path.join(tmp, "ckpt.pt"),
            csv_log_rows=rows, phase_name="bayesian_train",
            restore_best_checkpoint=False,
            synops_budget_fraction=0.01,
            synops_lambda_lr=0.05,
            synops_recount_every=1,
            synops_refresh_fn=refresh,
        )
    check("dual ascent raised lambda above zero while over budget", result["final_lambda"] > 0.0)
    check("csv rows carry lambda and train_synops", "lambda" in rows[0] and "train_synops" in rows[0])
    check("epoch 1 trained under lambda=0", rows[0]["lambda"] == 0.0)
    check("epoch 2 trained under the raised lambda", rows[1]["lambda"] > 0.0)

    err = None
    try:
        run_training(
            m4, loader_of(torch.ones(2, 3, 8, 8)), loader_of(torch.ones(2, 3, 8, 8)),
            epochs=1, lr=1e-3, weight_decay=0.0,
            optimizer_name="adam", scheduler_name="none",
            grad_clip_norm=5.0, use_amp=False, device=CPU,
            beta_max=0.01, kl_warmup_epochs=1,
            logger=logging.getLogger("test_synops"),
            checkpoint_path="unused.pt",
            csv_log_rows=[], phase_name="bayesian_train",
            restore_best_checkpoint=False,
            synops_budget_fraction=0.5,
        )
    except ValueError as e:
        err = e
    check("a budget without a refresh_fn is rejected loudly", err is not None)

    print("\nAll SynOps checks passed.")


if __name__ == "__main__":
    main()
