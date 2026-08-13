"""
Discriminating toy: does the gate machinery rank ground-truth channel
importance under dpap-like vs spear-like SNN settings?

Uses the repo's real classes (VGGStyleSNN, bayesian_snn_loss,
build_gate_split_optimizers) on a synthetic 4-class task small enough for
CPU. Ground-truth importance is measured by per-channel ablation (zero the
channel's hard_mask, measure val-loss increase). After a gate-training
phase, we report pooled log_alpha std and the Spearman correlation between
(-log_alpha) and ablation importance. A working criterion gives a clearly
positive correlation; an arbitrary one gives ~0.

Cells:
  A dpap-like SNN (T=8, thr 0.5, subtract, spikes readout) + adam gates/adamw weights
  B spear-like SNN (T=4, thr 1.0, zero reset, current readout) + adam gates/adamw weights
  C spear-like SNN + sgd gates / near-frozen sgd weights  (the failing runs' mechanics)
  D dpap-like SNN + sgd gates / near-frozen sgd weights
Then single-ingredient flips of the spear-like config under the dpap
mechanics if B fails.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from config import ArchConfig, SNNConfig, BayesianConfig
from models import SpikingBasicBlock, VGGStyleSNN, read_output, _make_leaky
from bayesian_layers import BayesianConv2d, collect_prunable_bayesian_layers, set_bayesian_mode
from losses import bayesian_snn_loss, get_task_loss, linear_warmup_schedule
from train import build_gate_split_optimizers

torch.manual_seed(0)
DEV = torch.device("cpu")
N_CLASSES = 4


def make_data(n_per_class=160, seed=1):
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(N_CLASSES, 3, 8, 8, generator=g)
    # smooth the prototypes a bit so conv features matter
    k = torch.ones(1, 1, 3, 3) / 9.0
    protos = torch.conv2d(protos.view(-1, 1, 8, 8), k, padding=1).view(N_CLASSES, 3, 8, 8)
    xs, ys = [], []
    for c in range(N_CLASSES):
        noise = torch.randn(n_per_class, 3, 8, 8, generator=g)
        xs.append(protos[c] * 1.0 + 0.8 * noise)
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    idx = torch.randperm(x.shape[0], generator=g)
    x, y = x[idx], y[idx]
    n_train = int(0.7 * x.shape[0])
    return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])


def batches(x, y, bs, shuffle=True, seed=0):
    n = x.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
    for i in range(0, n - bs + 1, bs):
        j = order[i : i + bs]
        yield x[j], y[j]


def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        pred = out.sum(dim=0).argmax(dim=1)
    return (pred == y).float().mean().item()


def val_loss(model, x, y, loss_name):
    model.eval()
    with torch.no_grad():
        out = model(x)
        return float(get_task_loss(loss_name)(out, y))


def build(snn_cfg):
    arch = ArchConfig(conv_spec=[8, "M", 8], fc_hidden=[], norm_type="batch", input_size=8)
    torch.manual_seed(7)
    m = VGGStyleSNN(arch, snn_cfg, BayesianConfig(), num_classes=N_CLASSES)
    return m


class TinyResSNN(nn.Module):
    """A minimal residual spiking net wired exactly like SpikingResNet18:
    non-prunable stem, real SpikingBasicBlocks (conv1 gated and prunable,
    conv2 residual-tied and non-prunable), global pool, fc_out. Exists to
    measure whether the residual bypass compresses the *ground-truth*
    importance spread of conv1 channels -- the direct test of the
    interchangeability hypothesis, independent of any gate machinery."""

    def __init__(self, snn_cfg, bayesian_cfg, width=8, n_blocks=2, num_classes=N_CLASSES):
        super().__init__()
        self.num_steps = snn_cfg.num_steps
        self.output_readout = snn_cfg.output_readout
        gate_kwargs = dict(
            log_alpha_init=bayesian_cfg.log_alpha_init,
            log_alpha_clamp_min=bayesian_cfg.log_alpha_clamp_min,
            log_alpha_clamp_max=bayesian_cfg.log_alpha_clamp_max,
        )
        self.stem_conv = BayesianConv2d(3, width, kernel_size=3, padding=1, **gate_kwargs)
        self.stem_conv.structurally_prunable = False
        self.stem_conv.defer_gate = True
        self.stem_bn = nn.BatchNorm2d(width)
        self.stem_lif = _make_leaky(snn_cfg)
        self.blocks = nn.ModuleList(
            [SpikingBasicBlock(width, width, 1, snn_cfg, bayesian_cfg) for _ in range(n_blocks)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(width, num_classes)
        self.lif_out = _make_leaky(snn_cfg, output=True)

    def forward(self, x):
        mems = [b.init_state() for b in self.blocks]
        stem_mem = self.stem_lif.init_leaky()
        mem_out = self.lif_out.init_leaky()
        rec = []
        for _ in range(self.num_steps):
            cur = self.stem_conv.apply_gate(self.stem_bn(self.stem_conv(x)))
            spk, stem_mem = self.stem_lif(cur, stem_mem)
            for i, b in enumerate(self.blocks):
                m1, m2 = mems[i]
                spk, m1, m2 = b(spk, m1, m2)
                mems[i] = (m1, m2)
            cur_out = self.fc_out(self.global_pool(spk).flatten(1))
            out_t, mem_out = read_output(self, cur_out, mem_out)
            rec.append(out_t)
        return torch.stack(rec, dim=0)


def build_res(snn_cfg):
    torch.manual_seed(7)
    return TinyResSNN(snn_cfg, BayesianConfig())


def pretrain(model, tr, va, loss_name, epochs=30, lr=1e-3, bs=32):
    set_bayesian_mode(model, False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = get_task_loss(loss_name)
    for ep in range(epochs):
        model.train()
        for xb, yb in batches(*tr, bs, seed=ep):
            opt.zero_grad()
            loss = lf(model(xb), yb)
            loss.backward()
            opt.step()
    return accuracy(model, *va)


def ablation_importance(model, va, loss_name):
    """Per-gate ground truth: val-loss increase when the channel is masked."""
    base = val_loss(model, *va, loss_name)
    imps = {}
    for layer in collect_prunable_bayesian_layers(model):
        n = layer.hard_mask.numel()
        v = torch.zeros(n)
        for j in range(n):
            layer.hard_mask[j] = 0.0
            v[j] = val_loss(model, *va, loss_name) - base
            layer.hard_mask[j] = 1.0
        imps[layer] = v
    return imps


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def gate_phase(model, tr, loss_name, mechanics, epochs=60, bs=32, beta_max=0.05, warm=10):
    set_bayesian_mode(model, True)
    if mechanics == "dpap":  # adam gates (constant lr), adamw weights
        main, gates = build_gate_split_optimizers(model, "adamw", 5e-4, 5e-5, "adam", gate_lr=4e-3)
    elif mechanics == "lagr":  # sgd gates, near-frozen sgd-momentum weights
        main, gates = build_gate_split_optimizers(model, "sgd", 5e-4, 5e-5, "sgd", gate_lr=0.15)
    else:
        raise ValueError(mechanics)
    for ep in range(epochs):
        beta = linear_warmup_schedule(ep, warm, beta_max)
        model.train()
        for xb, yb in batches(*tr, bs, seed=1000 + ep):
            main.zero_grad(set_to_none=True)
            gates.zero_grad(set_to_none=True)
            out = model(xb)
            loss, task, kl, cost, syn = bayesian_snn_loss(
                out, yb, model, beta, 0.0, 3.0, loss_name
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            main.step()
            gates.step()
    la = torch.cat([l.log_alpha.detach().flatten() for l in collect_prunable_bayesian_layers(model)])
    return la


def run_cell(name, snn_cfg, mechanics, loss_name="spike_rate_ce", builder=build):
    print(f"[running] {name} ...", flush=True)
    tr, va = make_data()
    m = builder(snn_cfg)
    acc = pretrain(m, tr, va, loss_name)
    imps = ablation_importance(m, va, loss_name)
    m2 = copy.deepcopy(m)
    la = gate_phase(m2, tr, loss_name, mechanics)
    layers = collect_prunable_bayesian_layers(m2)
    imp_vec = torch.cat([imps[l0] for l0 in collect_prunable_bayesian_layers(m)])
    # negative log_alpha = kept/important; correlate importance with -log_alpha
    rho = spearman(imp_vec, -la)
    per_layer = []
    for l0, l2 in zip(collect_prunable_bayesian_layers(m), layers):
        per_layer.append(spearman(imps[l0], -l2.log_alpha.detach()))
    med = la.median().item()
    # The ground-truth spread itself: if ablating any channel costs about the
    # same (std << mean), the channels are interchangeable and NO criterion
    # could rank them -- that is a property of the network, not the gates.
    print(
        f"{name:34s} acc={acc:.3f} gate_median={med:+.2f} std={la.std():.3f} "
        f"rho_pooled={rho:+.2f} rho_layers=[" + ", ".join(f"{r:+.2f}" for r in per_layer) + "] "
        f"true_imp={imp_vec.mean():.4f}+-{imp_vec.std():.4f}"
    )
    return rho, la.std().item()


def cfg_dpap(**kw):
    d = dict(num_steps=8, beta=0.5, threshold=0.5, reset_mechanism="subtract", output_readout="spikes")
    d.update(kw)
    return SNNConfig(**d)


def cfg_spear(**kw):
    d = dict(num_steps=4, beta=0.5, threshold=1.0, reset_mechanism="zero", output_readout="current")
    d.update(kw)
    return SNNConfig(**d)


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count() or 8)
    print("== main cells ==")
    run_cell("A dpap-like + dpap mechanics", cfg_dpap(), "dpap")
    run_cell("B spear-like + dpap mechanics", cfg_spear(), "dpap")
    run_cell("C spear-like + lagr mechanics", cfg_spear(), "lagr")
    run_cell("D dpap-like + lagr mechanics", cfg_dpap(), "lagr")
    print("== single-ingredient flips of spear-like, dpap mechanics ==")
    run_cell("E spear but T=8", cfg_spear(num_steps=8), "dpap")
    run_cell("F spear but thr=0.5", cfg_spear(threshold=0.5), "dpap")
    run_cell("G spear but subtract reset", cfg_spear(reset_mechanism="subtract"), "dpap")
    run_cell("H spear but spikes readout", cfg_spear(output_readout="spikes"), "dpap")
    print("== residual cells: does the bypass erase the ground truth? ==")
    run_cell("I residual, spear-like + dpap mech", cfg_spear(), "dpap", builder=build_res)
    run_cell("J residual, dpap-like + dpap mech", cfg_dpap(), "dpap", builder=build_res)
