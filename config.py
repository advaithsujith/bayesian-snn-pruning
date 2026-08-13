"""
Central configuration for all experiments.

Every hyperparameter used anywhere in the pipeline lives here so that a
single file fully determines a run's reproducibility. Nothing in
train.py / pruning.py / models.py should hard-code a hyperparameter that
belongs here.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union


@dataclass
class SNNConfig:
    """Spiking-neuron simulation settings shared by all architectures."""

    num_steps: int = 25  # number of simulated timesteps per forward pass
    beta: float = 0.95  # membrane potential decay (snntorch.Leaky 'beta')
    threshold: float = 1.0  # firing threshold
    spike_grad: str = "atan"  # snntorch surrogate-gradient function name
    # Learn the membrane decay instead of holding it fixed. Needed to
    # replicate DPAP, which uses a parametric LIF (PLIFNode) whose membrane
    # time constant is a trained parameter -- see docs/replication_targets.md.
    learn_beta: bool = False
    # snntorch reset behaviour: "subtract" (soft reset, subtract threshold)
    # or "zero" (hard reset to resting potential).
    reset_mechanism: str = "subtract"
    # What the network's per-timestep output is read from.
    #   "spikes"  -- the output layer's spikes, summed over time to classify.
    #                Every original experiment uses this; it is the default so
    #                they stay bit-identical.
    #   "current" -- the output layer's *pre-synaptic current*, i.e. fc_out's
    #                analog output, recorded before any spiking neuron.
    #
    # Lives here rather than on ArchConfig because it applies to every
    # architecture, including LeNetSNN and SpikingResNet18, which hard-code
    # their structure and never receive an ArchConfig at all.
    #
    # "current" exists for the SPEAR replication. Two reasons, both in
    # docs/replication_targets.md: TET (SPEAR's task loss) defines its O(t) as
    # "pre-synaptic input I(t) of the output layer", so running it on spikes
    # would put per-timestep cross-entropy on a binary {0,1} logit vector; and
    # at SPEAR's T=4 a summed spike count takes only five values per class, so
    # 10-way classification ties constantly and argmax breaks those ties
    # toward the lowest class index. Both problems vanish with an analog
    # readout.
    #
    # Provenance, because the distinction matters: TET *states* this
    # ("We use O(t) to represent pre-synaptic input I(t) of the output
    # layer", their Sec. 4.1), which is enough on its own since TET is the
    # loss SPEAR trains with. SPEAR itself never states its readout, and
    # SCA's was not checked. So this is [paper] for TET and [UNKNOWN] for
    # SPEAR, not a documented choice of SPEAR's.
    #
    # Nothing downstream needs to change: the returned tensor keeps its
    # [num_steps, batch, num_classes] shape, and both the accuracy helper and
    # spike_rate_cross_entropy reduce it with sum(dim=0), which is
    # argmax-identical on currents. SynOps counting is likewise unaffected --
    # metrics.measure_synops hooks Conv2d/Linear inputs, and fc_out's input is
    # the last spiking layer's output either way.
    output_readout: str = "spikes"  # "spikes" | "current"

    def __post_init__(self) -> None:
        if self.output_readout not in ("spikes", "current"):
            raise ValueError(
                f"output_readout must be 'spikes' or 'current', got "
                f"'{self.output_readout}'"
            )
        # Under "current" the output neuron is constructed but never called,
        # which is harmless only while it owns no trainable parameter. With
        # learn_beta it does: snnTorch registers `beta` as an nn.Parameter that
        # would then sit in the optimizer receiving no gradient forever, still
        # counted by count_parameters and still copied by transfer_leaky_state.
        # A frozen ghost parameter is invisible until it corrupts a
        # parameter-count comparison, so reject the combination.
        if self.output_readout == "current" and self.learn_beta:
            raise ValueError(
                "output_readout='current' skips the output neuron entirely, but "
                "learn_beta=True makes its decay a trainable parameter that would "
                "then never receive a gradient. Use learn_beta=False with the "
                "current readout, or the 'spikes' readout with learn_beta."
            )


@dataclass
class ArchConfig:
    """
    Architecture specification for the configurable VGG-style SNN family
    (models.VGGStyleSNN).

    One dataclass expresses every plain feedforward conv-stack architecture
    this project needs -- the existing VGG9, plus the three published setups
    being replicated (see docs/replication_targets.md). Defaults reproduce
    this project's original VGG9SNN exactly, so a bare ArchConfig() is a
    drop-in description of the already-validated architecture.

    `conv_spec` entries are either an int (a conv layer with that many
    output channels) or the string "M" (a pooling layer after the preceding
    conv). Pool type is chosen once via `pool_type` -- Chowdhury et al. use
    average pooling specifically to avoid the information loss max-pooling
    causes on binary spike trains.
    """

    conv_spec: List[Union[int, str]] = field(
        default_factory=lambda: [64, 64, "M", 128, 128, "M", 256, 256, 512, "M"]
    )
    fc_hidden: List[int] = field(default_factory=lambda: [800])
    kernel_size: int = 3
    padding: int = 1
    pool_type: str = "max"  # "max" | "avg"
    pool_size: int = 2
    norm_type: str = "none"  # "none" | "batch"
    conv_bias: bool = True
    # Dropout probability applied after each spiking layer. Chowdhury et al.
    # use dropout *instead of* batch-norm; 0.0 disables it entirely.
    dropout_p: float = 0.0
    encoding: str = "direct"  # "direct" | "poisson"
    in_channels: int = 3
    input_size: int = 32
    # Optional per-layer firing thresholds, one per conv/fc gated layer, in
    # depth order. Used by the ANN->SNN conversion path (Chowdhury), where
    # each layer's threshold is calibrated to a percentile of that layer's
    # activation distribution rather than shared globally. None => use
    # SNNConfig.threshold for every layer.
    layer_thresholds: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject specs whose geometry this class cannot describe correctly.

        These are guards against *silently wrong* flatten dimensions rather
        than style checks: an unnoticed mismatch here produces a model that
        trains happily on the wrong architecture, which is far worse than a
        constructor error.
        """
        if not self.conv_channels():
            raise ValueError("conv_spec must contain at least one conv layer")
        if self.conv_spec and self.conv_spec[0] == "M":
            raise ValueError("conv_spec cannot start with a pooling entry 'M'")
        # Consecutive "M"s would collapse onto one pooling stage in the build
        # loop (which sets pool_flags[-1] = True) while num_pools() counted
        # them separately, making flatten_dim wrong by a factor of pool_size^2.
        for prev, entry in zip(self.conv_spec, self.conv_spec[1:]):
            if prev == "M" and entry == "M":
                raise ValueError(
                    "conv_spec cannot contain consecutive 'M' entries; use one "
                    "'M' per pooling stage"
                )
        # spatial_after_convs assumes each conv preserves its input size, so
        # the padding must match the kernel. Otherwise flatten_dim silently
        # over-counts (e.g. kernel 5 / padding 1 shrinks by 2 per conv).
        if self.padding != (self.kernel_size - 1) // 2:
            raise ValueError(
                f"padding={self.padding} does not preserve spatial size for "
                f"kernel_size={self.kernel_size}; expected {(self.kernel_size - 1) // 2}. "
                "Size-changing convolutions are not supported by flatten_dim()."
            )
        if self.spatial_after_convs() < 1:
            raise ValueError(
                f"{self.num_pools()} pooling stages reduce a {self.input_size}px "
                "input below 1px; remove a pooling stage or use a larger input"
            )

    def conv_channels(self) -> List[int]:
        """Output channel count of each conv layer, in depth order."""
        return [entry for entry in self.conv_spec if entry != "M"]

    def num_pools(self) -> int:
        """How many pooling stages the spec contains."""
        return sum(1 for entry in self.conv_spec if entry == "M")

    def spatial_after_convs(self) -> int:
        """Feature-map side length after all pooling stages.

        Floor division matches PyTorch's default `ceil_mode=False` pooling.
        """
        size = self.input_size
        for _ in range(self.num_pools()):
            size //= self.pool_size
        return size

    def flatten_dim(self) -> int:
        """Flattened feature width entering the first fully-connected layer."""
        spatial = self.spatial_after_convs()
        return self.conv_channels()[-1] * spatial * spatial


@dataclass
class BayesianConfig:
    """
    Structured Bayesian pruning hyperparameters.

    Follows the log-normal / Gaussian multiplicative-noise gate formulation
    of Molchanov et al. (2017, "Variational Dropout Sparsifies Deep Neural
    Networks") and its structured (per-neuron / per-channel) extension in
    Neklyudov et al. (2017, "Structured Bayesian Pruning via Log-Normal
    Multiplicative Noise"). One stochastic gate z_j is attached per output
    neuron / output channel; each gate's posterior is parameterised by a
    single log_alpha_j = log(sigma_j^2 / mu_j^2), the noise-to-signal ratio.
    """

    log_alpha_init: float = -3.0  # initial log(alpha); negative => confident/on
    log_alpha_clamp_min: float = -8.0
    log_alpha_clamp_max: float = 8.0
    prune_threshold: float = 3.0  # log_alpha > threshold => prune (Molchanov et al., 2017)

    # How the learned gates become a keep/drop decision. The criterion
    # (posterior uncertainty, i.e. log_alpha) is the same in every mode --
    # only the cut point differs. See pruning.py's module docstring.
    #   "threshold"           -- log_alpha > prune_threshold. Faithful to
    #                            Molchanov et al.; sparsity is emergent and
    #                            can only be steered indirectly via beta_max.
    #   "uniform_ratio"       -- keep `keep_fraction` of the units in every
    #                            prunable layer. Identical layer widths to a
    #                            bio-inspired run at the same keep_fraction,
    #                            so the two differ only in *which* units.
    #   "global_ratio"        -- keep the best `keep_fraction` ranked across
    #                            the whole network, letting the criterion
    #                            allocate sparsity between layers.
    #   "param_target"        -- bisect a uniform keep_fraction until
    #                            `target_pruned_pct` of parameters are gone.
    #   "param_target_global" -- same, ranked globally.
    # Default stays "threshold" so every pre-existing experiment is
    # reproduced unchanged.
    prune_mode: str = "threshold"
    keep_fraction: float = 0.5  # used by uniform_ratio / global_ratio
    target_pruned_pct: float = 50.0  # used by the param_target modes
    # Floor on units kept per layer. A ranked cut is free to empty a whole
    # layer, which severs the network rather than sparsifying it -- LeNet's
    # fc2 collapsing to one unit cost ~43 accuracy points that fine-tuning
    # never recovered (HANDOFF.md bug #6).
    min_keep_per_layer: int = 1
    beta_max: float = 0.05  # final weight of the KL term in the loss
    kl_warmup_epochs: int = 15  # epochs over which beta is linearly annealed to beta_max
    bayesian_train_epochs: int = 40  # epochs spent training gates after "Converting to Bayesian"
    bayesian_train_lr: float = 5e-4  # typically lower than the initial-pretrain lr
    # Weight decay for the gate-training phase. None => inherit
    # TrainConfig.weight_decay (previous behaviour, unchanged for every
    # existing experiment). Set explicitly where the pretrain recipe's decay
    # is unsuitable for gate training -- e.g. the DPAP replication inherits
    # AdamW's 0.01, 200x this project's usual value.
    bayesian_train_weight_decay: Optional[float] = None
    gamma_max: float = 0.0  # final weight of the expected-FLOPs-cost term (0.0 = disabled)
    cost_warmup_epochs: int = 15  # epochs over which gamma is linearly annealed to gamma_max

    # -- Gate-phase optimizer split (see train.build_gate_split_optimizers) --
    # "inherit": log_alpha trains inside the same optimizer as the weights
    #            (previous behaviour, and the default). Note the gates then
    #            follow the phase's LR schedule, which anneals to ~0.
    # "sgd":     log_alpha gets its own plain SGD (no momentum) at a
    #            *constant* `gate_lr`, while the weights keep the phase's
    #            configured optimizer. Gradient-proportional steps, so the
    #            march slows as the task-vs-KL balance approaches.
    # "adam":    the same split with a plain Adam. Adam normalises each
    #            gate's step by that gate's own gradient magnitude, which is
    #            the quantity that actually varies per gate (~9x across
    #            layers on spear_repl_resnet18) while the KL push is
    #            identical everywhere -- so it converts per-gate gradient
    #            scale into ranking spread, where SGD's net displacement
    #            sees only the (tiny) per-gate mean differences. See
    #            train.build_gate_split_optimizers for the full rationale
    #            and the 2026-08-13 evidence behind each mode.
    gate_optimizer: str = "inherit"  # "inherit" | "sgd" | "adam"
    gate_lr: Optional[float] = None  # None => bayesian_train_lr

    # -- SynOps-budget loss term (dual-ascent Lagrangian; see train.run_training) --
    # None disables the term entirely (default; every existing experiment is
    # unchanged). A value b in (0, 1] constrains the expected SynOps of the
    # surviving network to b * (its measured SynOps at the start of gate
    # training):
    #     loss += lambda * (E[SynOps] / budget - 1)
    # with lambda >= 0 updated by dual ascent at the end of every epoch:
    #     lambda <- max(0, lambda + synops_lambda_lr * (E[SynOps]-budget)/budget)
    # The multiplier tunes itself: pressure rises while the model is over
    # budget and decays back toward zero once under it. This deliberately
    # replaces the fixed `gamma_max` weighting for compute-aware pruning,
    # which failed in four runs (calibrated values changed nothing, 30x
    # values collapsed the network); FALCON's ablations (arXiv 2403.07094)
    # found budget-constrained formulations strictly better than plain cost
    # minimisation, and HALP (arXiv 2110.10811) uses the same
    # pressure-follows-violation idea for latency. E[SynOps] uses the
    # measured per-unit SynOps costs (metrics.measure_synops_unit_costs) and
    # the well-conditioned keep-probability surrogate sigmoid(-log_alpha) --
    # see bayesian_layers.total_expected_synops. Mutually exclusive with
    # gamma_max > 0, since both write to the same `unit_cost` buffers.
    synops_budget_fraction: Optional[float] = None
    synops_lambda_lr: float = 0.05
    # Epochs between re-measurements of the per-unit SynOps costs during
    # gate training (firing rates drift as the network trains). 0 = measure
    # once at phase start and hold.
    synops_recount_every: int = 1
    # Batches of the training loader used per rate measurement.
    synops_measure_batches: int = 8


@dataclass
class TrainConfig:
    """Optimisation hyperparameters for a single training phase."""

    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-5
    optimizer: str = "adam"  # "adam" | "adamw" | "sgd"
    lr_scheduler: str = "cosine"  # "cosine" | "cosine_warmup" | "none"
    lr_warmup_epochs: int = 0  # only used by "cosine_warmup"
    min_lr: float = 0.0  # cosine floor; only used by "cosine_warmup"
    grad_clip_norm: float = 5.0
    use_amp: bool = True  # automatic mixed precision, used only if CUDA is available


@dataclass
class FineTuneConfig:
    """Optimisation hyperparameters for the post-pruning fine-tuning phase."""

    epochs: int = 20
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 5e-5
    optimizer: str = "adam"
    lr_scheduler: str = "cosine"
    # Only used by the "cosine_warmup" scheduler; the 0 / 0.0 defaults leave
    # every pre-existing fine-tune bit-identical. Added for the SPEAR
    # replication, whose paper fine-tunes "in the same configuration as
    # training" -- i.e. including the 10-epoch linear warm-up. That warm-up is
    # load-bearing rather than cosmetic there: their recipe's max LR is 0.1
    # under SGD, and applying it cold to a freshly rebuilt network would undo
    # the pruned weights before the gradient signal stabilises.
    lr_warmup_epochs: int = 0
    min_lr: float = 0.0
    grad_clip_norm: float = 5.0
    use_amp: bool = True


@dataclass
class BioPruningConfig:
    """
    Shared hyperparameters for the bio-inspired structured-pruning criteria
    in activity_pruning.py (naive static firing-rate, SCA, DPAP-structured).

    All three criteria fork from the same deterministic pretrained
    checkpoint (`trained_model.pt`) that the Bayesian pipeline also starts
    its "Converting to Bayesian" stage from, and share the identical
    physical-rebuild / fine-tune / evaluation infrastructure -- only the
    importance criterion and how it is trained differs. `keep_fractions`
    is swept to produce the accuracy-vs-sparsity curves needed to compare
    against the Bayesian side at matched sparsity, per the dissertation's
    controlled-comparison framing (see HANDOFF.md).
    """

    keep_fractions: List[float] = field(default_factory=lambda: [0.1])

    # -- naive static firing-rate pruning: one-shot, no training loop --
    naive_num_passes: int = 1  # passes over train_loader to average spike-rate stats over

    # -- SCA (Spiking Channel Activity-based pruning), Li et al. ICML 2024 --
    # Criterion: mean |membrane potential| per channel/neuron. Dynamic:
    # each cycle recomputes the full keep-set from freshly accumulated
    # activity (natural prune+regrow), while the target keep-count ramps
    # down linearly from "everything" to the final `keep_fraction` across
    # `sca_num_cycles` cycles of `sca_epochs_per_cycle` epochs each.
    sca_epochs_per_cycle: int = 5
    sca_num_cycles: int = 6
    sca_lr: float = 5e-4
    sca_weight_decay: float = 5e-5

    # -- DPAP-structured (Developmental Plasticity-inspired Adaptive
    # Pruning, arXiv 2211.12714), neuron/channel branch only --
    # Criterion: an exponential-moving-average "survival score" per
    # channel/neuron, driven by spike-rate activity with a constant
    # per-epoch decay ("use it or lose it, gradually decay"). Final
    # scores are ranked to hit the target `keep_fraction` -- see
    # activity_pruning.py's module docstring for why a fixed score<0
    # cutoff was replaced with explicit top-k selection.
    dpap_train_epochs: int = 30
    dpap_ema_decay: float = 0.9
    dpap_survival_decay: float = 0.02
    dpap_lr: float = 5e-4
    dpap_weight_decay: float = 5e-5

    # -- Network Slimming (Liu et al., ICCV 2017, arXiv 1708.06519) --
    # The only non-activity criterion in this file, and deliberately so: it
    # is the structured-pruning baseline SPEAR reports alongside SCA on the
    # same CIFAR-10 / VGG16 / T=4 setup, so running it here gives a
    # reimplementation whose published counterpart is known (91.16% at 87.3%
    # SynOps, 14.3% params). That makes it a check on the whole harness, not
    # just another comparator: if our Network Slimming lands near their row,
    # the SCA and DPAP reimplementations are more credible too.
    #
    # Criterion: an L1 penalty on the BatchNorm scale factors during a
    # sparsity-training phase, then rank channels by |gamma|. Channels whose
    # scale has been driven toward zero contribute almost nothing downstream,
    # so gamma doubles as a learned importance score.
    slim_train_epochs: int = 30
    # Weight of the L1 penalty on the BN gammas. 1e-4 is the value Liu et al.
    # use for CIFAR VGG; they note results are not sensitive across roughly
    # 1e-5 to 1e-3. Check the logged gamma statistics before trusting a run:
    # the signature of a working value is the gamma distribution developing a
    # clear near-zero mode while accuracy holds.
    slim_l1_lambda: float = 1e-4
    slim_lr: float = 5e-4
    slim_weight_decay: float = 5e-5


@dataclass
class DataConfig:
    """CIFAR-10 loading and augmentation settings."""

    data_dir: str = "./data"
    # Fraction of the official 50k training set held out for validation.
    # Set to 0.0 to train on all 50k and use the test set for validation --
    # what the papers being replicated do (see the leakage caveat in
    # datasets.get_cifar10_loaders).
    val_fraction: float = 0.1
    # Validation fraction used by the *pruning* stages only (gate training,
    # fine-tuning, and any hyperparameter chosen from validation accuracy).
    # None => use `val_fraction` for those too, which is right whenever
    # `val_fraction > 0`. Set it when replicating a paper that trains
    # without a validation split: the baseline then follows their protocol
    # while no pruning decision is ever made on the test set. See
    # datasets.get_pruning_phase_loaders.
    pruning_val_fraction: Optional[float] = None
    num_workers: int = 4
    pin_memory: bool = True
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
    normalize_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalize_std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])
    # Heavier augmentation, off by default so existing experiments are
    # unchanged. DPAP's pipeline (timm `create_transform` via BrainCog's
    # get_cifar10_data) enables all three; see docs/replication_targets.md.
    rand_augment: str = ""  # e.g. "rand-m9-mstd0.5-inc1"; "" disables
    color_jitter: float = 0.0  # timm passes 0.4 alongside RandAugment
    random_erasing_prob: float = 0.0  # timm's re_prob; 0.25 in DPAP's recipe


@dataclass
class ExperimentConfig:
    """Full configuration for one architecture's end-to-end experiment."""

    name: str
    snn: SNNConfig = field(default_factory=SNNConfig)
    # Only consulted by models.VGGStyleSNN (the configurable conv-stack
    # family used for the paper replications). The three original
    # architectures -- LeNetSNN / VGG9SNN / SpikingResNet18 -- hard-code
    # their own structure and ignore this field entirely.
    arch: ArchConfig = field(default_factory=ArchConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    bio: BioPruningConfig = field(default_factory=BioPruningConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    # Task loss for the pretrain phase: "spike_rate_ce" (this project's
    # default) or "unilateral_mse" (DPAP's, see docs/replication_targets.md).
    loss_type: str = "spike_rate_ce"
    # Task loss for the gate-training and fine-tuning phases. None => use
    # `loss_type` for those too (previous behaviour, unchanged for every
    # existing experiment).
    #
    # Split from `loss_type` for the replications: a replication's job is to
    # reproduce the paper's *baseline*, so the pretrain phase must match
    # their recipe exactly. The pruning criterion under test is this
    # project's own, and it was developed and calibrated against
    # cross-entropy. Those are separate concerns and forcing them to share a
    # loss conflates them -- see the note in get_dpap_repl_config on why
    # DPAP's MSE cannot drive the gates.
    pruning_loss_type: Optional[str] = None

    def pruning_loss(self) -> str:
        """Task loss used by the gate-training and fine-tuning phases."""
        return self.loss_type if self.pruning_loss_type is None else self.pruning_loss_type
    # Skip the pretrain phase and load `<output_dir>/trained_model.pt` if it
    # exists. Pretraining dominates runtime but is independent of every
    # pruning hyperparameter, so tuning beta_max / gamma_max does not need it
    # repeated -- and reusing one fixed baseline makes successive pruning
    # runs exactly comparable. Off by default: a full run from scratch stays
    # the default behaviour, and this must be a deliberate choice, since a
    # stale checkpoint from a different architecture or data recipe would
    # otherwise be picked up silently.
    reuse_pretrained: bool = False
    device: str = "cuda"  # falls back to cpu automatically if cuda is unavailable
    num_classes: int = 10
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    plot_dir: str = "./plots"
    latency_num_warmup: int = 10
    latency_num_runs: int = 100
    latency_batch_size: int = 1


def get_lenet_config() -> ExperimentConfig:
    """LeNet-SNN experiment config (~62K parameters, fastest of the three)."""
    cfg = ExperimentConfig(name="lenet")
    cfg.output_dir = "./outputs/lenet"
    cfg.train.epochs = 60
    cfg.finetune.epochs = 20
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 50
    cfg.bayesian.beta_max = 0.4
    # gamma_max=1e-5 (magnitude-matched to beta_max*KL at init, ~191) only
    # pruned fc1/fc2 -- identical to the gamma_max=0 baseline -- and never
    # touched conv1/conv2, the actually FLOPs-expensive layers. Per
    # Louizos et al. (ICLR 2018), variational-dropout-style pruning has no
    # inherent bias toward compute-heavy layers; they needed a conv-layer
    # weight ~20x the FC-layer weight to get proportional sparsification.
    # This value (30x the calibrated one) is a deliberate diagnostic probe,
    # not a re-calibration: does a stronger *global* dial ever reach
    # conv1/conv2, or does it just prune fc1/fc2 harder while conv stays
    # untouched (which would mean a global scalar is the wrong lever and a
    # per-layer weight, matching Louizos's fix, is needed instead)?
    cfg.bayesian.gamma_max = 3e-4
    cfg.bayesian.cost_warmup_epochs = 50
    return cfg


def get_vgg9_config() -> ExperimentConfig:
    """VGG9-SNN experiment config (~9M parameters)."""
    cfg = ExperimentConfig(name="vgg9")
    cfg.output_dir = "./outputs/vgg9"
    cfg.train.epochs = 100
    cfg.train.batch_size = 64
    cfg.finetune.epochs = 30
    cfg.finetune.batch_size = 64
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 45
    cfg.bayesian.beta_max = 0.4
    # gamma_max=1e-5 (30x the calibrated value, same probe factor that worked
    # for LeNet) caused a total collapse: accuracy held until epoch ~16, then
    # fell off a cliff and was fully dead (100% pruned, ~10% acc, random
    # chance) by epoch 36 -- well before gamma even finished ramping to 1e-5
    # at epoch 45. Unlike LeNet, we never actually tested VGG9 at its true
    # calibrated value before jumping to the 30x probe, so we don't have a
    # data point for what it does. Reverting to the calibrated value
    # (magnitude-matched to beta_max*KL at init, ~1868) to get that missing
    # baseline before trying anything in between.
    cfg.bayesian.gamma_max = 3e-7
    cfg.bayesian.cost_warmup_epochs = 45
    cfg.data.num_workers = 8
    return cfg


def get_resnet18_config() -> ExperimentConfig:
    """Spiking ResNet-18 experiment config (~11.7M parameters)."""
    cfg = ExperimentConfig(name="resnet18")
    cfg.output_dir = "./outputs/resnet18"
    cfg.train.epochs = 120
    cfg.train.batch_size = 64
    cfg.train.lr = 5e-4
    cfg.finetune.epochs = 30
    cfg.finetune.batch_size = 64
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 45
    cfg.bayesian.beta_max = 0.2
    # Calibrated the same way as LeNet's (see get_lenet_config): KL_init=8259,
    # expected_cost_init=13.7B, beta_max*KL_init=1652.
    cfg.bayesian.gamma_max = 1e-7
    cfg.bayesian.cost_warmup_epochs = 45
    cfg.data.num_workers = 8
    return cfg


def get_dpap_repl_config() -> ExperimentConfig:
    """
    Replication of DPAP's CIFAR-10 setup (Han et al., IEEE TPAMI 2024).

    Every value below is transcribed from docs/replication_targets.md, which
    records its provenance. Note that the paper itself does *not* state the
    CIFAR-10 training setup -- these come from the authors' released code
    (`BrainCog-X/Brain-Cog`, examples/Structural_Development/DPAP), so they
    are replication of what was actually run rather than of what was
    written down.

    Deliberately unlike this project's own experiments, and not to be
    "corrected" toward them: 8 timesteps (not 25), a learned membrane decay
    (their PLIFNode) initialised to tau=2.0 => beta=0.5 (not fixed 0.95), a
    firing threshold of 0.5 (not 1.0), a one-sided MSE task loss (not
    cross-entropy), and AdamW at weight_decay=0.01 (not Adam at 5e-5).

    Target: reproduce their 94.54% unpruned baseline. The go/no-go gate is
    ~1% -- if this lands materially below, diagnose before running any
    pruning on top of it, since comparing our pruned result against their
    published pruned numbers is only meaningful from a matched baseline.
    """
    cfg = ExperimentConfig(name="dpap_repl")
    cfg.output_dir = "./outputs/dpap_repl"

    # 128C3-BN-128C3-BN-MaxPool2-256C3-BN-256C3-BN-MaxPool2-512C3-BN-512C3-BN-512FC-10FC
    # Only 2 pooling stages, so the flattened width is 512*8*8 = 32768.
    cfg.arch = ArchConfig(
        conv_spec=[128, 128, "M", 256, 256, "M", 512, 512],
        fc_hidden=[512],
        norm_type="batch",
    )

    cfg.snn.num_steps = 8
    cfg.snn.beta = 0.5  # tau=2.0 under the standard 1 - 1/tau parameterisation
    cfg.snn.threshold = 0.5
    cfg.snn.learn_beta = True  # PLIFNode: the time constant is trained
    cfg.loss_type = "unilateral_mse"

    # lr is their 5e-3 after the linear batch-size scaling their code applies
    # (lr * batch_size / 1024, with batch_size 50) => 2.44e-4.
    cfg.train.epochs = 300
    cfg.train.batch_size = 50
    cfg.train.lr = 5e-3 * 50 / 1024
    cfg.train.weight_decay = 0.01
    cfg.train.optimizer = "adamw"
    cfg.train.lr_scheduler = "cosine_warmup"
    cfg.train.lr_warmup_epochs = 5
    cfg.train.min_lr = 1e-5

    # Pruning-side settings stay this project's own -- the replication fixes
    # the architecture and training recipe so the *baseline* is comparable;
    # the pruning criterion under test is deliberately ours.
    # Data pipeline, transcribed from BrainCog's get_cifar10_data (which
    # DPAP calls) rather than assumed. Three differences from this
    # project's default recipe, together worth most of an initial 2.6pp
    # baseline shortfall:
    #   1. no validation split at all -- trains on all 50k, evaluates on
    #      test (see datasets.get_cifar10_loaders for the leakage caveat);
    #   2. much heavier augmentation via timm's create_transform;
    #   3. a different normalisation std to this project's.
    cfg.data.val_fraction = 0.0
    # ...but the pruning stages get a genuine held-out split. Their protocol
    # governs the baseline we are trying to reproduce; it must not also
    # govern how our own pruning hyperparameters and checkpoints are
    # selected, which with val == test would mean selecting on the test set.
    # See datasets.get_pruning_phase_loaders.
    cfg.data.pruning_val_fraction = 0.1
    cfg.data.rand_augment = "rand-m9-mstd0.5-inc1"
    cfg.data.color_jitter = 0.4
    cfg.data.random_erasing_prob = 0.25
    cfg.data.normalize_std = [0.2023, 0.1994, 0.2010]
    cfg.data.num_workers = 8

    # Pruning phases use cross-entropy, not DPAP's MSE. Two runs established
    # that MSE cannot drive the Bayesian gates at all:
    #   beta_max=0.4   -> every layer collapsed to zero survivors by epoch 11
    #   beta_max=0.005 -> frac_prunable 0.998, val_acc at chance
    # Scaling beta_max by the ratio of loss *values* (78x) was the wrong
    # correction, because what balances the KL term is the task loss's
    # *gradient* on log_alpha, and MSE's is far weaker for three compounding
    # reasons: it averages over classes as well as examples (~10x), it reads
    # a firing rate bounded in [0,1] rather than a spike count spanning
    # 0..num_steps (~25x), and at DPAP's 8 timesteps that rate takes only 9
    # distinct values. The KL gradient is unchanged throughout, so the
    # balance is one-sided by orders of magnitude.
    #
    # Splitting the losses keeps the two concerns separate: pretrain
    # replicates DPAP exactly (that is what the 94.35% baseline must match),
    # while the pruning criterion under test runs under the loss it was
    # calibrated for. Accuracy is loss-independent -- it ranks by summed
    # spike count -- so comparison against their published pruned numbers
    # remains valid.
    cfg.pruning_loss_type = "spike_rate_ce"

    # The 94.35% baseline is already saved (outputs/dpap_repl/trained_model.pt),
    # and none of the pruning hyperparameters affect it, so successive pruning
    # attempts reuse it instead of repeating 5.4h of pretraining -- and all
    # fork from one identical baseline, making them directly comparable.
    # Set back to False to retrain, e.g. after any change to the architecture
    # or the data pipeline.
    cfg.reuse_pretrained = True

    # beta_max: 0.4 -> 0.01, measured rather than guessed.
    #
    # 0.4 was inherited from VGG9 on the grounds that the KL at the start of
    # gate training is nearly identical (4874 vs 4671). That reasoning was
    # wrong: what balances the KL is the task loss's *gradient* on log_alpha,
    # not the KL's value. train.gate_pressure_diagnostic on this exact
    # baseline reads |d task/d log_alpha| / (beta * |d KL/d log_alpha|)
    # between 1.4e-4 and 3.2e-3 at beta_max=0.4 -- the task loss has under a
    # third of a percent of the KL's pull. The run that produced those
    # numbers went on to lose all accuracy by epoch 15 and finished with 89%
    # of gates pinned at the clamp ceiling.
    #
    # A ratio of ~1 (the two terms in contest, so gates settle rather than
    # march) needs beta about 300x smaller. 0.01 is deliberately on the low
    # side of that: under ranked pruning nothing has to cross a threshold, so
    # gates that settle around log_alpha = -1 with real spread are a perfectly
    # good ranking, whereas gates that march past the clamp are no ranking at
    # all.
    #
    # Note that lowering beta_max does NOT slow the march. Adam normalises
    # each update by that parameter's own running gradient, so while the KL
    # dominates the step is about lr * sign(grad) regardless of beta -- the
    # observed 0.27/epoch is exactly lr=5e-4 over ~900 steps at ~60%
    # sign-following. beta_max moves where the sign flips; the gate LR below
    # is what governs how fast it gets there and whether it overshoots.
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 10
    cfg.bayesian.beta_max = 0.01
    # Halved so the approach to equilibrium is an approach rather than an
    # overshoot. Sign-following at 5e-4 moves log_alpha 0.45/epoch at most,
    # which steps straight over any balance point narrower than that.
    cfg.bayesian.bayesian_train_lr = 2e-4
    # DPAP's weight_decay of 0.01 belongs to its pretraining recipe; letting
    # it carry into gate training decays the weights 200x harder than this
    # project's phases expect, further weakening the task term's ability to
    # resist the KL pressure.
    cfg.bayesian.bayesian_train_weight_decay = 5e-5
    cfg.finetune.epochs = 30
    cfg.finetune.batch_size = 50
    cfg.finetune.weight_decay = 5e-5
    return cfg


def get_spear_repl_config() -> ExperimentConfig:
    """
    Replication of SPEAR's CIFAR-10 / VGG16 setup (Xie et al., arXiv
    2507.02945), the head-to-head comparison target for this project's
    SynOps-budget work.

    Every value is transcribed from docs/replication_targets.md section 4,
    which records provenance per row. SPEAR states its training recipe in the
    paper text -- unlike SCA -- but states neither the VGG16 specification nor
    an unpruned baseline accuracy, and no public code could be found. The five
    values that had to be assumed are listed in that document's "Assumptions"
    subsection and flagged inline below.

    Eight values had to be assumed and are numbered inline below; the same
    list is in the doc. Do not promote any of them to a replicated fact.

    Target operating point: **52.5% SynOps, 14.4% params, 91.77% top-1**.
    That SynOps figure sits almost exactly on this project's existing 0.5
    budget, which is why this row was chosen over their others.

    **There is no go/no-go baseline gate here**, unlike the DPAP replication's
    94.54%. SPEAR never publishes an unpruned number. The nearest published
    reference is SCA's 91.14% baseline, since SPEAR's table quotes SCA's
    pruned rows verbatim and both use VGG16 at T=4 -- treat it as context for
    sanity-checking the pretrain, not as a target to hit.
    """
    cfg = ExperimentConfig(name="spear_repl")
    cfg.output_dir = "./outputs/spear_repl"

    # ASSUMPTION (1 of 8): the standard CIFAR VGG16 -- 13 conv layers, 5
    # pooling stages, no hidden FC, a single 512->10 classifier. SPEAR names
    # "VGG16" and specifies nothing further; SCA, whose numbers SPEAR quotes,
    # does not either. Five pools take 32px to 1px, so flatten_dim = 512.
    # ASSUMPTION (2 of 8): BatchNorm present. Not mentioned by SPEAR; SCA
    # places BN between conv and spiking neuron. norm_type="batch" makes
    # VGGStyleSNN defer every conv gate to after its norm, which is mandatory
    # here -- gating before a BatchNorm is the bug that collapsed three DPAP
    # runs (see models.assert_gate_after_norm).
    # ASSUMPTION (6 of 8): max pooling, inherited from ArchConfig's default
    # and pinned explicitly here so it is a choice rather than an oversight.
    # SPEAR states no pooling type. Max is what torchvision's VGG16 uses, and
    # SPEAR names VGG16 without qualification. Noted against it: TET's own
    # VGGSNN uses average pooling throughout, and this class's docstring
    # argues avg pooling suits binary spike trains. Revisit if the baseline
    # lands well under SCA's 91.14% reference.
    # ASSUMPTION (7 of 8): conv bias present, also an ArchConfig default.
    # torchvision's vgg16_bn keeps conv bias too, and it is required to reach
    # the 14,728,266 parameter count asserted in tests/test_spear.py.
    cfg.arch = ArchConfig(
        conv_spec=[64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                   512, 512, 512, "M", 512, 512, 512, "M"],
        fc_hidden=[],
        norm_type="batch",
        pool_type="max",
        conv_bias=True,
    )
    # TET defines its per-timestep output as the output layer's pre-synaptic
    # current, and at T=4 a summed spike count takes only five values per
    # class. Both reasons are spelled out on SNNConfig.output_readout.
    cfg.snn.output_readout = "current"

    # "We copy the images 4 times along the timeline to obtain input for 4
    # time steps" => T=4 with direct encoding (the ArchConfig default).
    cfg.snn.num_steps = 4
    # tau=2.0 with "No decay for input currents": SpikingJelly's
    # LIFNode(decay_input=False) is v <- (1 - 1/tau) * v + x, and snnTorch's
    # Leaky is mem <- beta * mem + input, so beta = 1 - 1/tau = 0.5 exactly.
    # A cleaner correspondence than DPAP's PLIFNode needed.
    cfg.snn.beta = 0.5
    cfg.snn.threshold = 1.0
    cfg.snn.learn_beta = False  # fixed tau, unlike DPAP's parametric LIF
    cfg.snn.reset_mechanism = "zero"  # "hard reset mechanism"
    cfg.snn.spike_grad = "atan"  # "We use arctan function as the surrogate"

    # "TET is used as loss function." SPEAR gives neither TET hyperparameter;
    # losses.tet_loss defaults to the TET paper's own CIFAR values
    # (lam=0.05, phi=V_th=1.0) -- ASSUMPTION (3 of 8).
    cfg.loss_type = "tet"

    # SGD momentum 0.9, wd 5e-5, 210 epochs = 10 linear warm-up + 200 cosine,
    # max lr 0.1. All four stated in the paper.
    cfg.train.epochs = 210
    cfg.train.lr = 0.1
    cfg.train.weight_decay = 5e-5
    cfg.train.optimizer = "sgd"  # train._optimizer_by_name applies momentum 0.9
    cfg.train.lr_scheduler = "cosine_warmup"
    cfg.train.lr_warmup_epochs = 10
    cfg.train.min_lr = 0.0  # they state max lr only; cosine to 0 is the default
    # ASSUMPTION (5 of 8): batch size. Not stated. Note their lr is NOT
    # batch-size-scaled in the paper, unlike DPAP's, so the two are not
    # coupled here and this does not perturb the stated 0.1.
    cfg.train.batch_size = 128

    # DELIBERATE DEVIATION from the paper's literal text, decided on evidence
    # 2026-08-12. SPEAR says "For static datasets, no data augmentation is
    # applied", which was first implemented literally (padding 0, flip p=0).
    # That run is recorded: 210 epochs, **train_acc 1.0000 against val_acc
    # 0.8565**, validation loss flat-to-rising from epoch ~150, final test
    # accuracy **86.09%**. The network memorised all 50k images, and 86% is
    # exactly where VGG16 lands on CIFAR-10 with no augmentation.
    #
    # Their own table sits at 91.2-92.5% and SCA's baseline at 91.14%, which
    # are not reachable from an 86% baseline by pruning. So the literal
    # reading is inconsistent with their published numbers, and "no data
    # augmentation" must mean no augmentation *beyond* the standard crop and
    # flip -- the convention in most of this literature, where the term is
    # reserved for RandAugment / Cutout / Mixup. RandAugment, colour jitter
    # and random erasing therefore stay off (DataConfig defaults), and only
    # the standard pair is restored.
    #
    # Record this as an assumption in the write-up, not as replication, and
    # report the 86.09% no-augmentation run alongside it as the evidence.
    cfg.data.random_crop_padding = 4
    cfg.data.horizontal_flip_prob = 0.5
    # ASSUMPTION (4 of 8): normalisation. Not stated; these are the standard
    # CIFAR-10 values and the same std DPAP's code uses.
    cfg.data.normalize_std = [0.2023, 0.1994, 0.2010]
    # ASSUMPTION (8 of 8): the train/validation protocol. Not stated either.
    # Training on all 50k and evaluating on test is what
    # SCA and DPAP both do, and is the convention this field of work follows;
    # it is also the setting that reproduces their number rather than
    # handicapping it. Same split as DPAP's replication, with the same
    # leakage caveat documented in datasets.get_cifar10_loaders.
    cfg.data.val_fraction = 0.0
    # ...but the pruning stages still get a genuine held-out split, so no
    # decision of *ours* is ever made on the test set.
    cfg.data.pruning_val_fraction = 0.1
    cfg.data.num_workers = 8

    # As with DPAP: the replication fixes the architecture and training recipe
    # so the *baseline* is comparable, while the pruning criterion under test
    # stays this project's own and runs under the loss it was calibrated for.
    # TET is a training-dynamics loss aimed at flatter minima, not a pruning
    # signal, and the gate mechanism has never been characterised under it.
    # Accuracy is loss-independent, so the comparison against their published
    # pruned row is unaffected.
    cfg.pruning_loss_type = "spike_rate_ce"

    # Fine-tuning: "finetune the compressed SNN for 210 epochs in the same
    # configuration as training". The *optimisation* recipe below is matched
    # in full (210 epochs, SGD, lr 0.1, 10 warm-up, wd 5e-5), per the decision
    # to run 210 epochs at every sparsity point rather than only at the
    # matched one. This is ~7x the fine-tune budget of every other experiment
    # here and is the dominant GPU cost of the replication.
    #
    # Two deliberate departures from "the same configuration", both recorded
    # in docs/replication_targets.md under "Known deviations that remain":
    #   1. the fine-tune runs under `pruning_loss_type` (cross-entropy), not
    #      TET, for the reason given above;
    #   2. it trains on the 45k pruning split, not the 50k the pretrain uses,
    #      because pruning_val_fraction=0.1 holds out a genuine validation set
    #      so no decision of ours is made on the test set.
    cfg.finetune.epochs = 210
    cfg.finetune.batch_size = 128
    cfg.finetune.lr = 0.1
    cfg.finetune.weight_decay = 5e-5
    cfg.finetune.optimizer = "sgd"
    cfg.finetune.lr_scheduler = "cosine_warmup"
    cfg.finetune.lr_warmup_epochs = 10
    cfg.finetune.min_lr = 0.0

    # Gate-phase settings are carried over from the DPAP platform as a
    # STARTING POINT ONLY. beta_max is not transferable between setups: it
    # balances the KL against the task loss's *gradient* on log_alpha, and
    # that ratio moves with architecture, timestep count and -- new here --
    # the analog output readout, which removes the spike-count quantisation
    # that shaped the gradient on every previous run. Borrowing 0.4 from VGG9
    # on a value-matching argument is exactly the mistake that cost three
    # collapsed DPAP runs.
    #
    # So: run train.gate_pressure_diagnostic on the trained baseline and read
    # the |d task/d log_alpha| / (beta * |d KL/d log_alpha|) ratio BEFORE
    # trusting this number. Target a ratio near 1. It prints before epoch 1;
    # on the DPAP run it read 1.4e-4 to 3.2e-3 and was ignored, and the run
    # died by epoch 15.
    # beta_max: 0.01 -> 0.05, measured on this baseline rather than guessed.
    #
    # The 0.01 carried over from dpap_repl failed here, and in the *opposite*
    # direction to every previous failure in this project. Run 18500857
    # (2026-08-12) trained all 75 epochs with the network intact -- val_acc
    # 0.99, no collapse, frac_saturated 0.000 -- and produced
    # log_alpha min=-2.89 / median=-2.87 / max=-2.87, **std 0.002 across 4224
    # gates**. The gates never differentiated, so the ranking was arbitrary and
    # `KeepPlan.ranking_is_usable` correctly refused it. train_kl was
    # byte-identical for the last four epochs, so they had reached equilibrium
    # rather than run out of epochs: this is a *where* problem, not a *speed*
    # problem, which makes beta_max the lever and not the gate LR.
    #
    # Why 0.05 specifically. train.gate_pressure_diagnostic reads the weakest
    # layer's |d task/d log_alpha| / (beta * |d KL/d log_alpha|) at **2.62e-2**
    # here, against **5.46e-3** on dpap_repl -- and dpap_repl's 5.46e-3 is the
    # value that produced the project's one working gate run (std 0.308, zero
    # saturation). VGG16 therefore gives the task loss ~5x more pull relative
    # to the KL, so the gates settle almost on top of their initialisation,
    # where the injected noise is small enough that every gate is equally
    # tolerable and nothing separates them. Scaling beta by that same 5x puts
    # the ratio at ~5.2e-3, on the known-good point.
    #
    # The per-layer task gradients already vary 12x (5.5e-4 at conv_layers.3
    # down to 4.5e-5 at conv_layers.12), so the signal to rank on exists. The
    # gates simply never travel far enough for it to express itself.
    #
    # Success signature to check in the log, per HANDOFF.md: log_alpha stops
    # rising somewhere around -1 to 0 with **std climbing past ~0.25**, while
    # val_acc holds. Failure now has two directions -- std near zero means
    # beta is still too low, frac_saturated climbing means it is too high.
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 10
    cfg.bayesian.beta_max = 0.05
    cfg.bayesian.bayesian_train_lr = 2e-4
    cfg.bayesian.bayesian_train_weight_decay = 5e-5

    # The 90.62% baseline is saved (outputs/spear_repl/trained_model.pt, run
    # 2026-08-12, 210 epochs at 26 s/epoch). None of the pruning
    # hyperparameters affect it, so every pruning attempt reuses it instead of
    # repeating ~1.5h, and all of them fork from one identical baseline, which
    # is what makes successive runs comparable to each other.
    #
    # Set back to False after any change to the architecture or the data
    # pipeline -- e.g. testing pool_type="avg", which would invalidate this
    # checkpoint. A mismatched checkpoint raises with a message naming the
    # cause rather than silently training on the wrong baseline.
    cfg.reuse_pretrained = True
    return cfg


def get_spear_repl_resnet18_config() -> ExperimentConfig:
    """
    SPEAR's CIFAR-10 **ResNet18** setup (Xie et al., arXiv 2507.02945), the
    second architecture in the head-to-head comparison.

    Identical training recipe to `get_spear_repl_config` -- SPEAR states one
    recipe for all static datasets and both architectures -- so only the
    architecture and its target row differ. Read that function's docstring
    first; the eight assumptions it lists apply here too, except assumption 1
    (the VGG16 specification), which is replaced by this project's existing
    `SpikingResNet18`.

    **Target operating point: 39.2% SynOps, 30.3% params, 92.78% top-1.**
    A tighter SynOps budget than the VGG16 row's 52.5%, so this is the harder
    of the two and the one where the budget-in-the-loss result should show the
    most, if the dpap_repl pattern holds (its margin widened as the budget
    tightened: +2.3pp at 0.5 against +13.0pp at 0.3).

    As with VGG16, **SPEAR publishes no unpruned ResNet18 baseline**, so there
    is no go/no-go gate. Unlike VGG16 there is not even a SCA row to borrow as
    a reference, since SPEAR's ResNet18 comparisons are against their own
    ablations. Judge the pretrain against this project's own `resnet18`
    experiment (89.26% at T=25) bearing in mind T=4 here, and against the
    general expectation that a pruned 92.78% implies a dense baseline of
    roughly 93%.

    **Architecture caveat that must be stated in the write-up.** Only each
    BasicBlock's internal `conv1` is structurally prunable here; `conv2`'s
    output channels are tied to the residual addition and the stem is fixed
    (see models.py's residual pruning caveat). SPEAR prunes ResNet18 to 30.3%
    of parameters, which is almost certainly more than this constraint allows,
    so the parameter-percentage axis is **not** directly comparable between us
    on this architecture. The SynOps axis and the accuracy are.
    """
    cfg = ExperimentConfig(name="spear_repl_resnet18")
    cfg.output_dir = "./outputs/spear_repl_resnet18"

    # SpikingResNet18 is hard-coded and ignores ArchConfig entirely, so there
    # is no arch to configure -- which is exactly why output_readout lives on
    # SNNConfig rather than ArchConfig.
    cfg.snn.num_steps = 4
    cfg.snn.beta = 0.5  # tau=2.0, no input-current decay
    cfg.snn.threshold = 1.0
    cfg.snn.learn_beta = False
    cfg.snn.reset_mechanism = "zero"  # hard reset
    cfg.snn.spike_grad = "atan"
    cfg.snn.output_readout = "current"  # TET reads the pre-synaptic current

    cfg.loss_type = "tet"
    cfg.pruning_loss_type = "spike_rate_ce"

    cfg.train.epochs = 210
    cfg.train.lr = 0.1
    cfg.train.weight_decay = 5e-5
    cfg.train.optimizer = "sgd"
    cfg.train.lr_scheduler = "cosine_warmup"
    cfg.train.lr_warmup_epochs = 10
    cfg.train.min_lr = 0.0
    cfg.train.batch_size = 128

    # Standard crop+flip, for the reason established empirically on the VGG16
    # run: the literal "no data augmentation" reading gave train_acc 1.0000
    # and 86.09% test, 5pp under the published reference. See
    # get_spear_repl_config and docs/replication_targets.md section 4.
    cfg.data.random_crop_padding = 4
    cfg.data.horizontal_flip_prob = 0.5
    cfg.data.normalize_std = [0.2023, 0.1994, 0.2010]
    cfg.data.val_fraction = 0.0
    cfg.data.pruning_val_fraction = 0.1
    cfg.data.num_workers = 8

    cfg.finetune.epochs = 210
    cfg.finetune.batch_size = 128
    cfg.finetune.lr = 0.1
    cfg.finetune.weight_decay = 5e-5
    cfg.finetune.optimizer = "sgd"
    cfg.finetune.lr_scheduler = "cosine_warmup"
    cfg.finetune.lr_warmup_epochs = 10
    cfg.finetune.min_lr = 0.0

    # Placeholder, exactly as in the VGG16 config: beta_max does not transfer
    # across architectures, and ResNet18's gate population differs more than
    # most (half its gates sit on non-prunable layers and are excluded from
    # the KL entirely -- Session 3 bug #2). Run
    # `run_sparsity_curve.py --model spear_repl_resnet18 --diagnose-only`
    # before spending gate-training hours.
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 10
    cfg.bayesian.beta_max = 0.01
    cfg.bayesian.bayesian_train_lr = 2e-4
    cfg.bayesian.bayesian_train_weight_decay = 5e-5

    # Reuse outputs/spear_repl_resnet18/trained_model.pt once it exists, so
    # every pruning attempt forks from one identical baseline instead of
    # repeating the pretrain. Safe to leave True before the file exists:
    # run_all.py ANDs this with os.path.isfile and retrains if it is missing,
    # and slurm_compare.sh refuses to start without it.
    # Set back to False after any change to the architecture or data pipeline.
    cfg.reuse_pretrained = True
    return cfg


ALL_EXPERIMENTS = {
    "lenet": get_lenet_config,
    "vgg9": get_vgg9_config,
    "resnet18": get_resnet18_config,
    "dpap_repl": get_dpap_repl_config,
    "spear_repl": get_spear_repl_config,
    "spear_repl_resnet18": get_spear_repl_resnet18_config,
}
