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


@dataclass
class DataConfig:
    """CIFAR-10 loading and augmentation settings."""

    data_dir: str = "./data"
    # Fraction of the official 50k training set held out for validation.
    # Set to 0.0 to train on all 50k and use the test set for validation --
    # what the papers being replicated do (see the leakage caveat in
    # datasets.get_cifar10_loaders).
    val_fraction: float = 0.1
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

    # Back to VGG9's value, which is well-grounded here: the KL at the start
    # of gate training is nearly identical (4874 vs VGG9's 4671).
    cfg.bayesian.bayesian_train_epochs = 75
    cfg.bayesian.kl_warmup_epochs = 45
    cfg.bayesian.beta_max = 0.4
    # DPAP's weight_decay of 0.01 belongs to its pretraining recipe; letting
    # it carry into gate training decays the weights 200x harder than this
    # project's phases expect, further weakening the task term's ability to
    # resist the KL pressure.
    cfg.bayesian.bayesian_train_weight_decay = 5e-5
    cfg.finetune.epochs = 30
    cfg.finetune.batch_size = 50
    cfg.finetune.weight_decay = 5e-5
    return cfg


ALL_EXPERIMENTS = {
    "lenet": get_lenet_config,
    "vgg9": get_vgg9_config,
    "resnet18": get_resnet18_config,
    "dpap_repl": get_dpap_repl_config,
}
