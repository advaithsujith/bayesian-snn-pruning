"""
Central configuration for all experiments.

Every hyperparameter used anywhere in the pipeline lives here so that a
single file fully determines a run's reproducibility. Nothing in
train.py / pruning.py / models.py should hard-code a hyperparameter that
belongs here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SNNConfig:
    """Spiking-neuron simulation settings shared by all architectures."""

    num_steps: int = 25  # number of simulated timesteps per forward pass
    beta: float = 0.95  # membrane potential decay (snntorch.Leaky 'beta')
    threshold: float = 1.0  # firing threshold
    spike_grad: str = "atan"  # snntorch surrogate-gradient function name


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


@dataclass
class TrainConfig:
    """Optimisation hyperparameters for a single training phase."""

    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-5
    optimizer: str = "adam"
    lr_scheduler: str = "cosine"
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
class DataConfig:
    """CIFAR-10 loading and augmentation settings."""

    data_dir: str = "./data"
    val_fraction: float = 0.1  # fraction of the official training set held out for validation
    num_workers: int = 4
    pin_memory: bool = True
    random_crop_padding: int = 4
    horizontal_flip_prob: float = 0.5
    normalize_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalize_std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass
class ExperimentConfig:
    """Full configuration for one architecture's end-to-end experiment."""

    name: str
    snn: SNNConfig = field(default_factory=SNNConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
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
    cfg.bayesian.bayesian_train_epochs = 40
    cfg.bayesian.kl_warmup_epochs = 15
    cfg.bayesian.beta_max = 0.05
    return cfg


def get_vgg9_config() -> ExperimentConfig:
    """VGG9-SNN experiment config (~9M parameters)."""
    cfg = ExperimentConfig(name="vgg9")
    cfg.output_dir = "./outputs/vgg9"
    cfg.train.epochs = 100
    cfg.train.batch_size = 64
    cfg.finetune.epochs = 30
    cfg.finetune.batch_size = 64
    cfg.bayesian.bayesian_train_epochs = 60
    cfg.bayesian.kl_warmup_epochs = 25
    cfg.bayesian.beta_max = 0.02
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
    cfg.bayesian.bayesian_train_epochs = 60
    cfg.bayesian.kl_warmup_epochs = 30
    cfg.bayesian.beta_max = 0.01
    cfg.data.num_workers = 8
    return cfg


ALL_EXPERIMENTS = {
    "lenet": get_lenet_config,
    "vgg9": get_vgg9_config,
    "resnet18": get_resnet18_config,
}
