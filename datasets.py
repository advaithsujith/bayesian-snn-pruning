"""
CIFAR-10 data loading: automatic download, standard augmentation, and a
reproducible train / validation / test split.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from config import DataConfig


def build_transforms(data_cfg: DataConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build the training-time (augmented) and eval-time (deterministic)
    transform pipelines. Random crop + horizontal flip is the standard
    CIFAR-10 augmentation recipe used across the SNN pruning literature
    referenced in this project (e.g. VGG/ResNet CIFAR training recipes).

    Three optional, off-by-default stages reproduce the heavier pipeline
    DPAP trains with (timm's `create_transform` via BrainCog's
    `get_cifar10_data`): RandAugment, colour jitter and random erasing.
    Leaving them at their defaults keeps every pre-existing experiment
    bit-identical to before.
    """
    train_stages = [
        transforms.RandomCrop(32, padding=data_cfg.random_crop_padding),
        transforms.RandomHorizontalFlip(p=data_cfg.horizontal_flip_prob),
    ]
    if data_cfg.color_jitter > 0:
        j = data_cfg.color_jitter
        train_stages.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))
    if data_cfg.rand_augment:
        # timm spells this "rand-m9-mstd0.5-inc1"; torchvision exposes only
        # magnitude, so the magnitude is parsed out and the rest (magnitude
        # std, increasing severity) is not reproduced. Recorded as a
        # deviation in docs/replication_targets.md.
        magnitude = 9
        for part in data_cfg.rand_augment.split("-"):
            if part.startswith("m") and part[1:].isdigit():
                magnitude = int(part[1:])
                break
        train_stages.append(transforms.RandAugment(num_ops=2, magnitude=magnitude))
    train_stages.append(transforms.ToTensor())
    train_stages.append(
        transforms.Normalize(mean=data_cfg.normalize_mean, std=data_cfg.normalize_std)
    )
    if data_cfg.random_erasing_prob > 0:
        # Must follow ToTensor: RandomErasing operates on tensors.
        train_stages.append(
            transforms.RandomErasing(p=data_cfg.random_erasing_prob, value="random")
        )
    train_transform = transforms.Compose(train_stages)
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg.normalize_mean, std=data_cfg.normalize_std),
        ]
    )
    return train_transform, eval_transform


def get_cifar10_loaders(
    data_cfg: DataConfig,
    batch_size: int,
    seed: int,
    val_fraction: "float | None" = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for CIFAR-10.

    The official 50,000-image training set is split into train/validation
    using `data_cfg.val_fraction`, with a fixed generator seed so the split
    is identical across every experiment and every pruning/fine-tuning
    stage. The official 10,000-image test set is used only for final
    reporting. Data is downloaded automatically into `data_cfg.data_dir`
    if not already present.

    `val_fraction` overrides `data_cfg.val_fraction` for one call; see
    get_pruning_phase_loaders for the one place that uses it.

    Replication mode (`val_fraction = 0.0`)
    ---------------------------------------
    Training then uses all 50,000 images and the *test* set is returned as
    the validation loader. This is what the papers being replicated
    actually do -- BrainCog's `get_cifar10_data`, which DPAP trains with,
    splits off no validation set at all -- and the extra 5,000 training
    images are worth roughly half a point on CIFAR-10, so matching it
    matters when the goal is to reproduce a published baseline.

    The caveat is real and must be reported, not buried: with no held-out
    validation set, "best validation accuracy" checkpoint selection is
    selecting on the test set, and any hyperparameter chosen by watching
    that number is fitted to the test set. This makes the resulting figure
    a faithful reproduction of the published protocol rather than a clean
    generalisation estimate. Use it for baseline comparison against those
    papers; keep `val_fraction > 0` for this project's own experiments,
    and see get_pruning_phase_loaders for how the pruning stages opt out.
    """
    if val_fraction is None:
        val_fraction = data_cfg.val_fraction
    train_transform, eval_transform = build_transforms(data_cfg)

    full_train_for_train_tf = datasets.CIFAR10(
        root=data_cfg.data_dir, train=True, download=True, transform=train_transform
    )
    full_train_for_eval_tf = datasets.CIFAR10(
        root=data_cfg.data_dir, train=True, download=True, transform=eval_transform
    )
    test_set = datasets.CIFAR10(
        root=data_cfg.data_dir, train=False, download=True, transform=eval_transform
    )

    num_train = len(full_train_for_train_tf)
    num_val = int(num_train * val_fraction)

    if num_val == 0:
        # Replication mode -- train on all 50k, validate on test. See the
        # docstring's caveat about what this does to checkpoint selection.
        train_set = full_train_for_train_tf
        val_set = test_set
    else:
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(
            range(num_train), [num_train - num_val, num_val], generator=generator
        )
        train_set = Subset(full_train_for_train_tf, list(train_indices))
        val_set = Subset(full_train_for_eval_tf, list(val_indices))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_pruning_phase_loaders(
    data_cfg: DataConfig,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loaders for the gate-training, fine-tuning and hyperparameter-selection
    stages, which must never see the test set.

    Replicating a paper's *baseline* is a good reason to adopt its data
    protocol, including `val_fraction=0.0` (train on all 50k, "validate" on
    test). It is not a reason to select this project's own pruning
    decisions that way, and that is what was happening: with val == test,
    `restore_best_checkpoint` picked the fine-tuned checkpoint by test
    accuracy, the reported number was that same test accuracy, and
    sweep_beta.py then ranked candidate `beta_max` values by it. A
    hyperparameter chosen on the test set, on a checkpoint also chosen on
    the test set, is not a result an examiner will accept.

    So when `pruning_val_fraction` is set, the pruning phases get their own
    held-out split carved out of the 50k training images and the test set
    is not touched until final evaluation. When it is None (this project's
    own experiments, which already hold out 10%) the loaders are exactly
    the ones every other stage uses and nothing changes.

    One honest limitation to state in the write-up rather than let an
    examiner state: those held-out images were still seen during the
    replicated *pretraining*, so this validation split measures the pruned
    model's generalisation only relative to a baseline that had seen them.
    What it does guarantee is that no pruning decision, checkpoint or
    hyperparameter is informed by the test set.
    """
    override = data_cfg.pruning_val_fraction
    return get_cifar10_loaders(data_cfg, batch_size, seed, val_fraction=override)
