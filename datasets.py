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
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=data_cfg.random_crop_padding),
            transforms.RandomHorizontalFlip(p=data_cfg.horizontal_flip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg.normalize_mean, std=data_cfg.normalize_std),
        ]
    )
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
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) for CIFAR-10.

    The official 50,000-image training set is split into train/validation
    using `data_cfg.val_fraction`, with a fixed generator seed so the split
    is identical across every experiment and every pruning/fine-tuning
    stage. The official 10,000-image test set is used only for final
    reporting. Data is downloaded automatically into `data_cfg.data_dir`
    if not already present.
    """
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
    num_val = int(num_train * data_cfg.val_fraction)
    num_train_split = num_train - num_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(num_train), [num_train_split, num_val], generator=generator
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
