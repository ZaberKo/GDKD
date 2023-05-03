import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

from mdistiller.dataset.cifar100 import (
    get_data_folder,
    get_cifar100_train_transform,
    get_cifar100_train_transform_with_autoaugment,
    get_cifar100_test_transform,
    CIFAR100Instance,
)
from mdistiller.dataset.imagenet import (
    data_folder,
    get_imagenet_train_transform,
    get_imagenet_test_transform,
    ImageNet,
    get_imagenet_val_loader
)

"""
    This file contains the dataloader used for test.
    Add new option `use_val_transform` for trainset
"""


def get_cifar100_dataloaders(batch_size, val_batch_size, num_workers, use_val_transform, use_aug=False):
    data_folder = get_data_folder()
    if use_val_transform:
        train_transform = get_cifar100_test_transform()
    elif use_aug:
        train_transform = get_cifar100_train_transform_with_autoaugment()
    else:
        train_transform = get_cifar100_train_transform()

    test_transform = get_cifar100_test_transform()
    train_set = CIFAR100Instance(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data


def get_imagenet_dataloaders(batch_size, val_batch_size,
                             num_workers, use_val_transform,
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]):
    if use_val_transform:
        train_transform = get_imagenet_test_transform(mean, std)
    else:
        train_transform = get_imagenet_train_transform(mean, std)

    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = get_imagenet_val_loader(
        val_batch_size, num_workers, mean, std, is_distributed=False)
    return train_loader, test_loader, num_data


def get_dataset(cfg, use_val_transform=False):
    if cfg.DATASET.TYPE == "cifar100":
        train_loader, val_loader, num_data = get_cifar100_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            use_val_transform=use_val_transform,
            use_aug=cfg.DATASET.ENHANCE_AUGMENT,
        )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        train_loader, val_loader, num_data = get_imagenet_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS
        )
        num_classes = 1000
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)
    return train_loader, val_loader, num_data, num_classes
