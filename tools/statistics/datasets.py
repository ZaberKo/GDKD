import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

from mdistiller.dataset.cifar100 import (
    get_data_folder as get_cifar100_data_folder,
    get_cifar100_train_transform,
    get_cifar100_train_transform_with_autoaugment,
    get_cifar100_test_transform,
    CIFAR100Instance,
)
from mdistiller.dataset.imagenet import (
    data_folder as imagenet_data_folder,
    get_imagenet_train_transform,
    get_imagenet_train_transform_strong_aug,
    get_imagenet_test_transform,
    ImageNet,
    get_imagenet_val_loader
)
from mdistiller.dataset.tiny_imaganet import (
    data_folder as tiny_imagenet_data_folder,
    get_tiny_imagenet_train_transform,
    get_tiny_imagenet_train_transform_strong_aug,
    get_tiny_imagenet_test_transform,
    TinyImageNet,
    get_tiny_imagenet_val_loader,
)
from mdistiller.dataset.cub2011 import (
    data_folder as cub2011_data_folder,
    get_cub2011_train_transform,
    get_cub2011_train_transform_strong_aug,
    get_cub2011_test_transform,
    CUB2011,
    get_cub2011_val_loader,
)

"""
    This file contains the dataloader used for test.
    Add new option `use_val_transform` for trainset
"""


def get_cifar100_dataloaders(train: bool, batch_size, num_workers, use_val_transform, enhance_augment=False):
    data_folder = get_cifar100_data_folder()

    if train:
        if use_val_transform:
            train_transform = get_cifar100_test_transform()
        elif enhance_augment:
            train_transform = get_cifar100_train_transform_with_autoaugment()
        else:
            train_transform = get_cifar100_train_transform()

        train_set = CIFAR100Instance(
            root=data_folder, download=True, train=True, transform=train_transform
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader
    else:
        test_transform = get_cifar100_test_transform()

        test_set = datasets.CIFAR100(
            root=data_folder, download=True, train=False, transform=test_transform
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return test_loader


def get_imagenet_dataloaders(train: bool, batch_size,
                             num_workers, use_val_transform=False,
                             enhance_augment=False):
    if train:
        if use_val_transform:
            train_transform = get_imagenet_test_transform()
        elif enhance_augment:
            train_transform = get_imagenet_train_transform_strong_aug()
        else:
            train_transform = get_imagenet_train_transform()

        train_set = ImageNet(imagenet_data_folder, split='train', transform=train_transform)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader
    else:
        test_loader = get_imagenet_val_loader(
            batch_size, num_workers, is_distributed=False)
        return test_loader
    
def get_tiny_imagenet_dataloaders(train: bool, batch_size,
                                num_workers, use_val_transform=False,
                                enhance_augment=False):
        if train:
            if use_val_transform:
                train_transform = get_tiny_imagenet_test_transform()
            elif enhance_augment:
                train_transform = get_tiny_imagenet_train_transform_strong_aug()
            else:
                train_transform = get_tiny_imagenet_train_transform()
    
            train_folder = os.path.join(tiny_imagenet_data_folder, 'train')
            train_set = TinyImageNet(train_folder, transform=train_transform)
    
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
    
            return train_loader
        else:
            test_loader = get_tiny_imagenet_val_loader(
                batch_size, num_workers, is_distributed=False)
            return test_loader

def get_cub2011_dataloaders(train: bool, batch_size,
                            num_workers, use_val_transform=False,
                            enhance_augment=False):
    if train:
        if use_val_transform:
            train_transform = get_cub2011_test_transform()
        elif enhance_augment:
            train_transform = get_cub2011_train_transform_strong_aug()
        else:
            train_transform = get_cub2011_train_transform()

        train_folder = os.path.join(cub2011_data_folder, 'train')
        train_set = CUB2011(train_folder, transform=train_transform)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader
    else:
        test_loader = get_cub2011_val_loader(
            batch_size, num_workers, is_distributed=False)
        return test_loader


def get_dataset(cfg, train=False, use_val_transform=False):
    if cfg.DATASET.TYPE == "cifar100":
        dataloader = get_cifar100_dataloaders(
            train=train,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            use_val_transform=use_val_transform,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT,
        )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        dataloader = get_imagenet_dataloaders(
            train=train,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            use_val_transform=use_val_transform,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT,
        )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tiny_imagenet":
        dataloader = get_tiny_imagenet_dataloaders(
            train=train,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            use_val_transform=use_val_transform,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT,
        )
        num_classes = 200
    elif cfg.DATASET.TYPE == "cub2011":
        dataloader = get_cub2011_dataloaders(
            train=train,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            use_val_transform=use_val_transform,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT,
        )
        num_classes = 200
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)
    return dataloader, num_classes
