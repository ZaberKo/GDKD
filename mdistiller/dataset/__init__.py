import torch
from mdistiller.engine.utils import is_distributed

from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample
from .tiny_imaganet import get_tiny_imagenet_dataloaders, get_tiny_imagenet_dataloaders_sample


def get_dataset(cfg):
    return {
        "cifar100": get_cifar,
        "tiny-imagenet": get_tiny_imagenet,
        "imagenet": get_imagenet
    }[cfg.DATASET.TYPE](cfg)


def get_cifar(cfg):
    if is_distributed():
        raise NotImplementedError("cifar100 is not supported for DDP")

    if cfg.DISTILLER.TYPE == "CRD":
        train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            k=cfg.CRD.NCE.K,
            mode=cfg.CRD.MODE,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
        )
    else:
        train_loader, val_loader, num_data = get_cifar100_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
        )
    num_classes = 100

    return train_loader, val_loader, num_data, num_classes


def get_tiny_imagenet(cfg):
    if cfg.DATASET.ENHANCE_AUGMENT:
        raise ValueError(
            "Enhance augment is not supported for tiny-imagenet dataset")

    if cfg.DISTILLER.TYPE == "CRD":
        train_loader, val_loader, num_data = get_tiny_imagenet_dataloaders_sample(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            k=cfg.CRD.NCE.K,
            is_distributed=is_distributed()
        )
    else:
        train_loader, val_loader, num_data = get_tiny_imagenet_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            is_distributed=is_distributed()
        )
    num_classes = 200

    return train_loader, val_loader, num_data, num_classes


def get_imagenet(cfg):
    if cfg.DATASET.ENHANCE_AUGMENT:
        raise ValueError(
            "Enhance augment is not supported for imagenet dataset")

    if cfg.DISTILLER.TYPE == "CRD":
        train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            k=cfg.CRD.NCE.K,
            is_distributed=is_distributed()
        )
    else:
        train_loader, val_loader, num_data = get_imagenet_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            is_distributed=is_distributed()
        )
    num_classes = 1000

    return train_loader, val_loader, num_data, num_classes
