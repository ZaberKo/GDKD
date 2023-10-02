from mdistiller.engine.utils import is_distributed

from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .imagenet import get_imagenet_dataloaders
from .tiny_imaganet import get_tiny_imagenet_dataloaders
from .cub2011 import get_cub2011_dataloaders
from .dtd import get_dtd_dataloaders
from .food101 import get_food101_dataloaders

def get_dataset(cfg):
    return {
        "cifar100": get_cifar,
        "imagenet": get_imagenet,
        "tiny-imagenet": get_tiny_imagenet,
        "cub2011": get_cub2011,
        "dtd": get_dtd,
        "food101": get_food101
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


def get_imagenet(cfg):
    train_loader, val_loader, num_data, num_classes = get_imagenet_dataloaders(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        k=cfg.CRD.NCE.K if cfg.DISTILLER.TYPE == "CRD" else -1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        is_distributed=is_distributed(),
        enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
    )
    assert num_classes == 1000

    return train_loader, val_loader, num_data, num_classes


def get_tiny_imagenet(cfg):
    train_loader, val_loader, num_data, num_classes = get_tiny_imagenet_dataloaders(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        k=cfg.CRD.NCE.K if cfg.DISTILLER.TYPE == "CRD" else -1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        is_distributed=is_distributed(),
        enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
    )
    assert num_classes == 200

    return train_loader, val_loader, num_data, num_classes


def get_cub2011(cfg):
    train_loader, val_loader, num_data, num_classes = get_cub2011_dataloaders(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        k=cfg.CRD.NCE.K if cfg.DISTILLER.TYPE == "CRD" else -1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        is_distributed=is_distributed(),
        enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
    )

    assert num_classes == 200
    return train_loader, val_loader, num_data, num_classes


def get_dtd(cfg):
    train_loader, val_loader, num_data, num_classes = get_dtd_dataloaders(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        k=cfg.CRD.NCE.K if cfg.DISTILLER.TYPE == "CRD" else -1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        is_distributed=is_distributed(),
        enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
    )

    assert num_classes == 47
    return train_loader, val_loader, num_data, num_classes

def get_food101(cfg):
    train_loader, val_loader, num_data, num_classes = get_food101_dataloaders(
        batch_size=cfg.SOLVER.BATCH_SIZE,
        val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
        k=cfg.CRD.NCE.K if cfg.DISTILLER.TYPE == "CRD" else -1,
        num_workers=cfg.DATASET.NUM_WORKERS,
        is_distributed=is_distributed(),
        enhance_augment=cfg.DATASET.ENHANCE_AUGMENT
    )

    assert num_classes == 101
    return train_loader, val_loader, num_data, num_classes