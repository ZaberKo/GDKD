import os
from typing import Callable, Optional
import numpy as np
import torch
from torchvision.datasets import DTD
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, DataLoader
import pandas as pd

from functools import reduce
from pathlib import Path

from mdistiller.engine.utils import log_msg
from .sampler import DistributedEvalSampler
from .instance_sample import InstanceSample
from .imagenet import (
    get_imagenet_train_transform,
    get_imagenet_train_transform_strong_aug,
    get_imagenet_test_transform
)

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data')


class DTDInstanceSample(InstanceSample, DTD):
    def __init__(self,  *args, k=-1, **kwargs):
        DTD.__init__(self, *args, **kwargs)
        InstanceSample.__init__(self, k=k)

    @property
    def _base_folder(self):
        return Path(self.root)/"dtd"
    
    @_base_folder.setter
    def _base_folder(self, p):
        # avoid DTD change the value
        pass

def get_dtd_train_transform():
    return get_imagenet_train_transform()


def get_dtd_train_transform_with_strong_aug():
    return get_imagenet_train_transform_strong_aug()


def get_dtd_test_transform():
    return get_imagenet_test_transform()


def get_dtd_dataloaders(batch_size, val_batch_size, k=-1, num_workers=4, is_distributed=False, enhance_augment=False):
    if enhance_augment:
        train_transform = get_dtd_train_transform_with_strong_aug()
    else:
        train_transform = get_dtd_train_transform()
    train_set = DTDInstanceSample(
        data_folder, split="train", transform=train_transform, k=k, download=True)
    num_data = len(train_set)

    if is_distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not is_distributed,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    test_loader = get_dtd_val_loader(
        val_batch_size, num_workers, is_distributed)
    
    num_classes = len(train_set.classes)
    return train_loader, test_loader, num_data, num_classes


def get_dtd_val_loader(val_batch_size, num_workers=4, is_distributed=False):
    test_transform = get_dtd_test_transform()
    test_set = DTD(
        data_folder, split="val", transform=test_transform, download=True)
    if is_distributed:
        test_sampler = DistributedEvalSampler(test_set, shuffle=False)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler
    )
    return test_loader
