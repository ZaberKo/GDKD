import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, DataLoader

from mdistiller.engine.utils import log_msg
from .sampler import DistributedEvalSampler
from .instance_sample import InstanceSample
from .imagenet import (
    get_imagenet_train_transform,
    get_imagenet_train_transform_strong_aug,
    get_imagenet_test_transform
)

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data/tiny-imagenet-200')


class TinyImageNet(ImageFolder):
    def __init__(self, *args, on_memory=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_memory = on_memory

        if self.on_memory:
            self._init_cache()

    def _init_cache(self):
        cached_samples = []
        for path, target in self.samples:
            img = self.loader(path)
            cached_samples.append((img, target))

        self.samples = cached_samples

        print(log_msg(
            f"Finish loading TinyImageNet into memory, num data: {len(self)}", "INFO"))

    def __getitem__(self, index: int):
        if self.on_memory:
            img, target = self.samples[index]
        else:
            path, target = self.samples[index]
            img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TinyImageNetInstanceSample(InstanceSample, TinyImageNet):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self,  *args, k=-1, **kwargs):
        TinyImageNet.__init__(
            self, *args, **kwargs)
        InstanceSample.__init__(self, k=k)


def get_tiny_imagenet_train_transform():
    return get_imagenet_train_transform()

def get_tiny_imagenet_train_transform_with_strong_aug():
    return get_imagenet_train_transform_strong_aug()

def get_tiny_imagenet_test_transform():
    return get_imagenet_test_transform()


def get_tiny_imagenet_dataloaders(batch_size, val_batch_size, k=-1, num_workers=4,  is_distributed=False, enhance_augment=False):
    if enhance_augment:
        train_transform = get_tiny_imagenet_train_transform_with_strong_aug()
    else:
        train_transform = get_tiny_imagenet_train_transform()

    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNetInstanceSample(
        train_folder, transform=train_transform, k=k)
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

    test_loader = get_tiny_imagenet_val_loader(
        val_batch_size, num_workers, is_distributed)
    return train_loader, test_loader, num_data


def get_tiny_imagenet_val_loader(val_batch_size, num_workers=4, is_distributed=False):
    test_transform = get_tiny_imagenet_test_transform()
    test_folder = os.path.join(data_folder, 'val')
    test_set = TinyImageNet(test_folder, transform=test_transform)
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
