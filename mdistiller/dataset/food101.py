import os
import torch
from torchvision.datasets import Food101 as Food101Base

from torch.utils.data import DistributedSampler, DataLoader

import PIL.Image
import io

from mdistiller.engine.utils import log_msg
from .sampler import DistributedEvalSampler
from .instance_sample import InstanceSample
from .imagenet import (
    get_imagenet_train_transform,
    get_imagenet_train_transform_strong_aug,
    get_imagenet_test_transform
)

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data/food101')


class Food101(Food101Base):
    def __init__(self, *args, on_memory: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_memory = on_memory

        self.loader = lambda p: PIL.Image.open(p).convert("RGB")

        if self.on_memory:
            self._init_cache()

    def _init_cache(self):
        # Direadly load Image consumes too many memory, so we load bytes and then convert to Image at fetch time.
        self.cached_images_bytes = []

        for path in self._image_files:
            with open(path, "rb") as f:
                self.cached_images_bytes.append(io.BytesIO(f.read()))

        print(
            log_msg(f"Finish loading Food101 into memory, num data: {len(self)}", "INFO"))
        
    def __getitem__(self, idx):
        label = self._labels[idx]
        if self.on_memory:
            image_bytes = self.cached_images_bytes[idx]
            image = self.loader(image_bytes)
        else:
            image_file = self._image_files[idx]
            image = self.loader(image_file)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class Food101InstanceSample(InstanceSample, Food101):
    def __init__(self,  *args, k=-1, **kwargs):
        Food101.__init__(self, *args, **kwargs)
        InstanceSample.__init__(self, k=k)


def get_food101_train_transform():
    return get_imagenet_train_transform()


def get_food101_train_transform_strong_aug():
    return get_imagenet_train_transform_strong_aug()


def get_food101_test_transform():
    return get_imagenet_test_transform()


def get_food101_dataloaders(batch_size, val_batch_size, k=-1, num_workers=4, is_distributed=False, enhance_augment=False):
    if enhance_augment:
        train_transform = get_food101_train_transform_strong_aug()
    else:
        train_transform = get_food101_train_transform()
    train_set = Food101InstanceSample(
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

    test_loader = get_food101_val_loader(
        val_batch_size, num_workers, is_distributed)
    
    num_classes = len(train_set.classes)
    return train_loader, test_loader, num_data, num_classes


def get_food101_val_loader(val_batch_size, num_workers=4, is_distributed=False):
    test_transform = get_food101_test_transform()
    test_set = Food101(
        data_folder, split="test", transform=test_transform, download=True)
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
