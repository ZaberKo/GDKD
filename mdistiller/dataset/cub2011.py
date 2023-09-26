import os
from typing import Callable, Optional
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, DataLoader
import pandas as pd

from functools import reduce

from mdistiller.engine.utils import log_msg
from .sampler import DistributedEvalSampler
from .instance_sample import InstanceSample

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data/cub2011')


class CUB2011(VisionDataset):

    base_folder = 'CUB_200_2011'
    images_folder = 'CUB_200_2011/images'

    def __init__(self, root: str, transform=None, target_transform=None,
                 train: bool = True, on_memory: bool = True):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train
        self.on_memory = on_memory
        self.loader = default_loader

        self._load_data()

        if self.on_memory:
            self._init_cache()

    def _load_data(self):
        # Note: img_id and cls_id start at 1

        classes = pd.read_csv(os.path.join(self.root, self.base_folder, "classes.txt"), sep=" ",
                              names=["cls_id", "class_name"])

        self.classes = classes["class_name"].tolist()
        self.class_to_idx = {cls_name: i
                             for i, cls_name in enumerate(self.classes)}

        image_paths = pd.read_csv(os.path.join(self.root, self.base_folder, "images.txt"), sep=" ",
                                  names=["img_id", "filepath"])
        image_class_labels = pd.read_csv(os.path.join(self.root, self.base_folder, "image_class_labels.txt"),
                                         sep=" ", names=["img_id", "cls_id"])
        is_train = pd.read_csv(os.path.join(self.root, self.base_folder, "train_test_split.txt"),
                               sep=" ", names=["img_id", "is_training_img"])
        bbox = pd.read_csv(os.path.join(self.root, self.base_folder, "bounding_boxes.txt"),
                           sep=" ", names=["img_id", "x", "y", "width", "height"])

        self.data = reduce(lambda x, y: pd.merge(x, y, on="img_id"),
                           [image_paths, image_class_labels, bbox, is_train])

        self.data["target"] = self.data["cls_id"] - 1

        if self.train:
            self.data = self.data[self.data["is_training_img"] == 1]
        else:
            self.data = self.data[self.data["is_training_img"] == 0]

    def _init_cache(self):
        self.cached_images = [
            self.loader(os.path.join(
                self.root, self.images_folder, sample.filepath))
            for _, sample in self.data.iterrows()
        ]

        print(
            log_msg(f"Finish loading CUB2011 into memory, num data: {len(self)}", "INFO"))

    def __getitem__(self, index: int):
        sample = self.data.iloc[index]
        if self.on_memory:
            img = self.cached_images[index]
        else:
            img = self.loader(os.path.join(
                self.root, self.images_folder, sample.filepath))

        target = sample.target

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class CUB2011InstanceSample(InstanceSample, CUB2011):
    def __init__(self,  *args, k=-1, **kwargs):
        CUB2011.__init__(self, *args, **kwargs)
        InstanceSample.__init__(self, k=k)


def get_cub2011_train_transform():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
    )
    return train_transform


def get_cub2011_test_transform():
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]
    )
    return test_transform


def get_cub2011_dataloaders(batch_size, val_batch_size, k=-1, num_workers=4, is_distributed=False):
    train_transform = get_cub2011_train_transform()
    train_set = CUB2011InstanceSample(
        data_folder, transform=train_transform, train=True,  k=k)
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

    test_loader = get_cub2011_val_loader(
        val_batch_size, num_workers, is_distributed)
    return train_loader, test_loader, num_data


def get_cub2011_val_loader(val_batch_size, num_workers=4, is_distributed=False):
    test_transform = get_cub2011_test_transform()
    test_set = CUB2011(
        data_folder, transform=test_transform, train=False)
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
