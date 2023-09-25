import os
import sys
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler, DataLoader

from mdistiller.engine.utils import log_msg

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data/tiny-imagenet-200')


class TinyImageNet(ImageFolder):
    def __init__(self, *args, on_memory=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_memory = on_memory

        if self.on_memory:
            self.init_cache()

    def init_cache(self):
        cached_samples = []
        for path, target in self.samples:
            img = self.loader(path)
            cached_samples.append((img, target))

        self.samples = cached_samples

        print(log_msg(f"Finished loading TinyImageNet into memory", "INFO"))

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

        return img, target, index


class TinyImageNetInstanceSample(TinyImageNet):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, folder, on_memory=True, transform=None, is_sample=False, k=4096):
        super().__init__(folder, on_memory=on_memory, transform=transform)

        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(
                self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(
                self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


def get_tiny_imagenet_train_transform():
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


def get_tiny_imagenet_test_transform():
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


def get_tiny_imagenet_dataloaders(batch_size, val_batch_size, num_workers, is_distributed=False):
    train_transform = get_tiny_imagenet_train_transform()

    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNet(train_folder, transform=train_transform)
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

    test_loader = get_imagenet_val_loader(
        val_batch_size, num_workers, is_distributed)
    return train_loader, test_loader, num_data


def get_tiny_imagenet_dataloaders_sample(
    batch_size, val_batch_size, num_workers=16, k=4096, is_distributed=False
):

    train_transform = get_tiny_imagenet_train_transform()
    train_folder = os.path.join(data_folder, 'train')
    train_set = TinyImageNetInstanceSample(
        train_folder, transform=train_transform, is_sample=True, k=k)
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

    test_loader = get_imagenet_val_loader(
        val_batch_size, num_workers, is_distributed)
    return train_loader, test_loader, num_data


def get_imagenet_val_loader(val_batch_size, num_workers=16, is_distributed=False):
    test_transform = get_tiny_imagenet_test_transform()
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    if is_distributed:
        # TODO: use EvalDistributedSampler
        test_sampler = DistributedSampler(test_set, shuffle=False,)
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
