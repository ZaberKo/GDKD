import os
import numpy as np
import torch
from torchvision.datasets import ImageNet
import torchvision.transforms as transforms

from torch.utils.data.distributed import DistributedSampler
from .sampler import DistributedEvalSampler
from .instance_sample import InstanceSample

data_folder = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '../../data/imagenet')


class ImageNetInstanceSample(InstanceSample, ImageNet):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self,  *args, k=-1, **kwargs):
        ImageNet.__init__(self, *args, **kwargs)
        InstanceSample.__init__(self, k=k)


def get_imagenet_train_transform():
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


def get_imagenet_test_transform():
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

def get_imagenet_train_transform_strong_aug():
    # follows Swin
    try:
        from timm.data import create_transform
        train_transform = create_transform(
            input_size=224,
            is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            interpolation='bicubic'
        )
        return train_transform
    except:
        raise ImportError("timm is required")


def get_imagenet_dataloaders(batch_size, val_batch_size, k=-1, num_workers=16, is_distributed=False, enhance_augment=False):
    if enhance_augment:
        train_transform = get_imagenet_train_transform_strong_aug()
    else:
        train_transform = get_imagenet_train_transform()
    train_set = ImageNetInstanceSample(data_folder, split='train',
                         transform=train_transform,  k=k)
    num_data = len(train_set)
    if is_distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=not is_distributed,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    test_loader = get_imagenet_val_loader(
        val_batch_size, num_workers, is_distributed)
    
    num_classes = len(train_set.classes)
    return train_loader, test_loader, num_data, num_classes


def get_imagenet_val_loader(val_batch_size, num_workers=16, is_distributed=False):
    test_transform = get_imagenet_test_transform()
    test_set = ImageNet(data_folder, split='val', transform=test_transform)
    if is_distributed:
        # Note: use with caution: test_set must be divisible by #gpu
        # test_sampler = DistributedSampler(test_set, shuffle=False, drop_last=True)
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
