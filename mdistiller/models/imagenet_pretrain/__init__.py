from functools import partial
from pathlib import Path
from collections import OrderedDict
import torch


from ..imagenet import get_imagenet_model
from mdistiller.engine.utils import load_checkpoint

model_base_dir = Path(__file__).absolute(
).parent.parent.parent/"download_ckpts"

tiny_imagenet_model_dir = model_base_dir/"tiny_imagenet_teachers/"
cub2011_model_dir = model_base_dir/"cub2011_teachers/"
dtd_model_dir = model_base_dir/"dtd_teachers/"
food101_model_dir = model_base_dir/"food101_teachers/"


model_weights_dict = {
    "tiny-imagenet": {
        "ResNet34": tiny_imagenet_model_dir/"resnet34/xxx.pth",
        "ResNet50": tiny_imagenet_model_dir/"resnet50/xxx.pth",
    },
    "cub2011": {
        "ResNet34": cub2011_model_dir/"resnet34/xxx.pth",
        "ResNet50": cub2011_model_dir/"resnet50/xxx.pth",
    },
    "dtd": {
        "ResNet34": dtd_model_dir/"resnet34/xxx.pth",
        "ResNet50": dtd_model_dir/"resnet50/xxx.pth",
    },
    "food101": {
        "ResNet34": food101_model_dir/"resnet34/xxx.pth",
        "ResNet50": food101_model_dir/"resnet50/xxx.pth",
    }
}

dataset_num_classes_dict = {
    "tiny-imagenet": 200,
    "cub2011": 200,
    "dtd": 47,
    "food101": 101,
}


def get_imagenet_pretrained_model(name: str, dataset: str, pretrained: bool = False):
    """
        pretrained: if True, load the pretrained weights from the model_weights_dict;
                    if False, load the model with ImageNet-pretrained weights.
    """
    num_classes = dataset_num_classes_dict[dataset]
    if pretrained:
        model = get_imagenet_model(
            name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(load_checkpoint(
            model_weights_dict[dataset][name]))
    else:
        model = get_imagenet_model(
            name, pretrained=True, num_classes=num_classes)
        
    return model
