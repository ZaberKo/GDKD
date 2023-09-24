from functools import partial
from pathlib import Path
from collections import OrderedDict
import torch


from ..imagenet import get_imagenet_model
from mdistiller.engine.utils import load_checkpoint


tiny_imagenet_model_dir = Path(__file__).absolute(
).parent/"../../../download_ckpts/tiny_imagenet_teachers/"



tiny_imagenet_model_weights_dict = {
    "ResNet34": tiny_imagenet_model_dir/"resnet34/xxx.pth",
    "ResNet50": tiny_imagenet_model_dir/"resnet50/xxx.pth",
}


def get_tiny_imagenet_model(name, pretrained=False):
    if pretrained:
        model = get_imagenet_model(name, pretrained=False, num_classes=200)
        model.load_state_dict(load_checkpoint(tiny_imagenet_model_weights_dict[name]))
    else:
        model = get_imagenet_model(name, pretrained=True, num_classes=200)

    return model
