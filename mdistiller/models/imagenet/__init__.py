import os
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .mobilenetv1 import MobileNetV1

from functools import partial
from pathlib import Path

imagenet_model_dir = Path(__file__).absolute().parent/"../../../download_ckpts/imagenet_teachers/"


imagenet_model_dict = {
    "ResNet18": partial(resnet18, model_dir=imagenet_model_dir/"resnet18"),
    "ResNet34": partial(resnet34, model_dir=imagenet_model_dir/"resnet34"),
    "ResNet50": partial(resnet50, model_dir=imagenet_model_dir/"resnet50"),
    "ResNet101": partial(resnet101, model_dir=imagenet_model_dir/"resnet101"),
    "MobileNetV1": MobileNetV1, # use for student only
}
