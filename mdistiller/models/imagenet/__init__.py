import os

import torch.utils.model_zoo as model_zoo
from pathlib import Path
from collections import OrderedDict

from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, get_default_model_weights
from .mobilenetv1 import MobileNetV1

imagenet_model_dir = Path(__file__).absolute().parent/"../../../download_ckpts/imagenet_teachers/"



imagenet_model_dict = {
    "ResNet18": (resnet18, imagenet_model_dir/"resnet18"),
    "ResNet34": (resnet34, imagenet_model_dir/"resnet34"),
    "ResNet50": (resnet50, imagenet_model_dir/"resnet50"),
    "ResNet101": (resnet101, imagenet_model_dir/"resnet101"),
    "MobileNetV1": (MobileNetV1, None) # use for student only
}



def get_imagenet_model(name, pretrained=False, num_classes=1000):
    gen_model, model_dir = imagenet_model_dict[name]
    if num_classes == 1000:
        model = gen_model(pretrained=pretrained, model_dir=model_dir, num_classes=1000)
    else:
        model = gen_model(pretrained=False, num_classes=num_classes)

        if pretrained:
            state_dict = get_default_model_weights(model_dir.name, model_dir)
            new_state_dict = OrderedDict({
                k: v
                for k, v in state_dict.items()
                if not k.startswith("fc.")
            })

            model.load_state_dict(new_state_dict, strict=False)
                
    return model