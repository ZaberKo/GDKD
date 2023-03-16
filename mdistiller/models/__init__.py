from .cifar import cifar_model_dict, cifar_aug_model_dict
from .imagenet import imagenet_model_dict


model_tag_dict ={
    "resnet32x4": "res32x4",
    "resnet8x4": "res8x4",
    "ShuffleV1": "shuv1",
    "ShuffleV2": "shuv2",
    "ResNet18": "res18",
    "ResNet34": "res34",
    "ResNet50": "res50",
    "MobileNetV1": "mobilenetv1",
    "MobileNetV2": "mv2",
    "resnet20": "res20",
    "resnet32": "res32",
    "resnet56": "res56",
    "resnet110": "res110",
    "vgg13": "vgg13",
    "vgg8": "vgg8",
    "wrn_40_2": "wrn_40_2",
    "wrn_16_2": "wrn_16_2",
    "wrn_40_1": "wrn_40_1",
}