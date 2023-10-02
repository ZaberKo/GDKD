from .cifar import get_cifar100_model
from .imagenet import get_imagenet_model
from .imagenet_pretrain import get_imagenet_pretrained_model

model_tag_dict = {
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

transfer_learning_datasets = ["tiny-imagenet", "cub2011", "dtd", "food101"]


def get_model(cfg, name, pretrained=False):
    if cfg.DATASET.TYPE == "cifar100":
        model = get_cifar100_model(
            name, pretrained=pretrained, aug=cfg.DATASET.ENHANCE_AUGMENT)
    elif cfg.DATASET.TYPE == "imagenet":
        model = get_imagenet_model(name, pretrained=pretrained)
    elif cfg.DATASET.TYPE in transfer_learning_datasets:
        model = get_imagenet_pretrained_model(
            name, cfg.DATASET.TYPE, pretrained=pretrained)
    else:
        raise NotImplementedError

    return model
