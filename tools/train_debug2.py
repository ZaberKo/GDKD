from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import show_cfg, dump_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
# from mdistiller.dataset import get_dataset
from mdistiller.distillers import distiller_dict
from mdistiller.models import cifar_model_dict, imagenet_model_dict
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime

from tqdm import tqdm

from .statistics.utils import get_dataset
from mdistiller.distillers import DKDMod, CRD


cudnn.benchmark = True


def main(cfg, resume, opts, group_flag=False, id=""):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v)
                          for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += "|"+",".join(addtional_tags)

    # experiment_name = f"{cfg.EXPERIMENT.PROJECT}/{experiment_name}"
    experiment_name = cfg.EXPERIMENT.PROJECT + "/" + experiment_name

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(
        cfg,
        use_val_transform=False
    )

    print(log_msg("Loading teacher model", "INFO"))
    if cfg.DATASET.TYPE == "imagenet":
        model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](
            pretrained=True)
        model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](
            pretrained=False)
    else:
        net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
        assert (
            pretrain_model_path is not None
        ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
        model_teacher = net(num_classes=num_classes)
        model_teacher.load_state_dict(
            load_checkpoint(pretrain_model_path)["model"])
        model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
            num_classes=num_classes
        )

    distiller = CRD(model_student, model_teacher, cfg, num_data)

    # distiller = DKDMod(model_student, model_teacher, cfg)
    distiller = distiller.cuda()

    data=next(iter(train_loader))

    image, target, index = data
    image = image.cuda(non_blocking=True)


    t_logits, t_feats = distiller.teacher(image)
    print(f"Teacher {cfg.DISTILLER.TEACHER}:",t_feats["pooled_feat"].shape)

    s_logits, s_feats = distiller.student(image)
    print(f"Student {cfg.DISTILLER.STUDENT}:", s_feats["pooled_feat"].shape)


# for training with original python reqs
if __name__ == "__main__":

    cfg.merge_from_file("configs/cifar100/dkdmod/res32x4_shuv2.yaml")
    # cfg.merge_from_file("configs/cifar100/crd/res32x4_shuv2.yaml")
    cfg.SOLVER.TRAINER = "custom"
    # cfg.DKDMOD.BETA = 8

    cfg.DKDMOD.STRATEGY = "target"
    cfg.DATASET.ENHANCE_AUGMENT = False

    cfg.freeze()
    main(cfg, False, [], group_flag=False, id="")
