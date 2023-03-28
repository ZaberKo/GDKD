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

from .utils import get_dataset
from ..debug.DKDMod_debug import DKDMod

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

    distiller = DKDMod(model_student, model_teacher, cfg)

    # cfg2=cfg.clone()
    # cfg2.merge_from_file(
    #     "configs/cifar100/dkd/res32x4_res8x4.yaml"
    # )
    # net, pretrain_model_path = cifar_model_dict[cfg2.DISTILLER.TEACHER]
    # assert (
    #     pretrain_model_path is not None
    # ), "no pretrain model for teacher {}".format(cfg2.DISTILLER.TEACHER)
    # model_teacher2 = net(num_classes=num_classes)
    # model_teacher2.load_state_dict(
    #     load_checkpoint(pretrain_model_path)["model"])
    # model_student2 = cifar_model_dict[cfg2.DISTILLER.STUDENT][0](
    #     num_classes=num_classes
    # )
    # distiller2 = distiller_dict[cfg2.DISTILLER.TYPE](
    #             model_student2, model_teacher2, cfg2
    #         )

    distiller = distiller.cuda()
    distiller = torch.nn.DataParallel(distiller)
    # distiller2 = distiller2.cuda()

    # training
    if group_flag:
        if id == "":
            id = "default"
        experiment_name = experiment_name+"_"+id+"_" + \
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    experiment_name = "cifar100_baselines/dkdmod,res32x4,res8x4|LOG.WANDB:False_bak"
    log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)

    suffix = ""
    # load_flag = False
    load_flag = True

    if load_flag:
        epochs=[0]+list(range(40,241,40))
    else:
        epochs = [240]

    for epoch in epochs:
        if epoch != 0:
            print("load state:", os.path.join(log_path, f"epoch_{epoch}"))
            state = load_checkpoint(os.path.join(log_path, f"epoch_{epoch}"))
            distiller.load_state_dict(state["model"])

        distiller.eval()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            image, target, index = data
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)

            distiller.module.forward_train(
                image=image, target=target, epoch=epoch, suffix=suffix)
            
        distiller.module.clean_record()

        # distiller2(image=image, target=target, epoch=0)
    


# for training with original python reqs
if __name__ == "__main__":

    cfg.merge_from_file("configs/cifar100/dkdmod/res32x4_res8x4.yaml")
    cfg.SOLVER.TRAINER = "custom"
    # cfg.DKDMOD.BETA = 8

    cfg.DKDMOD.STRATEGY = "target"
    cfg.DATASET.ENHANCE_AUGMENT = False

    cfg.freeze()
    main(cfg, False, [], group_flag=False, id="")
