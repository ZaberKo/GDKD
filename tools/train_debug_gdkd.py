from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import show_cfg, dump_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
# from mdistiller.dataset import get_dataset
from mdistiller.distillers import distiller_dict
from mdistiller.models import cifar100_model_dict, imagenet_model_dict
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime

from tqdm import tqdm

from .statistics.datasets import get_dataset
from .debug.GDKD_debug import GDKD

cudnn.benchmark = True


def main(cfg, name, model_path):
    # cfg & loggers
    # show_cfg(cfg)
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
        net, pretrain_model_path = cifar100_model_dict[cfg.DISTILLER.TEACHER]
        assert (
            pretrain_model_path is not None
        ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
        model_teacher = net(num_classes=num_classes)
        model_teacher.load_state_dict(
            load_checkpoint(pretrain_model_path)["model"])
        model_student = cifar100_model_dict[cfg.DISTILLER.STUDENT][0](
            num_classes=num_classes
        )

    distiller = GDKD(model_student, model_teacher, cfg)

    distiller = distiller.cuda()
    distiller = torch.nn.DataParallel(distiller)

    log_path = os.path.join(cfg.LOG.PREFIX, model_path)

    epochs=list(range(0,241,40))

    suffix=""
    if cfg.DATASET.ENHANCE_AUGMENT:
        suffix += "_aug"

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

            distiller.module.record_grad(image=image, target=target)

        distiller.module.save_record(
            path=f"exp/grad/gdkd/{name}{suffix}/epoch{epoch}.npz")
        distiller.module.reset_record()


# for training with original python reqs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str,
                        default="configs/cifar100/gdkd/res32x4_res8x4.yaml")
    parser.add_argument("--model_path", type=str, default="cifar100_baselines/gdkd,res32x4,res8x4,aug")
    parser.add_argument("--name", type=str, default="resnet32x4_8x4_grad")
    parser.add_argument("--aug", type=str, default="auto_aug")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.TRAINER = "custom"

    if args.aug == "auto_aug":
        print("use auto_aug")
        cfg.DATASET.ENHANCE_AUGMENT = True
    else:
        cfg.DATASET.ENHANCE_AUGMENT = False

    cfg.freeze()
    main(cfg, args.name, args.model_path)
