import pickle
from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import show_cfg, dump_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import (
    load_checkpoint, log_msg, AverageMeter, accuracy)
from mdistiller.models import (
    cifar_model_dict, 
    cifar_aug_model_dict,
    imagenet_model_dict
)

import os
import shutil
import argparse
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np

from .utils import get_dataset


cudnn.benchmark = True

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        correct_flags = []
        for k in topk:
            correct_flags.append(correct[:k].sum(0) > 0)
        return correct_flags


def validate(dataloader, model, num_classes):
    logits_dict = {i: [] for i in range(num_classes)}
    num_iter = len(dataloader)
    pbar = tqdm(range(num_iter))

    model.eval()

    correct_dict={i: [] for i in range(num_classes)}
    

    with torch.no_grad():
        for i, (image, target, index) in enumerate(dataloader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            logits, _ = model(image)
            logits = logits.cpu()
            

            correct_flags, = accuracy(logits, target, topk=(1,))

            for j in range(num_classes):
                logits_dict[j].append(logits[target == j])

            for j in range(num_classes):
                correct_dict[j].append(correct_flags[target == j])

            pbar.update()
    pbar.close()

    for i in range(num_classes):
        correct_dict[i]=torch.cat(correct_dict[i]).to(dtype=torch.float32)

    for i in range(num_classes):
        acc = torch.mean(correct_dict[i])
        print(f"Class {i} accuracy: {acc:.4f}")

    currect_tuple=tuple(correct_dict.values())
    acc = torch.mean(torch.cat(currect_tuple))
    print(f"Total accuracy: {acc:.4f}")

    res = {}
    for i in range(num_classes):
        res[f"class{i}"] = torch.concat(logits_dict[i]).numpy()

    return res


def main(cfg, use_val_transform=False):
    teacher_model = cfg.DISTILLER.TEACHER

    show_cfg(cfg)
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg, use_val_transform)

    print(log_msg("Loading teacher model", "INFO"))
    if cfg.DATASET.TYPE == "imagenet":
        model_teacher = imagenet_model_dict[teacher_model](
            pretrained=True)
    else:
        if cfg.DATASET.ENHANCE_AUGMENT:
            net, pretrain_model_path = cifar_aug_model_dict[teacher_model]
        else:
            net, pretrain_model_path = cifar_model_dict[teacher_model]
        assert (
            pretrain_model_path is not None
        ), "no pretrain model for teacher {}".format(teacher_model)
        model_teacher = net(num_classes=num_classes)
        model_teacher.load_state_dict(
            load_checkpoint(pretrain_model_path)["model"])

    model_teacher.cuda()

    # data_path = f"exp/{cfg.DATASET.TYPE}/{teacher_model}"
    # if os.path.exists(data_path):
    #     shutil.rmtree(data_path)
    # os.makedirs(data_path)
    logits_dict = validate(train_loader, model_teacher, num_classes)

    np.savez(f'exp/{cfg.DATASET.TYPE}_{teacher_model}{"_aug" if cfg.DATASET.ENHANCE_AUGMENT else ""}_logits.npz', **logits_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="ResNet34")
    parser.add_argument("--use_val_transform", action="store_true")

    args = parser.parse_args()
    if args.dataset == "imagenet":
        cfg_path = "tools/statistics/imagenet.yaml"
    elif args.dataset == "cifar100":
        cfg_path = "tools/statistics/cifar100.yaml"
    elif args.dataset == "cifar100_aug":
        cfg_path = "tools/statistics/cifar100_aug.yaml"

    cfg.merge_from_file(cfg_path)
    cfg.DISTILLER.TEACHER = args.model
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, use_val_transform=args.use_val_transform)
