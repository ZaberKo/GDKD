import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from collections import defaultdict
from pathlib import Path
import numpy as np

from mdistiller.engine.cfg import show_cfg, dump_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import (
    load_checkpoint, log_msg, AverageMeter, accuracy)
from mdistiller.models import get_model

from .datasets import get_dataset


cudnn.benchmark = False


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        correct_flags_list = []
        for k in topk:
            correct_flags_list.append(correct[:k].any(dim=0))
    return correct_flags_list


def validate(dataloader, model, num_classes):
    model.eval()

    logits_dict = defaultdict(list)
    feats_dict = defaultdict(list)
    correct_dict = defaultdict(list)

    pbar = tqdm(total=len(dataloader))
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, target = data[:2]
            image = image.float()
            image = image.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            logits, feats = model(image)
            logits = logits.cpu()
            pooled_feat = feats['pooled_feat'].cpu()

            correct_flags, = accuracy(logits, target, topk=(1,))

            for j in range(num_classes):
                logits_dict[j].append(logits[target == j])
                feats_dict[j].append(pooled_feat[target == j])
                correct_dict[j].append(correct_flags[target == j])

            pbar.update()
    pbar.close()

    for i in range(num_classes):
        correct_dict[i] = torch.cat(correct_dict[i]).to(dtype=torch.float32)

    for i in range(num_classes):
        acc = torch.mean(correct_dict[i])
        print(f"Class {i} accuracy: {acc:.4f}")

    currect_tuple = tuple(correct_dict.values())
    acc = torch.mean(torch.cat(currect_tuple))
    print(f"Total accuracy: {acc:.4f}")

    res_logits = {}
    res_feats = {}
    for i in range(num_classes):
        res_logits[f"class{i}"] = torch.concat(logits_dict[i]).numpy()
        res_feats[f"class{i}"] = torch.concat(feats_dict[i]).numpy()

    return res_logits, res_feats


def get_filename(cfg, args, name):
    model_name = cfg.DISTILLER.TEACHER

    filename = f'{cfg.DATASET.TYPE}_{model_name}_{name}'

    if args.save_prefix:
        filename = f"{args.save_prefix}_{filename}"

    if cfg.DATASET.ENHANCE_AUGMENT:
        filename += "_aug"

    if args.train:
        if args.val_transform:
            filename += "_train(val_t)"
        else:
            filename += "_train"
    else:
        filename += "_val"

    filename += ".npz"

    return filename

def main(cfg, args):

    show_cfg(cfg)
    dataloader, num_classes = get_dataset(
        cfg,
        train=args.train,
        use_val_transform=args.val_transform
    )

    print(log_msg("Loading model", "INFO"))

    model_name = cfg.DISTILLER.TEACHER

    model = get_model(cfg, model_name, pretrained=False)
    model.load_state_dict(
        load_checkpoint(args.model_path)
    )

    model.cuda()

    logits_dict, feats_dict = validate(dataloader, model, num_classes)



    path = Path(args.save_dir).expanduser()
    if not path.exists():
        path.mkdir(parents=True)

    logits_filename = get_filename(cfg, args, "logits")
    feats_filename = get_filename(cfg, args, "feats")

    np.savez(path/logits_filename, **logits_dict)
    np.savez(path/feats_filename, **feats_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--model", type=str, default="ResNet34")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--train", action="store_true", help="use train set")
    parser.add_argument("--val-transform",
                        action="store_true", help="use val transform")
    parser.add_argument("--save-dir", type=str, default="exp/kd_logits_data")
    parser.add_argument("--save-prefix", type=str)

    args = parser.parse_args()
    if args.dataset == "imagenet":
        cfg_path = "tools/statistics/imagenet.yaml"
    elif args.dataset == "cifar100":
        cfg_path = "tools/statistics/cifar100.yaml"
    elif args.dataset == "cifar100_aug":
        cfg_path = "tools/statistics/cifar100_aug.yaml"
    else:
        raise NotImplementedError(args.dataset)

    cfg.merge_from_file(cfg_path)
    cfg.DISTILLER.TYPE = "NONE"
    cfg.DISTILLER.TEACHER = args.model
    # cfg.merge_from_list(args.opts)

    cfg.freeze()
    main(cfg, args)
