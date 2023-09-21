#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import detectron2.utils.comm as comm

from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.utils.events import EventStorage

from detectron2.checkpoint import DetectionCheckpointer

from trainer import Trainer
from config import get_distiller_config

import argparse
from tqdm import tqdm
import numpy as np
import torch
from pathlib import Path
import os

from .dkddebug import DKDDebug


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_distiller_config()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)

    if args.img_bs is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.img_bs
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


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


def calc_pred_logits(distiller, data_loader, num_iter):
    num_classes=81
    logits_dict = {i: [] for i in range(num_classes)}
    correct_dict = {i: [] for i in range(num_classes)}

    # distiller.eval()
    distiller.train()

    pbar = tqdm(range(num_iter))

    with EventStorage(0) as storage:
        with torch.no_grad():
            for data, iteration in zip(data_loader, range(num_iter)):
                storage.iter = iteration

                s_logits, t_logits, gt_classes = distiller.forward_predictions(data)

                correct_flags, = accuracy(t_logits, gt_classes)

                for i in range(num_classes):
                    logits_dict[i].append(t_logits[gt_classes == i].cpu())
                    correct_dict[i].append(correct_flags[gt_classes == i].to(dtype=torch.float32, device=torch.device("cpu")))

                pbar.update()
            pbar.close()

    for i in range(num_classes):
        correct_dict[i]=torch.cat(correct_dict[i])

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


def main(args):
    cfg = setup(args)

    output_dir = Path("debug/calc_logits")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    distiller: DKDDebug = Trainer.build_model(cfg)


    data_loader = build_detection_train_loader(cfg)

    # for i in range(0,180000, 9000):
    for i in range(3):
        id = 59999+60000*i
        ckpt_path = os.path.join(args.ckpt,f"model_{id:07d}.pth")
        print(f"load: {ckpt_path}")
        checkpointer = DetectionCheckpointer(distiller, output_dir)
        checkpointer.resume_or_load(ckpt_path, resume=False)
        logits_dict = calc_pred_logits(distiller, data_loader, args.num_iter)

        np.savez(output_dir/f"DKD-R18-R101-iter{id}.npz", **logits_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="debug/DKD-R18-R101.yaml",
                        metavar="FILE", help="path to config file")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-iter", type=int, default=1000,
                        help="max iteration to run")
    parser.add_argument("--img-bs", type=int)

    args = parser.parse_args()

    main(args)
