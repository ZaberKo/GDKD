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
from detectron2.checkpoint import DetectionCheckpointer

from trainer import Trainer
from config import get_distiller_config

import os
from datetime import datetime
import time
import logging
import wandb


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_distiller_config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.debug:
        cfg.defrost()
        cfg.OUTPUT_DIR = "./output/debug"
        cfg.freeze()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    if comm.is_main_process():

        tags = cfg.EXPERIMENT.TAG
        if tags is None or len(tags) == 0:
            tags = [cfg.KD.TYPE]
        # tags.append(cfg.KD.TYPE)

        if args.opts:
            addtional_tags = ["{}:{}".format(k, v)
                              for k, v in zip(args.opts[::2], args.opts[1::2])]
            tags += addtional_tags

        if not cfg.EXPERIMENT.NAME:
            experiment_name = "-".join(tags)
        else:
            experiment_name = cfg.EXPERIMENT.NAME

        wandb_cfg=cfg.clone()
        # wandb_cfg.defrost()

        if cfg.EXPERIMENT.WANDB:
            wandb.init(
                project=cfg.EXPERIMENT.PROJECT,
                name=experiment_name,
                # config=wandb_cfg, # set later at WandbWriter
                tags=tags,
                group=experiment_name+"_group" if args.group else None,
                settings=wandb.Settings(start_method="fork")
            )

        cfg.defrost()
        # cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f'{cfg.KD.TYPE}|{",".join(addtional_tags)}_{args.id}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}')
        output_dirname = f'{experiment_name}_{wandb.run.id}'
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_dirname)
        cfg.freeze()

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )

    trainer.train()

    comm.synchronize()
    if comm.is_main_process():
        wandb.finish()
    logger = logging.getLogger("detectron2")
    logger.info("Wait for 30 seconds before exiting")

    time.sleep(30)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--group", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
