import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from mdistiller.dataset import get_dataset
from mdistiller.distillers import get_distiller
from mdistiller.engine import Trainer
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import dump_cfg, show_cfg
from mdistiller.engine.utils import log_msg

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
    # experiment_name = cfg.EXPERIMENT.PROJECT + "/" + experiment_name
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(
                project=cfg.EXPERIMENT.PROJECT,
                name=experiment_name,
                tags=tags,
                config=dump_cfg(cfg),
                group=experiment_name+"_group" if group_flag else None,
                # helps resolve "InitStartError: Error communicating with wandb process"
                settings=wandb.Settings(start_method="fork")
            )
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.defrost()
            cfg.LOG.WANDB = False
            cfg.freeze()

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    distiller = get_distiller(cfg, num_data=num_data)

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # backward compatibility for `.module`:
    # We recommend to use one GPU since DP is deprecated:
    # https://github.com/pytorch/pytorch/issues/65936
    distiller = torch.nn.DataParallel(distiller.cuda())

    if int(os.environ.get("USE_TORCH_COMPILE", "0")):
        # require torch>=2.0
        print(log_msg("Enable torch.compile", "INFO"))
        distiller = torch.compile(distiller, mode="reduce-overhead")

    # training
    if group_flag:
        if id == "":
            id = "default"
        experiment_name = experiment_name+"_"+id+"_" + \
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    trainer = Trainer(
       experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


def setup_cfg(args):
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if cfg.DATASET.ENHANCE_AUGMENT:
        cfg.EXPERIMENT.TAG += ",aug"

    if args.suffix != "":
        cfg.EXPERIMENT.TAG += ","+args.suffix

    if args.data_workers is not None:
        cfg.DATASET.NUM_WORKERS = int(args.data_workers)

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cfg.EXPERIMENT.TAG += ",seed_"+str(seed)

    cfg.freeze()

# for training with original python reqs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--group", action="store_true")
    parser.add_argument("--id", type=str, default="",
                        help="identifier for training instance")
    parser.add_argument("--suffix", type=str, nargs="?", default="", const="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data_workers", type=int, default=None)
    parser.add_argument("opts", nargs="*")

    args = parser.parse_args()
    setup_cfg(args)
    
    main(cfg, args.resume, args.opts, group_flag=args.group, id=args.id)
