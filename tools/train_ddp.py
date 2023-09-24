
import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from mdistiller.dataset import get_dataset
from mdistiller.distillers import get_distiller
from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import dump_cfg, show_cfg
from mdistiller.engine.utils import log_msg, is_main_process, local_print

cudnn.benchmark = True


@record
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

    if is_main_process():
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

    # NOTE: set at train script cli level: "PXB"|"NVL"
    # os.environ["NCCL_NET_GDR_LEVEL"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    # os.environ["NCCL_P2P_LEVEL"] = "PXB"

    # init dist
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    distiller = get_distiller(cfg, num_data)

    if cfg.DISTILLER.TYPE != "NONE":
        local_print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.get_extra_parameters()
                ),
                "INFO",
            ))

    distiller = nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    distiller = distiller.cuda()
    distiller = DDP(
        distiller,
        device_ids=[local_rank],
        find_unused_parameters=True,
        static_graph=True
    )

    # training
    if group_flag:
        if id == "":
            id = "default"
        experiment_name = experiment_name+"_"+id+"_" + \
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
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

    local_print(f"resize batch_size {cfg.SOLVER.BATCH_SIZE} to {cfg.SOLVER.BATCH_SIZE // world_size}")
    cfg.SOLVER.BATCH_SIZE = cfg.SOLVER.BATCH_SIZE // world_size

    local_print(
        f"resize test batch_size {cfg.DATASET.TEST.BATCH_SIZE} to {cfg.DATASET.TEST.BATCH_SIZE // world_size}")
    cfg.DATASET.TEST.BATCH_SIZE = cfg.DATASET.TEST.BATCH_SIZE // world_size

    local_print(
        f"resize num_workers {cfg.DATASET.NUM_WORKERS} to {cfg.DATASET.NUM_WORKERS // world_size}")
    cfg.DATASET.NUM_WORKERS = cfg.DATASET.NUM_WORKERS // world_size

    if args.seed is not None:
        seed = args.seed + dist.get_rank()
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
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_workers", type=int, default=None)
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("opts", nargs="*")

    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_print(f"start local_rank: {local_rank}, world_size: {world_size}")

    setup_cfg(args)

    main(cfg, args.resume, args.opts, group_flag=args.group, id=args.id)
