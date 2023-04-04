from mdistiller.engine import trainer_dict
from mdistiller.engine.cfg import show_cfg, dump_cfg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.dataset import get_dataset
from mdistiller.distillers import distiller_dict
from mdistiller.models import cifar_model_dict, imagenet_model_dict
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record

cudnn.benchmark = True


def local_print(msg, local_rank):
    if local_rank == 0:
        print(msg)


@record
def main(cfg, resume, opts, group_flag=False, id="", local_rank=0):
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

    if local_rank == 0:
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
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(
        cfg, is_distributed=True)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](
                pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        local_print(log_msg("Loading teacher model", "INFO"), local_rank)
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
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )

    # distiller = torch.nn.DataParallel(distiller.cuda())
    distiller = nn.SyncBatchNorm.convert_sync_batchnorm(distiller)
    distiller = distiller.cuda()
    distiller = DDP(
        distiller,
        device_ids=[local_rank],
        find_unused_parameters=True,
        static_graph = True
    )

    if cfg.DISTILLER.TYPE != "NONE":
        local_print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            ), local_rank
        )

    # training
    if group_flag:
        if id == "":
            id = "default"
        experiment_name = experiment_name+"_"+id+"_" + \
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg, is_distributed=True, local_rank=local_rank
    )
    trainer.train(resume=resume)


# for training with original python reqs
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--group", action="store_true")
    parser.add_argument("--id", type=str, default="",
                        help="identifier for training instance")
    parser.add_argument("--record_loss", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data_workers", type=int, default=None)
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("opts", nargs="*")

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.suffix != "":
        cfg.EXPERIMENT.TAG += ","+args.suffix

    if args.record_loss:
        if cfg.DISTILLER.TYPE == "CRD":
            raise ValueError("CRD currently does not support record loss")
        cfg.SOLVER.TRAINER = "custom"

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    local_print(
        f"start local_rank: {local_rank}, world_size: {world_size}", local_rank)
    local_print(
        f"resize batch_size {cfg.SOLVER.BATCH_SIZE} to {cfg.SOLVER.BATCH_SIZE // world_size}", local_rank)
    cfg.SOLVER.BATCH_SIZE = cfg.SOLVER.BATCH_SIZE // world_size
    local_print(
        f"resize test batch_size {cfg.DATASET.TEST.BATCH_SIZE} to {cfg.DATASET.TEST.BATCH_SIZE // world_size}", local_rank)
    cfg.DATASET.TEST.BATCH_SIZE = cfg.DATASET.TEST.BATCH_SIZE // world_size

    if args.data_workers is not None:
        cfg.DATASET.NUM_WORKERS = int(args.data_workers)

    local_print(
        f"resize num_workers {cfg.DATASET.NUM_WORKERS} to {cfg.DATASET.NUM_WORKERS // world_size}", local_rank)
    cfg.DATASET.NUM_WORKERS = cfg.DATASET.NUM_WORKERS // world_size

    cfg.freeze()
    main(cfg, args.resume, args.opts, group_flag=args.group,
         id=args.id, local_rank=local_rank)
