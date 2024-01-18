import argparse
import os
import random
from datetime import datetime
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from mdistiller.dataset import get_dataset
from mdistiller.distillers import get_distiller
# from mdistiller.engine import Trainer
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import dump_cfg, show_cfg
from mdistiller.engine.utils import log_msg, is_distributed

cudnn.benchmark = True

class Trainer():
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.lr_scheduler = self.init_lr_scheduler(cfg, self.optimizer)
        self.best_acc = -1
        self.is_distributed = is_distributed()

        self.enable_progress_bar = cfg.LOG.ENABLE_PROGRESS_BAR

        self.train_meters = None
        self.train_info_meters = None

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                nesterov=cfg.SOLVER.SGD_NESTEROV,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def init_lr_scheduler(self, cfg, optimizer):
        if cfg.SOLVER.LR_SCHEDULER == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, cfg.SOLVER.EPOCHS, eta_min=cfg.SOLVER.LR_MIN)
        elif cfg.SOLVER.LR_SCHEDULER == "step":
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, cfg.SOLVER.LR_DECAY_STAGES, cfg.SOLVER.LR_DECAY_RATE
            )
        else:
            raise NotImplementedError(cfg.SOLVER.LR_SCHEDULER.TYPE)
        return lr_scheduler
    
    def _preprocess_data(self, data) -> dict:
        if self.cfg.DISTILLER.TYPE == "CRD":
            image, target, index, contrastive_index = data
            image = image.float()
            image = image.cuda()
            target = target.cuda()
            index = index.cuda()
            contrastive_index = contrastive_index.cuda()
            return image, target, dict(index=index, contrastive_index=contrastive_index)
        else:
            image, target, index = data
            image = image.float()
            image = image.cuda()
            target = target.cuda()
            return image, target, {}

class Timer:
    def __init__(self) -> None:
        self.time = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_time = 0.0
        self.cnt = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        self.total_time += time.perf_counter() - self.start_time
        self.cnt += 1
        self.time = self.total_time / self.cnt

    def __str__(self) -> str:
        return f"{self.time*1000:.2f}ms"


def get_train_speed(trainer, epochs=1):
    timer = Timer()
    trainer.distiller.train()

    torch.cuda.synchronize()
    for epoch in range(0, epochs):
        with tqdm(total=len(trainer.train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            dataloader=iter(trainer.train_loader)
            for i in range(len(trainer.train_loader)):
                with timer:
                    data = next(dataloader)
                    image, target, other_data_dict = trainer._preprocess_data(data)
                    # forward
                    preds, losses_dict = trainer.distiller(
                        image=image, target=target, epoch=epoch, **other_data_dict)

                    # backward
                    loss = sum([l.mean() for l in losses_dict.values()])
                    loss.backward()
                    trainer.optimizer.step()
                pbar.update(1)
                # pbar.set_description(f'[train speed: {timer}]')
                pbar.set_postfix_str(f'train speed: {timer}')


    print(f'Average train speed per iter: {timer}')



def get_train_speed2(trainer, epochs=1):
    timer = Timer()
    trainer.distiller.train()

    torch.cuda.synchronize()

    timer_all = Timer()
    with timer_all:
        for epoch in range(0, epochs):
            with tqdm(total=len(trainer.train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                dataloader=iter(trainer.train_loader)
                for i in range(len(trainer.train_loader)):
                    with timer:
                        data = next(dataloader)
                        image, target, other_data_dict = trainer._preprocess_data(data)
                        # forward
                        preds, losses_dict = trainer.distiller(
                            image=image, target=target, epoch=epoch, **other_data_dict)

                        # backward
                        loss = sum([l.mean() for l in losses_dict.values()])
                        loss.backward()
                        trainer.optimizer.step()
                    pbar.update(1)
                    # pbar.set_description(f'[train speed: {timer}]')
                    pbar.set_postfix_str(f'train speed: {timer}')
        torch.cuda.synchronize()

    print(f'Average train speed per iter: {timer}')
    avg_time=timer_all.time/(epochs*len(trainer.train_loader))
    print(f'Average train speed per iter (all): {avg_time*1000:.2f}ms')

def main(cfg, epochs, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v)
                          for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += "|"+",".join(addtional_tags)

    # cfg & loggers
    # show_cfg(cfg)
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

    # Test training
    print(f'Start test {experiment_name}')
    experiment_name = "train_speed"

    experiment_name = experiment_name + \
        f'_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

    trainer = Trainer(
        experiment_name, distiller, train_loader, val_loader, cfg
    )

    # get_train_speed2(trainer, epochs=epochs)

    print("="*10)


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
    parser.add_argument("--suffix", type=str, nargs="?", default="", const="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--data_workers", type=int, default=None)
    parser.add_argument("opts", nargs="*")

    args = parser.parse_args()

    if os.environ.get("KD_EXPERIMENTAL", "0") == "1":
        import mdistiller.distillers.experimental as experimental

    setup_cfg(args)

    main(cfg, args.epochs, args.opts)
