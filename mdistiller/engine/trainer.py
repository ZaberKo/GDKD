import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict
from tensorboardX import SummaryWriter

from .validate import validate
from .utils import (
    AverageMeter,
    accuracy,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
    reduce_tensor,
    is_distributed,
    is_main_process,
    Timer
)


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

        if is_main_process():
            # init loggers
            self.log_path = os.path.join(
                cfg.LOG.PREFIX, cfg.EXPERIMENT.PROJECT, experiment_name)
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tf_writer = SummaryWriter(
                os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
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

    def log(self, lr, epoch, log_dict):
        if is_main_process():
            # tensorboard log
            for k, v in log_dict.items():
                self.tf_writer.add_scalar(k, v, epoch)
            self.tf_writer.flush()
            # wandb log
            if self.cfg.LOG.WANDB:
                import wandb

                wandb_log_dict = {"current lr": lr, **log_dict}

                wandb.log(wandb_log_dict)
            if log_dict["test_acc"] > self.best_acc:
                self.best_acc = log_dict["test_acc"]
                if self.cfg.LOG.WANDB:
                    wandb.run.summary["best_acc"] = self.best_acc
            # worklog.txt
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                lines = [
                    "-" * 25 + os.linesep,
                    "epoch: {}".format(epoch) + os.linesep,
                    "lr: {:.8f}".format(float(lr)) + os.linesep,
                ]
                for k, v in log_dict.items():
                    lines.append("{}: {:.2f}".format(k, v) + os.linesep)
                lines.append("-" * 25 + os.linesep)
                writer.writelines(lines)

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1

        if is_main_process():
            print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                writer.write("best_acc\t" +
                             "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        # lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)

        self.train_meters = defaultdict(AverageMeter)
        self.train_info_meters = defaultdict(AverageMeter)

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        if self.enable_progress_bar and is_main_process():
            pbar = tqdm(range(len(self.train_loader)))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch)
            if self.enable_progress_bar and is_main_process():
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()

        if self.enable_progress_bar and is_main_process():
            pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(
            self.val_loader, self.distiller)

        lr = self.lr_scheduler.get_last_lr()[0]
        self.lr_scheduler.step()

        # log
        if is_main_process():
            log_dict = OrderedDict(
                {
                    "train_acc": self.train_meters["top1"].avg,
                    "train_loss": self.train_meters["losses"].avg,
                    "test_acc": test_acc,
                    "test_acc_top5": test_acc_top5,
                    "test_loss": test_loss,
                    "train_loss_ce": self.train_meters["loss_ce"].avg,
                }
            )
            if "loss_kd" in self.train_meters:
                log_dict["train_loss_kd"] = self.train_meters["loss_kd"].avg

            log_dict.update({
                k: v.avg for k, v in self.train_info_meters.items()
            })
            self.log(lr, epoch, log_dict)

            # saving checkpoint
            state = {
                "epoch": epoch,
                "model": self.distiller.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
            }
            student_state = {
                "model": self.distiller.module.student.state_dict()}
            save_checkpoint(state, os.path.join(self.log_path, "latest.pth"))
            save_checkpoint(
                student_state, os.path.join(
                    self.log_path, "student_latest.pth")
            )
            if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
                save_checkpoint(
                    state, os.path.join(self.log_path, f"epoch_{epoch}.pth")
                )
                save_checkpoint(
                    student_state,
                    os.path.join(self.log_path, f"student_{epoch}.pth"),
                )
            # update the best
            if test_acc >= self.best_acc:
                save_checkpoint(state, os.path.join(self.log_path, "best.pth"))
                save_checkpoint(
                    student_state, os.path.join(
                        self.log_path, "student_best.pth")
                )

    def train_iter(self, data, epoch):
        self.optimizer.zero_grad()

        train_meters = self.train_meters

        with Timer() as train_timer:
            with Timer() as data_timer:
                image, target, other_data_dict = self._preprocess_data(data)
            train_meters["data_time"].update(data_timer.interval)

            # forward
            preds, losses_dict = self.distiller(
                image=image, target=target, epoch=epoch, **other_data_dict)

            # backward
            loss = sum([l.mean() for l in losses_dict.values()])
            loss.backward()
            self.optimizer.step()

        train_meters["training_time"].update(train_timer.interval)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))

        train_info = self.distiller.module.get_train_info()
        for key, info in train_info.items():
            if isinstance(info, torch.Tensor):
                self.train_info_meters[key].update(
                    info.item(), batch_size
                ).all_reduce()
            else:
                # for non-tensor info, just update on the local process
                self.train_info_meters[key].update(
                    info, batch_size
                )

        train_meters["losses"].update(loss.tolist(), batch_size).all_reduce()
        train_meters["top1"].update(acc1.item(), batch_size).all_reduce()
        train_meters["top5"].update(acc5.item(), batch_size).all_reduce()

        # record "loss_ce" & "loss_kd"
        for name, loss in losses_dict.items():
            train_meters[name].update(loss.item(), batch_size).all_reduce()

        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg

    def _preprocess_data(self, data) -> dict:
        if type(self.train_loader.dataset).__name__.endswith("InstanceSample"):
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
