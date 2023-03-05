import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict, defaultdict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
    reduce_tensor
)


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg, is_distributed=False, local_rank=0):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1

        username = getpass.getuser()
        self.is_distributed = is_distributed
        self.local_rank = local_rank

        if self.local_rank == 0:
            # init loggers
            self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
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

    def log(self, lr, epoch, log_dict):
        if self.local_rank == 0:                
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
                    "lr: {:.2f}".format(float(lr)) + os.linesep,
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
        
        if self.local_rank == 0:
            print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
            with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
                writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)

        num_iter = len(self.train_loader)
        if self.local_rank == 0:
            pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            if self.local_rank == 0:
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
        if self.local_rank == 0:
            pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(
            self.val_loader, self.distiller)

        # TODO: check sync train_meters and test_acc

        if self.local_rank == 0:
            # log
            log_dict = OrderedDict(
                {
                    "train_acc": train_meters["top1"].avg,
                    "train_loss": train_meters["losses"].avg,
                    "test_acc": test_acc,
                    "test_acc_top5": test_acc_top5,
                    "test_loss": test_loss,
                }
            )
            self.log(lr, epoch, log_dict)
            # saving checkpoint
            state = {
                "epoch": epoch,
                "model": self.distiller.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
            }
            student_state = {"model": self.distiller.module.student.state_dict()}
            save_checkpoint(state, os.path.join(self.log_path, "latest"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_latest")
            )
            if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
                save_checkpoint(
                    state, os.path.join(self.log_path, "epoch_{}".format(epoch))
                )
                save_checkpoint(
                    student_state,
                    os.path.join(self.log_path, "student_{}".format(epoch)),
                )
            # update the best
            if test_acc >= self.best_acc:
                save_checkpoint(state, os.path.join(self.log_path, "best"))
                save_checkpoint(
                    student_state, os.path.join(self.log_path, "student_best")
                )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        
        # for dist training
        loss = reduce_tensor(loss.detach())
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        
        train_meters["losses"].update(
            loss.cpu().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1.item(), batch_size)
        train_meters["top5"].update(acc5.item(), batch_size)
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


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))

        # for dist training
        loss = reduce_tensor(loss.detach())
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        train_meters["losses"].update(
            loss.cpu().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1.item(), batch_size)
        train_meters["top5"].update(acc5.item(), batch_size)
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


class RecordTrainer(BaseTrainer):
    """
        Add record for ce_loss and kd_loss
    """
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg, is_distributed=False, local_rank=0):
        super(RecordTrainer,self).__init__(experiment_name, distiller, train_loader, val_loader, cfg, is_distributed, local_rank)

        self.enable_progress_bar = cfg.LOG.ENABLE_PROGRESS_BAR


    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "loss_ce": AverageMeter(),
            "loss_kd": AverageMeter(),
        }
        # train_meters = defaultdict(AverageMeter)

        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        num_iter = len(self.train_loader)
        if self.enable_progress_bar and self.local_rank==0:
            pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            if self.enable_progress_bar and self.local_rank==0:
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()

        if self.enable_progress_bar and self.local_rank==0:
            pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(
            self.val_loader, self.distiller)

        # log
        if self.local_rank == 0:
            log_dict = OrderedDict(
                {
                    "train_acc": train_meters["top1"].avg,
                    "train_loss": train_meters["losses"].avg,
                    "test_acc": test_acc,
                    "test_acc_top5": test_acc_top5,
                    "test_loss": test_loss,
                    "train_loss_ce": train_meters["loss_ce"].avg,
                    "train_loss_kd": train_meters["loss_kd"].avg,
                }
            )
            self.log(lr, epoch, log_dict)
            # saving checkpoint
            state = {
                "epoch": epoch,
                "model": self.distiller.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_acc": self.best_acc,
            }
            student_state = {"model": self.distiller.module.student.state_dict()}
            save_checkpoint(state, os.path.join(self.log_path, "latest"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_latest")
            )
            if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
                save_checkpoint(
                    state, os.path.join(self.log_path, "epoch_{}".format(epoch))
                )
                save_checkpoint(
                    student_state,
                    os.path.join(self.log_path, "student_{}".format(epoch)),
                )
            # update the best
            if test_acc >= self.best_acc:
                save_checkpoint(state, os.path.join(self.log_path, "best"))
                save_checkpoint(
                    student_state, os.path.join(self.log_path, "student_best")
                )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))

        # for dist training
        loss = reduce_tensor(loss.detach())
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        train_meters["losses"].update(
            loss.cpu().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1.item(), batch_size)
        train_meters["top5"].update(acc5.item(), batch_size)

        # record "loss_ce" & "loss_kd"
        for name, loss in losses_dict.items():
            loss = reduce_tensor(loss.detach())
            train_meters[name].update(loss.cpu().numpy().mean(), batch_size)

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