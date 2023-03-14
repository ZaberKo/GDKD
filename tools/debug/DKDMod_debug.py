import torch
import torch.nn as nn
import torch.nn.functional as F

from mdistiller.distillers._base import Distiller
from mdistiller.distillers.utils import kl_div

from pathlib import Path
import numpy as np


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, target.unsqueeze(1), 1)
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, target.unsqueeze(1), 0)
    return mask


def get_top1_masks(logits, target, return_idx=False):
    # NOTE: masks are calculated in cuda

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    if return_idx:
        return max_indices.squeeze(1), mask_u1, mask_u2
    else:
        return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def _loss_tckd(soft_logits_student, soft_logits_teacher, gt_mask, other_mask, temperature):
    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    log_p0_student = torch.log(p0_student)
    tckd_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    return tckd_loss


def _loss_nckd(soft_logits_student, soft_logits_teacher, gt_mask, other_mask, temperature, mask_magnitude, kl_type):
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * gt_mask, dim=1
    )
    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_p2_student,
                       log_p2_teacher, temperature, kl_type=kl_type)

    return nckd_loss


def _loss_dkd(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, strategy="target"):
    if strategy == "target":
        gt_mask = _get_gt_mask(logits_teacher, target)
        other_mask = _get_other_mask(logits_teacher, target)
    elif strategy == "top1":
        gt_mask, other_mask = get_top1_masks(logits_teacher, target)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    tckd_loss = _loss_tckd(
        soft_logits_student, soft_logits_teacher, gt_mask, other_mask, temperature)

    nckd_loss = _loss_nckd(soft_logits_student, soft_logits_teacher,
                           gt_mask, other_mask, temperature, mask_magnitude, kl_type)

    loss = alpha*tckd_loss + beta*nckd_loss
    return loss


def _loss_dkd_ce(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, strategy):
    _dkd_loss = dkd_loss(logits_student, logits_teacher, target,
                         alpha, beta, temperature, mask_magnitude, kl_type, strategy)

    ce_loss = F.cross_entropy(logits_student, target)
    loss = _dkd_loss + ce_loss
    return loss


def _loss_kd(logits_student, logits_teacher, target, temperature):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    log_p_student = F.log_softmax(soft_logits_student, dim=1)
    log_p_teacher = F.log_softmax(soft_logits_teacher, dim=1)

    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    ori_ratio = 1-p_teacher.gather(1, target.unsqueeze(1)).squeeze(1)

    loss = kl_div(log_p_student, log_p_teacher, temperature, kl_type="forward")

    return loss, ori_ratio


def record_grad(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, epoch, iters, suffix,  tckd_grad_list, nckd_grad_list, idx_list, strategy="target"):
    logits_student = logits_student.detach().cpu().clone()
    logits_student.requires_grad = True

    logits_teacher = logits_teacher.cpu()
    target = target.cpu()

    if strategy == "target":
        gt_mask = _get_gt_mask(logits_teacher, target)
        other_mask = _get_other_mask(logits_teacher, target)
        idx_list.append(target)
    elif strategy == "top1":
        max_idx, gt_mask, other_mask = get_top1_masks(logits_teacher, target, return_idx=True)
        idx_list.append(max_idx)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    tckd_loss = _loss_tckd(
        soft_logits_student, soft_logits_teacher, gt_mask, other_mask, temperature)

    # loss = tckd_loss
    tckd_loss.backward()
    tckd_grad = logits_student.grad.clone()
    logits_student.grad.zero_()
    tckd_grad_list.append(tckd_grad)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    nckd_loss = _loss_nckd(soft_logits_student, soft_logits_teacher,
                           gt_mask, other_mask, temperature, mask_magnitude, kl_type)

    nckd_loss.backward()
    nckd_grad = logits_student.grad.clone()
    logits_student.grad.zero_()
    nckd_grad_list.append(nckd_grad)

    # loss.backward()
    # grad = logits_student.grad.detach()
    # target_grad = grad.gather(1, target.unsqueeze(1)).squeeze(1)
    # other_grad = grad.scatter(1, target.unsqueeze(1), 0)
    # # target_norm = target_grad.abs().mean()
    # # other_norm = other_grad.norm(p=2, dim=1).mean()
    # target_norm = target_grad.abs()
    # # other_norm = other_grad.norm(p=2, dim=1)
    # other_norm = other_grad.abs().mean(dim=1)
    # global target_norm_list, other_norm_list, i
    # target_norm_list.append(target_norm)
    # other_norm_list.append(other_norm)

    # if suffix == "kd":
    #     ori_ratio_list.append(ori_ratio)
    #     teacher_logits_list.append(logits_teacher)

    # target_norm = target_grad.norm(p=2)
    # other_norm = other_grad.norm(p=2)
    # print(f"target/other: {(target_norm/other_norm).item()} target: {target_norm.item()} other: {other_norm.item()}")
    logits_student.grad.zero_()

    # data_str = [target_norm.item(), other_norm.item(),
    #             (target_norm/other_norm).item()]
    # data_str = [str(i) for i in data_str]
    # log_file.write(",".join(data_str)+"\n")
    if iters >= 782:
        tckd_grad_list = torch.cat(tckd_grad_list, dim=0).numpy()
        nckd_grad_list = torch.cat(nckd_grad_list, dim=0).numpy()
        idx_list = torch.cat(idx_list, dim=0).numpy()

        suffix = '_'+suffix if suffix != '' else ''

        np.savez(
            f"exp/grad/resnet32x4_8x4_grad_epoch{epoch}{suffix}.npz",
            tckd_grad=tckd_grad_list,
            nckd_grad=nckd_grad_list,
            idx=idx_list
        )


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, strategy="target"):
    return _loss_dkd(
        logits_student, logits_teacher, target, alpha, beta,
        temperature, mask_magnitude, kl_type, strategy="target"
    )


class DKDMod(Distiller):
    """DKD with some new losses"""

    def __init__(self, student, teacher, cfg):
        super(DKDMod, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKDMOD.CE_WEIGHT
        self.alpha = cfg.DKDMOD.ALPHA
        self.beta = cfg.DKDMOD.BETA
        self.temperature = cfg.DKDMOD.T
        self.warmup = cfg.DKDMOD.WARMUP
        self.kl_type = cfg.DKDMOD.KL_TYPE
        self.mask_magnitude = cfg.DKDMOD.MASK_MAGNITUDE
        self.strategy = cfg.DKDMOD.STRATEGY

        self.iters = 0
        self.tckd_grad_list = []
        self.nckd_grad_list = []
        self.idx_list = []
        # self.log_file = Path(f"{cfg.DISTILLER.TEACHER}_{cfg.DISTILLER.STUDENT}_DKDMod_grad_beta{cfg.DKDMOD.BETA}.csv").open("w")

        # self.log_file.write("target_norm,other_norm,ratio\n")

    def clean_record(self):
        self.tckd_grad_list = []
        self.nckd_grad_list = []
        self.idx_list = []
        self.iters = 0

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        self.iters += 1
        record_grad(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature=self.temperature,
            mask_magnitude=self.mask_magnitude,
            kl_type=self.kl_type,
            epoch=kwargs["epoch"],
            iters=self.iters,
            suffix=kwargs["suffix"],
            tckd_grad_list=self.tckd_grad_list,
            nckd_grad_list=self.nckd_grad_list,
            idx_list=self.idx_list,
            strategy=self.strategy,
        )

        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            temperature=self.temperature,
            mask_magnitude=self.mask_magnitude,
            kl_type=self.kl_type,
            strategy=self.strategy
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
