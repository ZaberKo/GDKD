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


def _loss_tckd(logits_student, logits_teacher, gt_mask, other_mask, temperature):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature
    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    log_p0_student = torch.log(p0_student)
    # Note: be careful for kl_div batchmean, which divides by batch size
    # Here we want individual grads, so not need average.
    tckd_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="sum")
        * (temperature**2)
    )

    return tckd_loss


def _loss_nckd(logits_student, logits_teacher, gt_mask, other_mask, temperature, mask_magnitude, kl_type):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * gt_mask, dim=1
    )
    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_p2_student,
                       log_p2_teacher, temperature, kl_type=kl_type, reduction="sum")

    return nckd_loss


def _loss_dkd(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, strategy="target"):
    if strategy == "target":
        gt_mask = _get_gt_mask(logits_teacher, target)
        other_mask = _get_other_mask(logits_teacher, target)
    elif strategy == "top1":
        gt_mask, other_mask = get_top1_masks(logits_teacher, target)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    tckd_loss = _loss_tckd(
        logits_student, logits_teacher, gt_mask, other_mask, temperature)

    nckd_loss = _loss_nckd(logits_student, logits_teacher,
                           gt_mask, other_mask, temperature, mask_magnitude, kl_type)

    loss = alpha*tckd_loss + beta*nckd_loss
    return loss


def _loss_kd(logits_student, logits_teacher, temperature):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    log_p_student = F.log_softmax(soft_logits_student, dim=1)
    log_p_teacher = F.log_softmax(soft_logits_teacher, dim=1)

    loss = kl_div(log_p_student, log_p_teacher, temperature,
                  kl_type="forward", reduction="sum")

    return loss


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

        self.reset_record()
        # self.log_file = Path(f"{cfg.DISTILLER.TEACHER}_{cfg.DISTILLER.STUDENT}_DKDMod_grad_beta{cfg.DKDMOD.BETA}.csv").open("w")

        # self.log_file.write("target_norm,other_norm,ratio\n")

    def reset_record(self):
        self.tckd_grad_list = []
        self.nckd_grad_list = []
        self.target_list = []
        self.logits_t_list = []
        self.logits_s_list = []
        self.kd_grad_list = []

    def save_record(self, path):
        tckd_grad_list = torch.cat(self.tckd_grad_list, dim=0).numpy()
        nckd_grad_list = torch.cat(self.nckd_grad_list, dim=0).numpy()
        target_list = torch.cat(self.target_list, dim=0).numpy()
        logits_t_list = torch.cat(self.logits_t_list, dim=0).numpy()
        logits_s_list = torch.cat(self.logits_s_list, dim=0).numpy()
        kd_grad_list = torch.cat(self.kd_grad_list, dim=0).numpy()

        folder_path = Path(path).parent
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)

        np.savez(
            # f"exp/grad/resnet32x4_8x4_grad_epoch{epoch}{suffix}.npz",
            path,
            tckd_grad=tckd_grad_list,
            nckd_grad=nckd_grad_list,
            logits_t=logits_t_list,
            logits_s=logits_s_list,
            target=target_list,
            kd_grad=kd_grad_list
        )

    def record_grad(self, image, target):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        self.logits_t_list.append(logits_teacher.detach().cpu())
        self.logits_s_list.append(logits_student.detach().cpu())

        logits_student = logits_student.detach().clone()
        logits_student.requires_grad = True

        # logits_teacher = logits_teacher.cpu()
        # target = target.cpu()

        if self.strategy == "target":
            gt_mask = _get_gt_mask(logits_teacher, target)
            other_mask = _get_other_mask(logits_teacher, target)
        elif self.strategy == "top1":
            gt_mask, other_mask = get_top1_masks(
                logits_teacher, target)
        else:
            raise ValueError("Unknown strategy: {}".format(self.strategy))

        self.target_list.append(target.cpu())

        # record grads
        tckd_loss = _loss_tckd(
            logits_student, logits_teacher, gt_mask, other_mask, self.temperature)
        tckd_grad = torch.autograd.grad(
            tckd_loss, logits_student
        )[0]
        self.tckd_grad_list.append(tckd_grad.detach().cpu())

        nckd_loss = _loss_nckd(logits_student, logits_teacher,
                               gt_mask, other_mask, self.temperature, self.mask_magnitude, self.kl_type)
        nckd_grad = torch.autograd.grad(
            nckd_loss, logits_student
        )[0]
        self.nckd_grad_list.append(nckd_grad.detach().cpu())

        kd_loss = _loss_kd(logits_student, logits_teacher, self.temperature)
        kd_grad = torch.autograd.grad(
            kd_loss, logits_student
        )[0]
        self.kd_grad_list.append(kd_grad.detach().cpu())
