import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils import kl_div


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def get_top1_masks(logits, target):
    # NOTE: masks are calculated in cuda

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type, strategy="target"):
    if strategy == "target":
        gt_mask = _get_gt_mask(logits_teacher, target)
        other_mask = _get_other_mask(logits_teacher, target)
    elif strategy == "top1":
        gt_mask, other_mask = get_top1_masks(logits_teacher, target)
    else:
        raise ValueError("Unknown strategy: {}".format(strategy))

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    # tckd_loss = (
    #     F.binary_cross_entropy(pred_student, pred_teacher, reduction="mean")
    #     * (temperature**2)
    # )
    log_p0_student = torch.log(p0_student)
    tckd_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * gt_mask, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_p2_student,
                       log_p2_teacher, temperature, kl_type=kl_type)

    return (
        alpha * tckd_loss + beta * nckd_loss,
        tckd_loss.detach(),
        nckd_loss.detach()
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

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd, self.tckd_loss, self.nckd_loss= dkd_loss(
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
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_dkd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "tckd_loss": self.tckd_loss,
            "nckd_loss": self.nckd_loss
        }