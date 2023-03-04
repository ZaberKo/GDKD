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


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, mask_magnitude, kl_type):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    tckd_loss = (
        F.binary_cross_entropy(pred_student, pred_teacher, reduction="mean")
        * (temperature**2)
    )

    log_pred_teacher_part2 = F.log_softmax(
        logits_teacher / temperature - mask_magnitude * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - mask_magnitude * gt_mask, dim=1
    )

    nckd_loss = kl_div(log_pred_student_part2,
                       log_pred_teacher_part2, temperature, kl_type=kl_type)

    return alpha * tckd_loss + beta * nckd_loss


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

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.mask_magnitude,
            self.kl_type
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
