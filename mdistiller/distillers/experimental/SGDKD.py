import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, target, eta=0.1):
    # splits to 3 parts: target, other, ignore

    # NOTE: masks are calculated in cuda
    mask_u0 = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, target.unsqueeze(1), 1)

    mask_u1 = torch.rand_like(logits) < eta
    mask_u2 = torch.logical_not(mask_u1)

    num_classes = logits.shape[-1]

    # make sure mask_u1 has at least one element
    rand_u2 = torch.randint_like(target, num_classes)
    rand_mask = rand_u2 == target
    # if rand_u2[i] = target[i]; then rand_u2[i]+=1#
    rand_u2 = (rand_u2 + rand_mask.to(dtype=rand_u2.dtype)) % num_classes
    rand_u2 = rand_u2.unsqueeze(1)

    mask_u1.scatter_(1, rand_u2, False)
    mask_u2.scatter_(1, rand_u2, True)

    # make sure not cover the target
    mask_u1[mask_u0] = False
    mask_u2[mask_u0] = False

    return mask_u0, mask_u1, mask_u2


def cat_mask(t, *masks):
    res = [
        (t * mask).sum(dim=1, keepdims=True)
        for mask in masks
    ]

    rt = torch.cat(res, dim=1)  # [B, k]
    return rt


def gdkd_loss(logits_student, logits_teacher, target, eta, w0, w1, temperature, kl_type):
    mask_u0, mask_u1, mask_u2 = get_masks(logits_teacher, target, eta)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u0, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u0, mask_u1, mask_u2)

    # Caution: p0_student.sum(1)!=1
    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    # low loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * ~mask_u1, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * ~mask_u1, dim=1
    )

    low_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    return (
        w0 * high_loss + w1 * low_loss,
        high_loss.detach(),
        low_loss.detach()
    )


class SGDKD(Distiller):
    """
        Stocastic GDKD with 3 splits: target, other, ignore.
        low_ignore_loss is not used.
    """
    def __init__(self, student, teacher, cfg):
        super(SGDKD, self).__init__(student, teacher)
        self.ce_weight = cfg.SGDKD.CE_WEIGHT
        self.w0 = cfg.SGDKD.W0
        self.w1 = cfg.SGDKD.W1
        self.temperature = cfg.SGDKD.T
        self.warmup = cfg.SGDKD.WARMUP
        self.eta = cfg.SGDKD.ETA
        self.kl_type = cfg.SGDKD.KL_TYPE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        loss_dkd, self.high_loss, self.low_loss = gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            eta=self.eta,
            w0=self.w0,
            w1=self.w1,
            temperature=self.temperature,
            kl_type=self.kl_type
        )

        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_dkd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "high_loss": self.high_loss,
            "low_loss": self.low_loss,
        }
