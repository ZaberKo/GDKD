import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, k=5, strategy="best"):
    if strategy == "best":
        largest_flag = True
    elif strategy == "worst":
        largest_flag = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = torch.topk(logits, k, dim=-1,
                       largest=largest_flag,
                       sorted=False).indices

    # topk mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def gdkd_loss_autow(logits_student, logits_teacher, k, m1, m2, w1, w2, T, mode="v1"):
    mask_u1, mask_u2 = get_masks(logits_teacher, k, "best")

    soft_logits_student = logits_student / T
    soft_logits_teacher = logits_teacher / T

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (T**2)
    )

    b_t = p0_teacher[:, 0]
    b_o = p0_teacher[:, 1]

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u2, dim=1
    )

    if mode == "v2":
        low_top_loss = kl_div(
            log_p1_student, log_p1_teacher,
            T, reduction='batchmean'
        )
    else:
        low_top_loss = b_t * kl_div(
            log_p1_student, log_p1_teacher,
            T, reduction='none'
        )
        low_top_loss = low_top_loss.mean()

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u1, dim=1
    )

    if mode == "v3":
        low_other_loss = kl_div(
            log_p2_student, log_p2_teacher,
            T, reduction='batchmean'
        )
    else:
        low_other_loss = b_o * kl_div(
            log_p2_student, log_p2_teacher,
            T, reduction='none'
        )
        low_other_loss = low_other_loss.mean()

    if mode == 'v1':
        loss = high_loss + m1*low_top_loss + m2*low_other_loss
    elif mode == 'v2':
        loss = high_loss + w1*low_top_loss + m2*low_other_loss
    elif mode == 'v3':
        loss = high_loss + m1*low_top_loss + w2*low_other_loss
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return (
        loss,
        high_loss.detach(),
        low_top_loss.detach(),
        low_other_loss.detach(),
        b_t.mean().detach(),
        b_o.mean().detach()
    )


class GDKDAutoW(Distiller):
    """
        mode:
        v1: loss = high_loss + m1 * b_t * low_top_loss + m2 * b_o * low_other_loss
        v2: loss = high_loss + w1 * low_top_loss + m2 * b_o * low_other_loss
        v3: loss = high_loss + m1 * b_t * low_top_loss + w2 * low_other_loss
    """

    def __init__(self, student, teacher, cfg):
        super(GDKDAutoW, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.GDKDAutoW.CE_WEIGHT

        self.m1 = cfg.GDKDAutoW.M1
        self.m2 = cfg.GDKDAutoW.M2
        self.w1 = cfg.GDKDAutoW.W1
        self.w2 = cfg.GDKDAutoW.W2
        self.mode = cfg.GDKDAutoW.MODE

        self.T = cfg.GDKDAutoW.T
        self.warmup = cfg.GDKDAutoW.WARMUP
        self.k = cfg.GDKDAutoW.TOPK

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        (loss_dkd, self.high_loss, self.low_top_loss, self.low_other_loss,
         self.b_t, self.b_o) = gdkd_loss_autow(
            logits_student,
            logits_teacher,
            k=self.k,
            m1=self.m1,
            m2=self.m2,
            w1=self.w1,
            w2=self.w2,
            T=self.T,
            mode=self.mode
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
            "low_top_loss_with_b_t": self.low_top_loss,
            "low_other_loss_with_b_o": self.low_other_loss,
            "b_t": self.b_t,
            "b_o": self.b_o,
        }
