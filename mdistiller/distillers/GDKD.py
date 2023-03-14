import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, k=5, strategy="best"):
    if strategy == "best":
        largest_flag = True
    elif strategy == "worst":
        largest_flag = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = torch.topk(logits, k, dim=1, largest=largest_flag, sorted=False).indices

    # top 5 mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def gdkd_loss(logits_student, logits_teacher, target, k, strategy, w0, w1, w2, temperature, kl_type):
    mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u1, dim=1
    )

    low_other_loss = kl_div(
        log_p2_student, log_p2_teacher, temperature, kl_type)

    return (
        w0 * high_loss + w1 * low_top_loss + w2 * low_other_loss,
        high_loss.detach(),
        low_top_loss.detach(),
        low_other_loss.detach()
    )


class GDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(GDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.GDKD.CE_WEIGHT
        # self.alpha = cfg.GDKD.ALPHA
        # self.beta = cfg.GDKD.BETA
        self.w0 = cfg.GDKD.W0
        self.w1 = cfg.GDKD.W1
        self.w2 = cfg.GDKD.W2
        self.temperature = cfg.GDKD.T
        self.warmup = cfg.GDKD.WARMUP
        self.k = cfg.GDKD.TOPK
        self.strategy = cfg.GDKD.STRATEGY
        self.kl_type = cfg.GDKD.KL_TYPE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd, self.high_loss, self.low_top_loss, self.low_other_loss = gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.k,
            self.strategy,
            self.w0,
            self.w1,
            self.w2,
            self.temperature,
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
            "low_top_loss": self.low_top_loss,
            "low_other_loss": self.low_other_loss,
        }
