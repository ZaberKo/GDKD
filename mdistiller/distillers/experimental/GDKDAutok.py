import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div, validate

from mdistiller.dataset import get_dataset

import yaml

MASK_MAGNITUDE = 1000.0


def get_masks(logits, topks):
    maxk = topks.max()
    ranks = torch.topk(logits, maxk, dim=-1,
                       largest=True,
                       sorted=True).indices

    for i in range(logits.shape[0]):
        ranks[i, topks[i]:] = ranks[i][0]
    # ranks = ranks.to(device)

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


def calc_topk(probs, topk_th, ratio_th):
    num_classes = probs.shape[1]
    x = torch.arange(1, num_classes+1).to(probs.device)

    probs = probs.sort(dim=1, descending=False).values

    cumavg = probs.cumsum(dim=1)/x
    ratio = probs/cumavg

    topk_arr = []
    for r in ratio:
        idx = torch.nonzero(r >= ratio_th).squeeze(1)
        if len(idx):
            topk = 100-idx[0].item()
        else:
            topk = 1

        topk = min(topk, topk_th)

        topk_arr.append(topk)

    topk_arr = torch.as_tensor(topk_arr)

    return topk_arr


def gdkd_loss(logits_student, logits_teacher, target, topk_th, ratio_th, w0, w1, w2, temperature, kl_type):

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    topks = calc_topk(p_teacher, topk_th, ratio_th)

    mask_u1, mask_u2 = get_masks(logits_teacher, topks)

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


class GDKDAutok(Distiller):
    """
        GDKD with autok: k is determined by each sample,
        which is controled by topk_th and ratio_th.
    """
    def __init__(self, student, teacher, cfg):
        super(GDKDAutok, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.GDKDAUTOK.CE_WEIGHT
        self.w0 = cfg.GDKDAUTOK.W0
        self.w1 = cfg.GDKDAUTOK.W1
        self.w2 = cfg.GDKDAUTOK.W2
        self.temperature = cfg.GDKDAUTOK.T
        self.warmup = cfg.GDKDAUTOK.WARMUP
        self.kl_type = cfg.GDKDAUTOK.KL_TYPE

        self.topk_th = cfg.GDKDAUTOK.TOPK_TH
        self.ratio_th = cfg.GDKDAUTOK.RATIO_TH

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
            self.topk_th,
            self.ratio_th,
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
