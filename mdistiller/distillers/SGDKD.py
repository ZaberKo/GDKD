import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

MASK_MAGNITUDE=1000.0

def get_masks(logits, target, eta=0.1):
    mask_u0 = torch.zeros_like(logits).scatter_(
        1, target.unsqueeze(1), 1).bool()

    # NOTE: use fixed number or probaility?
    num_classes = logits.shape[1]
    n1 = int(num_classes*eta)
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool)
    for i in range(logits.shape[0]):
        indices = torch.randperm(num_classes)[:n1]
        mask_u1[i, indices] = True
    mask_u2 = torch.logical_not(mask_u1)

    # mask_u1 = torch.rand_like(logits) < eta
    # mask_u2 = torch.logical_not(mask_u1)

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


def gdkd_loss(logits_student, logits_teacher, target, eta, w0, w1, temperature):
    mask_u0, mask_u1, mask_u2 = get_masks(logits_teacher, target, eta)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u0, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u0, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    loss0 = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )
    
    # p0_student = cat_mask(p_student, mask_u0, mask_u1)
    # p0_teacher = cat_mask(p_teacher, mask_u0, mask_u1)
    # loss0=F.binary_cross_entropy(p0_student,p0_teacher,reduction="mean")* (temperature**2)

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * ~mask_u1, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * ~mask_u1, dim=1
    )

    loss1 = (
        F.kl_div(log_p1_student, log_p1_teacher,
                 reduction="batchmean", log_target=True)
        * (temperature**2)
    )

    # log_p2_student = F.log_softmax(
    #     soft_logits_student - MASK_MAGNITUDE * ~mask_u1, dim=1
    # )
    # log_p2_teacher = F.log_softmax(
    #     soft_logits_teacher - MASK_MAGNITUDE * ~mask_u1, dim=1
    # )

    # loss2 = (
    #     F.kl_div(log_p2_student, log_p2_teacher,
    #              reduction="batchmean", log_target=True)
    #     * (temperature**2)
    # )

    return w0 * loss0 + w1 * loss1


class SGDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(SGDKD, self).__init__(student, teacher)
        self.ce_weight = cfg.SGDKD.CE_WEIGHT
        self.w0 = cfg.SGDKD.W0
        self.w1 = cfg.SGDKD.W1
        self.temperature = cfg.SGDKD.T
        self.warmup = cfg.SGDKD.WARMUP
        self.eta = cfg.SGDKD.ETA

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            eta=self.eta,
            w0=self.w0,
            w1=self.w1,
            temperature=self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
