import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ._base import Distiller
from .utils import kl_div, validate

from mdistiller.dataset import get_dataset


import math

MASK_MAGNITUDE = 1000.0


def get_top1_masks(logits, target):
    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    return max_indices, mask_u1, mask_u2


def get_target_masks(logits, target):
    # target mask
    target = target.unsqueeze(1)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, target, 1)

    # other mask
    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, target, 0)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, gamma, temperature, kl_type):
    mask_u1, mask_u2 = get_target_masks(logits_teacher, target)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    tckd_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    log_p2_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u1, dim=1
    )
    nckd = kl_div(log_p2_student, log_p2_teacher, temperature,
                  kl_type, reduction="none")  # [B]
    # nckd = kl_div(log_p2_student, log_p2_teacher, temperature,
    #               kl_type)

    # TODO: decay gamma
    # adaptive beta based on teacher logits:
    _beta = beta[target]

    nckd_loss = (_beta*nckd).sum()/nckd.shape[0]*gamma

    return (
        alpha*tckd_loss + nckd_loss,
        tckd_loss.detach(),
        (nckd_loss/beta.mean()).detach()
    )





def prebuild_beta(teacher, cfg, T=4.0, preload_path=None):
    # logits_dict = np.load(f"exp/{dataset}_{model}_logits.npz")
    if preload_path:
        raise NotImplementedError

    print("Prebuilding beta...")
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    logits_arr = validate(train_loader, teacher, num_classes)

    beta = torch.zeros(num_classes)
    for i, logits in enumerate(logits_arr):
        logits = torch.as_tensor(logits)
        probs = F.softmax(logits/T, dim=1)
        idx = torch.argsort(probs, dim=1)

        max_prob = torch.zeros(probs.shape[0])
        for j in range(idx.shape[0]):
            max_prob[j] = probs[j, idx[j, -1]]

        other_prob = torch.zeros(probs.shape[0])
        for j in range(idx.shape[0]):
            other_prob[j] = probs[j, idx[j, -2]]

        beta[i] = (max_prob/other_prob).mean()

    return beta

# Adaptive beta DKD


class ADKD(Distiller):
    """
        Auto adjust beta in DKD
    """
    def __init__(self, student, teacher, cfg):
        super(ADKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.ADKD.CE_WEIGHT
        # self.kd_loss_weight = cfg.ADKD.KD_WEIGHT
        self.alpha = cfg.ADKD.ALPHA
        self.temperature = cfg.ADKD.T
        self.warmup = cfg.ADKD.WARMUP
        self.kl_type = cfg.ADKD.KL_TYPE

        self.base_gamma = cfg.ADKD.BASE_GAMMA
        self.target_gamma = cfg.ADKD.TARGET_GAMMA

        self.total_epochs = cfg.SOLVER.EPOCHS

        self.register_buffer(
            "beta",
            prebuild_beta(self.teacher, cfg, self.temperature)
        )

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        epoch = kwargs["epoch"]

        self.gamma = (
            self.target_gamma +
            (self.base_gamma - self.target_gamma) * (
                1 + math.cos(
                    math.pi * epoch / self.total_epochs)
            ) / 2
        )

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd, self.tckd_loss, self.avg_nckd_loss = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
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
            "gamma": self.gamma,
            "tckd_loss": self.tckd_loss,
            "avg_nckd_loss": self.avg_nckd_loss
        }
