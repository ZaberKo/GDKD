import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, target, eta=0.1):
    mask = torch.rand_like(logits) < eta

    return mask


def kd_loss(logits_student, logits_teacher, target, eta, temperature, kl_type):
    mask = get_masks(logits_teacher, target, eta)
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * ~mask, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * ~mask, dim=1
    )

    loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    return (
        loss,
        loss.detach(),
    )


class SKD(Distiller):
    """
        KD with stochastic partial logits distillation
    """
    def __init__(self, student, teacher, cfg):
        super(SKD, self).__init__(student, teacher)
        self.ce_weight = cfg.SKD.CE_WEIGHT
        self.kd_weight = cfg.SKD.KD_WEIGHT
        self.temperature = cfg.SKD.T
        self.warmup = cfg.SKD.WARMUP
        self.eta = cfg.SKD.ETA
        self.kl_type = cfg.SKD.KL_TYPE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_weight * F.cross_entropy(logits_student, target)
        loss_skd, self.loss_skd = kd_loss(
            logits_student,
            logits_teacher,
            target,
            eta=self.eta,
            temperature=self.temperature,
            kl_type=self.kl_type
        )

        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * self.kd_weight * loss_skd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "skd_loss": self.loss_skd,
        }
