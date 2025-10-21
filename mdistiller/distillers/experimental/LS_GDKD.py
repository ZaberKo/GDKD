import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div

from ..LSKD import normalize
from ..GDKD import gdkd_loss


MASK_MAGNITUDE = 1000.0
EPS = 1e-7


class LS_GDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(LS_GDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.GDKD.CE_WEIGHT

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

        z_student, self.mean_student, self.std_student = normalize(logits_student)
        z_teacher, self.mean_teacher, self.std_teacher = normalize(logits_teacher)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_gdkd, self.high_loss, self.low_top_loss, self.low_other_loss = gdkd_loss(
            z_student,
            z_teacher,
            target,
            self.k,
            self.strategy,
            self.w0,
            self.w1,
            self.w2,
            self.temperature,
            kl_type=self.kl_type,
        )
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_gdkd
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
            "mean_student": self.mean_student,
            "std_student": self.std_student,
            "mean_teacher": self.mean_teacher,
            "std_teacher": self.std_teacher,
        }
