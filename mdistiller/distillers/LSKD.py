import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

EPS = 1e-7


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (EPS + stdv), mean.detach().mean(), stdv.detach().mean()


def kd_loss(logits_student, logits_teacher, temperature):
    z_student, mean_student, std_student = normalize(logits_student)
    z_teacher, mean_teacher, std_teacher = normalize(logits_teacher)

    log_pred_student = F.log_softmax(z_student / temperature, dim=1)
    pred_teacher = F.softmax(z_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return (loss_kd, mean_student, std_student, mean_teacher, std_teacher)


class LSKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(LSKD, self).__init__(student, teacher)
        self.temperature = cfg.LSKD.TEMPERATURE
        self.ce_loss_weight = cfg.LSKD.CE_WEIGHT
        self.kd_loss_weight = cfg.LSKD.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        (
            loss_kd,
            self.mean_student,
            self.std_student,
            self.mean_teacher,
            self.std_teacher,
        ) = kd_loss(logits_student, logits_teacher, self.temperature)
        self.kd_loss = loss_kd.detach()

        loss_kd = self.kd_loss_weight * loss_kd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "kd_loss": self.kd_loss,
            "mean_student": self.mean_student,
            "std_student": self.std_student,
            "mean_teacher": self.mean_teacher,
            "std_teacher": self.std_teacher,
        }
