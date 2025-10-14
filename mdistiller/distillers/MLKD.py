import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from .utils import kl_div


def get_kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = (
            F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        )
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature**2
    return loss_kd


def get_cc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)  # (C, C)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / class_num
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / class_num
    return consistency_loss


def get_bc_loss(logits_student, logits_teacher, temperature, reduce=True):
    batch_size, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))  # (B, B)
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    if reduce:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2).sum() / batch_size
    else:
        consistency_loss = ((teacher_matrix - student_matrix) ** 2) / batch_size
    return consistency_loss


class MLKD(Distiller):
    """Multi-level Knowledge Distillation

    Note: This implementation does not apply the strong-weak augmentation strategy in the original repo: https://github.com/Jin-Ying/Multi-Level-Logit-Distillation
    """

    def __init__(self, student, teacher, cfg):
        super(MLKD, self).__init__(student, teacher)
        self.temperatures = cfg.MLKD.TEMPERATURES
        self.ce_loss_weight = cfg.MLKD.CE_WEIGHT
        self.kd_loss_weight = cfg.MLKD.KD_WEIGHT
        self.enable_mask = cfg.MLKD.ENABLE_MASK
        self.confidence_percentage = cfg.MLKD.CONFIDENCE_PERCENTAGE
        self.class_confidence_percentage = cfg.MLKD.CLASS_CONFIDENCE_PERCENTAGE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        if self.enable_mask:
            pred_teacher = F.softmax(logits_teacher.detach(), dim=1)
            confidence, pseudo_labels = pred_teacher.max(dim=1)
            confidence = confidence.detach()
            conf_thresh = torch.quantile(confidence, self.confidence_percentage)
            mask = confidence.le(conf_thresh).bool()

            class_confidence = torch.sum(pred_teacher, dim=0)
            class_confidence = class_confidence.detach()
            class_confidence_thresh = torch.quantile(
                class_confidence, self.class_confidence_percentage
            )
            class_conf_mask = class_confidence.le(class_confidence_thresh).bool()
        else:
            mask = torch.ones_like(
                target, device=logits_student.device, dtype=torch.bool
            )
            class_conf_mask = torch.ones(
                logits_student.shape[1], device=logits_student.device, dtype=torch.bool
            )

        kd_loss = sum(
            (get_kd_loss(logits_student, logits_teacher, t) * mask).mean()
            for t in self.temperatures
        )

        cc_loss = sum(
            (get_cc_loss(logits_student, logits_teacher, t) * class_conf_mask).mean()
            for t in self.temperatures
        )

        bc_loss = sum(
            (get_bc_loss(logits_student, logits_teacher, t) * mask).mean()
            for t in self.temperatures
        )

        loss_kd = self.kd_loss_weight * (kd_loss + cc_loss + bc_loss)

        self.kd_loss = kd_loss.detach()
        self.cc_loss = cc_loss.detach()
        self.bc_loss = bc_loss.detach()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "kd_loss": self.kd_loss,
            "cc_loss": self.cc_loss,
            "bc_loss": self.bc_loss,
        }
