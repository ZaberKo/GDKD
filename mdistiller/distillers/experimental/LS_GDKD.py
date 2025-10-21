import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import Distiller
from ..utils import kl_div

from ..GDKD import get_masks, cat_mask


MASK_MAGNITUDE = 1000.0
EPS = 1e-7


def normalize(logits, mask=None):
    if mask is None:
        mean = logits.mean(dim=-1, keepdims=True)
        stdv = logits.std(dim=-1, keepdims=True)
    else:
        num_valid = torch.sum(mask, dim=-1, keepdims=True)  # we assume it is not zero
        mean = torch.sum(logits * mask, dim=-1, keepdims=True) / (num_valid)
        stdv = torch.sqrt(
            torch.sum(((logits - mean) * mask) ** 2, dim=-1, keepdims=True)
            / (num_valid)
        )

    return (logits - mean) / (EPS + stdv), mean.detach().mean(), stdv.detach().mean()


def gdkd_loss(
    logits_student,
    logits_teacher,
    target,
    k,
    strategy,
    w0,
    w1,
    w2,
    temperature,
    kl_type,
    norm_mode="global",
):
    assert norm_mode in ["global", "both", "other"]

    mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

    if norm_mode == "global":
        logits_student, mean_student, std_student = normalize(logits_student)
        logits_teacher, mean_teacher, std_teacher = normalize(logits_teacher)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )

    logits_student_top = logits_student
    logits_teacher_top = logits_teacher
    logits_student_other = logits_student
    logits_teacher_other = logits_teacher

    # topk loss
    if norm_mode == "both":
        logits_student_top, mean_student_top, std_student_top = normalize(
            logits_student, mask_u1
        )
        logits_teacher_top, mean_teacher_top, std_teacher_top = normalize(
            logits_teacher, mask_u1
        )
        logits_teacher_other, mean_teacher_other, std_teacher_other = normalize(
            logits_teacher, mask_u2
        )
        logits_student_other, mean_student_other, std_student_other = normalize(
            logits_student, mask_u2
        )
    elif norm_mode == "other":
        logits_teacher_other, mean_teacher_other, std_teacher_other = normalize(
            logits_teacher, mask_u2
        )
        logits_student_other, mean_student_other, std_student_other = normalize(
            logits_student, mask_u2
        )

    log_p1_student = F.log_softmax(
        logits_student_top / temperature - MASK_MAGNITUDE * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        logits_teacher_top / temperature - MASK_MAGNITUDE * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    # other classes loss
    log_p2_student = F.log_softmax(
        logits_student_other / temperature - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        logits_teacher_other / temperature - MASK_MAGNITUDE * mask_u1, dim=1
    )

    low_other_loss = kl_div(log_p2_student, log_p2_teacher, temperature, kl_type)

    metrics = {
        "high_loss": high_loss.detach(),
        "low_top_loss": low_top_loss.detach(),
        "low_other_loss": low_other_loss.detach(),
    }
    match norm_mode:
        case "global":
            metrics.update(
                {
                    "mean_student": mean_student.detach(),
                    "std_student": std_student.detach(),
                    "mean_teacher": mean_teacher.detach(),
                    "std_teacher": std_teacher.detach(),
                }
            )
        case "both":
            metrics.update(
                {
                    "mean_student_top": mean_student_top.detach(),
                    "std_student_top": std_student_top.detach(),
                    "mean_teacher_top": mean_teacher_top.detach(),
                    "std_teacher_top": std_teacher_top.detach(),
                    "mean_student_other": mean_student_other.detach(),
                    "std_student_other": std_student_other.detach(),
                    "mean_teacher_other": mean_teacher_other.detach(),
                    "std_teacher_other": std_teacher_other.detach(),
                }
            )
        case "other":
            metrics.update(
                {
                    "mean_teacher_other": mean_teacher_other.detach(),
                    "std_teacher_other": std_teacher_other.detach(),
                    "mean_student_other": mean_student_other.detach(),
                    "std_student_other": std_student_other.detach(),
                }
            )

    return (
        w0 * high_loss + w1 * low_top_loss + w2 * low_other_loss,
        metrics,
    )


class LS_GDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(LS_GDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.LS_GDKD.CE_WEIGHT

        self.w0 = cfg.LS_GDKD.W0
        self.w1 = cfg.LS_GDKD.W1
        self.w2 = cfg.LS_GDKD.W2
        self.temperature = cfg.LS_GDKD.T
        self.warmup = cfg.LS_GDKD.WARMUP
        self.k = cfg.LS_GDKD.TOPK
        self.strategy = cfg.LS_GDKD.STRATEGY
        self.kl_type = cfg.LS_GDKD.KL_TYPE
        self.norm_mode = cfg.LS_GDKD.NORM_MODE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_gdkd, self.metrics = gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.k,
            self.strategy,
            self.w0,
            self.w1,
            self.w2,
            self.temperature,
            kl_type=self.kl_type,
            norm_mode=self.norm_mode,
        )
        loss_kd = min(kwargs["epoch"] / self.warmup, 1.0) * loss_gdkd
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return self.metrics
