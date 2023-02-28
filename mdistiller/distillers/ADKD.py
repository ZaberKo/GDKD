import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .utils import kl_div

MASK_MAGNITUDE = 1000.0


def get_masks(logits, target):
    # NOTE: masks are calculated in cuda

    # target mask
    # mask_u1 = torch.zeros_like(logits).scatter_(
    #     1, target.unsqueeze(1), 1).bool()

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return max_indices, mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, gamma, temperature, kl_type):
    max_indices, mask_u1, mask_u2 = get_masks(logits_teacher, target)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    tckd = (
        F.binary_cross_entropy(p0_student, p0_teacher, reduction="mean")
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

    # adaptive beta based on teacher logits:
    top1_p = p_teacher.gather(1, max_indices).squeeze(1)  # [B]
    top2_p, _ = torch.max(p_teacher*mask_u2, dim=1) # [B]
    beta = top1_p / top2_p  # [B], requires_grad=False
    beta_mean = beta.mean()
    # avoid outlier beta value 
    # NOTE: beta >= 1.0 by definition
    beta = torch.clamp_max(beta, max=beta_mean)
    batch_size = beta.shape[0]

    return alpha*tckd + gamma*((beta*nckd).sum()/batch_size)


# Adaptive beta DKD
class ADKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ADKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.ADKD.CE_WEIGHT
        self.alpha = cfg.ADKD.ALPHA
        self.temperature = cfg.ADKD.T
        self.warmup = cfg.ADKD.WARMUP
        self.kl_type = cfg.ADKD.KL_TYPE
        self.gamma = cfg.ADKD.GAMMA # additional global control for nckd

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            temperature=self.temperature,
            kl_type=self.kl_type
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
