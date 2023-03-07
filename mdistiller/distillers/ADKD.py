import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from ._base import Distiller
from .utils import kl_div

MASK_MAGNITUDE = 1000.0

# from mdistiller.engine.utils import AverageMeter

# beta_meter = AverageMeter()
# nckd_meter = AverageMeter()


def get_top1_masks(logits, target):
    # NOTE: masks are calculated in cuda

    # target mask
    # mask_u1 = torch.zeros_like(logits).scatter_(
    #     1, target.unsqueeze(1), 1).bool()

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    return max_indices, mask_u1, mask_u2


def get_target_masks(logits, target):
    # NOTE: masks are calculated in cuda

    # target mask
    target = target.unsqueeze(1)
    mask_u1 = torch.zeros_like(
        logits, dtype=torch.bool).scatter_(1, target, 1)

    mask_u2 = torch.ones_like(
        logits, dtype=torch.bool).scatter_(1, target, 0)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, gamma, temperature, kl_type):
    max_indices, mask_u1, mask_u2 = get_top1_masks(logits_teacher, target)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    # tckd_loss = (
    #     F.binary_cross_entropy(p0_student, p0_teacher, reduction="mean")
    #     * (temperature**2)
    # )
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

    # adaptive beta based on teacher logits:
    _beta = torch.zeros_like(nckd)
    for i in range(_beta.shape[0]):
        _beta[i] = beta[target[i]]

    _beta = _beta*gamma
    # _beta = _beta*0.8
    # print(f"beta: {_beta.mean().item()} ± {_beta.std().item()}")

    nckd_loss = (_beta*nckd).sum()/nckd.shape[0]

    # top1_p = p_teacher.gather(1, max_indices).squeeze(1)  # [B]
    # top2_p, _ = torch.max(p_teacher*mask_u2, dim=1)  # [B]
    # beta = top1_p / top2_p  # [B], requires_grad=False

    # beta = gamma*beta
    # beta = gamma*beta
    # beta_mean = beta.mean()
    # avoid outlier beta value
    # NOTE: beta >= 1.0 by definition
    # beta = torch.clamp_max(beta, max=beta_mean*1.2)

    # beta_meter.update(beta.mean().item(), beta.shape[0])
    # print(f"beta_mean: {beta_mean.item()}")
    # print(f"beta_meter: {beta_meter.avg}")

    # nckd_loss = (beta*nckd).sum()/beta.shape[0]
    # nckd_loss = beta.mean()*nckd

    # nckd_meter.update(nckd_loss.item(), 1)

    # print("ADKD nckd:",nckd_meter.avg)

    return alpha*tckd_loss + nckd_loss


def dkd_loss2(logits_student, logits_teacher, target, warmup_weight, gamma, temperature, kl_type):
    logits_student_clone = logits_student.detach()
    logits_student_clone.requires_grad = True
    soft_logits_student = logits_student_clone / temperature
    soft_logits_teacher = logits_teacher / temperature

    log_p_student = F.log_softmax(soft_logits_student, dim=1)
    log_p_teacher = F.log_softmax(soft_logits_teacher, dim=1)

    # assume alpha=1 and dynamic beta
    loss = 8.0*(
        # warmup_weight *
        kl_div(log_p_student, log_p_teacher, temperature, kl_type)
    )
    loss.backward()
    grad = logits_student_clone.grad
    target_grad = grad.gather(1, target.unsqueeze(1)) # [B, 1]
    # other_grad = grad.scatter(1, target.unsqueeze(1), 0)

    # target_norm = target_grad.abs().mean()
    # other_norm = other_grad.norm(p=2, dim=1).mean()

    # target_norm = target_grad.abs()
    # other_norm = other_grad.abs().mean(dim=1)

    gamma = 0.5
    # let reweighted target_norm / new_other_norm = gamma
    # r = (target_norm / other_norm)/gamma
    # print(r.mean())

    # mask_t, mask_n = get_target_masks(logits_student_clone, target)

    # new_grad = grad*mask_t + r.unsqueeze(1) * grad * mask_n
    new_grad = grad + (gamma-1) * target_grad

    # meta update
    # Other part grads are updated in Trainer.
    logits_student.backward(grad, retain_graph=True)

    # NOTE: return detached loss for kd_loss record only.
    return loss.detach()


# def prebuild_beta(model, dataset="cifar100", T=4.0):
#     logits_dict = np.load(f"exp/{dataset}_{model}_logits.npz")

#     beta=torch.zeros(len(logits_dict))
#     for i, logits in enumerate(logits_dict.values()):
#         logits = torch.as_tensor(logits)
#         probs = F.softmax(logits/T, dim=1)
#         idx = torch.argsort(probs, dim=1)

#         max_prob = torch.zeros(probs.shape[0])
#         for j in range(idx.shape[0]):
#             max_prob[j] = probs[j, idx[j, -1]]

#         other_prob = torch.zeros(probs.shape[0])
#         for j in range(idx.shape[0]):
#             other_prob[j] = probs[j, idx[j, -2]]

#         beta[i] = (max_prob/other_prob).mean()

#     return beta

# Adaptive beta DKD
class ADKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(ADKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.ADKD.CE_WEIGHT
        # self.alpha = cfg.ADKD.ALPHA
        self.temperature = cfg.ADKD.T
        self.warmup = cfg.ADKD.WARMUP
        self.kl_type = cfg.ADKD.KL_TYPE
        self.gamma = cfg.ADKD.GAMMA  

        # self.beta = prebuild_beta(cfg.DISTILLER.TEACHER)

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        warmup_weight = min(kwargs["epoch"] / self.warmup, 1.0)

        loss_dkd = dkd_loss2(
            logits_student,
            logits_teacher,
            target,
            # alpha=self.alpha,
            # beta=self.beta,
            warmup_weight=warmup_weight,
            gamma=self.gamma,
            temperature=self.temperature,
            kl_type=self.kl_type
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
