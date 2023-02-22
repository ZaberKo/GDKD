import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def get_masks(logits, k=5, strategy="best"):
    ranks = logits.argsort(dim=-1)

    if strategy == "best":
        # use top k from teacher
        ranks = ranks[:, -k:]
    elif strategy == "worst":
        ranks = ranks[:, :k]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # top 5 mask
    mask_u1 = torch.zeros_like(logits).scatter_(1, ranks, 1).bool()
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def gdkd_loss(logits_student, logits_teacher, target, k, strategy, w0, w1, w2, temperature):
    mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    loss0 = (
        F.binary_cross_entropy(p0_student, p0_teacher, reduction="mean")
        * (temperature**2)
    )

    # topk loss
    p1_student = F.log_softmax(
        soft_logits_student - 1000.0 * mask_u2, dim=1
    )
    p1_teacher = F.softmax(
        soft_logits_teacher - 1000.0 * mask_u2, dim=1
    )

    loss1 = (
        F.kl_div(p1_student, p1_teacher, reduction="batchmean")
        * (temperature**2)
    )

    # other classes loss
    p2_student = F.log_softmax(
        soft_logits_student - 1000.0 * mask_u1, dim=1
    )
    p2_teacher = F.softmax(
        soft_logits_teacher - 1000.0 * mask_u1, dim=1
    )

    loss2 = (
        F.kl_div(p2_student, p2_teacher, reduction="batchmean")
        * (temperature**2)
    )

    return w0 * loss0 + w1 * loss1 + w2 * loss2

# More numerically stable?
def gdkd_loss_mod(logits_student, logits_teacher, target, k, strategy, w0, w1, w2, temperature):
    mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    # ======== bipartition loss =========
    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    loss0 = (
        F.binary_cross_entropy(p0_student, p0_teacher, reduction="mean")
        * (temperature**2)
    )

    # ========== topk loss ===========
    p1_teacher = F.softmax(
        soft_logits_teacher - 1000.0 * mask_u2, dim=1
    )

    loss1 = (
        # Note: not supported in pytorch 1.9
        F.cross_entropy(soft_logits_student - 1000.0 * mask_u2, p1_teacher)
        * (temperature**2)
    )

    # other classes loss
    p2_teacher = F.softmax(
        soft_logits_teacher - 1000.0 * mask_u1, dim=1
    )

    loss2 = (
        F.cross_entropy(soft_logits_student - 1000.0 * mask_u1, p2_teacher)
        * (temperature**2)
    )

    return w0 * loss0 + w1 * loss1 + w2 * loss2


class GDKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(GDKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.GDKD.CE_WEIGHT
        # self.alpha = cfg.GDKD.ALPHA
        # self.beta = cfg.GDKD.BETA
        self.w0 = cfg.GDKD.W0
        self.w1 = cfg.GDKD.W1
        self.w2 = cfg.GDKD.W2
        self.temperature = cfg.GDKD.T
        self.warmup = cfg.GDKD.WARMUP
        self.k = cfg.GDKD.TOPK
        self.strategy = cfg.GDKD.STRATEGY

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.k,
            self.strategy,
            self.w0,
            self.w1,
            self.w2,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
