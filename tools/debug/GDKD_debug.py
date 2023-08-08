import torch
import torch.nn as nn
import torch.nn.functional as F

from mdistiller.distillers._base import Distiller
from mdistiller.distillers.utils import kl_div

from pathlib import Path
import numpy as np

MASK_MAGNITUDE = 1000.0


def get_masks(logits, k=5, strategy="best"):
    if strategy == "best":
        largest_flag = True
    elif strategy == "worst":
        largest_flag = False
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    ranks = torch.topk(logits, k, dim=-1,
                       largest=largest_flag,
                       sorted=False).indices

    # topk mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def _loss_high(logits_student, logits_teacher, mask_u1, mask_u2, temperature):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature
    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="sum")
        * (temperature**2)
    )

    return high_loss

def _loss_low_topk(logits_student, logits_teacher, mask_u1, mask_u2, temperature, kl_type):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature
        # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type, reduction="sum")

    return low_top_loss

def _loss_low_other(logits_student, logits_teacher, mask_u1, mask_u2, temperature, kl_type):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature
    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u1, dim=1
    )

    low_other_loss = kl_div(
        log_p2_student, log_p2_teacher, temperature, kl_type, reduction="sum")
    
    return low_other_loss


def _loss_kd(logits_student, logits_teacher, temperature):
    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    log_p_student = F.log_softmax(soft_logits_student, dim=1)
    log_p_teacher = F.log_softmax(soft_logits_teacher, dim=1)

    loss = kl_div(log_p_student, log_p_teacher, temperature, kl_type="forward", reduction="sum") 

    return loss

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
        self.kl_type = cfg.GDKD.KL_TYPE

        self.reset_record()

    def reset_record(self):
        self.high_grad_list = []
        self.low_topk_grad_list = []
        self.low_other_grad_list = []
        self.target_list = []
        self.logits_t_list =[]
        self.logits_s_list = []
        self.kd_grad_list = []

    def save_record(self, path):
        high_grad_list = torch.cat(self.high_grad_list, dim=0).numpy()
        low_topk_grad_list = torch.cat(self.low_topk_grad_list, dim=0).numpy()
        low_other_grad_list = torch.cat(self.low_other_grad_list, dim=0).numpy()
        target_list = torch.cat(self.target_list, dim=0).numpy()
        logits_t_list = torch.cat(self.logits_t_list, dim=0).numpy()
        logits_s_list = torch.cat(self.logits_s_list, dim=0).numpy()
        kd_grad_list = torch.cat(self.kd_grad_list, dim=0).numpy()

        folder_path=Path(path).parent
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)


        np.savez(
            path,
            high_grad=high_grad_list,
            low_topk_grad=low_topk_grad_list,
            low_other_grad=low_other_grad_list,
            logits_t=logits_t_list,
            logits_s=logits_s_list,
            target=target_list,
            kd_grad=kd_grad_list
        )    

    def record_grad(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        k=self.k
        temperature=self.temperature
        kl_type=self.kl_type
        strategy=self.strategy


        logits_student = logits_student.detach().clone()
        logits_student.requires_grad = True

        self.logits_s_list.append(logits_student.detach().cpu())
        self.logits_t_list.append(logits_teacher.detach().cpu())
        self.target_list.append(target.cpu())

        mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

        high_loss = _loss_high(logits_student, logits_teacher, mask_u1, mask_u2, temperature)
        high_loss.backward()
        high_grad = logits_student.grad.clone()
        self.high_grad_list.append(high_grad.cpu())
        logits_student.grad.zero_()

        low_top_loss = _loss_low_topk(logits_student, logits_teacher, mask_u1, mask_u2, temperature, kl_type)
        low_top_loss.backward()
        low_top_grad = logits_student.grad.clone()
        self.low_topk_grad_list.append(low_top_grad.cpu())
        logits_student.grad.zero_()

        low_other_loss = _loss_low_other(logits_student, logits_teacher, mask_u1, mask_u2, temperature, kl_type)
        low_other_loss.backward()
        low_other_grad = logits_student.grad.clone()
        self.low_other_grad_list.append(low_other_grad.cpu())
        logits_student.grad.zero_()

        kd_loss = _loss_kd(logits_student, logits_teacher, temperature)
        kd_loss.backward()
        kd_grad = logits_student.grad.clone()
        self.kd_grad_list.append(kd_grad.cpu())
        logits_student.grad.zero_()
        

