import torch
import torch.nn.functional as F

from tqdm import tqdm

from detectron2.layers import FrozenBatchNorm2d

def kl_div(log_p, log_q, T, kl_type, reduction="batchmean"):
    if kl_type == "forward":
        res = F.kl_div(log_p, log_q, reduction=reduction,
                       log_target=True)
    elif kl_type == "reverse":
        res = F.kl_div(log_q, log_p, reduction=reduction,
                       log_target=True)
    elif kl_type == "both":
        res = 0.5 * (
            F.kl_div(log_p, log_q, reduction=reduction, log_target=True) +
            F.kl_div(log_q, log_p, reduction=reduction, log_target=True)
        )
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")

    if reduction == "none":
        res = res.sum(dim=1)  # [B,C]->[B]

    res = res * (T**2)

    return res


def freeze(model, freeze_bn=False):
    # model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    if freeze_bn:
        #TODO: also freeze BN1d?
        FrozenBatchNorm2d.convert_frozen_batchnorm(model)
