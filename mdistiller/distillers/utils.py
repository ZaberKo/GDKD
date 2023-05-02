import torch
import torch.nn.functional as F

from tqdm import tqdm

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


def validate(dataloader, model, num_classes):
    logits_dict = [[] for _ in range(num_classes)]

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (image, target, index) in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits, _ = model(image)

            for j in range(num_classes):
                logits_dict[j].append(logits[target == j])

    res = []
    for i in range(num_classes):
        res.append(
            torch.concat(logits_dict[i])
        )
    return res