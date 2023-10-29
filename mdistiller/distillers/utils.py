import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes


def kl_div(log_p, log_q, T, kl_type="forward", reduction="batchmean"):
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