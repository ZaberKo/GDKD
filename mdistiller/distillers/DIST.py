import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


def dist_loss(logits_student, logits_teacher, T, beta, gamma):
    y_s = F.softmax(logits_student / T, dim=1)
    y_t = F.softmax(logits_teacher / T, dim=1)
    inter_loss = inter_class_relation(y_s, y_t) * (T**2)
    intra_loss = intra_class_relation(y_s, y_t) * (T**2)

    return (
        beta * inter_loss + gamma * intra_loss,
        inter_loss.detach(),
        intra_loss.detach()
    )


class DIST(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DIST, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DIST.CE_WEIGHT
        self.temperature = cfg.DIST.T
        self.beta = cfg.DIST.BETA
        self.gamma = cfg.DIST.GAMMA

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_kd, self.inter_loss, self.intra_loss = dist_loss(
            logits_student, logits_teacher,
            T=self.temperature,
            beta=self.beta,
            gamma=self.gamma
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "inter_loss": self.inter_loss,
            "intra_loss": self.intra_loss
        }
