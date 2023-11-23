import torch
import torch.nn.functional as F

from ..ReviewKD import ReviewKD, hcl_loss
from ..GDKD import gdkd_loss


class ReviewKD_GDKD(ReviewKD):
    def __init__(self, student, teacher, cfg):
        super(ReviewKD_GDKD, self).__init__(student, teacher, cfg)

        self.w0 = cfg.GDKD.W0
        self.w1 = cfg.GDKD.W1
        self.w2 = cfg.GDKD.W2
        self.temperature = cfg.GDKD.T
        self.gdkd_warmup_epochs = cfg.GDKD.WARMUP
        self.k = cfg.GDKD.TOPK


    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["preact_feats"][1:] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_reviewkd = hcl_loss(results, features_teacher)
        self.reviewkd_loss = loss_reviewkd.detach()

        loss_reviewkd = (
            self.reviewkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * loss_reviewkd
        )

        loss_gdkd, self.high_loss, self.low_top_loss, self.low_other_loss = gdkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.k,
            "best",
            self.w0,
            self.w1,
            self.w2,
            self.temperature,
            kl_type="forward"
        )

        loss_gdkd = min(kwargs["epoch"] / self.gdkd_warmup_epochs, 1.0) * loss_gdkd


        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_reviewkd+loss_gdkd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "reviewkd_loss": self.reviewkd_loss,
            "high_loss": self.high_loss,
            "low_top_loss": self.low_top_loss,
            "low_other_loss": self.low_other_loss,
        }


