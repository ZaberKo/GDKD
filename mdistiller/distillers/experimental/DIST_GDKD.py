import torch
import torch.nn.functional as F

from ..DIST import DIST, intra_class_relation
from ..GDKD import GDKD, gdkd_loss

class DIST_GDKD(DIST):
    def __init__(self, student, teacher, cfg):
        super(DIST_GDKD, self).__init__(student, teacher, cfg)

        self.w0 = cfg.GDKD.W0
        self.w1 = cfg.GDKD.W1
        self.w2 = cfg.GDKD.W2
        self.gdkd_warmup_epochs = cfg.GDKD.WARMUP
        self.k = cfg.GDKD.TOPK

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)


        y_s = F.softmax(logits_student / self.temperature, dim=1)
        y_t = F.softmax(logits_teacher / self.temperature, dim=1)
        loss_dist_intra = intra_class_relation(y_s, y_t) * (self.temperature**2)
        self.dist_intra_loss = loss_dist_intra.detach()
        loss_dist_intra = self.gamma * loss_dist_intra

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
            "loss_kd": loss_dist_intra+loss_gdkd,
        }
        return logits_student, losses_dict

    def get_train_info(self):
        return {
            "dist_intra_loss": self.dist_intra_loss,
            "high_loss": self.high_loss,
            "low_top_loss": self.low_top_loss,
            "low_other_loss": self.low_other_loss,
        }