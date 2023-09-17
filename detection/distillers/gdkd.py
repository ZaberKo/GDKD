import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage

from .build import KD_REGISTRY
from .rcnnkd import RCNNKD
from .utils import kl_div


@KD_REGISTRY.register()
class GDKD(RCNNKD):
    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        kd_args,
    ):
        super().__init__(student=student, teacher=teacher, kd_args=kd_args)

    def _forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(images.tensor)
        t_features = self.teacher.backbone(images.tensor)

        losses = {}
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(
                images, s_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device)
                         for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.student.roi_heads(
            images, s_features, proposals, gt_instances)

        # TODO: avoid duplicate forward for student
        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals)
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals)

        losses["loss_kd"] = rcnn_gdkd_loss(
            s_predictions,
            t_predictions,
            k=self.kd_args.GDKD.TOPK,
            temperature=self.kd_args.GDKD.T,
            strategy="best",
            w0=self.kd_args.GDKD.W0,
            w1=self.kd_args.GDKD.W1,
            w2=self.kd_args.GDKD.W2,
            kl_type="forward"
        )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


def rcnn_gdkd_loss(s_predictions, t_predictions, k, strategy, w0, w1, w2, temperature, kl_type):
    s_logits, s_bbox_offsets = s_predictions
    t_logits, t_bbox_offsets = t_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)
    loss_gdkd = gdkd_loss(s_logits, t_logits,
                          k, strategy, w0, w1, w2, temperature, kl_type)

    return loss_gdkd


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


def gdkd_loss(logits_student, logits_teacher, k, strategy, w0, w1, w2, temperature, kl_type):
    mask_u1, mask_u2 = get_masks(logits_teacher, k, strategy)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = (
        F.kl_div(log_p0_student, p0_teacher, reduction="batchmean")
        * (temperature**2)
    )

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u2, dim=1
    )

    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - MASK_MAGNITUDE * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - MASK_MAGNITUDE * mask_u1, dim=1
    )

    low_other_loss = kl_div(
        log_p2_student, log_p2_teacher, temperature, kl_type)

    return (
        w0 * high_loss + w1 * low_top_loss + w2 * low_other_loss,
        high_loss.detach(),
        low_top_loss.detach(),
        low_other_loss.detach()
    )
