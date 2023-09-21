import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage

from .build import KD_REGISTRY
from .base import RCNNKD
from .gdkd import rcnn_gdkd_loss
from .reviewkd import hcl, build_abfs


@KD_REGISTRY.register()
class ReviewGDKD(RCNNKD):
    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        kd_args,
    ):
        super().__init__(student=student, teacher=teacher, kd_args=kd_args)

        self.abfs = build_abfs(
            in_channels=kd_args.REVIEWKD.IN_CHANNELS,
            out_channels=kd_args.REVIEWKD.OUT_CHANNELS,
            mid_channel=kd_args.REVIEWKD.MAX_MID_CHANNEL
        )

    def _forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def _forward_abf_trans(self, student_features):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            results.insert(0, out_features)

        return results

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        s_images = self.preprocess_image(batched_inputs)
        t_images = self.preprocess_image_teacher(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(s_images.tensor)
        t_features = self.teacher.backbone(t_images.tensor)

        losses = {}
        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(
                s_images, s_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device)
                         for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.student.roi_heads(
            s_images, s_features, proposals, gt_instances)

        # TODO: avoid duplicate forward for student
        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals)
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals)

        losses["loss_gdkd"], info_dict = rcnn_gdkd_loss(
            s_predictions,
            t_predictions,
            [x.gt_classes for x in sampled_proposals],
            k=self.kd_args.GDKD.TOPK,
            w0=self.kd_args.GDKD.W0,
            w1=self.kd_args.GDKD.W1,
            w2=self.kd_args.GDKD.W2,
            temperature=self.kd_args.GDKD.T,
            distill_type=self.kd_args.GDKD.DISTILL_TYPE
        )

        self.record_info(info_dict)

        s_features_flat = [s_features[f] for f in s_features]
        t_features_flat = [t_features[f] for f in t_features]

        s_features_flat_trans = self._forward_abf_trans(s_features_flat)
        losses['loss_reviewkd'] = hcl(
            s_features_flat_trans, t_features_flat) * self.kd_args.REVIEWKD.KD_WEIGHT

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
