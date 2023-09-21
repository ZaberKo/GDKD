import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple




from distillers import KD_REGISTRY, DKD


@KD_REGISTRY.register()
class DKDDebug(DKD):
    def forward_predictions(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        s_images = self.preprocess_image(batched_inputs)
        t_images = self.preprocess_image_teacher(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(s_images.tensor)
        t_features = self.teacher.backbone(t_images.tensor)

        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(
                s_images, s_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device)
                         for x in batched_inputs]

        sampled_proposals, detector_losses = self.student.roi_heads(
            s_images, s_features, proposals, gt_instances)

        gt_classes = torch.cat(
            [x.gt_classes for x in sampled_proposals], 0).reshape(-1)

        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals)
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals)

        return s_predictions[0], t_predictions[0], gt_classes
    
