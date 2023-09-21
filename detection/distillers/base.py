# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import copy

from detectron2.config import configurable
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.layers import move_device_like
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling import build_model

from .build import KD_REGISTRY
from .utils import freeze



class RCNNKD(nn.Module):
    """
    Base Distiller class for R-CNN style models.
    """

    @configurable
    def __init__(
        self,
        *,
        student: GeneralizedRCNN,
        teacher: GeneralizedRCNN,
        kd_args
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        if not (isinstance(student, GeneralizedRCNN) and isinstance(teacher, GeneralizedRCNN)):
            raise ValueError(
                "student and teacher must be GeneralizedRCNN instances")

        self.student = student
        self.teacher = teacher
        self.kd_args = kd_args

        # impl kd modules here in sub-class
        # ...

    @classmethod
    def from_config(cls, cfg):
        student = build_model(cfg)

        if cfg.KD.TYPE == "Vanilla":
            teacher = None
        else:
            teacher = build_model(cfg.TEACHER)
            freeze(teacher)

        return {
            "student": student,
            "teacher": teacher,
            "kd_args": cfg.KD,
        }
    
    def train(self, mode=True):
        """
        Override the default train() to freeze the teacher
        """
        super().train(mode)
        # always set teacher in eval mode:
        self.teacher.eval()

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        return self.student.preprocess_image(batched_inputs)

    def preprocess_image_teacher(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # images = self.teacher.preprocess_image(batched_inputs)

        images = [self.teacher._move_to_current_device(
            x["image"]) for x in batched_inputs]
        images = [(x - self.teacher.pixel_mean) /
                  self.teacher.pixel_std for x in images]
        if self.student.input_format != self.teacher.input_format:
            if self.student.input_format == "RGB" and self.teacher.input_format == "BGR":
                # images = [
                #     x.index_select(0, self.teacher._move_to_current_device(
                #         torch.LongTensor([2, 1, 0]))
                #     ) for x in images
                # ]
                images = [x.flip(0) for x in images]
            else:
                raise NotImplementedError
        images = ImageList.from_tensors(
            images, self.student.backbone.size_divisibility)
        return images

    @property
    def device(self):
        return self.student.device

    @property
    def vis_period(self):
        return self.student.vis_period
    
    def record_info(self, info_dict):
        storage=get_event_storage()
        for k,v in info_dict.items():
            storage.put_scalar(k, v)

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Default Distiller: no distillation (Vanilla)

        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        raise NotImplementedError

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        return self.student.inference(batched_inputs, detected_instances, do_postprocess)

    def visualize_training(self, batched_inputs, proposals):
        return self.student.visualize_training(batched_inputs, proposals)


