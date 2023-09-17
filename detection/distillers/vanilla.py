from typing import Dict, List, Optional, Tuple
import torch

from .build import KD_REGISTRY
from .base import RCNNKD

@KD_REGISTRY.register()
class Vanilla(RCNNKD):
    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        return self.student.forward(batched_inputs)