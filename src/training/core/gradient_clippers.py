"""

"""
from typing import Protocol
import torch


class GradientClipper(Protocol):
    def clip(self, model_parameters) -> None: ...

class NormClipper:
    def __init__(self, max_norm: float, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    def clip(self, model_parameters):
        torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=self.max_norm, norm_type=self.norm_type)

class ValueClipper:
    def __init__(self, clip_value: float):
        self.clip_value = clip_value
    def clip(self, model_parameters):
        torch.nn.utils.clip_grad_value_(model_parameters, clip_value=self.clip_value)