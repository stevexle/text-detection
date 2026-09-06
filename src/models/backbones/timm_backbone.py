"""
TIMM Backbone Adapter for arbitrary TIMM models.
"""

from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
from src.models.builder import BACKBONES


@BACKBONES.register_module(name="TIMMBackbone")
@BACKBONES.register_module(name="timm_backbone")
class TIMMBackbone(nn.Module):
    """Adapter for models from the timm library."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3, 4),
        **kwargs,
    ):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm via: uv pip install timm")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            **kwargs,
        )
        self.out_channels = self.model.feature_info.channels()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(self.model(x))
