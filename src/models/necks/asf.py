"""
Adaptive Scale Fusion (ASF) Neck for DBNet.
Enhances multi-scale text feature representation with spatial and scale attention.
"""

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.builder import NECKS
from src.models.necks.fpn import FPN


@NECKS.register_module(name="ASF")
@NECKS.register_module(name="asf")
class ASF(nn.Module):
    """Adaptive Scale Fusion module on top of FPN."""

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024, 2048),
        inner_channels: int = 256,
        out_channels: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.fpn = FPN(in_channels=in_channels, inner_channels=inner_channels, out_channels=out_channels)

        # Spatial attention module
        self.spatial_att = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        fused = self.fpn(features)
        att = self.spatial_att(fused)
        return fused * att + fused
