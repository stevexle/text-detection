"""
Feature Pyramid Network (FPN) Neck for DBNet Text Detection.
Merges multi-scale features and produces a single unified feature representation.
"""

from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.builder import NECKS


@NECKS.register_module(name="FPN")
@NECKS.register_module(name="fpn")
class FPN(nn.Module):
    """Feature Pyramid Network with multi-scale feature concatenation for DBNet."""

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024, 2048),
        inner_channels: int = 256,
        out_channels: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_channels = out_channels

        # Lateral 1x1 convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, inner_channels, kernel_size=1, bias=False)
            for in_c in in_channels
        ])

        # Smooth 3x3 convolutions
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(inner_channels, inner_channels // 4, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        ])

        # Final projection if needed
        concat_channels = (inner_channels // 4) * len(in_channels)
        if concat_channels != out_channels:
            self.final_conv = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.final_conv = nn.Identity()

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [C2, C3, C4, C5] with shapes 1/4, 1/8, 1/16, 1/32
        Returns:
            Fused feature map of shape (B, out_channels, H/4, W/4)
        """
        assert len(features) == len(self.in_channels), f"Expected {len(self.in_channels)} feature maps, got {len(features)}"

        # 1. Top-down lateral merge
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="nearest",
            )

        # 2. Smooth convs
        p_features = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]

        # 3. Upsample all P3, P4, P5 to P2 resolution (1/4 of input)
        target_size = p_features[0].shape[2:]
        upsampled = [p_features[0]]
        for p in p_features[1:]:
            upsampled.append(F.interpolate(p, size=target_size, mode="nearest"))

        # 4. Concatenate along channel dimension
        fused = torch.cat(upsampled, dim=1)
        fused = self.final_conv(fused)
        return fused
