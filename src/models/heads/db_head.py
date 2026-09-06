"""
Differentiable Binarization Head (DBHead) for Text Detection.
Predicts Probability Map, Threshold Map, and Approximate Binary Map using Differentiable Step Function.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from src.models.builder import HEADS


@HEADS.register_module(name="DBHead")
@HEADS.register_module(name="db_head")
class DBHead(nn.Module):
    """DBNet Prediction Head with Differentiable Binarization step function."""

    def __init__(
        self,
        in_channels: int = 256,
        k: float = 50.0,
        adaptive: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.adaptive = adaptive

        # 1. Probability Map Branch (1/4 -> 1/1)
        self.prob_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        # 2. Threshold Map Branch (1/4 -> 1/1)
        if self.adaptive:
            self.thresh_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels // 4, 1, kernel_size=2, stride=2),
                nn.Sigmoid(),
            )
        else:
            self.thresh_conv = None

    def step_function(self, prob_map: torch.Tensor, thresh_map: torch.Tensor) -> torch.Tensor:
        """Differentiable Binarization step function with amplification factor k."""
        return torch.reciprocal(1.0 + torch.exp(-self.k * (prob_map - thresh_map)))

    def forward(self, x: torch.Tensor, return_all_maps: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Fused feature map of shape (B, in_channels, H/4, W/4)
            return_all_maps: If True, forces computation of thresh_map and binary_map even in eval mode.
        Returns:
            Dict containing 'prob_map', and optionally 'thresh_map', and 'binary_map' of shape (B, 1, H, W)
        """
        prob_map = self.prob_conv(x)

        # In evaluation / inference mode, bypass threshold branch for 2x faster inference
        if not self.training and not return_all_maps:
            return {"prob_map": prob_map}

        if self.adaptive and self.thresh_conv is not None:
            thresh_map = self.thresh_conv(x)
            binary_map = self.step_function(prob_map, thresh_map)
            return {
                "prob_map": prob_map,
                "thresh_map": thresh_map,
                "binary_map": binary_map,
            }

        return {"prob_map": prob_map}
