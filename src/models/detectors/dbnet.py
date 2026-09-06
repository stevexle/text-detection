"""
Complete DBNet Architecture: Backbone + Neck + DBHead.
Supports flexible pretrained weight loading for transfer learning.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from src.models.builder import MODELS, build_backbone, build_neck, build_head
from src.utils.checkpoint import load_pretrained_weights


@MODELS.register_module(name="DBNet")
@MODELS.register_module(name="dbnet")
class DBNet(nn.Module):
    """DBNet (Real-time Scene Text Detection with Differentiable Binarization)."""

    def __init__(
        self,
        backbone: Union[Dict[str, Any], nn.Module],
        neck: Optional[Union[Dict[str, Any], nn.Module]] = None,
        head: Optional[Union[Dict[str, Any], nn.Module]] = None,
        pretrained_weights: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        # 1. Build Backbone
        if isinstance(backbone, dict):
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = backbone

        # Auto-infer neck in_channels from backbone if needed
        backbone_channels = getattr(self.backbone, "out_channels", [256, 512, 1024, 2048])

        # 2. Build Neck
        if neck is not None:
            if isinstance(neck, dict):
                neck_cfg = neck.copy()
                if "in_channels" not in neck_cfg:
                    neck_cfg["in_channels"] = backbone_channels
                self.neck = build_neck(neck_cfg)
            else:
                self.neck = neck
        else:
            self.neck = None

        # Auto-infer head in_channels from neck
        neck_channels = getattr(self.neck, "out_channels", 256) if self.neck is not None else backbone_channels[-1]

        # 3. Build Head
        if head is not None:
            if isinstance(head, dict):
                head_cfg = head.copy()
                if "in_channels" not in head_cfg:
                    head_cfg["in_channels"] = neck_channels
                self.head = build_head(head_cfg)
            else:
                self.head = head
        else:
            self.head = None

        # 4. Optional Pretrained OCR Weights Loading (Transfer Learning)
        if pretrained_weights:
            self._load_pretrained_ocr_weights(pretrained_weights)

    def _load_pretrained_ocr_weights(self, weights_path: str):
        """Load pretrained OCR weights into full DBNet model."""
        p = Path(weights_path)
        if p.exists():
            load_pretrained_weights(self, p)
        elif Path(f"weights/dbnet/{weights_path}").exists():
            load_pretrained_weights(self, Path(f"weights/dbnet/{weights_path}"))
        else:
            print(f"[DBNet] Warning: Pretrained weights not found at: {weights_path}")

    def forward(self, x: torch.Tensor, return_all_maps: bool = False) -> Dict[str, torch.Tensor]:
        """
        End-to-end forward pass.
        Args:
            x: Input image tensor (B, 3, H, W)
            return_all_maps: If True, forces threshold branch evaluation even in eval mode.
        Returns:
            Dict containing 'prob_map', and optionally 'thresh_map', 'binary_map' (B, 1, H, W)
        """
        features = self.backbone(x)

        if self.neck is not None:
            fused = self.neck(features)
        else:
            fused = features[-1]

        if self.head is not None:
            if hasattr(self.head, "forward") and "return_all_maps" in self.head.forward.__code__.co_varnames:
                return self.head(fused, return_all_maps=return_all_maps)
            return self.head(fused)

        return {"features": fused}
