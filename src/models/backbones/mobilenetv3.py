"""
MobileNetV3 Backbone (Large and Small) for Lightweight Text Detection.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.builder import BACKBONES
from src.utils.checkpoint import load_pretrained_weights


@BACKBONES.register_module(name="MobileNetV3")
@BACKBONES.register_module(name="mobilenetv3")
class MobileNetV3(nn.Module):
    """MobileNetV3 feature extractor returning 4 multi-scale stages."""

    def __init__(
        self,
        arch: str = "large",
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.arch = arch.lower()
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        if self.arch == "large":
            base_model = models.mobilenet_v3_large(weights=None)
            features = base_model.features
            # Stage boundaries for large: C2 (stride 4), C3 (stride 8), C4 (stride 16), C5 (stride 32)
            self.stage1 = nn.Sequential(*features[:4])    # 24 channels, 1/4
            self.stage2 = nn.Sequential(*features[4:7])   # 40 channels, 1/8
            self.stage3 = nn.Sequential(*features[7:13])  # 112 channels, 1/16
            self.stage4 = nn.Sequential(*features[13:])   # 960 channels, 1/32
            self.out_channels = [24, 40, 112, 960]
        elif self.arch == "small":
            base_model = models.mobilenet_v3_small(weights=None)
            features = base_model.features
            self.stage1 = nn.Sequential(*features[:2])    # 16 channels, 1/4
            self.stage2 = nn.Sequential(*features[2:4])   # 24 channels, 1/8
            self.stage3 = nn.Sequential(*features[4:9])   # 48 channels, 1/16
            self.stage4 = nn.Sequential(*features[9:])    # 576 channels, 1/32
            self.out_channels = [16, 24, 48, 576]
        else:
            raise ValueError(f"Unsupported MobileNetV3 arch: {arch}. Choose 'large' or 'small'.")

        if pretrained:
            self._load_pretrained(weights_path)

        if self.frozen_stages >= 0:
            self._freeze_stages()

    def _freeze_stages(self):
        """Freeze specified stages during warmup or fine-tuning."""
        for i in range(1, self.frozen_stages + 1):
            stage = getattr(self, f"stage{i}")
            stage.eval()
            for param in stage.parameters():
                param.requires_grad = False

    def _load_pretrained(self, weights_path: Optional[str] = None):
        target_path = None
        if weights_path and Path(weights_path).exists():
            target_path = Path(weights_path)
        elif Path(f"weights/dbnet/mobilenetv3_{self.arch}_imagenet.pth").exists():
            target_path = Path(f"weights/dbnet/mobilenetv3_{self.arch}_imagenet.pth")

        if target_path and target_path.exists():
            load_pretrained_weights(self, target_path)
        else:
            try:
                weights = getattr(models, f"MobileNet_V3_{self.arch.capitalize()}_Weights").DEFAULT
                state_dict = weights.get_state_dict(progress=False)
                load_pretrained_weights(self, state_dict)
            except Exception as e:
                print(f"[MobileNetV3] Could not load torchvision default weights: {e}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        stages = [c2, c3, c4, c5]
        return tuple(stages[i] for i in self.out_indices)
