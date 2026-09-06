"""
ResNet Backbone (18, 34, 50, 101) with Multi-Scale Feature Extraction.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from src.models.builder import BACKBONES
from src.utils.checkpoint import load_pretrained_weights


@BACKBONES.register_module(name="ResNet")
@BACKBONES.register_module(name="resnet")
class ResNet(nn.Module):
    """ResNet multi-scale feature extractor."""

    ARCH_MAP = {
        18: (models.resnet18, [64, 128, 256, 512]),
        34: (models.resnet34, [64, 128, 256, 512]),
        50: (models.resnet50, [256, 512, 1024, 2048]),
        101: (models.resnet101, [256, 512, 1024, 2048]),
    }

    def __init__(
        
        self,
        depth: int = 50,
        pretrained: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        weights_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if depth not in self.ARCH_MAP:
            raise KeyError(f"Unsupported ResNet depth {depth}. Supported: {list(self.ARCH_MAP.keys())}")

        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        resnet_fn, self.out_channels = self.ARCH_MAP[depth]
        base_model = resnet_fn(weights=None)

        # Stem layers (stride 4)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # Residual stages: layer1 (stride 4), layer2 (stride 8), layer3 (stride 16), layer4 (stride 32)
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Load weights if specified
        if pretrained:
            self._load_pretrained(weights_path)

        if self.frozen_stages >= 0:
            self._freeze_stages()

    def _load_pretrained(self, weights_path: Optional[str] = None):
        """Load pretrained ImageNet weights from local path or cache."""
        target_path = None
        if weights_path and Path(weights_path).exists():
            target_path = Path(weights_path)
        elif Path(f"weights/dbnet/resnet{self.depth}_imagenet.pth").exists():
            target_path = Path(f"weights/dbnet/resnet{self.depth}_imagenet.pth")

        if target_path and target_path.exists():
            load_pretrained_weights(self, target_path)
        else:
            try:
                weights = getattr(models, f"ResNet{self.depth}_Weights").DEFAULT
                state_dict = weights.get_state_dict(progress=False)
                load_pretrained_weights(self, state_dict)
            except Exception as e:
                print(f"[ResNet] Could not load torchvision default weights: {e}")

    def _freeze_stages(self):
        """Freeze specified stages during warmup or fine-tuning."""
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1]:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f"layer{i}")
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        stages = [c2, c3, c4, c5]
        return tuple(stages[i] for i in self.out_indices)
