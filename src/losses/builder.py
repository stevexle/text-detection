"""
Registries and Builders for Loss Functions.
"""

from typing import Any, Dict
import torch.nn as nn
from src.utils.registry import Registry

LOSSES = Registry("loss")


def build_loss(cfg: Dict[str, Any], **default_args) -> nn.Module:
    """Build a loss module from configuration dictionary using LOSSES registry."""
    return LOSSES.build(cfg, **default_args)
