"""
Registries and Model Builders for Backbones, Necks, Heads, and Full Models.
"""

from typing import Any, Dict
import torch.nn as nn
from src.utils.registry import Registry

BACKBONES = Registry("backbone")
NECKS = Registry("neck")
HEADS = Registry("head")
MODELS = Registry("model")


def build_backbone(cfg: Dict[str, Any], **default_args) -> nn.Module:
    """Build a backbone module from config."""
    return BACKBONES.build(cfg, **default_args)


def build_neck(cfg: Dict[str, Any], **default_args) -> nn.Module:
    """Build a neck module from config."""
    return NECKS.build(cfg, **default_args)


def build_head(cfg: Dict[str, Any], **default_args) -> nn.Module:
    """Build a head module from config."""
    return HEADS.build(cfg, **default_args)


def build_model(cfg: Dict[str, Any], **default_args) -> nn.Module:
    """Build a complete text detection model from config."""
    return MODELS.build(cfg, **default_args)
