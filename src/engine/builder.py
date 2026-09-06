"""
Engine Builder.
"""

from typing import Any, Dict
from .trainer import DBNetTrainer, ENGINE, build_optimizer, build_lr_scheduler


def build_trainer(cfg: Dict[str, Any], **default_args) -> DBNetTrainer:
    """Build trainer from configuration dictionary."""
    return ENGINE.build(cfg, default_args=default_args)
