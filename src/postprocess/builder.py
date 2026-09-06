"""
Registries and Builders for Post-Processors.
"""

from typing import Any, Dict
from src.utils.registry import Registry

POSTPROCESSORS = Registry("postprocessor")


def build_postprocessor(cfg: Dict[str, Any], **default_args) -> Any:
    """Build a postprocessor module from config."""
    return POSTPROCESSORS.build(cfg, **default_args)
