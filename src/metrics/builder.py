"""
Metrics Builder.
"""

from typing import Any, Dict
from .evaluator import ICDAREvaluator, METRICS


def build_evaluator(cfg: Dict[str, Any], **default_args) -> ICDAREvaluator:
    """Build evaluation metric from configuration dictionary."""
    return METRICS.build(cfg, default_args=default_args)
