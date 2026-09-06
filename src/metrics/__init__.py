"""
Evaluation Metrics for Text Detection.
"""

from .evaluator import METRICS, ICDAREvaluator, polygon_iou
from .builder import build_evaluator

__all__ = ["METRICS", "ICDAREvaluator", "polygon_iou", "build_evaluator"]
