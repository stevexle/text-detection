"""
Evaluation Metrics for Text Detection following the standard ICDAR 2015 Protocol.
Calculates Precision, Recall, and Hmean (F1-score) based on Polygon IoU matching.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from shapely.geometry import Polygon

from src.utils.registry import Registry

METRICS = Registry("metrics")


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two arbitrary polygons.
    """
    try:
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)

        if not p1.is_valid:
            p1 = p1.buffer(0)
        if not p2.is_valid:
            p2 = p2.buffer(0)

        if not p1.is_valid or not p2.is_valid or p1.is_empty or p2.is_empty:
            return 0.0

        inter_area = p1.intersection(p2).area
        union_area = p1.union(p2).area

        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)
    except Exception:
        return 0.0


@METRICS.register_module(name="ICDAREvaluator")
@METRICS.register_module(name="icdar_evaluator")
class ICDAREvaluator:
    """
    ICDAR 2015 standard evaluation protocol for scene text detection.
    Matches predicted polygons with ground truth polygons at IoU >= 0.5.
    """

    def __init__(self, iou_thresh: float = 0.5, **kwargs):
        self.iou_thresh = iou_thresh
        self.reset()

    def reset(self):
        """Reset accumulated evaluation counters."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_gt = 0
        self.total_pred = 0

    def evaluate_image(
        self,
        pred_polygons: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        pred_scores: Optional[Union[List[float], np.ndarray]] = None,
        gt_polygons: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        gt_ignore: Optional[Union[List[bool], np.ndarray]] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """
        Evaluate detections on a single image.
        Args:
            pred_polygons: List of predicted polygons [(N, 2), ...]
            pred_scores: Confidence scores for each predicted polygon
            gt_polygons: List of ground-truth polygons [(N, 2), ...]
            gt_ignore: Boolean flag for each GT polygon (True = ignore tag)
            detections: List of unified detection dicts: [{"label": ..., "confidence": ..., "polygon": ...}]
        Returns:
            Dict with {tp, fp, fn, num_gt, num_pred}
        """
        # Parse from unified detection format if provided
        if detections is not None:
            pred_polygons = [np.array(d["polygon"], dtype=np.float32) for d in detections]
            pred_scores = [d.get("confidence", 1.0) for d in detections]
        elif pred_polygons is None:
            pred_polygons = []
        if gt_polygons is None or len(gt_polygons) == 0:
            gt_polygons = []
            gt_ignore = []

        if gt_ignore is None:
            gt_ignore = [False] * len(gt_polygons)

        n_pred = len(pred_polygons)
        n_gt = len(gt_polygons)

        # Sort predictions by score descending if scores provided
        if pred_scores is not None and len(pred_scores) == n_pred and n_pred > 0:
            sort_indices = np.argsort(pred_scores)[::-1]
            pred_polygons = [pred_polygons[i] for i in sort_indices]

        gt_matched = [False] * n_gt
        pred_matched = [False] * n_pred
        pred_ignored = [False] * n_pred

        tp = 0
        fp = 0

        # Match predictions to ground truth
        for p_idx, pred_poly in enumerate(pred_polygons):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt_poly in enumerate(gt_polygons):
                iou = polygon_iou(pred_poly, gt_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= self.iou_thresh and best_gt_idx >= 0:
                if gt_ignore[best_gt_idx]:
                    # Matched with an ignored ground truth -> ignore this prediction
                    pred_ignored[p_idx] = True
                elif not gt_matched[best_gt_idx]:
                    # First match with valid ground truth -> True Positive
                    gt_matched[best_gt_idx] = True
                    pred_matched[p_idx] = True
                    tp += 1
                else:
                    # Duplicate prediction on already matched GT -> False Positive
                    fp += 1
            else:
                fp += 1

        # Count ground truth false negatives (valid GTs not matched)
        valid_gt_count = sum(1 for ig in gt_ignore if not ig)
        fn = valid_gt_count - tp

        # Accumulate
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_gt += valid_gt_count
        self.total_pred += n_pred

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt": valid_gt_count,
            "num_pred": n_pred,
        }

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute accumulated Precision, Recall, and Hmean (F1-score).
        """
        tp = self.total_tp
        fp = self.total_fp
        fn = self.total_fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            hmean = 2.0 * precision * recall / (precision + recall)
        else:
            hmean = 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "hmean": float(hmean),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total_gt": self.total_gt,
            "total_pred": self.total_pred,
        }
