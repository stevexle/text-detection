"""
DBNet Fast Polygon Post-Processor.
Binarizes probability map, extracts contours, expands polygons via Vatti Unclipping, and filters scores.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np
import pyclipper
import torch

from src.postprocess.builder import POSTPROCESSORS


@POSTPROCESSORS.register_module(name="DBPostprocessor")
@POSTPROCESSORS.register_module(name="DBPostProcessor")
@POSTPROCESSORS.register_module(name="db_postprocessor")
class DBPostprocessor:
    """Fast Polygon Extractor for DBNet inference."""

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.5,
        min_size: int = 3,
        **kwargs,
    ):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def _unclip(self, box: np.ndarray, unclip_ratio: float) -> np.ndarray:
        """Expand polygon outwards using Vatti clipping algorithm."""
        poly = box.reshape(-1, 2)
        area = cv2.contourArea(poly)
        length = cv2.arcLength(poly, True)
        if length <= 0:
            return box

        distance = area * unclip_ratio / length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = pco.Execute(distance)
        if len(expanded) == 0:
            return box
        return np.array(expanded[0], dtype=np.float32)

    def _get_box_score(self, prob_map: np.ndarray, poly: np.ndarray) -> float:
        """Calculate mean probability inside the polygon box."""
        h, w = prob_map.shape[:2]
        x_min = max(0, int(np.floor(poly[:, 0].min())))
        x_max = min(w - 1, int(np.ceil(poly[:, 0].max())))
        y_min = max(0, int(np.floor(poly[:, 1].min())))
        y_max = min(h - 1, int(np.ceil(poly[:, 1].max())))

        if x_max <= x_min or y_max <= y_min:
            return 0.0

        mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        shifted_poly = np.empty_like(poly, dtype=np.int32)
        shifted_poly[:, 0] = poly[:, 0] - x_min
        shifted_poly[:, 1] = poly[:, 1] - y_min
        cv2.fillPoly(mask, [shifted_poly], 1)

        cropped_prob = prob_map[y_min : y_max + 1, x_min : x_max + 1]
        score = cv2.mean(cropped_prob, mask=mask)[0]
        return float(score)

    def _order_points_clockwise(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points in Clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
        pts = pts.reshape(4, 2)
        # Sort by y (top two and bottom two)
        y_sorted = pts[np.argsort(pts[:, 1])]
        top_two = y_sorted[:2]
        bottom_two = y_sorted[2:]

        # Sort top two by x (Top-Left, Top-Right)
        top_left, top_right = top_two[np.argsort(top_two[:, 0])]
        # Sort bottom two by x (Bottom-Right, Bottom-Left)
        bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0])]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _boxes_from_bitmap(
        self,
        prob_map: np.ndarray,
        bitmap: np.ndarray,
        dest_width: Optional[int] = None,
        dest_height: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract bounding polygons and scores from a single probability map."""
        src_h, src_w = prob_map.shape[:2]
        dest_w = dest_width or src_w
        dest_h = dest_height or src_h

        scale_x = dest_w / src_w
        scale_y = dest_h / src_h

        contours, _ = cv2.findContours(
            bitmap.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        num_contours = min(len(contours), self.max_candidates)
        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for i in range(num_contours):
            contour = contours[i]
            # Minimum area bounding box
            min_rect = cv2.minAreaRect(contour)
            w, h = min_rect[1]
            if min(w, h) < self.min_size:
                continue

            # Convert to 4 points
            box = cv2.boxPoints(min_rect)

            score = self._get_box_score(prob_map, box)
            if score < self.box_thresh:
                continue

            # Unclip polygon back to original text boundary
            unclipped = self._unclip(box, self.unclip_ratio)
            if len(unclipped) < 3:
                continue

            # Minimum area box on unclipped polygon
            unclip_rect = cv2.minAreaRect(unclipped.astype(np.float32))
            unclip_w, unclip_h = unclip_rect[1]
            if min(unclip_w, unclip_h) < self.min_size + 2:
                continue

            quad = cv2.boxPoints(unclip_rect)
            quad = self._order_points_clockwise(quad)

            # Rescale to destination image size
            quad[:, 0] = np.clip(quad[:, 0] * scale_x, 0, dest_w)
            quad[:, 1] = np.clip(quad[:, 1] * scale_y, 0, dest_h)

            boxes.append(quad)
            scores.append(score)

        if len(boxes) == 0:
            return np.empty((0, 4, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32)

    def extract_detections(
        self,
        prob_map: Union[torch.Tensor, np.ndarray],
        orig_shape: Optional[Tuple[int, int]] = None,
        label: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Extract text detections in the unified 3-field schema:
        [
            {
                "label": "text",
                "confidence": 0.9842,
                "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            },
            ...
        ]
        """
        if isinstance(prob_map, torch.Tensor):
            prob_map = prob_map.detach().cpu().numpy()
        if prob_map.ndim == 4:
            prob_map = prob_map[0, 0]
        elif prob_map.ndim == 3:
            prob_map = prob_map[0]

        bitmap = prob_map > self.thresh
        dest_h, dest_w = orig_shape if orig_shape is not None else (None, None)
        boxes, scores = self._boxes_from_bitmap(prob_map, bitmap, dest_w, dest_h)

        detections = []
        for poly, score in zip(boxes, scores):
            detections.append({
                "label": label,
                "confidence": round(float(score), 4),
                "polygon": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly],
            })
        return detections

    def __call__(
        self,
        preds: Union[torch.Tensor, np.ndarray, Dict[str, Any]],
        orig_shape: Optional[Tuple[int, int]] = None,
        shape_list: Optional[Sequence[Tuple[int, int]]] = None,
        return_dict: bool = False,
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Args:
            preds: Tensor (B, 1, H, W) or Dict containing 'prob_map'
            orig_shape: (dest_h, dest_w) for single image
            shape_list: List of (dest_h, dest_w) for each image in batch
            return_dict: If True, returns legacy {"polygons": ..., "scores": ...} format
        Returns:
            Unified List of Dicts: [{"label": "text", "confidence": float, "polygon": [[x, y], ...]}, ...]
        """
        if isinstance(preds, dict):
            prob_maps = preds["prob_map"]
        else:
            prob_maps = preds

        if isinstance(prob_maps, torch.Tensor):
            prob_maps = prob_maps.detach().cpu().numpy()

        is_single = False
        if prob_maps.ndim == 4:
            if prob_maps.shape[0] == 1 and shape_list is None:
                is_single = True
            prob_maps = prob_maps[:, 0, :, :]
        elif prob_maps.ndim == 3:
            if prob_maps.shape[0] == 1 and shape_list is None:
                is_single = True
        elif prob_maps.ndim == 2:
            is_single = True
            prob_maps = prob_maps[None, :, :]

        batch_size = prob_maps.shape[0]
        all_detections = []
        all_legacy = []

        for b in range(batch_size):
            prob = prob_maps[b]
            bitmap = prob > self.thresh

            dest_h, dest_w = None, None
            if orig_shape is not None and b == 0:
                dest_h, dest_w = orig_shape
            elif shape_list is not None and b < len(shape_list):
                dest_h, dest_w = shape_list[b]

            boxes, scores = self._boxes_from_bitmap(prob, bitmap, dest_w, dest_h)

            detections = []
            for poly, score in zip(boxes, scores):
                detections.append({
                    "label": "text",
                    "confidence": round(float(score), 4),
                    "polygon": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly],
                })
            all_detections.append(detections)
            all_legacy.append({"polygons": boxes, "scores": scores})

        if return_dict:
            return all_legacy[0] if is_single else all_legacy

        return all_detections[0] if is_single else all_detections


DBPostProcessor = DBPostprocessor

