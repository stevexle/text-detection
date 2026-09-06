"""
Target Generator for Differentiable Binarization (DBNet).
Generates Probability Map (gt_prob_map), Threshold Map (gt_thresh_map),
and Threshold Mask (gt_thresh_mask) using Vatti Clipping algorithm.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np
import pyclipper


class DBTargetGenerator:
    """Target generator for DBNet training maps."""

    def __init__(
        self,
        shrink_ratio: float = 0.4,
        min_thresh: float = 0.3,
        max_thresh: float = 0.7,
    ):
        self.shrink_ratio = shrink_ratio
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def generate_targets(
        self,
        image_shape: Tuple[int, int],
        polygons: List[np.ndarray],
        ignore_tags: List[bool],
    ) -> Dict[str, np.ndarray]:
        """
        Generate ground truth maps.
        Args:
            image_shape: (H, W)
            polygons: List of (N, 2) numpy arrays
            ignore_tags: List of booleans
        Returns:
            Dict containing gt_prob_map, gt_mask, gt_thresh_map, gt_thresh_mask
        """
        h, w = image_shape
        gt_prob_map = np.zeros((h, w), dtype=np.float32)
        gt_mask = np.ones((h, w), dtype=np.float32)
        gt_thresh_map = np.zeros((h, w), dtype=np.float32)
        gt_thresh_mask = np.zeros((h, w), dtype=np.float32)

        for poly, ignore in zip(polygons, ignore_tags):
            if poly.shape[0] < 3:
                continue

            if ignore:
                cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0)
                continue

            # 1. Generate shrunk polygon (Probability Map)
            shrunk_poly = self._shrink_polygon(poly)
            if len(shrunk_poly) == 0:
                cv2.fillPoly(gt_mask, [poly.astype(np.int32)], 0)
                continue

            cv2.fillPoly(gt_prob_map, [np.array(shrunk_poly[0], dtype=np.int32)], 1.0)

            # 2. Generate threshold map around boundary
            self._draw_border_map(poly, gt_thresh_map, mask=gt_thresh_mask)

        # Baseline threshold value for background
        gt_thresh_map = gt_thresh_map * (self.max_thresh - self.min_thresh) + self.min_thresh

        return {
            "gt_prob_map": gt_prob_map,
            "gt_mask": gt_mask,
            "gt_thresh_map": gt_thresh_map,
            "gt_thresh_mask": gt_thresh_mask,
        }

    def _shrink_polygon(self, poly: np.ndarray) -> List[np.ndarray]:
        """Shrink polygon by distance D = A * (1 - r^2) / L using Vatti clipping."""
        area = cv2.contourArea(poly)
        length = cv2.arcLength(poly, True)
        if length <= 0:
            return []

        distance = area * (1.0 - self.shrink_ratio ** 2) / length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrunk = pco.Execute(-distance)
        return shrunk

    def _draw_border_map(
        self,
        poly: np.ndarray,
        canvas: np.ndarray,
        mask: np.ndarray,
    ):
        """Draw normalized distance transform map in canvas and mark mask."""
        area = cv2.contourArea(poly)
        length = cv2.arcLength(poly, True)
        if length <= 0:
            return

        distance = area * (1.0 - self.shrink_ratio ** 2) / length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        dilated = pco.Execute(distance)
        if len(dilated) == 0:
            return

        dilated_poly = np.array(dilated[0], dtype=np.int32)
        cv2.fillPoly(mask, [dilated_poly], 1.0)

        # Bounding box of dilated polygon
        x_min = max(0, int(np.min(dilated_poly[:, 0])))
        x_max = min(canvas.shape[1], int(np.max(dilated_poly[:, 0])) + 1)
        y_min = max(0, int(np.min(dilated_poly[:, 1])))
        y_max = min(canvas.shape[0], int(np.max(dilated_poly[:, 1])) + 1)

        if x_max <= x_min or y_max <= y_min:
            return

        # Grid of coordinates in local ROI
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Distance to polygon boundary for each point
        dist_map = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
        poly_pts = poly.astype(np.float32)

        for i in range(len(poly_pts)):
            p1 = poly_pts[i]
            p2 = poly_pts[(i + 1) % len(poly_pts)]

            # Point to line segment distance
            line_vec = p2 - p1
            line_len_sq = np.sum(line_vec ** 2)
            if line_len_sq == 0:
                continue

            t = ((grid_x - p1[0]) * line_vec[0] + (grid_y - p1[1]) * line_vec[1]) / line_len_sq
            t = np.clip(t, 0.0, 1.0)

            proj_x = p1[0] + t * line_vec[0]
            proj_y = p1[1] + t * line_vec[1]
            dist_sq = (grid_x - proj_x) ** 2 + (grid_y - proj_y) ** 2
            cur_dist = np.sqrt(dist_sq)

            if i == 0:
                dist_map = cur_dist
            else:
                dist_map = np.minimum(dist_map, cur_dist)

        # Normalize distance within threshold boundary region: 1 at polygon boundary, 0 at dilated edge
        norm_dist = np.clip(1.0 - (dist_map / max(distance, 1e-4)), 0.0, 1.0)

        # Merge with existing canvas
        roi = canvas[y_min:y_max, x_min:x_max]
        canvas[y_min:y_max, x_min:x_max] = np.maximum(roi, norm_dist)
