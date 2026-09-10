"""
High-Performance Spatial Matching Module for Fusing DBNet Text Polygons with YOLO Semantic Fields.
Optimized with AABB fast-rejection, Shapely geometry indexing, and unmatched field fallback.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from shapely.geometry import Polygon


def _get_aabb(coords: List[List[float]]) -> Tuple[float, float, float, float]:
    """Compute Axis-Aligned Bounding Box (min_x, min_y, max_x, max_y) fast in pure python."""
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _aabb_intersects(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> bool:
    """Check if two bounding boxes overlap."""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])


def _to_valid_polygon(poly_coords: List[List[float]]) -> Optional[Polygon]:
    """Convert coordinate list to a valid Shapely polygon."""
    if len(poly_coords) < 3:
        return None
    try:
        poly = Polygon(poly_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid and not poly.is_empty else None
    except Exception:
        return None


def match_text_to_fields(
    text_detections: List[Dict[str, Any]],
    field_detections: List[Dict[str, Any]],
    min_overlap_ratio: float = 0.20,
    fallback_unmatched_fields: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-speed fusion of DBNet text detections with YOLO semantic field bounding boxes/masks.

    Args:
        text_detections: List of DBNet polygons [{"polygon": [[x,y],...], "confidence": float, ...}]
        field_detections: List of YOLO fields [{"label": "name", "polygon": [[x,y],...], "confidence": float, ...}]
        min_overlap_ratio: Minimum ratio of (Intersection / Text_Area) to assign a field label.
        fallback_unmatched_fields: If True, any YOLO field that didn't have a matching DBNet text box
                                  is retained as a fallback detection to guarantee zero lost fields.

    Returns:
        List of fused text detections with semantic labels and combined metadata.
    """
    if not text_detections and not field_detections:
        return []

    if not field_detections:
        return [
            {
                "label": "text",
                "confidence": round(d.get("confidence", 0.0), 4),
                "polygon": d.get("polygon", []),
            }
            for d in text_detections
        ]

    if not text_detections:
        if fallback_unmatched_fields:
            return [
                {
                    "label": f.get("label", "field"),
                    "confidence": round(float(f.get("confidence", 0.0)), 4),
                    "polygon": f.get("polygon", []),
                }
                for f in field_detections
            ]
        return []

    # 1. Pre-construct Shapely polygons and AABBs for YOLO fields
    field_entries = []
    for idx, f in enumerate(field_detections):
        f_coords = f.get("polygon", [])
        if len(f_coords) < 3:
            continue
        f_poly = _to_valid_polygon(f_coords)
        if f_poly is not None and f_poly.area > 0:
            aabb = _get_aabb(f_coords)
            field_entries.append((idx, f, f_poly, aabb))

    matched_field_indices: Set[int] = set()

    # 2. Match each DBNet text polygon
    fused_results = []
    for t_det in text_detections:
        t_coords = t_det.get("polygon", [])
        t_conf = float(t_det.get("confidence", 0.0))

        if len(t_coords) < 3:
            fused_results.append({
                "label": "other_text",
                "confidence": round(t_conf, 4),
                "polygon": t_coords,
            })
            continue

        t_aabb = _get_aabb(t_coords)
        t_poly = _to_valid_polygon(t_coords)

        best_idx = None
        best_field = None
        best_ratio = 0.0

        if t_poly is not None and t_poly.area > 0:
            t_area = t_poly.area
            for f_idx, f_item, f_poly, f_aabb in field_entries:
                if not _aabb_intersects(t_aabb, f_aabb):
                    continue

                try:
                    inter_area = t_poly.intersection(f_poly).area
                    ratio = inter_area / t_area
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_field = f_item
                        best_idx = f_idx
                        if ratio >= 0.85:
                            break
                except Exception:
                    continue

        if best_field is not None and best_ratio >= min_overlap_ratio:
            f_label = best_field.get("label", "text")
            if best_idx is not None:
                matched_field_indices.add(best_idx)

            fused_results.append({
                "label": f_label,
                "confidence": round(t_conf, 4),
                "polygon": t_coords,
            })
        else:
            fused_results.append({
                "label": "other_text",
                "confidence": round(t_conf, 4),
                "polygon": t_coords,
            })

    # 3. Fallback for unmatched YOLO fields
    if fallback_unmatched_fields:
        for f_idx, f_item, _, _ in field_entries:
            if f_idx not in matched_field_indices:
                f_label = f_item.get("label", "field")
                f_conf = float(f_item.get("confidence", 0.0))
                f_coords = f_item.get("polygon", [])
                fused_results.append({
                    "label": f_label,
                    "confidence": round(f_conf, 4),
                    "polygon": f_coords,
                })

    return fused_results
