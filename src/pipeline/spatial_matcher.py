"""
High-Performance Spatial Matching Module for Fusing DBNet Text Polygons with YOLO Semantic Fields.
Optimized with AABB (Axis-Aligned Bounding Box) fast-rejection and Shapely geometry indexing.
"""

from typing import Any, Dict, List, Optional, Tuple
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
) -> List[Dict[str, Any]]:
    """
    High-speed fusion of DBNet text detections with YOLO semantic field bounding boxes/masks.
    Uses AABB fast-rejection before full polygon intersection.

    Args:
        text_detections: List of DBNet polygons [{"polygon": [[x,y],...], "confidence": float, ...}]
        field_detections: List of YOLO fields [{"label": "name", "polygon": [[x,y],...], "confidence": float, ...}]
        min_overlap_ratio: Minimum ratio of (Intersection / Text_Area) to assign a field label.

    Returns:
        List of fused text detections with semantic labels and combined metadata.
    """
    if not text_detections:
        return []

    if not field_detections:
        return [
            {
                "label": "text",
                "text_confidence": round(d.get("confidence", 0.0), 4),
                "field_confidence": 0.0,
                "confidence": round(d.get("confidence", 0.0), 4),
                "overlap_ratio": 0.0,
                "polygon": d.get("polygon", []),
            }
            for d in text_detections
        ]

    # 1. Pre-construct Shapely polygons and AABBs for YOLO fields
    field_entries = []
    for f in field_detections:
        f_coords = f.get("polygon", [])
        if len(f_coords) < 3:
            continue
        f_poly = _to_valid_polygon(f_coords)
        if f_poly is not None and f_poly.area > 0:
            aabb = _get_aabb(f_coords)
            field_entries.append((f, f_poly, aabb))

    if not field_entries:
        return [
            {
                "label": "other_text",
                "text_confidence": round(d.get("confidence", 0.0), 4),
                "field_confidence": 0.0,
                "confidence": round(d.get("confidence", 0.0), 4),
                "overlap_ratio": 0.0,
                "polygon": d.get("polygon", []),
            }
            for d in text_detections
        ]

    # 2. Fast match each DBNet text polygon
    fused_results = []
    for t_det in text_detections:
        t_coords = t_det.get("polygon", [])
        t_conf = float(t_det.get("confidence", 0.0))

        if len(t_coords) < 3:
            fused_results.append({
                "label": "other_text",
                "text_confidence": round(t_conf, 4),
                "field_confidence": 0.0,
                "confidence": round(t_conf, 4),
                "overlap_ratio": 0.0,
                "polygon": t_coords,
            })
            continue

        t_aabb = _get_aabb(t_coords)
        t_poly = _to_valid_polygon(t_coords)

        best_field = None
        best_ratio = 0.0

        if t_poly is not None and t_poly.area > 0:
            t_area = t_poly.area
            for f_item, f_poly, f_aabb in field_entries:
                # Fast AABB reject
                if not _aabb_intersects(t_aabb, f_aabb):
                    continue

                try:
                    inter_area = t_poly.intersection(f_poly).area
                    ratio = inter_area / t_area
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_field = f_item
                        if ratio >= 0.85:
                            # Early exit on strong overlap
                            break
                except Exception:
                    continue

        if best_field is not None and best_ratio >= min_overlap_ratio:
            f_label = best_field.get("label", "text")
            f_conf = float(best_field.get("confidence", 0.0))
            fused_results.append({
                "label": f_label,
                "text_confidence": round(t_conf, 4),
                "field_confidence": round(f_conf, 4),
                "confidence": round(t_conf, 4),
                "overlap_ratio": round(best_ratio, 4),
                "polygon": t_coords,
            })
        else:
            fused_results.append({
                "label": "other_text",
                "text_confidence": round(t_conf, 4),
                "field_confidence": 0.0,
                "confidence": round(t_conf, 4),
                "overlap_ratio": round(best_ratio, 4),
                "polygon": t_coords,
            })

    return fused_results
