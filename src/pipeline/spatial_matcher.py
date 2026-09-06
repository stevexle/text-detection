"""
Spatial Matching Module for Fusing DBNet Text Polygons with YOLO Semantic Fields.
Uses polygon intersection and overlap analysis to map semantic labels to sharp text contours.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from shapely.geometry import Polygon


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
    Fuse DBNet text detections with YOLO semantic field bounding boxes/masks.

    Args:
        text_detections: List of DBNet polygons [{"polygon": [[x,y],...], "confidence": float, ...}]
        field_detections: List of YOLO fields [{"label": "name", "polygon": [[x,y],...], "confidence": float, ...}]
        min_overlap_ratio: Minimum ratio of (Intersection / Text_Area) to assign a field label.

    Returns:
        List of fused text detections with semantic labels and combined metadata:
        [
            {
                "label": "id",
                "text_confidence": 0.85,
                "field_confidence": 0.98,
                "confidence": 0.85,
                "overlap_ratio": 0.94,
                "polygon": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            },
            ...
        ]
    """
    if not text_detections:
        return []

    if not field_detections:
        # Return DBNet results with fallback label
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

    # 1. Pre-construct Shapely polygons for YOLO fields
    field_polys = []
    for f in field_detections:
        f_poly = _to_valid_polygon(f.get("polygon", []))
        if f_poly is not None and f_poly.area > 0:
            field_polys.append((f, f_poly))

    # 2. Match each DBNet text polygon
    fused_results = []
    for t_det in text_detections:
        t_coords = t_det.get("polygon", [])
        t_conf = float(t_det.get("confidence", 0.0))
        t_poly = _to_valid_polygon(t_coords)

        best_field = None
        best_ratio = 0.0

        if t_poly is not None and t_poly.area > 0 and field_polys:
            t_area = t_poly.area
            for f_item, f_poly in field_polys:
                try:
                    inter_area = t_poly.intersection(f_poly).area
                    ratio = inter_area / t_area
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_field = f_item
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
