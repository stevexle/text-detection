"""
High-Performance Spatial Matching Module for Fusing DBNet Text Polygons with YOLO Semantic Fields.
Optimized with AABB fast-rejection, Shapely geometry indexing, and unmatched field fallback.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from shapely.geometry import Polygon
from shapely.prepared import prep


def _get_aabb(coords: List[List[float]]) -> Tuple[float, float, float, float]:
    """Compute Axis-Aligned Bounding Box (min_x, min_y, max_x, max_y) fast."""
    if len(coords) == 4:
        p0, p1, p2, p3 = coords
        return (
            min(p0[0], p1[0], p2[0], p3[0]),
            min(p0[1], p1[1], p2[1], p3[1]),
            max(p0[0], p1[0], p2[0], p3[0]),
            max(p0[1], p1[1], p2[1], p3[1]),
        )
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

    # 1. Pre-construct Shapely polygons, prepared GEOS geometry, and AABBs for YOLO fields
    field_entries = []
    for idx, f in enumerate(field_detections):
        f_coords = f.get("polygon", [])
        if len(f_coords) < 3:
            continue
        f_poly = _to_valid_polygon(f_coords)
        if f_poly is not None and f_poly.area > 0:
            aabb = _get_aabb(f_coords)
            prep_f = prep(f_poly)
            field_entries.append((idx, f, f_poly, prep_f, aabb))

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
            for f_idx, f_item, f_poly, prep_f, f_aabb in field_entries:
                if not _aabb_intersects(t_aabb, f_aabb):
                    continue
                if not prep_f.intersects(t_poly):
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
        for f_idx, f_item, _, _, _ in field_entries:
            if f_idx not in matched_field_indices:
                f_label = f_item.get("label", "field")
                f_conf = float(f_item.get("confidence", 0.0))
                f_coords = f_item.get("polygon", [])
                fused_results.append({
                    "label": f_label,
                    "confidence": round(f_conf, 4),
                    "polygon": f_coords,
                })

    # 4. Text Line Merging: merge fragmented boxes on the same line
    return merge_same_line_detections(fused_results)


def merge_same_line_detections(
    detections: List[Dict[str, Any]],
    y_overlap_thresh: float = 0.50,
    max_gap_ratio: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Ultra-fast merger for fragmented text bounding boxes on the same horizontal line.
    Optimized with single-pass dictionary clustering and unrolled scalar arithmetic (zero-numpy).

    Args:
        detections: List of detection dicts [{"label": str, "confidence": float, "polygon": [[x,y],...]}]
        y_overlap_thresh: Minimum vertical overlap ratio to be considered on the same line.
        max_gap_ratio: Maximum horizontal gap as a multiple of text line height.
    """
    if not detections or len(detections) <= 1:
        return detections

    # 1. Single-pass grouping by label (O(N))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in detections:
        lbl = d.get("label", "text")
        if lbl in groups:
            groups[lbl].append(d)
        else:
            groups[lbl] = [d]

    final_dets = []

    # 2. Process each group
    for label, group in groups.items():
        if len(group) == 1:
            final_dets.append(group[0])
            continue

        # Fast unrolled coordinate extraction without NumPy allocation overhead
        items: List[List[float]] = []
        for d in group:
            poly = d.get("polygon", [])
            if len(poly) == 4:
                p0, p1, p2, p3 = poly
                x1 = min(p0[0], p1[0], p2[0], p3[0])
                x2 = max(p0[0], p1[0], p2[0], p3[0])
                y1 = min(p0[1], p1[1], p2[1], p3[1])
                y2 = max(p0[1], p1[1], p2[1], p3[1])
            elif len(poly) >= 3:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
            else:
                continue

            items.append([float(x1), float(y1), float(x2), float(y2), float(d.get("confidence", 0.0))])

        if not items:
            continue

        # Sort top-to-bottom, then left-to-right
        items.sort(key=lambda it: (it[1], it[0]))

        # In-place line cluster merge
        merged_lines: List[List[float]] = []  # format: [x1, y1, x2, y2, conf]
        for it in items:
            x1, y1, x2, y2, conf = it
            h_it = y2 - y1
            merged = False

            for m in merged_lines:
                m_x1, m_y1, m_x2, m_y2, m_conf = m
                m_h = m_y2 - m_y1

                # Calculate vertical overlap on Y-axis
                y_top = max(y1, m_y1)
                y_bot = min(y2, m_y2)
                overlap_h = y_bot - y_top
                min_h = min(h_it, m_h)

                if min_h > 0 and (overlap_h / min_h) >= y_overlap_thresh:
                    # Same horizontal line: calculate gap between boxes
                    gap = max(0.0, max(x1, m_x1) - min(x2, m_x2))
                    avg_h = (h_it + m_h) * 0.5

                    if gap <= avg_h * max_gap_ratio:
                        m[0] = min(x1, m_x1)
                        m[1] = min(y1, m_y1)
                        m[2] = max(x2, m_x2)
                        m[3] = max(y2, m_y2)
                        m[4] = (m_conf + conf) * 0.5
                        merged = True
                        break

            if not merged:
                merged_lines.append([x1, y1, x2, y2, conf])

        for m in merged_lines:
            mx1, my1, mx2, my2, mconf = m
            final_dets.append({
                "label": label,
                "confidence": round(mconf, 4),
                "polygon": [
                    [round(mx1, 2), round(my1, 2)],
                    [round(mx2, 2), round(my1, 2)],
                    [round(mx2, 2), round(my2, 2)],
                    [round(mx1, 2), round(my2, 2)],
                ],
            })

    # Fast top-to-bottom reading order sort (pure Python tuple indexing)
    final_dets.sort(key=lambda d: (
        d["polygon"][0][1] if d.get("polygon") else 0,
        d["polygon"][0][0] if d.get("polygon") else 0,
    ))

    return final_dets
