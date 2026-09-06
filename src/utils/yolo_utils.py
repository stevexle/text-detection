"""
YOLO Coordinate and Format Conversion Utilities.
"""

from typing import Dict, List, Tuple


CLASS_NAMES: List[str] = [
    "id",
    "name",
    "dob",
    "gender",
    "nationality",
    "origin_place",
    "current_place",
    "expire_date",
    "issue_date",
    "features",
    "mrz",
]

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def polygon_to_bbox(poly: List[List[float]], img_w: float, img_h: float) -> Tuple[float, float, float, float]:
    """Convert polygon coordinates to normalized YOLO bounding box [xc, yc, w, h]."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = max(0.0, min(xs)), min(img_w, max(xs))
    min_y, max_y = max(0.0, min(ys)), min(img_h, max(ys))

    w = max(0.0, max_x - min_x)
    h = max(0.0, max_y - min_y)
    xc = min_x + w / 2.0
    yc = min_y + h / 2.0

    return (
        max(0.0, min(1.0, xc / img_w)),
        max(0.0, min(1.0, yc / img_h)),
        max(0.0, min(1.0, w / img_w)),
        max(0.0, min(1.0, h / img_h)),
    )


def polygon_to_normalized_coords(poly: List[List[float]], img_w: float, img_h: float) -> List[float]:
    """Convert polygon coordinates to flat normalized coordinates [x1, y1, x2, y2, ...]."""
    coords = []
    for pt in poly:
        coords.extend([
            max(0.0, min(1.0, float(pt[0]) / img_w)),
            max(0.0, min(1.0, float(pt[1]) / img_h)),
        ])
    return coords
