"""
Prediction / Inference script for YOLO26 Field Detection & Segmentation on CCCD.
Accepts an image path or a directory and returns standard structured polygon/box outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import cv2
import numpy as np

from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logger import get_logger

logger = get_logger("PredictYOLO")


def predict_yolo(
    weights: str,
    source: Union[str, Path],
    task: str = "segment",
    min_conf: float = 0.5,
    imgsz: int = 640,
    device: str = "",
    save_json: str = None,
    save_visualizations: str = None,
) -> List[Dict[str, Any]]:
    """
    Run YOLO segmentation/detection inference on image(s).
    """
    wrapper = YOLOWrapper(model_path=weights, task=task)
    src_path = Path(source)

    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    logger.info(f"Loaded YOLO ({task}) from '{weights}'. Running inference on {len(image_paths)} image(s)...")

    all_results = []
    vis_dir = Path(save_visualizations) if save_visualizations else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for img_p in image_paths:
        dets = wrapper.detect(image=str(img_p), min_conf=min_conf, imgsz=imgsz, device=device if device else None)
        item = {
            "image": img_p.name,
            "detections": dets,
            "total_fields": len(dets),
        }
        all_results.append(item)
        logger.info(f"[{img_p.name}] -> Found {len(dets)} fields: {[d['label'] for d in dets]}")

        # Visualization
        if vis_dir:
            img = cv2.imread(str(img_p))
            if img is not None:
                for d in dets:
                    pts = np.array(d["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    label_text = f"{d['label']} {d['confidence']:.2f}"
                    cv2.putText(
                        img,
                        label_text,
                        (int(pts[0][0][0]), max(15, int(pts[0][0][1]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                out_path = vis_dir / f"vis_{img_p.name}"
                cv2.imwrite(str(out_path), img)

    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to: {save_json}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Inference for YOLO26 Field Detection / Segmentation")
    parser.add_argument("--weights", type=str, default="weights/yolo/yolo26_seg_best.pt", help="Path to weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--task", type=str, default="segment", choices=["segment", "detect"], help="Task type")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    parser.add_argument("--save-json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--save-vis", type=str, default=None, help="Output visualization directory")
    args = parser.parse_args()

    predict_yolo(
        weights=args.weights,
        source=args.source,
        task=args.task,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
        save_visualizations=args.save_vis,
    )


if __name__ == "__main__":
    main()
