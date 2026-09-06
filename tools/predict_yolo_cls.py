"""
Prediction / Inference script for YOLO26 Document Classifier.
Accepts an image path or a directory of images and returns clean structured predictions.
"""

import argparse
import json
from pathlib import Path
from typing import List, Union

from src.models.wrappers.yolo_classifier import YOLOClassifier
from src.utils.logger import get_logger

logger = get_logger("PredictYOLOCls")


def predict(
    model_path: str,
    source: Union[str, Path],
    min_conf: float = 0.5,
    imgsz: int = 224,
    device: str = "",
    save_json: str = None,
):
    """
    Run classification inference on image(s).
    """
    classifier = YOLOClassifier(model_path=model_path, device=device if device else None)
    src_path = Path(source)

    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    logger.info(f"Loaded classifier from '{model_path}'. Running inference on {len(image_paths)} image(s)...")

    batch_preds = classifier.classify_batch(image_paths, batch_size=32, min_conf=min_conf, imgsz=imgsz)
    results = []
    for img_p, res in zip(image_paths, batch_preds):
        output_item = {
            "image": img_p.name,
            **res,
        }
        results.append(output_item)
        logger.info(
            f"[{img_p.name}] -> {res['card_type']} (conf: {res['confidence']:.4f})"
        )

    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to: {save_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference for YOLO26 Document Classifier")
    parser.add_argument("--weights", type=str, default="weights/yolo/yolo26_cls_best.pt", help="Path to classifier weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Minimum confidence threshold")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    parser.add_argument("--save-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    predict(
        model_path=args.weights,
        source=args.source,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
