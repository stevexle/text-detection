"""
1-Click Training Script for YOLO26 Text / Field Detection & Segmentation on CCCD.
"""

import argparse
from pathlib import Path
import shutil
from ultralytics import YOLO

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("TrainYOLODet")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 Field Detection / Segmentation")
    parser.add_argument("--config", type=str, default="configs/yolo/yolo_seg.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="weights/yolo/yolo26n-seg.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--save-best-to", type=str, default="weights/yolo/yolo26_seg_best.pt", help="Path to save best weights")
    args = parser.parse_args()

    # 1. Resolve weights
    weights = args.weights
    if not Path(weights).exists() and Path(f"weights/yolo/{weights}").exists():
        weights = f"weights/yolo/{weights}"
    elif not Path(weights).exists() and "seg" in args.config:
        weights = "yolo11n-seg.pt"
    elif not Path(weights).exists():
        weights = "yolo11n.pt"

    logger.info(f"Initializing YOLO model with weights: {weights}")
    task = "segment" if "seg" in str(args.config) else "detect"
    model = YOLO(weights, task=task)

    # 2. Train
    logger.info(f"Starting training on {args.config} (Epochs={args.epochs}, Batch={args.batch_size}, ImgSz={args.imgsz})...")
    model.train(
        data=args.config,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device if args.device else "",
        project="work_dirs/yolo_seg" if task == "segment" else "work_dirs/yolo_detect",
        name="exp",
        exist_ok=True,
    )

    # 3. Save best weights
    save_dest = Path(args.save_best_to)
    save_dest.parent.mkdir(parents=True, exist_ok=True)
    proj_dir = "work_dirs/yolo_seg" if task == "segment" else "work_dirs/yolo_detect"

    possible_paths = []
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        possible_paths.append(Path(model.trainer.best))
    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
        possible_paths.append(Path(model.trainer.save_dir) / "weights" / "best.pt")

    possible_paths.extend([
        Path(f"runs/{task}") / proj_dir / "exp" / "weights" / "best.pt",
        Path(proj_dir) / "exp" / "weights" / "best.pt",
        Path(f"runs/{task}") / "exp" / "weights" / "best.pt",
    ])

    saved = False
    for p in possible_paths:
        if p.exists():
            shutil.copy(p, save_dest)
            logger.info(f"Saved best YOLO weights from {p} to: {save_dest}")
            saved = True
            break

    if not saved:
        logger.warning(f"best.pt not found automatically. Check {possible_paths[0] if possible_paths else 'runs/'}")


if __name__ == "__main__":
    main()
