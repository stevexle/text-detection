"""
1-Click Training Script for YOLO26 Document Classification (CCCD 2021 vs 2024 Front/Back).
Performs deterministic dynamic in-memory Train/Val splitting without duplicate disk copies.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Tuple
from ultralytics import YOLO

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("TrainYOLOCls")


def create_dynamic_cls_staging(
    anno_file: str = "data/labels/dbnet/dataset.jsonl",
    val_ratio: float = 0.15,
    split_seed: int = 42,
    staging_dir: str = "scratch/yolo_cls",
) -> Path:
    """
    Read dataset.jsonl and build a clean symlinked staging directory for YOLO classification.
    Zero image duplication; deterministic split matching the detection dataset.
    """
    p_anno = Path(anno_file)
    if not p_anno.exists():
        raise FileNotFoundError(f"Annotation file not found: {p_anno}")

    p_stage = Path(staging_dir)
    if p_stage.exists():
        shutil.rmtree(p_stage)

    # 1. Load samples and map to 4 classes
    samples = []
    with open(p_anno, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_path = item.get("img_path") or item.get("image_path")
            std = item.get("card_standard", "CCCD_2021_CHIP")
            side = item.get("card_side", "front")

            # Resolve absolute image path
            img_path = Path("data") / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
            if not img_path.exists():
                img_path = Path(raw_path)

            # Class mapping
            if "2024" in std:
                cls_name = f"front_2024" if side == "front" else f"back_2024"
            else:
                cls_name = f"front_2021" if side == "front" else f"back_2021"

            samples.append((str(img_path.resolve()), cls_name))

    logger.info(f"Loaded {len(samples)} samples from {anno_file}")

    # 2. Deterministic split
    rng = random.Random(split_seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    val_size = max(1, int(len(samples) * val_ratio))
    val_indices = set(indices[:val_size])

    # 3. Create symlinks in staging dir
    for i, (img_path, cls_name) in enumerate(samples):
        subset = "val" if i in val_indices else "train"
        target_dir = p_stage / subset / cls_name
        target_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = target_dir / Path(img_path).name
        if not symlink_path.exists():
            os.symlink(img_path, symlink_path)

    n_train = len(samples) - val_size
    n_val = val_size
    logger.info(f"Created dynamic staging in '{staging_dir}': Train={n_train} images, Val={n_val} images (Seed={split_seed})")
    return p_stage


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 Document Classifier")
    parser.add_argument("--config", type=str, default="configs/yolo/yolo_cls.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override input image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device to use (mps, cuda, cpu)")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # 1. Create dynamic staging
    staging_dir = create_dynamic_cls_staging(
        anno_file=cfg.data.anno_file,
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
    )

    # 2. Resolve model weights
    weights_path = cfg.model.model_path
    if not Path(weights_path).exists() and Path(f"weights/yolo/{weights_path}").exists():
        weights_path = f"weights/yolo/{weights_path}"
    elif not Path(weights_path).exists():
        weights_path = "yolo11n-cls.pt"

    logger.info(f"Initializing YOLO Classifier with base weights: {weights_path}")
    model = YOLO(weights_path, task="classify")

    # 3. Train
    epochs = args.epochs or cfg.train.epochs
    batch_size = args.batch_size or cfg.data.batch_size
    imgsz = args.imgsz or cfg.data.imgsz
    device = args.device if args.device is not None else cfg.train.device

    logger.info(f"Starting training: Epochs={epochs}, BatchSize={batch_size}, ImgSz={imgsz}, Device={device or 'auto'}")
    results = model.train(
        data=str(staging_dir.resolve()),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer=cfg.train.optimizer,
        lr0=cfg.train.lr0,
        device=device,
        project=cfg.train.project,
        name=cfg.train.name,
        exist_ok=True,
    )

    # 4. Save best model to destination
    save_dest = Path(cfg.train.save_best_to)
    save_dest.parent.mkdir(parents=True, exist_ok=True)

    possible_paths = []
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        possible_paths.append(Path(model.trainer.best))
    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
        possible_paths.append(Path(model.trainer.save_dir) / "weights" / "best.pt")
    if hasattr(results, "save_dir"):
        possible_paths.append(Path(results.save_dir) / "weights" / "best.pt")

    possible_paths.extend([
        Path("runs/classify") / cfg.train.project / cfg.train.name / "weights" / "best.pt",
        Path(cfg.train.project) / cfg.train.name / "weights" / "best.pt",
        Path("runs/classify") / cfg.train.name / "weights" / "best.pt",
    ])

    saved = False
    for p in possible_paths:
        if p.exists():
            shutil.copy(p, save_dest)
            logger.info(f"Successfully saved best classification weights from {p} to: {save_dest}")
            saved = True
            break

    if not saved:
        logger.warning(f"best.pt not found automatically. Please check {possible_paths[0] if possible_paths else 'runs/'}")


if __name__ == "__main__":
    main()
