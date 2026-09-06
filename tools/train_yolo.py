"""
1-Click Training Script for YOLO26 Text / Field Detection & Segmentation on CCCD.
Performs deterministic dynamic in-memory Train/Val splitting without duplicate disk copies.
"""

import argparse
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import yaml

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("TrainYOLODet")


def create_dynamic_yolo_staging(
    config_path: str = "configs/yolo/yolo_seg.yaml",
    val_ratio: float = 0.15,
    split_seed: int = 42,
    staging_dir: Optional[str] = None,
) -> Path:
    """
    Build a clean symlinked staging directory for YOLO detection/segmentation.
    Zero image duplication; deterministic split matching DBNet and YOLO-cls.
    """
    p_cfg = Path(config_path)
    cfg_data = {}
    if p_cfg.exists():
        with open(p_cfg, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    is_seg = "seg" in str(config_path).lower()
    task_name = "yolo_seg" if is_seg else "yolo_detect"

    if staging_dir is None:
        staging_dir = f"scratch/{task_name}"

    p_stage = Path(staging_dir)
    if p_stage.exists():
        shutil.rmtree(p_stage)

    # 1. Determine image & label source directories
    img_dir = Path("data/images")
    if not img_dir.exists() and Path("images").exists():
        img_dir = Path("images")

    lbl_dir = Path(f"data/labels/{task_name}/labels")
    if not lbl_dir.exists() and Path(f"labels/{task_name}/labels").exists():
        lbl_dir = Path(f"labels/{task_name}/labels")

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    # 2. Match image and label pairs
    label_files = sorted(list(lbl_dir.glob("*.txt")))
    pairs: List[Tuple[Path, Path]] = []

    for lbl_file in label_files:
        stem = lbl_file.stem
        # Check matching image extensions
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            img_file = img_dir / f"{stem}{ext}"
            if img_file.exists():
                pairs.append((img_file.resolve(), lbl_file.resolve()))
                break

    if not pairs:
        raise ValueError(f"No matching image-label pairs found between {img_dir} and {lbl_dir}")

    logger.info(f"Loaded {len(pairs)} image-label pairs for {task_name}")

    # 3. Deterministic split
    rng = random.Random(split_seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)

    val_size = max(1, int(len(pairs) * val_ratio))
    val_indices = set(indices[:val_size])

    # 4. Create symlinked directories
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (p_stage / sub).mkdir(parents=True, exist_ok=True)

    for i, (img_path, lbl_path) in enumerate(pairs):
        subset = "val" if i in val_indices else "train"

        sym_img = p_stage / "images" / subset / img_path.name
        if not sym_img.exists():
            os.symlink(img_path, sym_img)

        sym_lbl = p_stage / "labels" / subset / lbl_path.name
        if not sym_lbl.exists():
            os.symlink(lbl_path, sym_lbl)

    # 5. Create staged data.yaml
    stage_yaml_path = p_stage / "data.yaml"
    stage_yaml_content = {
        "path": str(p_stage.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": cfg_data.get("nc", 11),
        "names": cfg_data.get("names", {
            0: "id", 1: "name", 2: "dob", 3: "gender", 4: "nationality",
            5: "origin_place", 6: "current_place", 7: "expire_date",
            8: "issue_date", 9: "features", 10: "mrz"
        }),
    }

    with open(stage_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(stage_yaml_content, f, sort_keys=False)

    n_train = len(pairs) - val_size
    n_val = val_size
    logger.info(
        f"Created dynamic staging in '{staging_dir}': Train={n_train} pairs, Val={n_val} pairs (Seed={split_seed})"
    )
    return stage_yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26 Field Detection / Segmentation")
    parser.add_argument("--config", type=str, default="configs/yolo/yolo_seg.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--save-best-to", type=str, default="", help="Path to save best weights")
    args = parser.parse_args()

    task = "segment" if "seg" in str(args.config) else "detect"

    # 1. Create dynamic staging
    staged_data_yaml = create_dynamic_yolo_staging(
        config_path=args.config,
        val_ratio=0.15,
        split_seed=42,
    )

    # 2. Resolve weights
    weights = args.weights
    if not weights:
        if task == "segment":
            weights = "weights/yolo/yolo26n-seg.pt" if Path("weights/yolo/yolo26n-seg.pt").exists() else "yolo11n-seg.pt"
        else:
            weights = "weights/yolo/yolo26n.pt" if Path("weights/yolo/yolo26n.pt").exists() else "yolo11n.pt"

    if not Path(weights).exists() and Path(f"weights/yolo/{weights}").exists():
        weights = f"weights/yolo/{weights}"

    logger.info(f"Initializing YOLO model ({task}) with weights: {weights}")
    model = YOLO(weights, task=task)

    # 3. Train
    proj_dir = "work_dirs/yolo_seg" if task == "segment" else "work_dirs/yolo_detect"
    logger.info(f"Starting training on {staged_data_yaml} (Epochs={args.epochs}, Batch={args.batch_size}, ImgSz={args.imgsz})...")
    results = model.train(
        data=str(staged_data_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device if args.device else "",
        project=proj_dir,
        name="exp",
        exist_ok=True,
    )

    # 4. Save best weights
    default_save = "weights/yolo/yolo26_seg_best.pt" if task == "segment" else "weights/yolo/yolo26_det_best.pt"
    save_dest_str = args.save_best_to if args.save_best_to else default_save
    save_dest = Path(save_dest_str)
    save_dest.parent.mkdir(parents=True, exist_ok=True)

    possible_paths = []
    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        possible_paths.append(Path(model.trainer.best))
    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None):
        possible_paths.append(Path(model.trainer.save_dir) / "weights" / "best.pt")
    if hasattr(results, "save_dir"):
        possible_paths.append(Path(results.save_dir) / "weights" / "best.pt")

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
