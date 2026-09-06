"""
Inference & Visualization CLI for DBNet Text Detection.
Takes an image or folder, detects text polygons, draws overlays, and saves predictions.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

from tqdm import tqdm

from src.models.builder import build_model
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("DemoDBNet")


def draw_polygons(
    image: np.ndarray,
    polygons: list,
    scores: list = None,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw polygon overlays with confidence labels on image."""
    vis_img = image.copy()
    overlay = image.copy()

    for i, poly in enumerate(polygons):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=thickness)

        if scores is not None and i < len(scores):
            score_text = f"{scores[i]:.2f}"
            org = (int(pts[0][0][0]), max(15, int(pts[0][0][1]) - 5))
            cv2.putText(
                vis_img,
                score_text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # 30% transparency overlay
    cv2.addWeighted(overlay, 0.25, vis_img, 0.75, 0, vis_img)
    return vis_img


def predict_demo(
    source: str,
    config_path: str = "configs/dbnet/dbnet.yaml",
    weights_path: str = "weights/dbnet/dbnet_cccd_best.pth",
    output_dir: str = "runs/predict",
    device: str = "",
    save_json: str = None,
    box_thresh: float = None,
    unclip_ratio: float = None,
):
    """Run DBNet text detection on source image(s) and save visualizations."""
    cfg = Config.fromfile(config_path)

    # 1. Device setup
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # 2. Build Model & Load Weights
    model = build_model(cfg.model).to(dev)
    w_path = Path(weights_path)
    if not w_path.exists() and Path(f"work_dirs/dbnet/{weights_path}").exists():
        w_path = Path(f"work_dirs/dbnet/{weights_path}")

    if w_path.exists():
        try:
            ckpt = torch.load(w_path, map_location=dev, weights_only=False)
        except Exception:
            ckpt = torch.load(w_path, map_location=dev)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
    else:
        logger.warning(f"Weights file not found at '{weights_path}'. Running with randomly initialized model.")

    model.eval()

    # 3. Postprocessor
    post_cfg = dict(cfg.postprocess)
    if box_thresh is not None:
        post_cfg["box_thresh"] = box_thresh
    if unclip_ratio is not None:
        post_cfg["unclip_ratio"] = unclip_ratio
    postprocessor = build_postprocessor(post_cfg)

    # 4. Resolve source files
    src_p = Path(source)
    if src_p.is_file():
        image_paths = [src_p]
    elif src_p.is_dir():
        image_paths = sorted(
            [p for p in src_p.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    p_out = Path(output_dir)
    p_out.mkdir(parents=True, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    target_size = tuple(cfg.data.target_size)

    all_results = []
    logger.info(f"Running text detection demo on {len(image_paths)} image(s)...")

    with torch.no_grad():
        pbar = tqdm(image_paths, desc="[Detecting Text]", unit="img")
        for img_path in pbar:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            orig_h, orig_w = img_bgr.shape[:2]

            # Preprocessing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, target_size)
            norm_img = ((img_resized / 255.0) - mean) / std
            tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(dev)

            # Forward pass
            preds = model(tensor_img)
            prob_map = preds["prob_map"][0]

            # Post-processing (Unified Detection Format)
            detections = postprocessor(prob_map, orig_shape=(orig_h, orig_w))
            polygons = [d["polygon"] for d in detections]
            scores = [d["confidence"] for d in detections]

            # Draw & Save visualization
            vis_img = draw_polygons(img_bgr, polygons, scores)
            save_dest = p_out / f"pred_{img_path.name}"
            cv2.imwrite(str(save_dest), vis_img)

            # Prepare structured result in unified format
            res_item = {
                "image": img_path.name,
                "detections": detections,
                "saved_to": str(save_dest),
            }
            all_results.append(res_item)
            logger.info(f"[{img_path.name}] -> Detected {len(detections)} text polygons | Saved: {save_dest}")

    if save_json:
        json_path = Path(save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved detection results JSON to: {save_json}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="DBNet Text Detection Demo / Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or directory of images")
    parser.add_argument("--config", type=str, default="configs/dbnet/dbnet.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="weights/dbnet/dbnet_cccd_best.pth", help="Path to model weights")
    parser.add_argument("--output-dir", type=str, default="runs/predict", help="Directory to save visualized predictions")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--box-thresh", type=float, default=None, help="Minimum polygon confidence threshold")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="Vatti unclip expansion ratio")
    parser.add_argument("--save-json", type=str, default=None, help="Optional output JSON path for polygons")
    args = parser.parse_args()

    predict_demo(
        source=args.source,
        config_path=args.config,
        weights_path=args.weights,
        output_dir=args.output_dir,
        device=args.device,
        save_json=args.save_json,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )


if __name__ == "__main__":
    main()
