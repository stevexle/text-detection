"""
Production ONNX Exporter for CCCD eKYC Pipeline.
Exports DBNet, YOLO-seg, and YOLO-cls to optimized ONNX format with dynamic shapes,
operator simplification (onnxsim), and onnxruntime numerical verification.
"""

import argparse
from pathlib import Path
import shutil
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

from src.models.builder import build_model
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("ExportONNX")


class DBNetExportWrapper(nn.Module):
    """
    Clean wrapper around DBNet detector for ONNX export.
    Directly returns single `prob_map` tensor of shape (B, 1, H, W)
    for seamless compatibility with TensorRT, Triton, and C++ runtimes.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["prob_map"]
        return out


def export_dbnet_onnx(
    config_path: str = "configs/dbnet/dbnet.yaml",
    weights_path: Optional[str] = "weights/dbnet/dbnet_cccd_best.pth",
    output_path: str = "weights/onnx/dbnet.onnx",
    imgsz: Tuple[int, int] = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    verify: bool = True,
) -> Path:
    """
    Export DBNet PyTorch model to ONNX with dynamic spatial and batch axes.
    """
    logger.info("=" * 60)
    logger.info(f"Starting DBNet ONNX Export (opset={opset})")
    logger.info("=" * 60)

    cfg = Config.fromfile(config_path)
    model = build_model(cfg.model)
    model.eval()

    # Load weights if available
    w_p = Path(weights_path) if weights_path else None
    if w_p and w_p.exists():
        logger.info(f"Loading checkpoint weights from: {w_p}")
        checkpoint = torch.load(w_p, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
    else:
        logger.warning(
            f"Checkpoint '{weights_path}' not found. Exporting model architecture with initialized weights."
        )

    wrapper = DBNetExportWrapper(model)
    wrapper.eval()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    h, w = imgsz
    dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)

    logger.info(f"Exporting PyTorch model to: {out_p}")
    t0 = time.perf_counter()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_p),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["prob_map"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "prob_map": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    t_export = time.perf_counter() - t0
    logger.info(f"Base ONNX export completed in {t_export:.2f}s | Size: {out_p.stat().st_size / (1024 * 1024):.2f} MB")

    # Simplify with onnxslim (recommended) or onnxsim
    if simplify:
        simplified = False
        try:
            import onnxslim

            logger.info("Running onnxslim to fuse operators and fold constants...")
            model_slimmed = onnxslim.slim(str(out_p))
            onnx.save(model_slimmed, str(out_p))
            logger.info(
                f"ONNX graph successfully simplified via onnxslim | Optimized Size: {out_p.stat().st_size / (1024 * 1024):.2f} MB"
            )
            simplified = True
        except Exception as e:
            logger.warning(f"onnxslim not available or failed ({e}), attempting onnxsim fallback...")

        if not simplified:
            try:
                import onnxsim

                logger.info("Running onnxsim to fuse operators and fold constants...")
                model_proto = onnx.load(str(out_p))
                model_simplified, check = onnxsim.simplify(model_proto)
                if check:
                    onnx.save(model_simplified, str(out_p))
                    logger.info(
                        f"ONNX graph successfully simplified | Optimized Size: {out_p.stat().st_size / (1024 * 1024):.2f} MB"
                    )
            except Exception as e:
                logger.warning(f"Failed to run onnxsim: {e}")

    # Check ONNX graph validity
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph structural validation passed.")

    # Numerical verification with onnxruntime
    if verify:
        logger.info("Validating ONNX Runtime execution with dynamic shapes...")
        session = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])

        # Test Shape 1: 1x3x640x640
        x1 = np.random.randn(1, 3, 640, 640).astype(np.float32)
        ort_out1 = session.run(["prob_map"], {"input": x1})[0]

        with torch.no_grad():
            torch_out1 = wrapper(torch.from_numpy(x1)).numpy()

        max_diff1 = np.max(np.abs(ort_out1 - torch_out1))
        logger.info(f"Test 1 [1, 3, 640, 640] -> Output: {ort_out1.shape} | Max Diff: {max_diff1:.2e}")
        assert max_diff1 < 1e-4, f"Numerical discrepancy too large in Test 1: {max_diff1}"

        # Test Shape 2: Dynamic Batch & Resolution 2x3x800x800
        x2 = np.random.randn(2, 3, 800, 800).astype(np.float32)
        ort_out2 = session.run(["prob_map"], {"input": x2})[0]

        with torch.no_grad():
            torch_out2 = wrapper(torch.from_numpy(x2)).numpy()

        max_diff2 = np.max(np.abs(ort_out2 - torch_out2))
        logger.info(f"Test 2 [2, 3, 800, 800] -> Output: {ort_out2.shape} | Max Diff: {max_diff2:.2e}")
        assert max_diff2 < 1e-4, f"Numerical discrepancy too large in Test 2: {max_diff2}"

        logger.info("Dynamic axes validation passed (tested sizes 640x640 and 800x800 across batches).")

    return out_p


def export_yolo_onnx(
    weights_path: str,
    output_path: str,
    imgsz: Union[int, Tuple[int, int]] = 640,
    opset: int = 17,
    simplify: bool = True,
) -> Path:
    """
    Export Ultralytics YOLO model (cls or seg) to ONNX with dynamic shapes.
    """
    from ultralytics import YOLO

    w_p = Path(weights_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not w_p.exists():
        logger.error(f"Weights file not found at: {w_p}")
        raise FileNotFoundError(f"YOLO weights not found: {w_p}")

    logger.info("=" * 60)
    logger.info(f"Starting YOLO ONNX Export: {w_p.name} (imgsz={imgsz}, opset={opset})")
    logger.info("=" * 60)

    model = YOLO(str(w_p))
    exported_file = model.export(
        format="onnx",
        dynamic=True,
        simplify=simplify,
        opset=opset,
        imgsz=imgsz,
    )

    exp_p = Path(exported_file)
    if exp_p != out_p:
        shutil.move(str(exp_p), str(out_p))

    logger.info(f"YOLO ONNX exported successfully to: {out_p} | Size: {out_p.stat().st_size / (1024 * 1024):.2f} MB")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Export CCCD Pipeline Models to Optimized ONNX Format")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "dbnet", "yolo-seg", "yolo-cls"],
        help="Target model to export",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/onnx",
        help="Directory to store exported ONNX files",
    )
    parser.add_argument(
        "--dbnet-config",
        type=str,
        default="configs/dbnet/dbnet.yaml",
        help="Path to DBNet configuration YAML",
    )
    parser.add_argument(
        "--dbnet-weights",
        type=str,
        default="weights/dbnet/dbnet_cccd_best.pth",
        help="Path to DBNet PyTorch weights (.pth)",
    )
    parser.add_argument(
        "--yolo-seg-weights",
        type=str,
        default="weights/yolo/yolo26_seg_best.pt",
        help="Path to YOLO-seg weights (.pt)",
    )
    parser.add_argument(
        "--yolo-cls-weights",
        type=str,
        default="weights/yolo/yolo26_cls_best.pt",
        help="Path to YOLO-cls weights (.pt)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable onnxsim graph simplification",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable numerical verification against ONNX Runtime",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    simplify = not args.no_simplify
    verify = not args.no_verify

    results: Dict[str, Path] = {}

    if args.model in ["all", "dbnet"]:
        dbnet_out = out_dir / "dbnet.onnx"
        results["dbnet"] = export_dbnet_onnx(
            config_path=args.dbnet_config,
            weights_path=args.dbnet_weights,
            output_path=str(dbnet_out),
            opset=args.opset,
            simplify=simplify,
            verify=verify,
        )

    if args.model in ["all", "yolo-seg"]:
        yolo_seg_out = out_dir / "yolo26_seg.onnx"
        if Path(args.yolo_seg_weights).exists():
            results["yolo_seg"] = export_yolo_onnx(
                weights_path=args.yolo_seg_weights,
                output_path=str(yolo_seg_out),
                imgsz=640,
                opset=args.opset,
                simplify=simplify,
            )
        else:
            logger.warning(
                f"YOLO-seg checkpoint not found at '{args.yolo_seg_weights}'. Skipping. (You can run once weights are synced)."
            )

    if args.model in ["all", "yolo-cls"]:
        yolo_cls_out = out_dir / "yolo26_cls.onnx"
        if Path(args.yolo_cls_weights).exists():
            results["yolo_cls"] = export_yolo_onnx(
                weights_path=args.yolo_cls_weights,
                output_path=str(yolo_cls_out),
                imgsz=224,
                opset=args.opset,
                simplify=simplify,
            )
        else:
            logger.warning(
                f"YOLO-cls checkpoint not found at '{args.yolo_cls_weights}'. Skipping. (You can run once weights are synced)."
            )

    logger.info("=" * 60)
    logger.info("ONNX Export Process Finished Summary:")
    for name, p in results.items():
        logger.info(f"  * {name.upper()}: {p} ({p.stat().st_size / (1024 * 1024):.2f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
