"""
High-Performance ONNX Exporter for CCCD eKYC Pipeline.
Exports DBNet, YOLO-seg, and YOLO-cls into optimized, self-contained ONNX models
featuring dynamic batch/spatial shapes, graph simplification (onnxslim/onnxsim),
embedded metadata, and multi-shape numerical verification.
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

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


def simplify_onnx_graph(onnx_path: Path) -> Path:
    """
    Simplify and optimize ONNX computational graph.
    Prioritizes modern onnxslim (faster, no nanobind issues) with fallback to onnxsim.
    """
    simplified = False

    # 1. Try onnxslim (modern industry standard)
    try:
        import onnxslim

        logger.info("Simplifying ONNX graph using onnxslim (constant folding & operator fusion)...")
        slimmed_model = onnxslim.slim(str(onnx_path))
        onnx.save(slimmed_model, str(onnx_path))
        simplified = True
        logger.info("Graph simplification with onnxslim completed successfully.")
    except Exception as e:
        logger.warning(f"onnxslim simplification bypassed: {e}")

    # 2. Fallback to onnxsim if onnxslim was unavailable
    if not simplified:
        try:
            import onnxsim

            logger.info("Falling back to onnxsim graph simplification...")
            model_proto = onnx.load(str(onnx_path))
            model_simp, check = onnxsim.simplify(model_proto)
            if check:
                onnx.save(model_simp, str(onnx_path))
                logger.info("Graph simplification with onnxsim completed successfully.")
        except Exception as e:
            logger.warning(f"onnxsim fallback bypassed: {e}")

    return onnx_path


def add_onnx_metadata(
    onnx_path: Path,
    meta_dict: Dict[str, str],
) -> None:
    """
    Embed descriptive metadata into the ONNX graph for inspectability in Netron.
    """
    try:
        model = onnx.load(str(onnx_path))
        for k, v in meta_dict.items():
            entry = model.metadata_props.add()
            entry.key = k
            entry.value = str(v)
        onnx.save(model, str(onnx_path))
    except Exception as e:
        logger.warning(f"Could not write metadata to {onnx_path.name}: {e}")


def export_dbnet_onnx(
    config_path: str = "configs/dbnet/dbnet.yaml",
    weights_path: Optional[str] = "weights/dbnet/dbnet_cccd_best.pth",
    output_path: str = "weights/onnx/dbnet.onnx",
    imgsz: Tuple[int, int] = (640, 640),
    opset: int = 18,
    simplify: bool = True,
    verify: bool = True,
) -> Path:
    """
    Export DBNet PyTorch model to a single, self-contained ONNX file with dynamic axes.
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

    logger.info(f"Exporting PyTorch model to self-contained ONNX: {out_p}")
    t0 = time.perf_counter()

    # Export using dynamo=False for single self-contained .onnx without .onnx.data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
            dynamo=False,
        )

    t_export = time.perf_counter() - t0
    logger.info(f"Base export completed in {t_export:.2f}s | Size: {out_p.stat().st_size / (1024 * 1024):.2f} MB")

    # Simplify graph
    if simplify:
        simplify_onnx_graph(out_p)

    # Embed metadata
    add_onnx_metadata(
        out_p,
        {
            "model_type": "DBNet",
            "backbone": str(cfg.model.get("backbone", {}).get("type", "ResNet")),
            "depth": str(cfg.model.get("backbone", {}).get("depth", 18)),
            "task": "Scene Text Detection",
            "input_shape": "B x 3 x H x W (dynamic)",
            "output_shape": "B x 1 x H x W (probability map)",
            "created_at": datetime.now().isoformat(),
        },
    )

    # Verify structural integrity
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph structural validation passed.")

    # Numerical verification with onnxruntime across dynamic shapes
    if verify:
        logger.info("Validating ONNX Runtime execution across dynamic shapes...")
        session = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])

        test_shapes = [
            (1, 3, 640, 640),
            (2, 3, 800, 800),
            (4, 3, 512, 512),
        ]

        for b, c, th, tw in test_shapes:
            x = np.random.randn(b, c, th, tw).astype(np.float32)
            ort_out = session.run(["prob_map"], {"input": x})[0]

            with torch.no_grad():
                torch_out = wrapper(torch.from_numpy(x)).numpy()

            max_diff = float(np.max(np.abs(ort_out - torch_out)))
            logger.info(
                f"Shape [{b}, {c}, {th}, {tw}] -> Output: {ort_out.shape} | Max Diff: {max_diff:.2e}"
            )
            assert max_diff < 1e-4, f"Discrepancy too large for shape [{b},{c},{th},{tw}]: {max_diff}"

        logger.info("All dynamic shape verification tests passed successfully.")

    return out_p


def export_yolo_onnx(
    weights_path: str,
    output_path: str,
    imgsz: Union[int, Tuple[int, int]] = 640,
    opset: int = 18,
    simplify: bool = True,
    verify: bool = True,
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
    if exp_p.resolve() != out_p.resolve():
        shutil.move(str(exp_p), str(out_p))

    # Embed metadata
    add_onnx_metadata(
        out_p,
        {
            "model_type": "YOLO26",
            "source_weights": w_p.name,
            "created_at": datetime.now().isoformat(),
        },
    )

    # Verification with ONNX Runtime
    if verify:
        logger.info(f"Validating {out_p.name} with ONNX Runtime...")
        session = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        in_meta = session.get_inputs()[0]
        in_name = in_meta.name

        dim_h = imgsz if isinstance(imgsz, int) else imgsz[0]
        dim_w = imgsz if isinstance(imgsz, int) else imgsz[1]
        dummy = np.random.randn(1, 3, dim_h, dim_w).astype(np.float32)

        out = session.run(None, {in_name: dummy})
        out_shapes = [list(o.shape) for o in out]
        logger.info(f"Verified {out_p.name} -> Input: {in_name} | Outputs: {out_shapes}")

    logger.info(f"YOLO ONNX exported successfully: {out_p} ({out_p.stat().st_size / (1024 * 1024):.2f} MB)")
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Export CCCD Pipeline Models to Optimized ONNX Format")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "dbnet", "yolo-seg", "yolo-cls"],
        help="Target model to export (default: all)",
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
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable graph simplification",
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

    # Clean up any residual external weight sidecars
    for sidecar in out_dir.glob("*.onnx.data"):
        try:
            sidecar.unlink()
        except OSError:
            pass

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
                verify=verify,
            )
        else:
            logger.warning(f"YOLO-seg checkpoint not found at '{args.yolo_seg_weights}'. Skipping.")

    if args.model in ["all", "yolo-cls"]:
        yolo_cls_out = out_dir / "yolo26_cls.onnx"
        if Path(args.yolo_cls_weights).exists():
            results["yolo_cls"] = export_yolo_onnx(
                weights_path=args.yolo_cls_weights,
                output_path=str(yolo_cls_out),
                imgsz=224,
                opset=args.opset,
                simplify=simplify,
                verify=verify,
            )
        else:
            logger.warning(f"YOLO-cls checkpoint not found at '{args.yolo_cls_weights}'. Skipping.")

    logger.info("=" * 60)
    logger.info("ONNX Export Process Finished Summary:")
    for name, p in results.items():
        logger.info(f"  * {name.upper()}: {p} ({p.stat().st_size / (1024 * 1024):.2f} MB)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
