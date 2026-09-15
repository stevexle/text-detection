"""
Automated NVIDIA TensorRT Engine Builder for CCCD Detection Pipeline.
Compiles DBNet, YOLO-seg, and YOLO-cls ONNX models into high-performance
FP16 TensorRT execution engines (.engine) with dynamic shape profiles.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger("BuildTensorRT")

# Default Dynamic Shape Profiles for Production Inference
SHAPE_PROFILES: Dict[str, Dict[str, str]] = {
    "dbnet": {
        "input_name": "input",
        "min": "1x3x480x480",
        "opt": "1x3x960x704",
        "max": "8x3x960x960",
    },
    "yolo_seg": {
        "input_name": "images",
        "min": "1x3x640x640",
        "opt": "1x3x640x640",
        "max": "8x3x640x640",
    },
    "yolo_cls": {
        "input_name": "images",
        "min": "1x3x224x224",
        "opt": "1x3x224x224",
        "max": "8x3x224x224",
    },
}


def find_trtexec() -> Optional[str]:
    """
    Auto-detect trtexec compiler binary across virtualenv, PATH, and standard system paths.
    """
    # 1. Check system PATH
    found = shutil.which("trtexec")
    if found:
        return found

    # 2. Check current Python virtualenv bin directory
    venv_bin = Path(sys.prefix) / "bin" / "trtexec"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

    # 3. Standard Linux / Docker / CUDA installation paths
    candidate_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/cuda/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for p in candidate_paths:
        if Path(p).exists() and os.access(p, os.X_OK):
            return p

    return None


def construct_trtexec_cmd(
    trtexec_bin: str,
    onnx_path: str,
    engine_path: str,
    input_name: str,
    min_shape: str,
    opt_shape: str,
    max_shape: str,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 2048,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Construct the full trtexec command with dynamic shape profile arguments.
    """
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={input_name}:{min_shape}",
        f"--optShapes={input_name}:{opt_shape}",
        f"--maxShapes={input_name}:{max_shape}",
        f"--memPoolSize=workspace:{workspace_mb}M",
    ]

    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def build_dbnet_engine(
    trtexec_bin: str,
    onnx_path: str = "weights/onnx/dbnet.onnx",
    output_dir: str = "weights/tensorrt",
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
) -> Path:
    """Build DBNet TensorRT Engine with dynamic spatial and batch profiles."""
    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        raise FileNotFoundError(f"DBNet ONNX model not found at: {onnx_p}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_p = out_dir / "dbnet.engine"

    prof = SHAPE_PROFILES["dbnet"]
    cmd = construct_trtexec_cmd(
        trtexec_bin=trtexec_bin,
        onnx_path=str(onnx_p),
        engine_path=str(engine_p),
        input_name=prof["input_name"],
        min_shape=prof["min"],
        opt_shape=prof["opt"],
        max_shape=prof["max"],
        fp16=fp16,
        workspace_mb=workspace_mb,
    )

    logger.info(f"Building DBNet TensorRT engine: {engine_p.name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    if not dry_run:
        res = subprocess.run(cmd, check=True)
        if res.returncode == 0:
            logger.info(f"Successfully generated DBNet TensorRT engine at: {engine_p}")
    else:
        logger.info("[Dry Run] Skipped actual engine build execution.")

    return engine_p


def build_yolo_engine(
    trtexec_bin: str,
    model_type: str,  # 'seg' or 'cls'
    onnx_path: str,
    pt_path: Optional[str] = None,
    output_dir: str = "weights/tensorrt",
    fp16: bool = True,
    workspace_mb: int = 2048,
    use_ultralytics_export: bool = True,
    dry_run: bool = False,
) -> Path:
    """
    Build YOLO TensorRT Engine.
    Can either use native Ultralytics engine exporter or trtexec directly.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_p = out_dir / f"yolo26_{model_type}.engine"

    if use_ultralytics_export and not dry_run:
        # Check if PyTorch .pt model exists, which allows Ultralytics to export with native TRT NMS plugin
        source_model = pt_path if (pt_path and Path(pt_path).exists()) else onnx_path
        if Path(source_model).exists():
            logger.info(f"Using Ultralytics native TensorRT export on {source_model}...")
            from ultralytics import YOLO

            model = YOLO(source_model)
            exported = model.export(
                format="engine",
                half=fp16,
                workspace=workspace_mb / 1024.0,
                dynamic=True,
                verbose=True,
            )
            # Move / copy to target output directory
            exported_p = Path(exported)
            if exported_p != engine_p and exported_p.exists():
                shutil.copy2(exported_p, engine_p)
            logger.info(f"Successfully generated YOLO-{model_type} TensorRT engine at: {engine_p}")
            return engine_p

    # Fallback to direct trtexec command
    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        raise FileNotFoundError(f"YOLO-{model_type} ONNX model not found at: {onnx_p}")

    prof_key = f"yolo_{model_type}"
    prof = SHAPE_PROFILES[prof_key]
    cmd = construct_trtexec_cmd(
        trtexec_bin=trtexec_bin,
        onnx_path=str(onnx_p),
        engine_path=str(engine_p),
        input_name=prof["input_name"],
        min_shape=prof["min"],
        opt_shape=prof["opt"],
        max_shape=prof["max"],
        fp16=fp16,
        workspace_mb=workspace_mb,
    )

    logger.info(f"Building YOLO-{model_type} TensorRT engine via trtexec: {engine_p.name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    if not dry_run:
        res = subprocess.run(cmd, check=True)
        if res.returncode == 0:
            logger.info(f"Successfully generated YOLO-{model_type} engine at: {engine_p}")
    else:
        logger.info("[Dry Run] Skipped actual engine build execution.")

    return engine_p


def main():
    parser = argparse.ArgumentParser(description="NVIDIA TensorRT Engine Compilation Tool for CCCD Detection Pipeline")
    parser.add_argument(
        "--model",
        choices=["all", "dbnet", "yolo-seg", "yolo-cls"],
        default="all",
        help="Target model to compile into TensorRT engine (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="weights/tensorrt",
        help="Output directory for compiled .engine files (default: weights/tensorrt)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision compilation (default: True)",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2048,
        help="TensorRT workspace memory allocation in Megabytes (default: 2048 MB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display generated trtexec commands without executing compilation",
    )
    parser.add_argument(
        "--use-yolo-export",
        action="store_true",
        default=True,
        help="Use Ultralytics native TRT engine exporter for YOLO models (default: True)",
    )
    parser.add_argument(
        "--trtexec-path",
        default=None,
        help="Explicit path to trtexec binary if not automatically discovered",
    )
    args = parser.parse_args()

    trtexec_bin = args.trtexec_path or find_trtexec()
    if not trtexec_bin and not args.dry_run:
        logger.error("Could not find 'trtexec' compiler binary on this system!")
        logger.error("To install TensorRT on Linux:")
        logger.error("  uv pip install tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs")
        logger.error("Or specify path explicitly using --trtexec-path=/path/to/trtexec")
        sys.exit(1)

    trtexec_bin = trtexec_bin or "trtexec"
    logger.info("=" * 60)
    logger.info("Starting NVIDIA TensorRT Engine Compilation Pipeline")
    logger.info(f"Compiler:    {trtexec_bin}")
    logger.info(f"Target:      {args.model}")
    logger.info(f"Precision:   {'FP16' if args.fp16 else 'FP32'}")
    logger.info(f"Output Dir:  {args.output_dir}")
    logger.info("=" * 60)

    try:
        if args.model in ["all", "dbnet"]:
            build_dbnet_engine(
                trtexec_bin=trtexec_bin,
                onnx_path="weights/onnx/dbnet.onnx",
                output_dir=args.output_dir,
                fp16=args.fp16,
                workspace_mb=args.workspace,
                dry_run=args.dry_run,
            )

        if args.model in ["all", "yolo-seg"]:
            build_yolo_engine(
                trtexec_bin=trtexec_bin,
                model_type="seg",
                onnx_path="weights/onnx/yolo26_seg.onnx",
                pt_path="weights/yolo/yolo26_seg_best.pt",
                output_dir=args.output_dir,
                fp16=args.fp16,
                workspace_mb=args.workspace,
                use_ultralytics_export=args.use_yolo_export,
                dry_run=args.dry_run,
            )

        if args.model in ["all", "yolo-cls"]:
            build_yolo_engine(
                trtexec_bin=trtexec_bin,
                model_type="cls",
                onnx_path="weights/onnx/yolo26_cls.onnx",
                pt_path="weights/yolo/yolo26_cls_best.pt",
                output_dir=args.output_dir,
                fp16=args.fp16,
                workspace_mb=args.workspace,
                use_ultralytics_export=args.use_yolo_export,
                dry_run=args.dry_run,
            )

        logger.info("=" * 60)
        logger.info("TensorRT Compilation completed successfully!")
        logger.info(f"Compiled engines saved in: {args.output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"TensorRT Engine build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
