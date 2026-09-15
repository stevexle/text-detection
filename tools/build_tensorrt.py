"""
Automated NVIDIA TensorRT Engine Builder for CCCD Detection Pipeline.
Compiles DBNet, YOLO-seg, and YOLO-cls ONNX models into high-performance
FP16 TensorRT execution engines (.engine) with dynamic shape profiles.

Supports both:
  1. Python native TensorRT Builder (via `tensorrt` package - no trtexec binary required).
  2. Standalone `trtexec` CLI compiler if available on system PATH.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Auto-detect and register NVIDIA CUDA, cuDNN, and TensorRT shared libraries on Linux
if sys.platform == "linux":
    import ctypes
    import site
    try:
        for site_pkg in site.getsitepackages():
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "tensorrt"]:
                    lib_dir = os.path.join(nvidia_dir, sub, "lib")
                    if os.path.isdir(lib_dir):
                        if "LD_LIBRARY_PATH" in os.environ:
                            if lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
                        else:
                            os.environ["LD_LIBRARY_PATH"] = lib_dir
                        for f in sorted(os.listdir(lib_dir)):
                            if f.endswith(".so") or ".so." in f:
                                try:
                                    ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass
            trt_dir = os.path.join(site_pkg, "tensorrt")
            if os.path.isdir(trt_dir):
                for f in sorted(os.listdir(trt_dir)):
                    if f.endswith(".so") or ".so." in f:
                        try:
                            ctypes.CDLL(os.path.join(trt_dir, f), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception:
        pass

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


def parse_shape_str(s: str) -> Tuple[int, ...]:
    """Parse '1x3x480x480' string into tuple of ints (1, 3, 480, 480)."""
    return tuple(int(x) for x in s.split("x"))


def find_trtexec() -> Optional[str]:
    """
    Auto-detect trtexec compiler binary across virtualenv, PATH, and standard system paths.
    """
    found = shutil.which("trtexec")
    if found:
        return found

    venv_bin = Path(sys.prefix) / "bin" / "trtexec"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

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
    """Construct the full trtexec command with dynamic shape profile arguments."""
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


def build_engine_from_onnx_python(
    onnx_path: str,
    engine_path: str,
    input_name: str,
    min_shape: str,
    opt_shape: str,
    max_shape: str,
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
) -> bool:
    """
    Compile ONNX model to TensorRT engine natively via Python TensorRT API.
    Does not require trtexec CLI binary!
    """
    if dry_run:
        logger.info(f"[Dry Run] Compiling {onnx_path} -> {engine_path} via TensorRT Python API")
        logger.info(f"[Dry Run] Shapes: min={min_shape}, opt={opt_shape}, max={max_shape}, fp16={fp16}")
        return True

    try:
        import tensorrt as trt
    except ImportError as e:
        logger.error(f"tensorrt Python package is required: {e}")
        return False

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    # TensorRT 8 requires EXPLICIT_BATCH, while TensorRT 10+ uses explicit batch by default
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    logger.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(error)}")
            return False

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType") and hasattr(trt.MemoryPoolType, "WORKSPACE"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mb * 1024 * 1024

    # TensorRT 8/10 supports BuilderFlag.FP16, while TensorRT 11 uses strongly typed graph definitions
    if fp16 and hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

    # Configure dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        parse_shape_str(min_shape),
        parse_shape_str(opt_shape),
        parse_shape_str(max_shape),
    )
    config.add_optimization_profile(profile)

    logger.info(f"Building serialized TensorRT engine: {engine_path} (this may take 1-3 minutes)...")
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT failed to build serialized network for {onnx_path}")
        with open(engine_path, "wb") as f:
            f.write(plan)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build engine for {onnx_path}")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

    logger.info(f"Successfully compiled engine via Python API: {engine_path}")
    return True


def build_dbnet_engine(
    trtexec_bin: Optional[str] = None,
    onnx_path: str = "weights/onnx/dbnet.onnx",
    output_dir: str = "weights/tensorrt",
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
) -> Path:
    """Build DBNet TensorRT Engine with dynamic spatial and batch profiles."""
    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        logger.info(f"DBNet ONNX model not found at {onnx_p}. Exporting now...")
        if not dry_run:
            from tools.export_onnx import export_dbnet_onnx
            export_dbnet_onnx()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_p = out_dir / "dbnet.engine"

    prof = SHAPE_PROFILES["dbnet"]

    # Option A: If trtexec binary is available, construct command
    if trtexec_bin:
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
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully generated DBNet engine at: {engine_p}")
        else:
            logger.info("[Dry Run] Skipped actual engine build execution.")
    else:
        # Option B: Use native Python TensorRT builder (no trtexec binary needed)
        logger.info(f"Building DBNet TensorRT engine via Python API: {engine_p.name}...")
        build_engine_from_onnx_python(
            onnx_path=str(onnx_p),
            engine_path=str(engine_p),
            input_name=prof["input_name"],
            min_shape=prof["min"],
            opt_shape=prof["opt"],
            max_shape=prof["max"],
            fp16=fp16,
            workspace_mb=workspace_mb,
            dry_run=dry_run,
        )

    return engine_p


def build_yolo_engine(
    trtexec_bin: Optional[str] = None,
    model_type: str = "seg",
    onnx_path: str = "weights/onnx/yolo26_seg.onnx",
    pt_path: str = "weights/yolo/yolo26_seg_best.pt",
    output_dir: str = "weights/tensorrt",
    fp16: bool = True,
    workspace_mb: int = 2048,
    use_ultralytics_export: bool = True,
    dry_run: bool = False,
) -> Path:
    """Build YOLO TensorRT Engine via Ultralytics native export or trtexec/Python builder."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_p = out_dir / f"yolo26_{model_type}.engine"

    pt_file = Path(pt_path)
    if use_ultralytics_export and pt_file.exists():
        logger.info(f"Building YOLO-{model_type} via Ultralytics native TensorRT exporter: {pt_file.name}")
        if not dry_run:
            from ultralytics import YOLO
            model = YOLO(str(pt_file))
            exported = model.export(
                format="engine",
                half=fp16,
                dynamic=True,
                workspace=max(1, workspace_mb // 1024),
            )
            exported_p = Path(exported) if exported else pt_file.with_suffix(".engine")
            if exported_p.exists():
                shutil.copy(exported_p, engine_p)
                logger.info(f"Successfully generated YOLO-{model_type} engine at: {engine_p}")
            else:
                raise RuntimeError(f"Ultralytics export finished but {exported_p} was not found.")
        else:
            logger.info(f"[Dry Run] Ultralytics export model={pt_file} format=engine half={fp16} dynamic=True")
        return engine_p

    # Fallback to ONNX conversion
    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        raise FileNotFoundError(f"Neither PyTorch weights ({pt_path}) nor ONNX model ({onnx_path}) found.")

    prof = SHAPE_PROFILES[f"yolo_{model_type}"]
    if trtexec_bin:
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
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully generated YOLO-{model_type} engine at: {engine_p}")
        else:
            logger.info("[Dry Run] Skipped actual engine build execution.")
    else:
        build_engine_from_onnx_python(
            onnx_path=str(onnx_p),
            engine_path=str(engine_p),
            input_name=prof["input_name"],
            min_shape=prof["min"],
            opt_shape=prof["opt"],
            max_shape=prof["max"],
            fp16=fp16,
            workspace_mb=workspace_mb,
            dry_run=dry_run,
        )

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
        help="Display generated compilation steps without executing",
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
        help="Explicit path to trtexec binary if preferred over Python API",
    )
    args = parser.parse_args()

    trtexec_bin = args.trtexec_path or find_trtexec()
    has_trt_py = False
    try:
        import tensorrt as trt
        has_trt_py = True
    except ImportError:
        pass

    if not trtexec_bin and not has_trt_py and not args.dry_run:
        logger.error("Neither 'trtexec' binary nor Python 'tensorrt' package was found on this system!")
        logger.error("Please run: uv sync")
        sys.exit(1)

    backend = f"trtexec ({trtexec_bin})" if trtexec_bin else "Python TensorRT API (tensorrt)"
    logger.info("=" * 60)
    logger.info("Starting NVIDIA TensorRT Engine Compilation Pipeline")
    logger.info(f"Compiler:    {backend}")
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
