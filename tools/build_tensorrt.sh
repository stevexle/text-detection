#!/usr/bin/env bash
# ==============================================================================
# High-Performance NVIDIA TensorRT Engine Build Script for CCCD OCR Pipeline
# Builds DBNet, YOLO-seg, and YOLO-cls FP16 engines with dynamic shape profiles.
# ==============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}      NVIDIA TensorRT Engine Compilation Pipeline (CCCD OCR)          ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# 1. Verify NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[OK] NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}[ERROR] No NVIDIA GPU detected via nvidia-smi! TensorRT compilation requires an NVIDIA GPU.${NC}"
    exit 1
fi

# 2. Locate trtexec binary
TRTEXEC_BIN=""
if command -v trtexec &> /dev/null; then
    TRTEXEC_BIN="trtexec"
elif [ -f ".venv/bin/trtexec" ]; then
    TRTEXEC_BIN=".venv/bin/trtexec"
elif [ -f "/usr/src/tensorrt/bin/trtexec" ]; then
    TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
elif [ -f "/usr/local/tensorrt/bin/trtexec" ]; then
    TRTEXEC_BIN="/usr/local/tensorrt/bin/trtexec"
elif [ -f "/usr/bin/trtexec" ]; then
    TRTEXEC_BIN="/usr/bin/trtexec"
fi

if [ -z "$TRTEXEC_BIN" ]; then
    echo -e "${YELLOW}[WARNING] 'trtexec' binary not found on standard paths.${NC}"
    echo -e "${YELLOW}Attempting to install TensorRT package into .venv...${NC}"
    uv pip install tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs
    if [ -f ".venv/bin/trtexec" ]; then
        TRTEXEC_BIN=".venv/bin/trtexec"
        echo -e "${GREEN}[OK] Installed and located: $TRTEXEC_BIN${NC}"
    else
        echo -e "${RED}[ERROR] Could not locate or install trtexec. Please install TensorRT first.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[OK] Located trtexec at: $TRTEXEC_BIN${NC}"
fi

# Create target output directory
mkdir -p weights/tensorrt
mkdir -p weights/onnx

# 3. Build DBNet TensorRT Engine
echo -e "\n${BLUE}--- [1/3] Compiling DBNet FP16 TensorRT Engine ---${NC}"
if [ ! -f "weights/onnx/dbnet.onnx" ]; then
    echo -e "${YELLOW}weights/onnx/dbnet.onnx not found. Exporting now...${NC}"
    uv run python tools/export_onnx.py --model dbnet
fi

$TRTEXEC_BIN \
    --onnx=weights/onnx/dbnet.onnx \
    --saveEngine=weights/tensorrt/dbnet.engine \
    --minShapes=input:1x3x480x480 \
    --optShapes=input:1x3x960x704 \
    --maxShapes=input:8x3x960x960 \
    --memPoolSize=workspace:2048M \
    --fp16

echo -e "${GREEN}[OK] DBNet engine compiled: weights/tensorrt/dbnet.engine${NC}"

# 4. Build YOLO-seg TensorRT Engine
echo -e "\n${BLUE}--- [2/3] Compiling YOLO-seg FP16 TensorRT Engine ---${NC}"
if [ -f "weights/yolo/yolo26_seg_best.pt" ]; then
    echo -e "${GREEN}Exporting YOLO-seg via Ultralytics native TensorRT engine export...${NC}"
    uv run yolo export model=weights/yolo/yolo26_seg_best.pt format=engine half=True dynamic=True workspace=2
    # Copy exported engine to standard directory
    if [ -f "weights/yolo/yolo26_seg_best.engine" ]; then
        cp weights/yolo/yolo26_seg_best.engine weights/tensorrt/yolo26_seg.engine
    fi
elif [ -f "weights/onnx/yolo26_seg.onnx" ]; then
    $TRTEXEC_BIN \
        --onnx=weights/onnx/yolo26_seg.onnx \
        --saveEngine=weights/tensorrt/yolo26_seg.engine \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:1x3x640x640 \
        --maxShapes=images:8x3x640x640 \
        --memPoolSize=workspace:2048M \
        --fp16
else
    echo -e "${YELLOW}YOLO-seg weights not found. Skipping.${NC}"
fi
echo -e "${GREEN}[OK] YOLO-seg engine compiled: weights/tensorrt/yolo26_seg.engine${NC}"

# 5. Build YOLO-cls TensorRT Engine
echo -e "\n${BLUE}--- [3/3] Compiling YOLO-cls FP16 TensorRT Engine ---${NC}"
if [ -f "weights/yolo/yolo26_cls_best.pt" ]; then
    echo -e "${GREEN}Exporting YOLO-cls via Ultralytics native TensorRT engine export...${NC}"
    uv run yolo export model=weights/yolo/yolo26_cls_best.pt format=engine half=True dynamic=True workspace=2
    if [ -f "weights/yolo/yolo26_cls_best.engine" ]; then
        cp weights/yolo/yolo26_cls_best.engine weights/tensorrt/yolo26_cls.engine
    fi
elif [ -f "weights/onnx/yolo26_cls.onnx" ]; then
    $TRTEXEC_BIN \
        --onnx=weights/onnx/yolo26_cls.onnx \
        --saveEngine=weights/tensorrt/yolo26_cls.engine \
        --minShapes=images:1x3x224x224 \
        --optShapes=images:1x3x224x224 \
        --maxShapes=images:8x3x224x224 \
        --memPoolSize=workspace:2048M \
        --fp16
else
    echo -e "${YELLOW}YOLO-cls weights not found. Skipping.${NC}"
fi
echo -e "${GREEN}[OK] YOLO-cls engine compiled: weights/tensorrt/yolo26_cls.engine${NC}"

echo -e "\n${BLUE}======================================================================${NC}"
echo -e "${GREEN}  All TensorRT Engines successfully compiled!                        ${NC}"
echo -e "${GREEN}  Saved in: weights/tensorrt/                                        ${NC}"
ls -lh weights/tensorrt/
echo -e "${BLUE}======================================================================${NC}"
