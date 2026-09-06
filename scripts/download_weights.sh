#!/usr/bin/env bash
# ==============================================================================
# 1-Click Pretrained Weights Downloader for Linux / Server Environments
# Usage:
#   bash scripts/download_weights.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "======================================================================"
echo "  Vietnamese ID Card (CCCD) Pipeline - Pretrained Weights Setup"
echo "======================================================================"

mkdir -p weights/dbnet weights/yolo

# Method 1: Using Python via uv / python
if command -v uv &> /dev/null; then
    echo "[Info] Using 'uv run python' to download weights..."
    uv run python tools/download_weights.py
elif command -v python3 &> /dev/null; then
    echo "[Info] Using 'python3' to download weights..."
    python3 tools/download_weights.py
else
    echo "[Error] Python 3 / uv is required to download weights."
    exit 1
fi

echo ""
echo "======================================================================"
echo "  Pretrained weights successfully prepared in 'weights/' directory!"
echo "======================================================================"
ls -lh weights/dbnet/ weights/yolo/
