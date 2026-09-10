# Vietnamese ID Card (CCCD) Text Detection & Classification Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![uv Package Manager](https://img.shields.io/badge/uv-fast%20packaging-blueviolet.svg)](https://github.com/astral-sh/uv)
[![Tests Passing](https://img.shields.io/badge/tests-43%20passed-success.svg)]()

A high-performance, modular, and production-ready **Text Detection, Field Segmentation, and Document Classification Framework** specifically tailored for Vietnamese Citizen Identity Cards (**CCCD 2021 chip** and **CCCD 2024** standards). Built with modern deep learning backbones (**DBNet**, **YOLO26-cls**, **YOLO26-seg**), Spatial Matching Hybrid Fusion, Automatic Mixed Precision (AMP FP16), and the standard **ICDAR 2015 Evaluation Protocol**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A["Raw Input Image<br>(CCCD 2021 / 2024)"] --> B["1. Document Classification<br>(YOLO26-cls)"]
    B -->|"card_type: front_2021 / back_2024"| C["Pipeline Routing Logic"]
    
    subgraph "Parallel High-Precision Detection"
        C --> D["2a. Text Contour Detection<br>(DBNet: ResNet-50 / MobileNetV3)"]
        C --> E["2b. Field Segmentation<br>(YOLO26-seg / YOLO26-detect)"]
        
        D --> D1["DBPostProcessor<br>(Vatti Polygon Unclipping r=1.5)"]
        E --> E1["11 Field Bounding Masks / Boxes"]
    end
    
    D1 & E1 --> F["3. Spatial Matching & Fusion<br>(AABB Fast-Reject + Polygon Overlap)"]
    F --> G["Labeled Text Polygons<br>(id, name, dob, origin_place,...)"]
    
    G --> H["4. Downstream Text Recognition<br>(VietOCR / CRNN)"]
    H --> I["Structured eKYC JSON Output"]
```

---

## ✨ Key Features

- **Modular Registry Pattern:** Dynamic component management for `BACKBONES`, `NECKS`, `HEADS`, `LOSSES`, `POSTPROCESSORS`, and `METRICS`.
- **Differentiable Binarization (DBNet):**
  - **Differentiable Binarization Step Function:** $\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}$ with amplification factor $k=50$.
  - **Multi-Task DBLoss:** Hard Negative Mining (OHEM 1:3) Binary Cross-Entropy + Smooth L1 Threshold Loss + Dice Loss.
  - **Vatti Unclipping Algorithm:** Restores original polygon contours using `pyclipper` polygon expansion ($r'=1.5$).
- **Lean Document Classifier (YOLO26-cls):**
  - Instant classification of card standard (`2021` vs `2024`) and side (`front` vs `back`).
  - Zero-disk-duplication dynamic Train/Val staging via in-memory symbolic links (`scratch/yolo_cls/`).
  - High-throughput vectorized batch inference (`classify_batch`).
  - 1-Line export to ONNX / TensorRT (`export(format="onnx", half=True)`).
- **Semantic Field Segmentation (YOLO26-seg):**
  - Segments 11 standardized CCCD information fields: `id`, `name`, `dob`, `gender`, `nationality`, `origin_place`, `current_place`, `expire_date`, `issue_date`, `features`, `mrz`.
  - Dynamic in-memory Train/Val staging via symlinks (`scratch/yolo_seg/`).
- **High-Speed Spatial Matching Hybrid Pipeline:**
  - Fast AABB (Axis-Aligned Bounding Box) filtering + Shapely polygon intersection.
  - Automatically maps semantic field labels (`name`, `id`, `dob`,...) to sharp DBNet character polygon contours.
  - Guaranteed Zero-Lost-Field fallback mechanism.
  - GPU/JIT Warmup & `torch.inference_mode()` execution (~40-50 ms / card).
- **Multi-View Visualizations:**
  - `fused_{image}.jpg`: Sharp DBNet text contours colored by CCCD field categories.
  - `raw_yolo_{image}.jpg`: Raw field bounding masks and layout regions from YOLO-seg.
  - `compare_{image}.jpg`: Side-by-side composite comparison (Raw Fields vs Labeled Text Polygons).
- **Production Training Engine:**
  - Automatic Mixed Precision (**AMP FP16**) on CUDA and Apple Silicon (**MPS**).
  - Cosine Annealing Learning Rate scheduler with 3-epoch Linear Warmup.
  - Gradient norm clipping (`max_norm=5.0`) for numerical stability.
  - Periodic and auto-best checkpoint saving based on validation **Hmean (F1-score)**.
  - Full checkpoint resume support (`--resume`).

---

## 📁 Repository Structure

```
text_detection/
├── configs/                          # Experiment configuration files
│   ├── dbnet/
│   │   └── dbnet.yaml                # Master DBNet configuration (ResNet-50 / MobileNetV3)
│   └── yolo/
│       ├── yolo.yaml                 # YOLO Bounding Box Detection configuration
│       ├── yolo_cls.yaml             # YOLO Document Classification configuration
│       └── yolo_seg.yaml             # YOLO Instance Segmentation (11 fields) configuration
├── data/                             # Dataset storage
│   ├── images/                       # Real CCCD images (640x640)
│   └── labels/
│       ├── dbnet/dataset.jsonl       # Master DBNet JSONL annotations
│       ├── yolo_detect/              # YOLO Bounding Box labels
│       └── yolo_seg/                 # YOLO Segmentation labels
├── docs/                             # Technical documentation and architecture diagrams
│   ├── dbnet.md                      # Comprehensive DBNet technical specification
│   └── images/                       # High-resolution architectural diagrams
├── src/                              # Core framework source code
│   ├── data/                         # Data loaders, target generators, and transforms
│   ├── engine/                       # Training loop, AMP, and checkpointing
│   ├── losses/                       # DBLoss, BCE with OHEM 1:3, Dice Loss, Smooth L1
│   ├── metrics/                      # ICDAR 2015 Evaluator & Polygon IoU
│   ├── models/                       # Model Zoo (Backbones, Necks, Heads, Detectors)
│   │   ├── backbones/                # ResNet, MobileNetV3, TIMMBackbone
│   │   ├── necks/                    # FPN, Adaptive Scale Fusion (ASF)
│   │   ├── heads/                    # DBHead
│   │   ├── detectors/                # DBNet assembly
│   │   └── wrappers/                 # YOLOClassifier & YOLOWrapper
│   ├── pipeline/                     # End-to-End Hybrid Processing Pipeline
│   │   ├── spatial_matcher.py        # AABB & Polygon Overlap Matching with Fallback
│   │   └── cccd_pipeline.py          # Unified eKYC CCCD Detection Pipeline
│   ├── postprocess/                  # DBPostProcessor & Vatti Unclipping
│   └── utils/                        # Config loader, Registry, Logger, Checkpoint helpers
├── tools/                            # 1-Click Command-Line Tools
│   ├── train.py                      # DBNet training CLI
│   ├── eval.py                       # DBNet ICDAR 2015 evaluation CLI
│   ├── demo.py                       # DBNet inference & polygon visualization CLI
│   ├── train_yolo.py                 # YOLO Field Detection / Segmentation training CLI
│   ├── eval_yolo.py                  # YOLO Detection / Segmentation evaluation CLI
│   ├── predict_yolo.py               # YOLO Field Detection & visualization CLI
│   ├── train_yolo_cls.py             # YOLO Document Classification training CLI
│   ├── eval_yolo_cls.py              # YOLO Document Classification evaluation CLI
│   ├── predict_yolo_cls.py           # YOLO Document Classification inference CLI
│   └── predict_pipeline.py           # 1-Click Hybrid Pipeline (DBNet + YOLO Labeled Polygons)
├── weights/                          # Pretrained & Best production weights
│   ├── dbnet/dbnet_cccd_best.pth     # Production DBNet weights
│   └── yolo/                         # Base & fine-tuned YOLO weights
├── tests/                            # Comprehensive Unit Test Suite (43/43 passing)
├── pyproject.toml                    # PEP 517 / PEP 621 package build configuration
└── README.md
```

---

## ⚡ Quick Start

### 1. Installation

The project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management and editable package installation:

```bash
# Clone the repository
git clone https://github.com/stevexle/text-detection.git
cd text-detection

# Synchronize virtual environment dependencies
uv sync

# Install package in editable mode
uv pip install -e .

# Run the complete test suite (43 unit tests)
uv run pytest
```

### 2. Download Pretrained Base Weights (1-Click)

Run the automated script to download ImageNet backbones (ResNet-50/18, MobileNetV3) and YOLO base models directly into `weights/`:

```bash
# On Linux / macOS server:
bash scripts/download_weights.sh
```

---

## 🚀 Execution & Usage

### Module 1: Document Classification (YOLO26-cls)

Classifies identity cards into 4 distinct classes: `front_2021`, `back_2021`, `front_2024`, `back_2024`.

#### Train Classifier:
```bash
uv run python tools/train_yolo_cls.py --config configs/yolo/yolo_cls.yaml --epochs 30 --batch-size 32
```
*Automatically splits `dataset.jsonl` dynamically into 85% Train / 15% Val using symbolic links in `scratch/yolo_cls/`.*

#### Evaluate Classifier:
```bash
uv run python tools/eval_yolo_cls.py --config configs/yolo/yolo_cls.yaml
```

#### Run Classification Inference:
```bash
uv run python tools/predict_yolo_cls.py --weights weights/yolo/yolo26_cls_best.pt --source data/images/sample.jpg --save-json output.json
```

**Structured JSON Output:**
```json
{
  "card_type": "front_2021",
  "confidence": 0.9982
}
```

---

### Module 2: Text Detection (DBNet)

Detects tight text polygon contours across the identity card.

#### Train DBNet:
```bash
uv run python tools/train.py --config configs/dbnet/dbnet.yaml --epochs 50 --batch-size 8 --lr 0.0005
```

#### Resume Training from Checkpoint:
```bash
uv run python tools/train.py --config configs/dbnet/dbnet.yaml --resume work_dirs/dbnet/latest.pth
```

#### Benchmark Evaluation (ICDAR 2015 Protocol):
```bash
uv run python tools/eval.py --config configs/dbnet/dbnet.yaml --weights weights/dbnet/dbnet_cccd_best.pth --iou-thresh 0.5
```
*Outputs Precision, Recall, Hmean (F1-score), Latency (ms), and FPS.*

#### Run Inference & Polygon Visualization:
```bash
uv run python tools/demo.py --source data/images/sample.jpg --output-dir runs/predict/ --save-json polygons.json
```
*Draws high-visibility polygon overlays and confidence badges, saving visualized outputs to `runs/predict/pred_sample.jpg`.*

---

### Module 3: Field Detection & Segmentation (YOLO26-seg)

Detects and segments the 11 standardized CCCD information fields: `id`, `name`, `dob`, `gender`, `nationality`, `origin_place`, `current_place`, `expire_date`, `issue_date`, `features`, `mrz`.

#### Train YOLO Field Segmentation:
```bash
uv run python tools/train_yolo.py --config configs/yolo/yolo_seg.yaml --epochs 50 --batch-size 16 --imgsz 640
```

#### Evaluate YOLO Segmentation (mAP50 / mAP50-95):
```bash
uv run python tools/eval_yolo.py --config configs/yolo/yolo_seg.yaml --weights weights/yolo/yolo26_seg_best.pt
```

#### Run Field Prediction & Visualization:
```bash
uv run python tools/predict_yolo.py --source data/images/sample.jpg --save-vis runs/predict_yolo/
```

---

### Module 4: 🌟 End-to-End Hybrid Fusion Pipeline (DBNet + YOLO)

Combines **Document Classification + Field Segmentation + DBNet Text Contour Extraction + Spatial Matching** into a single 1-click execution. Maps semantic field labels (`id`, `name`, `dob`, `origin_place`,...) onto pixel-sharp DBNet polygon text contours.

#### 1-Click Pipeline Execution on an Image:
```bash
# Default pure JSON execution (ultra-fast, saves to runs/pipeline/result.json):
uv run python tools/predict_pipeline.py --source data/cccd-minh2.jpg

# Optional: With side-by-side comparison image output:
uv run python tools/predict_pipeline.py --source data/cccd-minh2.jpg --save-vis runs/pipeline/
```

#### High-Throughput Batch Processing (GPU + FP16):
```bash
uv run python tools/predict_pipeline.py \
  --source data/images/ \
  --batch-size 16 \
  --fp16
```

#### Generated Visualizations (Optional via `--save-vis`):
- `runs/pipeline/fused_sample.jpg`: Side-by-side composite comparison (Left: Raw YOLO Fields | Right: Labeled DBNet Text Polygons).

#### Complete Structured Hybrid JSON Output:
```json
{
  "classification": {
    "card_type": "front_2021",
    "confidence": 0.9982
  },
  "total_texts": 9,
  "detections": [
    {
      "label": "id",
      "confidence": 0.7882,
      "polygon": [[1118.75, 936.90], [1813.25, 920.10], [1817.25, 1013.10], [1122.75, 1029.90]]
    },
    {
      "label": "name",
      "confidence": 0.7544,
      "polygon": [[908.00, 1113.00], [1792.00, 1113.00], [1792.00, 1185.00], [908.00, 1185.00]]
    },
    {
      "label": "dob",
      "confidence": 0.8474,
      "polygon": [[1500.00, 1188.00], [1844.00, 1188.00], [1844.00, 1260.00], [1500.00, 1260.00]]
    }
  ],
  "latency_ms": 40.8
}
```

**JSON Schema Field Definitions:**
- `classification`: Card type classification (`front_2021`, `back_2021`, `front_2024`, `back_2024`) and confidence score.
- `total_texts`: Total number of detected text polygon segments.
- `detections`: Fused character polygon detections with semantic field labels (`id`, `name`, `dob`, `origin_place`,...) and confidence score.
- `latency_ms`: Total execution time for the image in milliseconds.

---

## 🧪 Verification & Unit Tests

The test suite ensures 100% reliability across all core components:

```bash
uv run pytest -v
```

```text
============================== 43 passed in 4.34s ==============================
tests/test_classifier.py .........                                       [ 20%]
tests/test_data.py ...                                                   [ 27%]
tests/test_engine.py ....                                                [ 37%]
tests/test_losses.py ....                                                [ 46%]
tests/test_metrics.py .......                                            [ 62%]
tests/test_models.py .......                                             [ 79%]
tests/test_pipeline.py ..                                                [ 83%]
tests/test_postprocess.py ...                                            [ 90%]
tests/test_utils.py ....                                                 [100%]
```

---

## 📖 Technical Documentation

For detailed mathematical formulations, ablation studies, loss curves, and architectural deep-dives, please refer to:
* [DBNet Technical Master Documentation](docs/dbnet.md)

---

## 📄 License

This project is licensed under the Apache 2.0 License.
