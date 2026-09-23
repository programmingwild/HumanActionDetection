# Human Action Detection — Complete Project Documentation

## 1. Overview

**ActionScope** is an end-to-end human action detection system built in PyTorch. It covers
three complementary approaches: (a) a 2D frame classifier for 15 everyday actions,
(b) a true-temporal 3D CNN for 6 KTH video actions, and (c) YOLOv8 person detection fused
with the 2D classifier for per-person action labels. A premium dark-themed Gradio web app
exposes all three behind one interface, and the project ships with training, evaluation,
inference (image / video / webcam), deployment (Hugging Face Spaces, Docker), and reporting
tooling.

| Branch | Model | Dataset | Result |
|---|---|---|---|
| 2D frame classifier | EfficientNet-B3 (B0 mid-step, MobileNetV3 baseline) | HAR-15, 10,710 train / 1,890 test | **85.24% test**; B0 82.75% (83.49% TTA); baseline 74.02% |
| 3D temporal classifier | R3D-18 | KTH, 600 videos, 6 classes | **91.6% validation** |
| Detection + action | YOLOv8n + 2D head | Same 2D checkpoint | Verified qualitatively |

## 2. Objectives

1. Classify human actions in still images across 15 routine activities.
2. Model motion over time with a 3D CNN on real video clips.
3. Detect *who* is *where* (YOLO boxes) and label each person's action.
4. Serve everything through a polished, user-friendly web demo.
5. Document, evaluate honestly, and make the work deployable and reproducible.

## 3. Datasets (both from Kaggle)

### 3.1 HAR-15 — `shashankrapolu/human-action-recognition-dataset`
- ~218 MB, 12,600 images, pre-split `train/` + `test/`, `ImageFolder`-ready.
- 15 classes: calling, clapping, cycling, dancing, drinking, eating, fighting, hugging,
  laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop.
- Split used: 10,710 train / 1,890 test (126 per class in test).

### 3.2 KTH — `vafaeii/kth-action-recognition-dataset`
- 600 clips (160x120, 25 fps), 25 subjects x 6 actions x 4 scenarios.
- 6 classes: boxing, handclapping, handwaving, jogging, running, walking.
- Flat filenames (`person01_boxing_d1_uncomp.avi`) are auto-organized into
  per-class folders by `organize_kth()` in `src/video_dataset.py`.

## 4. Methodology

### 4.1 2D classifier (`src/model.py`, `src/train.py`, `src/dataset.py`)
- Transfer learning from ImageNet weights; only the final head is replaced.
- Baseline: MobileNetV3-Small, 12 epochs, Adam, cosine schedule → 74.02%.
- Champion: **EfficientNet-B3 @300px, 40 epochs, batch 32, AdamW (lr 3e-4, wd 1e-4),
  cosine annealing, label smoothing 0.1, mixed precision (AMP), RandAugment + MixUp 0.4,
  seed 42** → 85.24%. Mid-step EfficientNet-B0 @224px reached 82.75% (83.49% TTA).
- Augmentation: RandomResizedCrop, flip, rotation, color jitter, RandomErasing;
  optional RandAugment (`--randaugment`) and MixUp (`--mixup`) flags for further runs.
- Resolution-aware: `--img-size` is stored in the checkpoint; all inference paths
  (`infer`, `yolo_action`, `evaluate`, Gradio app) read it back automatically.

### 4.2 3D video classifier (`src/model_3d.py`, `src/train_video.py`, `src/video_dataset.py`)
- R3D-18 (also supports MC3-18) with replaced FC head; 16 uniformly sampled
  frames per clip at 112px → tensor (C, T, H, W).
- Adam, cosine, 10 epochs on RTX 5060 → 91.6% val. Note: torchvision S3D was
  evaluated and **removed** — its final `avg_pool3d(2,7,7)` crashes on 112px clips.
- Sliding-window inference (`src/inference_video_3d.py`): buffer 16 frames, run the
  net every 4th frame, average recent clip probabilities.

### 4.3 YOLO detection + action head (`src/yolo_action.py`)
- YOLOv8n detects persons (class 0); each box crop is classified by the 2D model;
  full-frame fallback when nobody is found. Adjustable confidence threshold.

### 4.4 Web app (`src/app.py`)
- Gradio "Noir Lab" dark UI: aurora hero, bento status cards with accuracy rings,
  verdict banners with animated confidence meters, per-person results table,
  confidence slider, examples, webcam/clipboard input, `SKIP_3D=1` lite mode.

## 5. Experiments and Results

### 5.1 2D results (test set, n=1890)

| Model | Epochs | Val/Test acc | TTA |
|---|---|---|---|
| MobileNetV3-Small | 12 | 74.02% | — |
| EfficientNet-B0 | 25 | **82.75%** | 83.49% TTA |
| EfficientNet-B3 @300px | 23 so far | **85.24%** | RandAugment + MixUp |

Per-class F1 (EfficientNet-B3): cycling 0.984, eating 0.949, dancing 0.887, sleeping 0.887,
running 0.882, drinking 0.877, hugging 0.867, laughing 0.864, fighting 0.849, clapping 0.846,
using_laptop 0.799, listening_to_music 0.793, texting 0.789, calling 0.777, sitting 0.706.
Confusable seated/phone poses (sitting, calling, texting) remain the hardest — a data limit,
not just a model limit.

### 5.2 3D results (KTH validation)
- R3D-18, 10 epochs: train 99.8% / **val 91.6%**.
- App spot checks: boxing clip → boxing 99.9%; cycling photo → cycling 99.9%.

### 5.3 Honest ceiling
95% top-1 on HAR-15 is beyond what the data supports (near-identical classes +
web-scrape label noise). Realistic path: B3 @300px + RandAugment + MixUp + TTA +
ensemble ≈ high-80s to ~91%. Reported numbers are measured, not projected.

## 6. How to Reproduce

```powershell
pip install -r requirements.txt
pip install --force-reinstall --no-cache-dir "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2"
# Kaggle token -> $env:USERPROFILE\.kaggle\kaggle.json
python -m src.download_data
python -m src.train --arch efficientnet_b0 --epochs 25 --batch-size 64 --out checkpoints/best_effb0.pth
python -m src.evaluate --checkpoint checkpoints/best_effb0.pth --tta
python -m src.download_data --dataset vafaeii/kth-action-recognition-dataset --dest data_kth
python -m src.train_video --data data_kth --epochs 10
python -m src.app
```

## 7. Inference Cheat Sheet

```powershell
python -m src.infer --image photo.jpg
python -m src.infer --video in.mp4 --out out.mp4
python -m src.infer --webcam
python -m src.yolo_action --source 0 --out out.mp4
python -m src.inference_video_3d clip.avi -o out_3d.mp4
python -m src.app --share   # public link
```

## 8. Deployment
- **Hugging Face Spaces** (free): README carries the Spaces frontmatter; enable Git
  LFS for `*.pth`; set secret `SKIP_3D=1` for CPU lite mode.
- **Temporary link**: `python -m src.app --share`.
- **Docker/GPU VM** (RunPod etc.): `Dockerfile` (CUDA 12.8) + `.dockerignore` included.

## 9. Repository Layout

```
src/config.py  model.py  model_3d.py  dataset.py  video_dataset.py
    download_data.py  train.py  train_video.py  evaluate.py
    infer.py  yolo_action.py  inference_video_3d.py  app.py
data/  data_kth/  checkpoints/  docs/  Dockerfile  README.md
Human_Action_Detection_Report.pdf
```

## 10. Limitations and Future Work
- 2D model sees single frames (no motion); 3D model covers only 6 KTH actions.
- YOLO crops inherit 2D confusions on small/occluded people.
- Next: finish B3 schedule + ensemble, KTH cross-subject
  split eval; pose-based branch (MediaPipe/ST-GCN); ONNX export for edge serving.
