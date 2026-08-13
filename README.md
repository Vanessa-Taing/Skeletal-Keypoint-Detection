# InnoCount Skeletal Keypoint Detection

A reproducible computer vision pipeline for **structural steel keypoint detection** using **YOLO26-Pose**. This repository contains the frozen dataset, dataset conversion utilities, validation scripts, and baseline training code.

## Overview

The project follows a strict **freeze → validate → convert → train** workflow to ensure the original annotations are never modified.

**Pipeline**

1. Create frozen COCO dataset (`Frozen_v2`)
2. Optimize train/valid/test split by source group
3. Validate class-specific keypoint schema
4. Generate ModelReady adapter (global K=6)
5. Convert to YOLO26-Pose format
6. Train YOLO26-Pose baseline

---

## Dataset

### Object classes

| ID | Class | Keypoints |
|---|---|---:|
| 0 | Channel | 4 |
| 1 | T-beam | 4 |
| 2 | HI-Beam | 6 |
| 3 | Angle Bar | 3 |
| 4 | Sheet Pile | 4 |

### Frozen dataset statistics

| Item | Value |
|---|---:|
| Images | 742 |
| Annotations | 15,377 |
| Train | 594 images |
| Validation | 74 images |
| Test | 74 images |

The dataset split is performed **by source group**, preventing data leakage between splits.

---

## Repository structure

```text
.
├── InnoCount_KeyPoint_Frozen_v2/
│   ├── splits/
│   └── reports/
│
├── InnoCount_KeyPoint_ModelReady_v1/
│   ├── splits/
│   └── reports/
│
├── InnoCount_KeyPoint_YOLO26_Pose_v1/
│   ├── images/
│   ├── labels/
│   ├── dataset.yaml
│   └── reports/
│
├── create_frozen_dataset.py
├── optimize_frozen_split_v2.py
├── validate_keypoint_schema.py
├── create_model_ready_adapter.py
├── create_yolo26_pose_dataset.py
├── inspect_yolo_pose.py
├── train_yolo26_pose.py
├── requirements.txt
└── environment.yml
```

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate innocount-freeze
```

Install PyTorch with CUDA 12.8 if required:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## Validation workflow

### 1. Validate keypoint schema

```bash
python validate_keypoint_schema.py \
    --dataset InnoCount_KeyPoint_Frozen_v2
```

Expected result:

```text
PASS: all splits match the class-specific schema.
```

### 2. Create ModelReady adapter

```bash
python create_model_ready_adapter.py
```

This creates a **global 6-keypoint representation** while preserving the original coordinates in `original_keypoints`.

### 3. Convert to YOLO26-Pose

```bash
python create_yolo26_pose_dataset.py
```

Expected result:

```text
PASS: YOLO26-Pose dataset conversion and validation passed.
```

---

## Training

Train the baseline model:

```bash
python train_yolo26_pose.py \
    --epochs 100 \
    --batch 4 \
    --imgsz 640 \
    --device 0
```

Training outputs are written to:

```text
YOLO26_Pose_Runs/
```

---

## Design principles

- **Frozen dataset:** Original COCO annotations are never modified.
- **Reproducible:** Fixed random seed and deterministic training settings.
- **Leakage-free:** Split optimization preserves complete source groups.
- **Traceable:** Every conversion stage produces validation reports.
- **Model-ready:** Class-specific keypoints are mapped into a unified global K=6 representation.

---

## License

This repository is intended for academic and research purposes. Please cite or acknowledge the project if the dataset or pipeline is used in published work.