# InnoCount Skeletal Keypoint Detection

A reproducible computer vision pipeline for **structural steel keypoint
detection** using **YOLO26-Pose**. This repository contains the frozen
dataset, dataset conversion utilities, validation scripts, quality
assurance tools, and baseline training code.

## Overview

The project follows a strict **freeze → validate → convert → train**
workflow to ensure the original COCO annotations are never modified.

### Pipeline

1.  Create frozen COCO dataset (`Frozen_v2`)
2.  Optimize train/validation/test split by source group
3.  Validate class-specific keypoint schema
4.  Generate ModelReady adapter (global K=6)
5.  Convert to YOLO26-Pose format
6.  Train YOLO26-Pose baseline
7.  Evaluate trained models (future experiments)

------------------------------------------------------------------------

## Dataset

### Object classes

  ID   Class          Keypoints
  ---- ------------ -----------
  0    Channel                4
  1    T-beam                 4
  2    HI-Beam                6
  3    Angle Bar              3
  4    Sheet Pile             4

### Frozen dataset statistics

  Item                 Value
  ------------- ------------
  Images                 742
  Annotations         15,377
  Train           594 images
  Validation       74 images
  Test             74 images

The dataset split is performed **by source group**, preventing data
leakage between training and evaluation.

------------------------------------------------------------------------

## Repository structure

``` text
.
├── datasets/
│   ├── InnoCount_KeyPoint_Frozen_v2/
│   ├── InnoCount_KeyPoint_ModelReady_v1/
│   └── InnoCount_KeyPoint_YOLO26_Pose_v1/
│
├── schema/
│   └── InnoCount_Keypoint_Schema_v2_FROZEN/
│
├── scripts/
│   ├── create_frozen_dataset.py
│   ├── optimize_frozen_split_v2.py
│   ├── validate_keypoint_schema.py
│   ├── create_model_ready_adapter.py
│   ├── create_yolo26_pose_dataset.py
│   ├── inspect_keypoint_schema.py
│   ├── inspect_yolo_pose.py
│   ├── train_yolo26_pose.py
│   └── qa/
│       ├── create_keypoint_integrity_qa.py
│       ├── create_visual_keypoint_qa.py
│       └── create_semantic_keypoint_inspection.py
│
├── docs/
│   ├── reports/
│   ├── semantic_inspection/
│   └── visual_qa/
│
├── experiments/
│   └── YOLO26_Pose_Runs/
│
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

## Installation

Create the Conda environment:

``` bash
conda env create -f environment.yml
conda activate innocount-freeze
```

Install PyTorch with CUDA 12.8 (if not already installed):

``` bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

``` bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

> **Note:** The pretrained checkpoint `yolo26n-pose.pt` is **not
> included** in this repository. Ultralytics will automatically download
> it during the first training run if it is unavailable.

------------------------------------------------------------------------

# Reproducible workflow

## Step 1 --- Create the frozen dataset

``` bash
python scripts/create_frozen_dataset.py
```

Output:

``` text
datasets/InnoCount_KeyPoint_Frozen_v2/
```

## Step 2 --- Optimize dataset split

``` bash
python scripts/optimize_frozen_split_v2.py
```

Output:

``` text
datasets/InnoCount_KeyPoint_Frozen_v2/reports/
```

## Step 3 --- Validate the keypoint schema

``` bash
python scripts/validate_keypoint_schema.py \
    --dataset datasets/InnoCount_KeyPoint_Frozen_v2
```

Expected result:

``` text
PASS: all splits match the class-specific schema.
```

## Step 4 --- Create the ModelReady adapter

``` bash
python scripts/create_model_ready_adapter.py
```

Output:

``` text
datasets/InnoCount_KeyPoint_ModelReady_v1/
```

The original COCO keypoints remain available in
`annotation.original_keypoints`.

## Step 5 --- Convert to YOLO26-Pose

``` bash
python scripts/create_yolo26_pose_dataset.py
```

Output:

``` text
datasets/InnoCount_KeyPoint_YOLO26_Pose_v1/
```

Expected result:

``` text
PASS: YOLO26-Pose dataset conversion and validation passed.
```

## Step 6 --- Inspect the converted labels (optional)

``` bash
python scripts/inspect_yolo_pose.py
```

Inspect the semantic keypoint ordering:

``` bash
python scripts/inspect_keypoint_schema.py
```

## Step 7 --- Train the baseline model

``` bash
python scripts/train_yolo26_pose.py \
    --epochs 100 \
    --batch 4 \
    --imgsz 640 \
    --device 0
```

Outputs are written to:

``` text
experiments/YOLO26_Pose_Runs/E01_YOLO26n_Baseline/
```

The experiment folder contains:

-   `best.pt` --- best-performing trained model
-   `last.pt` --- final checkpoint
-   `results.csv` --- training metrics
-   `results.png` --- learning curves
-   Confusion matrices
-   Precision--Recall curves
-   Validation prediction visualizations

------------------------------------------------------------------------

# Quality assurance (optional)

These scripts regenerate the documentation figures stored in `docs/`.

## Integrity report

``` bash
python scripts/qa/create_keypoint_integrity_qa.py
```

Output:

``` text
docs/reports/InnoCount_Keypoint_Integrity_QA_v2/
```

## Visual QA samples

``` bash
python scripts/qa/create_visual_keypoint_qa.py
```

Output:

``` text
docs/visual_qa/InnoCount_KeyPoint_Visual_QA/
```

## Semantic inspection sheets

``` bash
python scripts/qa/create_semantic_keypoint_inspection.py
```

Output:

``` text
docs/semantic_inspection/InnoCount_Semantic_Keypoint_Inspection/
```

------------------------------------------------------------------------

## Design principles

-   **Frozen dataset:** Original COCO annotations are never modified.
-   **Leakage-free:** Dataset splitting preserves complete source
    groups.
-   **Schema-validated:** Every annotation is verified against the
    frozen keypoint specification.
-   **Traceable:** Every conversion stage produces validation reports
    and metadata.
-   **Model-ready:** Class-specific keypoints are mapped into a unified
    global K=6 representation.
-   **Reproducible:** Fixed random seed and deterministic training
    configuration are used throughout the baseline experiment.

------------------------------------------------------------------------

## Baseline experiment

  Item         Value
  ------------ ----------------------------
  Model        YOLO26n-Pose
  Input size   640 × 640
  Epochs       100
  Batch size   4
  Optimizer    Ultralytics Auto
  GPU          NVIDIA RTX 2080 Ti (11 GB)
  CUDA         12.8
  PyTorch      2.11.0

The baseline experiment is stored under:

``` text
experiments/YOLO26_Pose_Runs/E01_YOLO26n_Baseline/
```

------------------------------------------------------------------------

## License

This repository is intended for **academic and research purposes**. If
you use the dataset, annotation pipeline, or training framework in
published work, please acknowledge or cite this project appropriately.
