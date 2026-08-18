#!/usr/bin/env python3

"""
InnoCount KeyPoint - YOLO26-Pose Baseline Training

Purpose
-------
Train a YOLO26-Pose baseline on the frozen InnoCount dataset.

Dataset:
    InnoCount_KeyPoint_YOLO26_Pose_v1

Keypoints:
    Global K = 6

Classes:
    0 Channel
    1 T-beam
    2 HI-Beam
    3 Angle Bar
    4 Sheet Pile

This script intentionally does NOT introduce custom augmentation.
The first run is intended to be a clean experimental baseline.

The frozen dataset and labels are never modified.
"""

from pathlib import Path
import argparse
import json
import platform
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# EXPECTED DATASET
# ============================================================

CLASS_NAMES = [
    "Channel",
    "T-beam",
    "HI-Beam",
    "Angle Bar",
    "Sheet Pile",
]

EXPECTED_KPT_SHAPE = [6, 3]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def get_system_info():

    info = {
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }

    if torch.cuda.is_available():

        info["gpu_count"] = torch.cuda.device_count()

        info["gpus"] = []

        for i in range(torch.cuda.device_count()):

            props = torch.cuda.get_device_properties(i)

            info["gpus"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb":
                    round(props.total_memory / (1024 ** 3), 2),
                "compute_capability":
                    f"{props.major}.{props.minor}",
            })

    return info


def print_system_info(info):

    print()
    print("=" * 70)
    print("SYSTEM")
    print("=" * 70)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {info['pytorch']}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"CUDA version : {info['cuda_version']}")

    if info["cuda_available"]:

        for gpu in info["gpus"]:

            print(
                f"GPU {gpu['index']}        : "
                f"{gpu['name']} "
                f"({gpu['total_memory_gb']} GB)"
            )

    print()


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset_yaml(data_yaml: Path):

    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found:\n{data_yaml}")

    print(f"Dataset YAML : {data_yaml}")

    import yaml

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    kpt_shape = data.get("kpt_shape")

    # Support both list and dict YAML formats
    if isinstance(names, dict):
        ordered_names = [names[i] for i in sorted(names.keys())]
    elif isinstance(names, list):
        ordered_names = names
    else:
        raise ValueError("Invalid 'names' field in dataset YAML.")

    nc = data.get("nc")
    if nc is None:
        nc = len(ordered_names)

    print(f"Classes      : {ordered_names}")
    print(f"nc           : {nc}")
    print(f"kpt_shape    : {kpt_shape}")

    if nc != 5:
        raise ValueError(f"Expected nc=5, got {nc}")

    if kpt_shape != EXPECTED_KPT_SHAPE:
        raise ValueError(
            f"Expected kpt_shape={EXPECTED_KPT_SHAPE}, got {kpt_shape}"
        )

    if ordered_names != CLASS_NAMES:
        raise ValueError(
            "Class names do not match expected schema.\n"
            f"Expected: {CLASS_NAMES}\n"
            f"Found:    {ordered_names}"
        )

    print()
    print("PASS: dataset schema is correct.")
    print()


# ============================================================
# DATASET STRUCTURE
# ============================================================

def validate_dataset_structure(dataset_dir: Path):

    print("=" * 70)
    print("DATASET STRUCTURE")
    print("=" * 70)

    for split in ["train", "valid", "test"]:

        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split

        if not image_dir.exists():

            raise FileNotFoundError(
                f"Missing image directory: {image_dir}"
            )

        if not label_dir.exists():

            raise FileNotFoundError(
                f"Missing label directory: {label_dir}"
            )

        image_files = [
            p for p in image_dir.iterdir()
            if p.is_file()
        ]

        label_files = [
            p for p in label_dir.iterdir()
            if p.suffix.lower() == ".txt"
        ]

        print(
            f"{split:5s}: "
            f"{len(image_files):5d} images, "
            f"{len(label_files):5d} labels"
        )

    print()
    print("PASS: dataset directories exist.")
    print()


# ============================================================
# LABEL SANITY CHECK
# ============================================================

def validate_labels(dataset_dir: Path):

    print("=" * 70)
    print("LABEL SANITY CHECK")
    print("=" * 70)

    total_objects = 0

    for split in ["train", "valid", "test"]:

        label_dir = dataset_dir / "labels" / split

        split_objects = 0

        for label_file in sorted(label_dir.glob("*.txt")):

            with open(label_file, "r", encoding="utf-8") as f:

                for line_no, line in enumerate(f, 1):

                    line = line.strip()

                    if not line:
                        continue

                    values = line.split()

                    # 1 class + 4 bbox + 18 keypoint values
                    if len(values) != 23:

                        raise ValueError(
                            f"{label_file}:{line_no}: "
                            f"expected 23 values, "
                            f"got {len(values)}"
                        )

                    cls = int(values[0])

                    if cls < 0 or cls >= 5:

                        raise ValueError(
                            f"{label_file}:{line_no}: "
                            f"invalid class {cls}"
                        )

                    # bbox
                    bbox = [
                        float(v)
                        for v in values[1:5]
                    ]

                    for value in bbox:

                        if not 0.0 <= value <= 1.0:

                            raise ValueError(
                                f"{label_file}:{line_no}: "
                                f"bbox value outside [0,1]"
                            )

                    # keypoints
                    for i in range(6):

                        x = float(values[5 + i * 3])
                        y = float(values[5 + i * 3 + 1])
                        v = int(float(values[5 + i * 3 + 2]))

                        if v not in (0, 1, 2):

                            raise ValueError(
                                f"{label_file}:{line_no}: "
                                f"K{i+1} invalid visibility {v}"
                            )

                        if v == 0:

                            if x != 0.0 or y != 0.0:

                                raise ValueError(
                                    f"{label_file}:{line_no}: "
                                    f"masked keypoint K{i+1} "
                                    f"has non-zero coordinates"
                                )

                        else:

                            if not 0.0 <= x <= 1.0:
                                raise ValueError(
                                    f"{label_file}:{line_no}: "
                                    f"K{i+1} x outside [0,1]"
                                )

                            if not 0.0 <= y <= 1.0:
                                raise ValueError(
                                    f"{label_file}:{line_no}: "
                                    f"K{i+1} y outside [0,1]"
                                )

                    split_objects += 1

        print(
            f"{split:5s}: "
            f"{split_objects:6d} objects"
        )

        total_objects += split_objects

    print()
    print(f"Total objects: {total_objects}")
    print()
    print("PASS: YOLO labels are structurally valid.")
    print()


# ============================================================
# MODEL INFORMATION
# ============================================================

def inspect_model(model):

    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print(f"Model: {model.ckpt_path if hasattr(model, 'ckpt_path') else 'loaded'}")

    if hasattr(model, "model"):

        print(
            f"Task : {getattr(model, 'task', 'unknown')}"
        )

    print()


# ============================================================
# TRAINING
# ============================================================

def train(args):

    dataset_dir = Path(args.dataset).resolve()
    data_yaml = Path(args.data).resolve()

    # make Ultralytics interpret path: . correctly
    os.chdir(data_yaml.parent)

    output_dir = Path(args.project).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(args.seed)

    # --------------------------------------------------------
    # System information
    # --------------------------------------------------------

    system_info = get_system_info()

    print_system_info(system_info)

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    validate_dataset_yaml(data_yaml)
    validate_dataset_structure(dataset_dir)
    validate_labels(dataset_dir)

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": str(dataset_dir),
        "dataset_yaml": str(data_yaml),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "workers": args.workers,
        "patience": args.patience,
        "pretrained": True,
        "augmentation_policy": "baseline",
        "additional_custom_augmentation": False,
        "system": system_info,
    }

    run_dir = output_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = run_dir / "experiment_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Experiment metadata: {metadata_path}"
    )

    # --------------------------------------------------------
    # Load pretrained YOLO26-Pose
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING YOLO26-POSE")
    print("=" * 70)

    print(f"Model: {args.model}")

    model = YOLO(args.model)

    inspect_model(model)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("=" * 70)
    print("STARTING BASELINE TRAINING")
    print("=" * 70)

    print()
    print("Dataset:")
    print(f"  {dataset_dir}")

    print()
    print("Training configuration:")
    print(f"  model       = {args.model}")
    print(f"  epochs      = {args.epochs}")
    print(f"  imgsz       = {args.imgsz}")
    print(f"  batch       = {args.batch}")
    print(f"  device      = {args.device}")
    print(f"  seed        = {args.seed}")
    print(f"  workers     = {args.workers}")
    print(f"  patience    = {args.patience}")

    print()
    print("Baseline policy:")
    print("  No custom augmentation")
    print("  No synthetic class balancing")
    print("  Frozen train/valid/test split")
    print()

    results = model.train(

        data=str(data_yaml),

        epochs=args.epochs,

        imgsz=args.imgsz,

        batch=args.batch,

        device=args.device,

        workers=args.workers,

        seed=args.seed,

        patience=args.patience,

        project=str(output_dir),

        name=args.name,

        resume=args.resume,

        exist_ok=False,

        pretrained=True,

        # ----------------------------------------------------
        # Baseline augmentation
        # ----------------------------------------------------
        #
        # These are deliberately not specified here.
        #
        # Ultralytics defaults therefore remain active.
        #
        # We are NOT adding the nine custom augmentations
        # previously discussed.
        #
        # ----- Freeze augmentation -----
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.0,

        # --------------------------------
        # ----------------------------------------------------
        # Save / validation
        # ----------------------------------------------------

        save=True,

        val=True,

        plots=True,

        verbose=True,
    )

    # --------------------------------------------------------
    # Training completed
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved under:\n"
        f"{output_dir / args.name}"
    )

    return results


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train YOLO26-Pose baseline."
    )

    parser.add_argument(
        "--dataset",
        default="InnoCount_KeyPoint_YOLO26_Pose_v1",
        help="YOLO dataset directory",
    )

    parser.add_argument(
        "--data",
        default=(
            "InnoCount_KeyPoint_YOLO26_Pose_v1/"
            "dataset.yaml"
        ),
        help="YOLO dataset YAML",
    )

    parser.add_argument(
        "--model",
        default="yolo26n-pose.pt",
        help="YOLO26-Pose pretrained checkpoint",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPU device, e.g. 0 or cpu",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--project",
        default="experiments/YOLO26_Pose_Runs",
    )

    parser.add_argument(
        "--name",
        default="InnoCount_YOLO26nPose_baseline_v1",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
