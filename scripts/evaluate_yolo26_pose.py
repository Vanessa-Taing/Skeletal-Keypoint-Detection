#!/usr/bin/env python3
"""
InnoCount KeyPoint - YOLO26-Pose Test Evaluation

Evaluate a trained YOLO26-Pose model on the frozen test split.

Outputs:
    experiments/.../evaluation/
        evaluation_metrics.json
        evaluation_report.txt
        predictions/
"""

from pathlib import Path
import argparse
import json
from datetime import datetime

import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def find_dataset_yaml(path_arg: str) -> Path:
    """
    Accept either:
      --data datasets/.../dataset.yaml
    or automatically locate dataset.yaml from repo root.
    """
    p = Path(path_arg)

    if p.exists():
        return p.resolve()

    repo = Path(__file__).resolve().parents[1]
    candidate = repo / "datasets" / "InnoCount_KeyPoint_YOLO26_Pose_v1" / "dataset.yaml"

    if candidate.exists():
        return candidate

    raise FileNotFoundError("dataset.yaml not found.")


def make_report(metrics, save_dir, weights_path, data_yaml):

    box = metrics.box
    pose = metrics.pose

    report = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(weights_path),
        "dataset": str(data_yaml),
        "box": {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "pose": {
            "map50": float(pose.map50),
            "map50_95": float(pose.map),
            "precision": float(pose.mp),
            "recall": float(pose.mr),
        },
    }

    json_path = save_dir / "evaluation_metrics.json"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    txt_path = save_dir / "evaluation_report.txt"

    with open(txt_path, "w") as f:

        f.write("YOLO26-POSE TEST EVALUATION\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Checkpoint : {weights_path}\n")
        f.write(f"Dataset    : {data_yaml}\n")
        f.write(f"Timestamp  : {report['timestamp']}\n\n")

        f.write("BOX DETECTION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Precision : {box.mp:.4f}\n")
        f.write(f"Recall    : {box.mr:.4f}\n")
        f.write(f"mAP@50    : {box.map50:.4f}\n")
        f.write(f"mAP50-95  : {box.map:.4f}\n\n")

        f.write("POSE ESTIMATION\n")
        f.write("-" * 20 + "\n")
        f.write(f"Precision : {pose.mp:.4f}\n")
        f.write(f"Recall    : {pose.mr:.4f}\n")
        f.write(f"mAP@50    : {pose.map50:.4f}\n")
        f.write(f"mAP50-95  : {pose.map:.4f}\n")

    return report


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(args):

    weights = Path(args.weights).resolve()

    if not weights.exists():
        raise FileNotFoundError(weights)

    data_yaml = find_dataset_yaml(args.data)

    print("=" * 70)
    print("YOLO26-POSE TEST EVALUATION")
    print("=" * 70)
    print(f"Weights : {weights}")
    print(f"Dataset : {data_yaml}")
    print(f"Device  : {args.device}")
    print()

    # Use dataset folder as working directory so path: . works everywhere
    import os
    os.chdir(data_yaml.parent)

    model = YOLO(str(weights))

    metrics = model.val(
        data="dataset.yaml",
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True,
        save_json=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir)
    report = make_report(metrics, save_dir, weights, data_yaml)

    # -----------------------------------------------------
    # Export prediction images
    # -----------------------------------------------------

    pred_dir = save_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    test_img_dir = data_yaml.parent / "images" / "test"

    model.predict(
        source=str(test_img_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(pred_dir),
        name="images",
        exist_ok=True,
        verbose=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nBox Detection")
    print(f"Precision : {report['box']['precision']:.4f}")
    print(f"Recall    : {report['box']['recall']:.4f}")
    print(f"mAP@50    : {report['box']['map50']:.4f}")
    print(f"mAP50-95  : {report['box']['map50_95']:.4f}")

    print("\nPose Estimation")
    print(f"Precision : {report['pose']['precision']:.4f}")
    print(f"Recall    : {report['pose']['recall']:.4f}")
    print(f"mAP@50    : {report['pose']['map50']:.4f}")
    print(f"mAP50-95  : {report['pose']['map50_95']:.4f}")

    print("\nSaved to:")
    print(save_dir)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

def parse_args():

    repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        required=True,
        help="best.pt checkpoint",
    )

    parser.add_argument(
        "--data",
        default=str(
            repo /
            "datasets" /
            "InnoCount_KeyPoint_YOLO26_Pose_v1" /
            "dataset.yaml"
        ),
    )

    parser.add_argument(
        "--project",
        default=str(
            repo /
            "experiments" /
            "YOLO26_Pose_Runs"
        ),
    )

    parser.add_argument(
        "--name",
        default="E01_YOLO26n_Baseline_TestEval",
    )

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)

    return parser.parse_args()


def main():
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
