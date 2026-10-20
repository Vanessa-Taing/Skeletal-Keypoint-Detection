#!/usr/bin/env python3
"""
Qualitative out-of-domain inference for a trained YOLO26-Pose checkpoint.

Runs a steel-profile keypoint model on a sample of container-defect images,
saves annotated images, and exports predictions to CSV.

IMPORTANT:
- This is a qualitative transfer test, NOT a valid defect-keypoint evaluation.
- The checkpoint predicts the steel-profile classes and its original keypoint
  schema. It has not been trained to recognise container defects.
- Meaningful defect measurement requires a defect-specific keypoint schema,
  annotations, and fine-tuning/evaluation.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(source: Path):
    if source.is_file() and source.suffix.lower() in IMAGE_EXTS:
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(
            f"Source must be an image or an extracted dataset directory: {source}"
        )
    return sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO26-Pose on sampled container-defect images."
    )
    parser.add_argument(
        "--weights", required=True,
        help="Path to E04 best.pt checkpoint."
    )
    parser.add_argument(
        "--source", required=True,
        help="One image or directory containing extracted dataset images."
    )
    parser.add_argument(
        "--output", default="runs/defect_ood_E04",
        help="Output directory for annotated images and predictions.csv."
    )
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of images to sample (default: 50).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0",
                        help="GPU index such as 0, or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-all-images", action="store_true",
        help="Also save source images where no detections were produced."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    all_images = collect_images(source)
    if not all_images:
        raise RuntimeError(f"No supported image files found under: {source}")

    random.seed(args.seed)
    sample_count = min(max(1, args.samples), len(all_images))
    sampled = random.sample(all_images, sample_count) if len(all_images) > sample_count else all_images

    print(f"Checkpoint: {weights}")
    print(f"Source:     {source}")
    print(f"Found {len(all_images)} images; testing {len(sampled)}.")
    print("WARNING: this is out-of-domain inference. Class names/keypoints retain")
    print("the steel-profile training meaning; do not interpret them as defect labels.\n")

    model = YOLO(str(weights))
    class_names = model.names
    print(f"Checkpoint class names: {class_names}")

    csv_path = out_dir / "predictions.csv"
    rows = []

    for i, image_path in enumerate(sampled, start=1):
        print(f"[{i}/{len(sampled)}] {image_path.name}")
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        if not results:
            continue

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        # Ultralytics' plot() overlays detected boxes, class labels and pose points.
        if boxes is not None and len(boxes) > 0:
            plotted = result.plot()
            target = annotated_dir / f"{i:03d}_{image_path.stem}_pred.jpg"
            cv2.imwrite(str(target), plotted)

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            kp_data = None
            if keypoints is not None and keypoints.data is not None:
                kp_data = keypoints.data.cpu().numpy()

            for det_idx, (box, box_conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
                kpts = []
                if kp_data is not None and det_idx < len(kp_data):
                    for kp_idx, kp in enumerate(kp_data[det_idx]):
                        # For pose results, the third value is normally keypoint confidence.
                        point = {
                            "index_1_based": kp_idx + 1,
                            "x_px": float(kp[0]),
                            "y_px": float(kp[1]),
                        }
                        if len(kp) >= 3:
                            point["confidence"] = float(kp[2])
                        kpts.append(point)

                rows.append({
                    "image": str(image_path),
                    "detection_index": det_idx + 1,
                    "predicted_class_id": int(cls_id),
                    "predicted_class_name": str(class_names.get(int(cls_id), cls_id))
                        if isinstance(class_names, dict)
                        else str(class_names[int(cls_id)]),
                    "box_confidence": float(box_conf),
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "keypoints_json": json.dumps(kpts),
                    "annotated_image": str(target),
                })
        else:
            if args.save_all_images:
                original = cv2.imread(str(image_path))
                if original is not None:
                    target = annotated_dir / f"{i:03d}_{image_path.stem}_no_detection.jpg"
                    cv2.imwrite(str(target), original)
            rows.append({
                "image": str(image_path),
                "detection_index": "",
                "predicted_class_id": "",
                "predicted_class_name": "",
                "box_confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "keypoints_json": "[]",
                "annotated_image": "",
            })

    fieldnames = [
        "image", "detection_index", "predicted_class_id", "predicted_class_name",
        "box_confidence", "x1", "y1", "x2", "y2", "keypoints_json",
        "annotated_image",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detected_images = sum(
        1 for r in rows if r.get("detection_index") not in ("", None)
    )
    print("\nFinished.")
    print(f"Annotated images: {annotated_dir}")
    print(f"Prediction CSV:   {csv_path}")
    print(f"Images with >=1 detection: {detected_images} / {len(sampled)}")
    print("\nInterpretation reminder: a prediction is not evidence of defect recognition.")
    print("Inspect whether the predicted steel-profile keypoints align meaningfully")
    print("with visible defect geometry before deciding on a defect-specific schema.")


if __name__ == "__main__":
    main()
