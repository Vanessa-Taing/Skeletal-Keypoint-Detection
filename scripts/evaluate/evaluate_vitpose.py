#!/usr/bin/env python3
"""Evaluate InnoCount ViTPose-B on the frozen test split using GT boxes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "evaluate"))

from common_pose_metrics import calculate_all_metrics, make_category_map_from_coco


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/innocount/td-hm_ViTPose-base_innocount.py",
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/ViTPose_Runs/E02_ViTPoseB_Full/epoch_100.pth",
    )
    parser.add_argument(
        "--annotations",
        default="datasets/ViTPose_COCO/annotations/test.json",
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/ViTPose_COCO/images/test",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/ViTPose_Runs/E02_ViTPoseB_FinalTest",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    root = PROJECT_ROOT
    config_path = (root / args.config).resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    annotations_path = (root / args.annotations).resolve()
    image_dir = (root / args.image_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (config_path, checkpoint_path, annotations_path, image_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations = sorted(
        coco["annotations"],
        key=lambda ann: (int(ann["image_id"]), int(ann["id"])),
    )
    if len(images_by_id) != 74 or len(annotations) != 1626:
        raise ValueError(
            "Unexpected test split size: "
            f"{len(images_by_id)} images, {len(annotations)} annotations. "
            "Expected the frozen split (74 images, 1626 annotations)."
        )

    # Keep the full COCO annotation identity; annotation IDs may repeat across images.
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[int(ann["image_id"])].append(ann)

    from mmpose.apis import init_model, inference_topdown
    from mmpose.utils import register_all_modules

    register_all_modules()
    model = init_model(str(config_path), str(checkpoint_path), device=args.device)
    model.eval()

    predictions = []
    gt = []
    gt_vis = []
    bboxes = []
    category_ids = []
    image_ids = []
    annotation_ids = []
    prediction_rows = []

    with torch.no_grad():
        for image_id in sorted(annotations_by_image):
            image_info = images_by_id[image_id]
            image_path = image_dir / Path(image_info["file_name"]).name
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image referenced by COCO annotation not found: {image_path}"
                )

            anns = annotations_by_image[image_id]
            boxes_xyxy = []
            for ann in anns:
                x, y, w, h = map(float, ann["bbox"])
                if w <= 0 or h <= 0:
                    raise ValueError(
                        f"Invalid bbox for image_id={image_id}, "
                        f"annotation_id={ann['id']}: {ann['bbox']}"
                    )
                boxes_xyxy.append([x, y, x + w, y + h])

            samples = inference_topdown(
                model,
                str(image_path),
                bboxes=np.asarray(boxes_xyxy, dtype=np.float32),
                bbox_format="xyxy",
            )
            if len(samples) != len(anns):
                raise RuntimeError(
                    f"Inference returned {len(samples)} instances for "
                    f"{len(anns)} annotations in image_id={image_id}; "
                    "refusing to silently misalign predictions."
                )

            for ann, sample in zip(anns, samples):
                pred_instances = sample.pred_instances
                pred_xy = np.asarray(pred_instances.keypoints, dtype=np.float32)
                pred_xy = pred_xy.reshape(-1, 6, 2)[0]

                keypoints = np.asarray(ann["keypoints"], dtype=np.float32).reshape(6, 3)
                gt_xy = keypoints[:, :2]
                visibility = keypoints[:, 2]
                bbox = np.asarray(ann["bbox"], dtype=np.float32)

                if not np.isfinite(pred_xy).all():
                    raise RuntimeError(
                        f"Non-finite prediction for image_id={image_id}, "
                        f"annotation_id={ann['id']}"
                    )

                predictions.append(pred_xy)
                gt.append(gt_xy)
                gt_vis.append(visibility)
                bboxes.append(bbox)
                category_ids.append(int(ann["category_id"]))
                image_ids.append(int(image_id))
                annotation_ids.append(int(ann["id"]))

                prediction_rows.append({
                    "image_id": int(image_id),
                    "annotation_id": int(ann["id"]),
                    "category_id": int(ann["category_id"]),
                    "file_name": image_info["file_name"],
                    "bbox_xywh": bbox.tolist(),
                    "gt_keypoints_xyv": keypoints.tolist(),
                    "pred_keypoints_xy": pred_xy.tolist(),
                    "keypoint_scores": (
                        np.asarray(pred_instances.keypoint_scores, dtype=np.float32)
                        .reshape(-1, 6)[0].tolist()
                        if hasattr(pred_instances, "keypoint_scores")
                        else None
                    ),
                })

    predictions = np.asarray(predictions, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    gt_vis = np.asarray(gt_vis, dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    category_ids = np.asarray(category_ids, dtype=np.int64)
    image_ids = np.asarray(image_ids, dtype=np.int64)
    annotation_ids = np.asarray(annotation_ids, dtype=np.int64)

    n = len(annotations)
    if predictions.shape != (n, 6, 2) or gt.shape != (n, 6, 2):
        raise RuntimeError(
            f"Unexpected array shapes: predictions={predictions.shape}, gt={gt.shape}"
        )

    category_map = make_category_map_from_coco(coco)
    metrics = calculate_all_metrics(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=args.threshold,
    )

    np.save(output_dir / "vitpose_predictions.npy", predictions)
    np.save(output_dir / "vitpose_gt.npy", gt)
    np.save(output_dir / "vitpose_gt_visibility.npy", gt_vis)
    np.save(output_dir / "vitpose_bboxes.npy", bboxes)
    np.save(output_dir / "vitpose_image_ids.npy", image_ids)
    np.save(output_dir / "vitpose_category_ids.npy", category_ids)
    np.save(output_dir / "vitpose_annotation_ids.npy", annotation_ids)

    # Explicit composite identity audit because annotation IDs need not be globally unique.
    with (output_dir / "vitpose_prediction_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")

    report = {
        "model": "ViTPose-B",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "annotations": str(annotations_path),
        "image_dir": str(image_dir),
        "split": "frozen_v3_optimized_test",
        "num_images": len(images_by_id),
        "num_instances": n,
        "threshold": args.threshold,
        "bbox_source": "ground-truth COCO annotations (top-down pose evaluation)",
        "coordinate_space": "original image pixel coordinates",
        "identity": "composite (image_id, annotation_id)",
        "metrics": metrics,
    }
    with (output_dir / "vitpose_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    standard = metrics["standard_pck"]
    order_inv = metrics["order_invariant_pck"]
    edges = metrics["structural_edges"]
    summary_rows = [
        ("Standard fixed-index PCK@0.05", standard["overall"]),
        ("Order-invariant PCK@0.05", order_inv["overall"]),
        ("Structural edge accuracy@0.05", edges["overall"]),
    ]
    with (output_dir / "vitpose_metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    lines = [
        "ViTPose-B Final Test Evaluation",
        f"Images: {len(images_by_id)}",
        f"Instances: {n}",
        f"Threshold: {args.threshold}",
        "Boxes: ground-truth boxes",
        "",
        f"Standard fixed-index PCK@0.05: {standard['overall']:.6f} "
        f"({standard['total_correct']}/{standard['total_valid']})",
        f"Order-invariant PCK@0.05: {order_inv['overall']:.6f} "
        f"({order_inv['correct']}/{order_inv['total']})",
        f"Structural edge accuracy@0.05: {edges['overall']:.6f} "
        f"({edges['correct']}/{edges['total']})",
        "",
        "Profile PCK:",
    ]
    for profile, values in metrics["profile_pck"].items():
        lines.append(
            f"  {profile}: {values['pck']:.6f} "
            f"({values['correct']}/{values['valid']}; "
            f"{values['instances']} instances)"
        )
    lines += ["", f"Full report: {output_dir / 'vitpose_metrics.json'}"]
    text_report = "\n".join(lines) + "\n"
    (output_dir / "vitpose_metrics_summary.txt").write_text(
        text_report, encoding="utf-8"
    )
    print(text_report)


if __name__ == "__main__":
    main()
