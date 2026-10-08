#!/usr/bin/env python3
"""
InnoCount KeyPoint - YOLO26-Pose Profile-Aware Test Evaluation

Evaluates a trained YOLO26-Pose model on the frozen InnoCount
test split.

Two evaluation layers are produced:

1. Native Ultralytics metrics
   - Box Precision
   - Box Recall
   - Box mAP@50
   - Box mAP@50-95
   - Pose Precision
   - Pose Recall
   - Pose mAP@50
   - Pose mAP@50-95

2. Common thesis metrics
   - Standard fixed-index PCK@0.05
   - Profile-level PCK
   - Semantic-role PCK
   - Order-invariant PCK
   - Structural-edge accuracy

The common thesis metrics are implemented in:

    common_pose_metrics.py

IMPORTANT
---------
The frozen test annotations are loaded directly from:

    InnoCount_KeyPoint_ModelReady_v2/test/annotations.json

The YOLO test images are loaded from:

    InnoCount_KeyPoint_YOLO26_Pose_v2/images/test

No training data, validation data, or modified annotations are used.

YOLO instance matching
----------------------
Unlike HRNet, YOLO predicts a variable number of detected instances.

Therefore each YOLO prediction must be matched to a GT instance.

Matching is:

    1. same class
    2. maximum-IoU one-to-one assignment
    3. minimum IoU threshold

Unmatched GT instances remain in the evaluation and are counted
as localization failures rather than being discarded.

This prevents missed detections from artificially increasing PCK.

cd /home/owner/skeletal_keypoint_detection

python scripts/evaluate/evaluate_yolo26_pose.py \
    --weights "experiments/YOLO26_Pose_Runs/E06_YOLO26n_Corrected-2/weights/best.pt" \
    --data "datasets/InnoCount_KeyPoint_YOLO26_Pose_v2/dataset.yaml" \
    --dataset "datasets/InnoCount_KeyPoint_YOLO26_Pose_v2" \
    --annotation-file "datasets/InnoCount_KeyPoint_ModelReady_v2/test/annotations.json" \
    --project "experiments/YOLO26_Pose_Runs" \
    --name "E06_YOLO26n_Corrected-2_FinalTest" \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --threshold 0.05 \
    --match-iou 0.50 \
    --conf 0.001 \
    --nms-iou 0.70 \
    --max-det 300
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ============================================================
# IMPORT COMMON METRICS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )

from common_pose_metrics import (  # noqa: E402
    CLASS_ID_TO_NAME,
    CLASS_NAMES,
    GLOBAL_K,
    PROFILE_SCHEMAS,
    calculate_all_metrics,
    make_category_map_from_coco,
)


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(path_arg: str | Path) -> Path:
    return (
        Path(path_arg)
        .expanduser()
        .resolve()
    )


# ============================================================
# COCO LOADING
# ============================================================

def load_coco(annotation_file: Path):
    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_test_ground_truth(annotation_file: Path):
    """
    Load all GT instances from the exact frozen ModelReady
    test annotation file.

    Returns a dictionary keyed by image file name.
    """

    data = load_coco(annotation_file)

    images = data.get(
        "images",
        [],
    )

    annotations = data.get(
        "annotations",
        [],
    )

    image_by_id = {
        int(image["id"]): image
        for image in images
    }

    annotations_by_image = {}

    for annotation in annotations:

        image_id = int(
            annotation["image_id"]
        )

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    return (
        data,
        image_by_id,
        annotations_by_image,
    )


# ============================================================
# GT CONVERSION
# ============================================================

def annotation_to_gt(
    annotation,
    image,
):
    """
    Convert one ModelReady COCO annotation into:

        bbox_xyxy_pixels
        keypoints_xy_pixels
        visibility
    """

    image_w = float(
        image["width"]
    )

    image_h = float(
        image["height"]
    )

    x, y, w, h = map(
        float,
        annotation["bbox"],
    )

    bbox_xyxy = np.array(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        dtype=np.float32,
    )

    raw_keypoints = annotation[
        "keypoints"
    ]

    if len(raw_keypoints) != GLOBAL_K * 3:
        raise ValueError(
            f"Annotation {annotation['id']} "
            f"has {len(raw_keypoints)} keypoint values; "
            f"expected {GLOBAL_K * 3}."
        )

    keypoints = np.zeros(
        (GLOBAL_K, 2),
        dtype=np.float32,
    )

    visibility = np.zeros(
        GLOBAL_K,
        dtype=np.float32,
    )

    for j in range(GLOBAL_K):

        keypoints[j, 0] = float(
            raw_keypoints[j * 3]
        )

        keypoints[j, 1] = float(
            raw_keypoints[j * 3 + 1]
        )

        visibility[j] = float(
            raw_keypoints[j * 3 + 2]
        )

    # Basic sanity check.
    if image_w <= 0 or image_h <= 0:
        raise ValueError(
            f"Invalid image dimensions for "
            f"{image['file_name']}: "
            f"{image_w}x{image_h}"
        )

    return (
        bbox_xyxy,
        keypoints,
        visibility,
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou_matrix(
    gt_boxes,
    pred_boxes,
):
    """
    Calculate pairwise IoU matrix.

    Args:
        gt_boxes:
            [G,4] xyxy

        pred_boxes:
            [P,4] xyxy

    Returns:
        [G,P]
    """

    gt_boxes = np.asarray(
        gt_boxes,
        dtype=np.float32,
    )

    pred_boxes = np.asarray(
        pred_boxes,
        dtype=np.float32,
    )

    if len(gt_boxes) == 0:
        return np.zeros(
            (0, len(pred_boxes)),
            dtype=np.float32,
        )

    if len(pred_boxes) == 0:
        return np.zeros(
            (len(gt_boxes), 0),
            dtype=np.float32,
        )

    gt_x1 = gt_boxes[:, 0][:, None]
    gt_y1 = gt_boxes[:, 1][:, None]
    gt_x2 = gt_boxes[:, 2][:, None]
    gt_y2 = gt_boxes[:, 3][:, None]

    pred_x1 = pred_boxes[:, 0][None, :]
    pred_y1 = pred_boxes[:, 1][None, :]
    pred_x2 = pred_boxes[:, 2][None, :]
    pred_y2 = pred_boxes[:, 3][None, :]

    inter_x1 = np.maximum(
        gt_x1,
        pred_x1,
    )

    inter_y1 = np.maximum(
        gt_y1,
        pred_y1,
    )

    inter_x2 = np.minimum(
        gt_x2,
        pred_x2,
    )

    inter_y2 = np.minimum(
        gt_y2,
        pred_y2,
    )

    inter_w = np.maximum(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = np.maximum(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_w * inter_h
    )

    gt_area = (
        np.maximum(
            0.0,
            gt_x2 - gt_x1,
        )
        *
        np.maximum(
            0.0,
            gt_y2 - gt_y1,
        )
    )

    pred_area = (
        np.maximum(
            0.0,
            pred_x2 - pred_x1,
        )
        *
        np.maximum(
            0.0,
            pred_y2 - pred_y1,
        )
    )

    union = (
        gt_area
        + pred_area
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )

    return iou.astype(
        np.float32
    )


# ============================================================
# INSTANCE MATCHING
# ============================================================

def match_instances(
    gt_boxes,
    gt_classes,
    pred_boxes,
    pred_classes,
    pred_confidences,
    iou_threshold=0.50,
):
    """
    Match YOLO detections to GT instances.

    Matching is class-aware and one-to-one.

    For each class, Hungarian assignment maximizes IoU.

    Only pairs whose IoU >= iou_threshold are accepted.

    Returns:
        gt_to_pred
            dictionary:
                gt_index -> pred_index

        unmatched_gt
            list of GT indices

        unmatched_pred
            list of prediction indices
    """

    gt_boxes = np.asarray(
        gt_boxes,
        dtype=np.float32,
    )

    pred_boxes = np.asarray(
        pred_boxes,
        dtype=np.float32,
    )

    gt_classes = np.asarray(
        gt_classes,
        dtype=np.int64,
    )

    pred_classes = np.asarray(
        pred_classes,
        dtype=np.int64,
    )

    pred_confidences = np.asarray(
        pred_confidences,
        dtype=np.float32,
    )

    gt_to_pred = {}

    matched_pred = set()

    all_classes = sorted(
        set(gt_classes.tolist())
        |
        set(pred_classes.tolist())
    )

    for class_id in all_classes:

        gt_indices = [
            i
            for i, cls in enumerate(
                gt_classes
            )
            if int(cls) == int(class_id)
        ]

        pred_indices = [
            i
            for i, cls in enumerate(
                pred_classes
            )
            if int(cls) == int(class_id)
        ]

        if not gt_indices:
            continue

        if not pred_indices:
            continue

        class_gt_boxes = (
            gt_boxes[gt_indices]
        )

        class_pred_boxes = (
            pred_boxes[pred_indices]
        )

        iou = calculate_iou_matrix(
            class_gt_boxes,
            class_pred_boxes,
        )

        if iou.size == 0:
            continue

        # Hungarian minimizes cost.
        # Therefore use negative IoU.
        row_ind, col_ind = (
            linear_sum_assignment(
                -iou
            )
        )

        candidates = []

        for r, c in zip(
            row_ind,
            col_ind,
        ):

            gt_index = gt_indices[
                int(r)
            ]

            pred_index = pred_indices[
                int(c)
            ]

            score = float(
                iou[r, c]
            )

            candidates.append(
                (
                    score,
                    gt_index,
                    pred_index,
                )
            )

        # Deterministic ordering:
        # highest IoU first, then GT index,
        # then prediction index.
        candidates.sort(
            key=lambda x: (
                -x[0],
                x[1],
                x[2],
            )
        )

        for score, gt_index, pred_index in candidates:

            if score < iou_threshold:
                continue

            if pred_index in matched_pred:
                continue

            gt_to_pred[
                gt_index
            ] = pred_index

            matched_pred.add(
                pred_index
            )

    unmatched_gt = [
        i
        for i in range(len(gt_boxes))
        if i not in gt_to_pred
    ]

    unmatched_pred = [
        i
        for i in range(len(pred_boxes))
        if i not in matched_pred
    ]

    return (
        gt_to_pred,
        unmatched_gt,
        unmatched_pred,
    )


# ============================================================
# YOLO PREDICTION EXTRACTION
# ============================================================

def extract_yolo_predictions(
    result,
):
    """
    Extract YOLO pose predictions from one Ultralytics
    Results object.

    Returns:
        boxes_xyxy
        classes
        confidences
        keypoints_xy
    """

    if result.boxes is None:
        return (
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
            np.zeros(
                (0,),
                dtype=np.int64,
            ),
            np.zeros(
                (0,),
                dtype=np.float32,
            ),
            np.zeros(
                (0, GLOBAL_K, 2),
                dtype=np.float32,
            ),
        )

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if result.keypoints is None:

        keypoints = np.zeros(
            (
                len(boxes),
                GLOBAL_K,
                2,
            ),
            dtype=np.float32,
        )

    else:

        keypoints = (
            result.keypoints.xy
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if keypoints.ndim != 3:
        raise ValueError(
            "Unexpected YOLO keypoint shape: "
            f"{keypoints.shape}"
        )

    if keypoints.shape[1] != GLOBAL_K:
        raise ValueError(
            "YOLO model returned "
            f"{keypoints.shape[1]} keypoints; "
            f"expected {GLOBAL_K}."
        )

    return (
        boxes,
        classes,
        confidences,
        keypoints,
    )


# ============================================================
# BUILD ALIGNED EVALUATION ARRAYS
# ============================================================

def build_aligned_arrays(
    model,
    image_paths,
    image_by_id,
    annotations_by_image,
    iou_threshold,
    prediction_confidence,
    device,
    imgsz,
    max_det,
    verbose=True,
):
    """
    Run YOLO inference and construct arrays aligned one-to-one
    with every GT instance.

    Missing detections remain in the arrays with:

        prediction_valid = False

    This means missed detections count as localization failures.
    """

    all_predictions = []
    all_gt = []
    all_gt_vis = []
    all_bboxes = []
    all_category_ids = []
    all_image_ids = []
    all_annotation_ids = []

    matching_audit = []

    total_gt = 0
    total_pred = 0
    total_matched = 0

    results = model.predict(
        source=[
            str(path)
            for path in image_paths
        ],
        imgsz=imgsz,
        batch=1,
        device=device,
        conf=prediction_confidence,
        max_det=max_det,
        verbose=False,
        stream=False,
        save=False,
    )

    if len(results) != len(
        image_paths
    ):
        raise RuntimeError(
            "YOLO returned "
            f"{len(results)} results for "
            f"{len(image_paths)} images."
        )

    prediction_valid_rows = []

    for image_path, result in zip(
        image_paths,
        results,
    ):

        filename = image_path.name

        # ----------------------------------------------------
        # Find corresponding COCO image record
        # ----------------------------------------------------

        image_record = None

        for image in image_by_id.values():

            if Path(
                image["file_name"]
            ).name == filename:

                image_record = image
                break

        if image_record is None:
            raise KeyError(
                f"Could not find COCO image "
                f"record for {filename}"
            )

        image_id = int(
            image_record["id"]
        )

        gt_annotations = (
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_boxes = []
        gt_classes = []
        gt_keypoints = []
        gt_visibility = []
        gt_annotation_ids = []

        for annotation in gt_annotations:

            (
                bbox_xyxy,
                keypoints,
                visibility,
            ) = annotation_to_gt(
                annotation,
                image_record,
            )

            gt_boxes.append(
                bbox_xyxy
            )

            gt_classes.append(
                int(
                    annotation[
                        "category_id"
                    ]
                )
            )

            gt_keypoints.append(
                keypoints
            )

            gt_visibility.append(
                visibility
            )

            gt_annotation_ids.append(
                int(
                    annotation["id"]
                )
            )

        gt_boxes = np.asarray(
            gt_boxes,
            dtype=np.float32,
        ).reshape(
            (-1, 4)
        )

        gt_classes = np.asarray(
            gt_classes,
            dtype=np.int64,
        )

        gt_keypoints = np.asarray(
            gt_keypoints,
            dtype=np.float32,
        ).reshape(
            (-1, GLOBAL_K, 2)
        )

        gt_visibility = np.asarray(
            gt_visibility,
            dtype=np.float32,
        ).reshape(
            (-1, GLOBAL_K)
        )

        # ----------------------------------------------------
        # YOLO predictions
        # ----------------------------------------------------

        (
            pred_boxes,
            pred_classes,
            pred_confidences,
            pred_keypoints,
        ) = extract_yolo_predictions(
            result
        )

        total_gt += len(
            gt_boxes
        )

        total_pred += len(
            pred_boxes
        )

        # ----------------------------------------------------
        # Match
        # ----------------------------------------------------

        (
            gt_to_pred,
            unmatched_gt,
            unmatched_pred,
        ) = match_instances(
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_confidences=pred_confidences,
            iou_threshold=iou_threshold,
        )

        total_matched += len(
            gt_to_pred
        )

        # ----------------------------------------------------
        # Save matching audit
        # ----------------------------------------------------

        audit_matches = []

        for gt_index, pred_index in (
            gt_to_pred.items()
        ):

            iou_matrix = calculate_iou_matrix(
                gt_boxes[gt_index:gt_index + 1],
                pred_boxes[pred_index:pred_index + 1],
            )

            iou_value = float(
                iou_matrix[0, 0]
            )

            audit_matches.append(
                {
                    "gt_index": int(
                        gt_index
                    ),
                    "gt_annotation_id": int(
                        gt_annotation_ids[
                            gt_index
                        ]
                    ),
                    "prediction_index": int(
                        pred_index
                    ),
                    "class_id": int(
                        gt_classes[
                            gt_index
                        ]
                    ),
                    "class_name": CLASS_ID_TO_NAME.get(
                        int(
                            gt_classes[
                                gt_index
                            ]
                        ),
                        "unknown",
                    ),
                    "prediction_confidence": float(
                        pred_confidences[
                            pred_index
                        ]
                    ),
                    "iou": iou_value,
                }
            )

        matching_audit.append(
            {
                "image": filename,
                "image_id": image_id,
                "gt_instances": len(
                    gt_boxes
                ),
                "pred_instances": len(
                    pred_boxes
                ),
                "matched_instances": len(
                    gt_to_pred
                ),
                "unmatched_gt": [
                    int(x)
                    for x in unmatched_gt
                ],
                "unmatched_predictions": [
                    int(x)
                    for x in unmatched_pred
                ],
                "matches": audit_matches,
            }
        )

        # ----------------------------------------------------
        # One aligned row per GT instance
        # ----------------------------------------------------

        for gt_index in range(
            len(gt_boxes)
        ):

            all_gt.append(
                gt_keypoints[
                    gt_index
                ]
            )

            all_gt_vis.append(
                gt_visibility[
                    gt_index
                ]
            )

            x1, y1, x2, y2 = (
                gt_boxes[
                    gt_index
                ]
            )

            all_bboxes.append(
                [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                ]
            )

            all_category_ids.append(
                gt_classes[
                    gt_index
                ]
            )

            all_image_ids.append(
                image_id
            )

            all_annotation_ids.append(
                gt_annotation_ids[
                    gt_index
                ]
            )

            if gt_index in gt_to_pred:

                pred_index = gt_to_pred[
                    gt_index
                ]

                all_predictions.append(
                    pred_keypoints[
                        pred_index
                    ]
                )

                # A matched detection gives all six YOLO
                # keypoint outputs to the metric layer.
                prediction_valid = np.ones(
                    GLOBAL_K,
                    dtype=bool,
                )

            else:

                # Missing detection:
                #
                # Keep a dummy prediction but explicitly mark
                # all keypoints invalid so they count as failures.
                all_predictions.append(
                    np.zeros(
                        (GLOBAL_K, 2),
                        dtype=np.float32,
                    )
                )

                prediction_valid = np.zeros(
                    GLOBAL_K,
                    dtype=bool,
                )

            prediction_valid_rows.append(
                prediction_valid
            )

    predictions = np.asarray(
        all_predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        all_gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        all_gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        all_bboxes,
        dtype=np.float32,
    )

    category_ids = np.asarray(
        all_category_ids,
        dtype=np.int64,
    )

    image_ids = np.asarray(
        all_image_ids,
        dtype=np.int64,
    )

    annotation_ids = np.asarray(
        all_annotation_ids,
        dtype=np.int64,
    )

    prediction_valid = np.asarray(
        prediction_valid_rows,
        dtype=bool,
    )

    if verbose:

        print(
            "\nYOLO INSTANCE MATCHING"
        )
        print(
            "-" * 50
        )

        print(
            f"GT instances       : {total_gt}"
        )

        print(
            f"YOLO predictions   : {total_pred}"
        )

        print(
            f"Matched instances  : {total_matched}"
        )

        print(
            f"Unmatched GT       : "
            f"{total_gt - total_matched}"
        )

        print(
            f"Unmatched YOLO     : "
            f"{total_pred - total_matched}"
        )

        if total_gt > 0:

            print(
                f"GT match rate      : "
                f"{total_matched / total_gt:.6f}"
            )

    return {
        "predictions": predictions,
        "gt": gt,
        "gt_vis": gt_vis,
        "bboxes": bboxes,
        "category_ids": category_ids,
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
        "prediction_valid": prediction_valid,
        "matching_audit": matching_audit,
        "total_gt": total_gt,
        "total_predictions": total_pred,
        "total_matched": total_matched,
    }


# ============================================================
# NORMALIZE COORDINATES
# ============================================================

def normalize_evaluation_coordinates(
    predictions,
    gt,
    bboxes,
    image_lookup,
    image_ids,
):
    """
    Convert pixel coordinates to normalized image coordinates.

    Bboxes are converted to normalized COCO-style:

        [x/W, y/H, w/W, h/H]

    This is the representation expected by common_pose_metrics.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    ).copy()

    gt = np.asarray(
        gt,
        dtype=np.float32,
    ).copy()

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    ).copy()

    for i, image_id in enumerate(
        image_ids
    ):

        image = image_lookup[
            int(image_id)
        ]

        width = float(
            image["width"]
        )

        height = float(
            image["height"]
        )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid image dimensions "
                f"for image ID {image_id}"
            )

        predictions[i, :, 0] /= width
        predictions[i, :, 1] /= height

        gt[i, :, 0] /= width
        gt[i, :, 1] /= height

        bboxes[i, 0] /= width
        bboxes[i, 1] /= height
        bboxes[i, 2] /= width
        bboxes[i, 3] /= height

    return (
        predictions,
        gt,
        bboxes,
    )


# ============================================================
# PROFILE CSV
# ============================================================

def save_profile_csv(
    path,
    profile_pck,
    role_pck,
    edge_accuracy,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Profile",
                "Instances",
                "PCK",
                "Role_PCK",
                "Edge_Accuracy",
            ]
        )

        for profile in PROFILE_SCHEMAS:

            instances = profile_pck[
                profile
            ][
                "instances"
            ]

            pck = profile_pck[
                profile
            ][
                "pck"
            ]

            roles = role_pck[
                "profile_role_pck"
            ][profile]

            role_values = [
                item["pck"]
                for item in roles.values()
                if item["valid"] > 0
            ]

            role_average = (
                float(
                    np.mean(
                        role_values
                    )
                )
                if role_values
                else 0.0
            )

            edge = edge_accuracy[
                "profile"
            ][profile][
                "accuracy"
            ]

            writer.writerow(
                [
                    profile,
                    instances,
                    f"{pck:.6f}",
                    f"{role_average:.6f}",
                    f"{edge:.6f}",
                ]
            )


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    path,
    report,
):
    metrics = report[
        "metrics"
    ]

    standard = metrics[
        "standard_pck"
    ]

    invariant = metrics[
        "order_invariant_pck"
    ]

    profile_pck = metrics[
        "profile_pck"
    ]

    roles = metrics[
        "semantic_roles"
    ]

    edges = metrics[
        "structural_edges"
    ]

    native = report.get(
        "native_metrics",
        {},
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "YOLO26-POSE PROFILE-AWARE "
            "TEST EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Start time : "
            f"{report['start_time']}\n"
        )

        f.write(
            f"End time   : "
            f"{report['end_time']}\n"
        )

        f.write(
            f"Checkpoint : "
            f"{report['weights']}\n"
        )

        f.write(
            f"Dataset    : "
            f"{report['dataset']}\n"
        )

        f.write(
            f"Annotations: "
            f"{report['annotation_file']}\n"
        )

        f.write(
            "Test split : test\n\n"
        )

        # ----------------------------------------------------
        # Native YOLO
        # ----------------------------------------------------

        f.write(
            "NATIVE ULTRALYTICS METRICS\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for key, value in native.items():

            if isinstance(
                value,
                (int, float),
            ):

                f.write(
                    f"{key:<25}: "
                    f"{value:.6f}\n"
                )

        # ----------------------------------------------------
        # Standard PCK
        # ----------------------------------------------------

        f.write(
            "\nSTANDARD PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Overall : "
            f"{standard['overall']:.6f}\n"
        )

        for joint, value in (
            standard[
                "joint_pck"
            ].items()
        ):

            f.write(
                f"{joint:<12}: "
                f"{value:.6f}\n"
            )

        f.write(
            f"Correct : "
            f"{standard['total_correct']}\n"
        )

        f.write(
            f"Valid   : "
            f"{standard['total_valid']}\n"
        )

        # ----------------------------------------------------
        # Order invariant
        # ----------------------------------------------------

        f.write(
            "\nORDER-INVARIANT PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Overall : "
            f"{invariant['overall']:.6f}\n"
        )

        for profile, item in (
            invariant[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile:<15} "
                f"{item['pck']:.6f}\n"
            )

        gain = (
            invariant["overall"]
            - standard["overall"]
        )

        f.write(
            "\nOrder-invariance gain : "
            f"{gain:.6f}\n"
        )

        f.write(
            "Percentage points     : "
            f"{gain * 100:.2f}\n"
        )

        # ----------------------------------------------------
        # Profile PCK
        # ----------------------------------------------------

        f.write(
            "\nPROFILE-LEVEL PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for profile, item in (
            profile_pck.items()
        ):

            f.write(
                f"{profile:<15} "
                f"instances="
                f"{item['instances']:<5} "
                f"PCK="
                f"{item['pck']:.6f}\n"
            )

        # ----------------------------------------------------
        # Semantic roles
        # ----------------------------------------------------

        f.write(
            "\nSEMANTIC ROLE PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for role, value in (
            roles[
                "overall_role_pck"
            ].items()
        ):

            f.write(
                f"{role:<35}: "
                f"{value:.6f}\n"
            )

        # ----------------------------------------------------
        # Structural edges
        # ----------------------------------------------------

        f.write(
            "\nSTRUCTURAL EDGE ACCURACY\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Overall : "
            f"{edges['overall']:.6f}\n"
        )

        for profile, item in (
            edges[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile:<15} "
                f"{item['accuracy']:.6f}\n"
            )

        # ----------------------------------------------------
        # Matching
        # ----------------------------------------------------

        matching = report[
            "matching"
        ]

        f.write(
            "\nINSTANCE MATCHING\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"GT instances      : "
            f"{matching['total_gt']}\n"
        )

        f.write(
            f"YOLO predictions   : "
            f"{matching['total_predictions']}\n"
        )

        f.write(
            f"Matched instances : "
            f"{matching['total_matched']}\n"
        )

        f.write(
            f"Unmatched GT       : "
            f"{matching['total_gt'] - matching['total_matched']}\n"
        )

        f.write(
            f"Match rate         : "
            f"{matching['match_rate']:.6f}\n"
        )

        f.write(
            "\nEvaluation settings\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"PCK threshold      : "
            f"{report['threshold']}\n"
        )

        f.write(
            f"Instance IoU       : "
            f"{report['match_iou_threshold']}\n"
        )

        f.write(
            f"YOLO confidence    : "
            f"{report['prediction_confidence']}\n"
        )

        f.write(
            f"Image size         : "
            f"{report['imgsz']}\n"
        )

        f.write(
            "\nNo source annotations were modified.\n"
        )


# ============================================================
# NATIVE ULTRALYTICS METRICS
# ============================================================

def extract_native_metrics(metrics):
    """
    Extract native Ultralytics metrics defensively.
    """

    output = {}

    try:
        box = metrics.box

        output[
            "box_precision"
        ] = float(box.mp)

        output[
            "box_recall"
        ] = float(box.mr)

        output[
            "box_map50"
        ] = float(box.map50)

        output[
            "box_map50_95"
        ] = float(box.map)

    except Exception:
        pass

    try:
        pose = metrics.pose

        output[
            "pose_precision"
        ] = float(pose.mp)

        output[
            "pose_recall"
        ] = float(pose.mr)

        output[
            "pose_map50"
        ] = float(pose.map50)

        output[
            "pose_map50_95"
        ] = float(pose.map)

    except Exception:
        pass

    return output


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(args):

    start_time = datetime.now().isoformat()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    weights = resolve_path(
        args.weights
    )

    data_yaml = resolve_path(
        args.data
    )

    annotation_file = resolve_path(
        args.annotation_file
    )

    dataset_root = resolve_path(
        args.dataset
    )

    project = resolve_path(
        args.project
    )

    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{weights}"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n"
            f"{data_yaml}"
        )

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{annotation_file}"
        )

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"YOLO dataset not found:\n"
            f"{dataset_root}"
        )

    test_image_dir = (
        dataset_root
        / "images"
        / "test"
    )

    if not test_image_dir.exists():
        raise FileNotFoundError(
            f"YOLO test image directory not found:\n"
            f"{test_image_dir}"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    save_dir = (
        project
        / args.name
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_dir = (
        save_dir
        / "predictions"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 70
    )

    print(
        "YOLO26-POSE PROFILE-AWARE "
        "TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Weights       : {weights}"
    )

    print(
        f"Dataset YAML  : {data_yaml}"
    )

    print(
        f"Dataset root  : {dataset_root}"
    )

    print(
        f"Annotations   : {annotation_file}"
    )

    print(
        f"Test images   : {test_image_dir}"
    )

    print(
        f"Device        : {args.device}"
    )

    print(
        f"Image size    : {args.imgsz}"
    )

    print(
        f"PCK threshold : {args.threshold}"
    )

    print(
        f"Match IoU     : {args.match_iou}"
    )

    print(
        f"YOLO conf     : {args.conf}"
    )

    # --------------------------------------------------------
    # Load COCO
    # --------------------------------------------------------

    (
        coco_data,
        image_by_id,
        annotations_by_image,
    ) = load_test_ground_truth(
        annotation_file
    )

    category_map = (
        make_category_map_from_coco(
            coco_data
        )
    )

    # --------------------------------------------------------
    # Validate categories
    # --------------------------------------------------------

    for category_id, name in (
        category_map.items()
    ):

        if name not in PROFILE_SCHEMAS:

            print(
                f"WARNING: category "
                f"{category_id} -> {name} "
                f"is not in PROFILE_SCHEMAS."
            )

    # --------------------------------------------------------
    # Load test images
    # --------------------------------------------------------

    image_paths = sorted(
        [
            p
            for p in test_image_dir.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if not image_paths:
        raise RuntimeError(
            "No test images found."
        )

    print(
        f"\nTest images found: "
        f"{len(image_paths)}"
    )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "\nLoading YOLO26-Pose..."
    )

    model = YOLO(
        str(weights)
    )

    # --------------------------------------------------------
    # Native Ultralytics evaluation
    # --------------------------------------------------------

    print(
        "\nRunning native Ultralytics "
        "test evaluation..."
    )

    native_metrics_obj = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        plots=True,
        save_json=False,
        project=str(
            save_dir
        ),
        name="native_val",
        exist_ok=True,
        verbose=True,
    )

    native_metrics = (
        extract_native_metrics(
            native_metrics_obj
        )
    )

    # --------------------------------------------------------
    # Custom inference
    # --------------------------------------------------------

    print(
        "\nRunning custom test inference "
        "for common pose metrics..."
    )

    evaluation_data = (
        build_aligned_arrays(
            model=model,
            image_paths=image_paths,
            image_by_id=image_by_id,
            annotations_by_image=(
                annotations_by_image
            ),
            iou_threshold=args.match_iou,
            prediction_confidence=args.conf,
            device=args.device,
            imgsz=args.imgsz,
            max_det=args.max_det,
            verbose=True,
        )
    )

    predictions_px = (
        evaluation_data[
            "predictions"
        ]
    )

    gt_px = (
        evaluation_data[
            "gt"
        ]
    )

    gt_vis = (
        evaluation_data[
            "gt_vis"
        ]
    )

    bboxes_px = (
        evaluation_data[
            "bboxes"
        ]
    )

    category_ids = (
        evaluation_data[
            "category_ids"
        ]
    )

    image_ids = (
        evaluation_data[
            "image_ids"
        ]
    )

    annotation_ids = (
        evaluation_data[
            "annotation_ids"
        ]
    )

    prediction_valid = (
        evaluation_data[
            "prediction_valid"
        ]
    )

    # --------------------------------------------------------
    # Normalize to image coordinates
    # --------------------------------------------------------

    (
        predictions,
        gt,
        bboxes,
    ) = normalize_evaluation_coordinates(
        predictions=predictions_px,
        gt=gt_px,
        bboxes=bboxes_px,
        image_lookup=image_by_id,
        image_ids=image_ids,
    )

    # --------------------------------------------------------
    # Common thesis metrics
    # --------------------------------------------------------

    print(
        "\nCalculating common thesis metrics..."
    )

    metrics = calculate_all_metrics(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=args.threshold,
        prediction_valid=prediction_valid,
    )

    # --------------------------------------------------------
    # Matching summary
    # --------------------------------------------------------

    matching_total_gt = (
        evaluation_data[
            "total_gt"
        ]
    )

    matching_total_predictions = (
        evaluation_data[
            "total_predictions"
        ]
    )

    matching_total_matched = (
        evaluation_data[
            "total_matched"
        ]
    )

    match_rate = (
        float(
            matching_total_matched
            / matching_total_gt
        )
        if matching_total_gt > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Save raw aligned arrays
    # --------------------------------------------------------

    np.save(
        predictions_dir
        / "yolo_predictions.npy",
        predictions,
    )

    np.save(
        predictions_dir
        / "yolo_predictions_pixels.npy",
        predictions_px,
    )

    np.save(
        predictions_dir
        / "gt.npy",
        gt,
    )

    np.save(
        predictions_dir
        / "gt_pixels.npy",
        gt_px,
    )

    np.save(
        predictions_dir
        / "gt_visibility.npy",
        gt_vis,
    )

    np.save(
        predictions_dir
        / "bboxes.npy",
        bboxes,
    )

    np.save(
        predictions_dir
        / "bboxes_pixels.npy",
        bboxes_px,
    )

    np.save(
        predictions_dir
        / "category_ids.npy",
        category_ids,
    )

    np.save(
        predictions_dir
        / "image_ids.npy",
        image_ids,
    )

    np.save(
        predictions_dir
        / "annotation_ids.npy",
        annotation_ids,
    )

    np.save(
        predictions_dir
        / "prediction_valid.npy",
        prediction_valid,
    )

    # --------------------------------------------------------
    # Save matching audit
    # --------------------------------------------------------

    with (
        predictions_dir
        / "matching_audit.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_data[
                "matching_audit"
            ],
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    end_time = datetime.now().isoformat()

    report = {
        "timestamp": end_time,

        "model": "YOLO26-Pose",

        "weights": str(
            weights
        ),

        "dataset": str(
            data_yaml
        ),

        "dataset_root": str(
            dataset_root
        ),

        "annotation_file": str(
            annotation_file
        ),

        "test_split": "test",

        "start_time": start_time,

        "end_time": end_time,

        "imgsz": args.imgsz,

        "batch": args.batch,

        "device": str(
            args.device
        ),

        "threshold": float(
            args.threshold
        ),

        "match_iou_threshold": float(
            args.match_iou
        ),

        "prediction_confidence": float(
            args.conf
        ),

        "nms_iou": float(
            args.nms_iou
        ),

        "max_det": int(
            args.max_det
        ),

        "native_metrics": native_metrics,

        "matching": {
            "total_gt":
                int(
                    matching_total_gt
                ),

            "total_predictions":
                int(
                    matching_total_predictions
                ),

            "total_matched":
                int(
                    matching_total_matched
                ),

            "unmatched_gt":
                int(
                    matching_total_gt
                    - matching_total_matched
                ),

            "unmatched_predictions":
                int(
                    matching_total_predictions
                    - matching_total_matched
                ),

            "match_rate":
                match_rate,
        },

        "metrics": metrics,

        "prediction_files": {
            "normalized":
                str(
                    predictions_dir
                    / "yolo_predictions.npy"
                ),

            "pixels":
                str(
                    predictions_dir
                    / "yolo_predictions_pixels.npy"
                ),

            "gt":
                str(
                    predictions_dir
                    / "gt.npy"
                ),

            "visibility":
                str(
                    predictions_dir
                    / "gt_visibility.npy"
                ),

            "bboxes":
                str(
                    predictions_dir
                    / "bboxes.npy"
                ),

            "category_ids":
                str(
                    predictions_dir
                    / "category_ids.npy"
                ),

            "image_ids":
                str(
                    predictions_dir
                    / "image_ids.npy"
                ),

            "annotation_ids":
                str(
                    predictions_dir
                    / "annotation_ids.npy"
                ),

            "prediction_valid":
                str(
                    predictions_dir
                    / "prediction_valid.npy"
                ),

            "matching_audit":
                str(
                    predictions_dir
                    / "matching_audit.json"
                ),
        },

        "profile_schemas":
            PROFILE_SCHEMAS,
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        save_dir
        / "evaluation_metrics.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    save_profile_csv(
        save_dir
        / "profile_metrics.csv",
        metrics[
            "profile_pck"
        ],
        metrics[
            "semantic_roles"
        ],
        metrics[
            "structural_edges"
        ],
    )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    write_text_report(
        save_dir
        / "evaluation_report.txt",
        report,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    standard = metrics[
        "standard_pck"
    ]

    invariant = metrics[
        "order_invariant_pck"
    ]

    profile_pck = metrics[
        "profile_pck"
    ]

    roles = metrics[
        "semantic_roles"
    ]

    edges = metrics[
        "structural_edges"
    ]

    print()
    print(
        "=" * 70
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nNative YOLO Metrics"
    )

    for key, value in native_metrics.items():

        print(
            f"  {key:<25}: "
            f"{value:.6f}"
        )

    print(
        "\nStandard PCK@0.05"
    )

    print(
        f"  Overall : "
        f"{standard['overall']:.6f}"
    )

    print(
        f"  Correct : "
        f"{standard['total_correct']}"
    )

    print(
        f"  Valid   : "
        f"{standard['total_valid']}"
    )

    print(
        "\nOrder-Invariant PCK"
    )

    print(
        f"  Overall : "
        f"{invariant['overall']:.6f}"
    )

    gain = (
        invariant["overall"]
        - standard["overall"]
    )

    print(
        f"  Gain    : "
        f"{gain:.6f} "
        f"({gain * 100:.2f} pp)"
    )

    print(
        "\nProfile PCK"
    )

    for profile, item in (
        profile_pck.items()
    ):

        print(
            f"  {profile:<15}: "
            f"{item['pck']:.6f} "
            f"(n={item['instances']})"
        )

    print(
        "\nSemantic Role PCK"
    )

    for role, value in (
        roles[
            "overall_role_pck"
        ].items()
    ):

        print(
            f"  {role:<35}: "
            f"{value:.6f}"
        )

    print(
        "\nStructural Edge Accuracy"
    )

    print(
        f"  Overall : "
        f"{edges['overall']:.6f}"
    )

    print(
        "\nInstance Matching"
    )

    print(
        f"  GT instances      : "
        f"{matching_total_gt}"
    )

    print(
        f"  YOLO predictions  : "
        f"{matching_total_predictions}"
    )

    print(
        f"  Matched            : "
        f"{matching_total_matched}"
    )

    print(
        f"  Match rate         : "
        f"{match_rate:.6f}"
    )

    print(
        "\nSaved to:"
    )

    print(
        save_dir
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    repo = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate YOLO26-Pose on the "
            "InnoCount frozen test split "
            "using common profile-aware "
            "pose metrics."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        help=(
            "YOLO26-Pose checkpoint, "
            "e.g. runs/.../weights/best.pt"
        ),
    )

    parser.add_argument(
        "--data",
        default=str(
            repo
            / "datasets"
            / "InnoCount_KeyPoint_YOLO26_Pose_v2"
            / "dataset.yaml"
        ),
        help=(
            "YOLO dataset YAML. "
            "Defaults to corrected v2."
        ),
    )

    parser.add_argument(
        "--dataset",
        default=str(
            repo
            / "datasets"
            / "InnoCount_KeyPoint_YOLO26_Pose_v2"
        ),
        help=(
            "YOLO dataset root. "
            "Defaults to corrected v2."
        ),
    )

    parser.add_argument(
        "--annotation-file",
        default=str(
            repo
            / "datasets"
            / "InnoCount_KeyPoint_ModelReady_v2"
            / "test"
            / "annotations.json"
        ),
        help=(
            "Exact frozen ModelReady v2 "
            "test annotations."
        ),
    )

    parser.add_argument(
        "--project",
        default=str(
            repo
            / "experiments"
            / "YOLO26_Pose_Runs"
        ),
        help=(
            "Evaluation output project."
        ),
    )

    parser.add_argument(
        "--name",
        default=(
            "E06_YOLO26n_Corrected-2_FinalTest"
        ),
        help=(
            "Evaluation output directory name."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help=(
            "PCK normalized localization "
            "threshold."
        ),
    )

    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.50,
        help=(
            "Minimum IoU required for a "
            "YOLO prediction to match a GT "
            "instance."
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help=(
            "Low inference confidence threshold "
            "used for custom pose evaluation."
        ),
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.70,
        help=(
            "IoU threshold used by YOLO NMS "
            "during inference."
        ),
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help=(
            "Maximum detections per image."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    evaluate(
        args
    )


if __name__ == "__main__":
    main()