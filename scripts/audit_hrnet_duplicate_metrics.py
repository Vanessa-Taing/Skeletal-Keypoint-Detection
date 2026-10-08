#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from evaluate.common_pose_metrics import (
    calculate_all_metrics,
    make_category_map_from_coco,
)


ROOT = Path("/home/owner/skeletal_keypoint_detection")

ANN_FILE = (
    ROOT
    / "datasets/InnoCount_KeyPoint_ModelReady_v2"
    / "splits/test/annotations.json"
)

EVAL_DIR = (
    ROOT
    / "experiments/HRNet_Runs/E08_HRNet_W32_FinalTest"
    / "2026-09-04_00-17-06"
)

ARRAY_DIR = (
    EVAL_DIR
    / "innocount"
    / "pose_hrnet"
    / "hrnet_eval_config"
)


# ----------------------------------------------------------------------
# Load frozen test annotations
# ----------------------------------------------------------------------

with open(ANN_FILE, encoding="utf-8") as f:
    coco = json.load(f)

category_map = make_category_map_from_coco(coco)


# ----------------------------------------------------------------------
# Load existing HRNet evaluation arrays
# ----------------------------------------------------------------------

predictions = np.load(
    ARRAY_DIR / "hrnet_predictions.npy",
    allow_pickle=True,
)

gt = np.load(
    ARRAY_DIR / "hrnet_gt.npy",
    allow_pickle=True,
)

gt_vis = np.load(
    ARRAY_DIR / "hrnet_gt_visibility.npy",
    allow_pickle=True,
)

bboxes = np.load(
    ARRAY_DIR / "hrnet_bboxes.npy",
    allow_pickle=True,
)

category_ids = np.load(
    ARRAY_DIR / "hrnet_category_ids.npy",
    allow_pickle=True,
)

annotation_ids = np.load(
    ARRAY_DIR / "hrnet_annotation_ids.npy",
    allow_pickle=True,
)


# ----------------------------------------------------------------------
# Identify exact duplicate annotation rows
# ----------------------------------------------------------------------

def annotation_signature(ann):
    return (
        ann["category_id"],
        tuple(ann.get("bbox", [])),
        tuple(ann.get("keypoints", [])),
    )


duplicate_ids = set()
seen = {}

for ann in coco["annotations"]:

    sig = (
        ann["image_id"],
        annotation_signature(ann),
    )

    if sig in seen:
        duplicate_ids.add(int(ann["id"]))
    else:
        seen[sig] = int(ann["id"])


all_ids = annotation_ids.astype(int)

keep_mask = np.array(
    [
        int(annotation_id) not in duplicate_ids
        for annotation_id in all_ids
    ],
    dtype=bool,
)


# ----------------------------------------------------------------------
# Verify alignment
# ----------------------------------------------------------------------

n = len(all_ids)

for name, array in [
    ("predictions", predictions),
    ("gt", gt),
    ("gt_vis", gt_vis),
    ("bboxes", bboxes),
    ("category_ids", category_ids),
]:
    if len(array) != n:
        raise ValueError(
            f"{name} has {len(array)} rows, "
            f"but annotation_ids has {n} rows."
        )


# ----------------------------------------------------------------------
# Common metric calculation
# ----------------------------------------------------------------------

def run_metrics(mask):

    return calculate_all_metrics(
        predictions=predictions[mask],
        gt=gt[mask],
        gt_vis=gt_vis[mask],
        bboxes=bboxes[mask],
        category_ids=category_ids[mask],
        category_map=category_map,
        threshold=0.05,
    )


original_mask = np.ones(
    n,
    dtype=bool,
)

original = run_metrics(original_mask)
dedup = run_metrics(keep_mask)


# ----------------------------------------------------------------------
# Print headline comparison
# ----------------------------------------------------------------------

print("=" * 75)
print("HRNET E08 TEST DUPLICATE METRIC SENSITIVITY")
print("=" * 75)

print()
print(f"Prediction rows        : {n}")
print(f"Duplicate IDs detected : {len(duplicate_ids)}")
print(f"Rows after dedup       : {int(keep_mask.sum())}")

print()
print("Duplicate annotation IDs:")

for annotation_id in sorted(duplicate_ids):
    print(f"  {annotation_id}")


print()
print("=" * 75)
print("HEADLINE METRICS")
print("=" * 75)

print(
    f"{'Metric':32s}"
    f"{'All 1626':>15s}"
    f"{'Dedup 1623':>15s}"
    f"{'Difference':>15s}"
)


def print_metric(label, key):

    a = original[key]["overall"]
    b = dedup[key]["overall"]

    print(
        f"{label:32s}"
        f"{a:15.6f}"
        f"{b:15.6f}"
        f"{b-a:+15.6f}"
    )


print_metric(
    "Standard PCK",
    "standard_pck",
)

print_metric(
    "Order-invariant PCK",
    "order_invariant_pck",
)

print_metric(
    "Structural Edge Accuracy",
    "structural_edges",
)


# ----------------------------------------------------------------------
# Profile-level comparison
# ----------------------------------------------------------------------

print()
print("=" * 75)
print("PROFILE-LEVEL PCK")
print("=" * 75)

print(
    f"{'Profile':20s}"
    f"{'All 1626':>15s}"
    f"{'Dedup 1623':>15s}"
    f"{'Difference':>15s}"
)

for profile in original["profile_pck"]:

    a = original["profile_pck"][profile]["pck"]
    b = dedup["profile_pck"][profile]["pck"]

    print(
        f"{profile:20s}"
        f"{a:15.6f}"
        f"{b:15.6f}"
        f"{b-a:+15.6f}"
    )


# ----------------------------------------------------------------------
# Save sensitivity result
# ----------------------------------------------------------------------

output = {
    "test_annotations_original": int(n),
    "test_annotations_deduplicated": int(keep_mask.sum()),
    "duplicate_annotation_ids": sorted(
        int(x)
        for x in duplicate_ids
    ),
    "metrics_original": original,
    "metrics_deduplicated": dedup,
}

output_file = (
    EVAL_DIR
    / "duplicate_metric_sensitivity.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=2,
    )

print()
print(f"Saved: {output_file}")
