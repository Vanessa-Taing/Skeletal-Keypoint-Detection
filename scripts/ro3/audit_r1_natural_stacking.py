#!/usr/bin/env python3
"""
RO3 R1-A: Read-only natural stacking / occlusion proxy audit.

Inputs:
  Frozen patched COCO test annotations
  Frozen clean ViTPose-B E03 predictions

No inference, augmentation, training or annotation changes.

Definitions:
  local overlap = max intersection(target GT bbox,
                       another GT bbox) / target GT bbox area
  crowding      = GT object count in original photograph
  visibility    = COCO original v=1 or v=2

These are descriptive proxies, NOT verified occlusion depth.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

COCO_PATH = ROOT / (
    "datasets/InnoCount_COCO_Patched_UniqueIDs_v1/"
    "annotations/test.json"
)

PRED_DIR = ROOT / (
    "experiments/RO3/R1_Eval_v1/"
    "test_uniform_p00_r0"
)

OUT = ROOT / (
    "experiments/RO3/"
    "R1_NaturalStackingAudit_v1"
)

THRESHOLD = 0.05
BOOTSTRAP_REPS = 2000
SEED = 20261119

EXPECTED_COUNTS = {
    "Angle Bar": 173,
    "Channel": 598,
    "T-beam": 397,
    "HI-Beam": 375,
    "Sheet Pile": 83,
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(row):
    return (
        int(row["image_id"]),
        int(row["id"]),
    )


def density_group(count):
    if count == 0:
        return "0_empty"
    if count < 10:
        return "1_low_1to9"
    if count < 50:
        return "2_medium_10to49"
    return "3_high_50plus"


def overlap_group(value):
    if value <= 1e-12:
        return "0_none"
    if value <= 0.10:
        return "1_low_0to10pct"
    if value <= 0.30:
        return "2_medium_10to30pct"
    return "3_high_over30pct"


def clipped_xyxy(boxes, image_width, image_height):
    """Float pixel boxes clipped to original image bounds."""
    boxes = np.asarray(boxes, dtype=float)

    x0 = np.clip(boxes[:, 0], 0, image_width)
    y0 = np.clip(boxes[:, 1], 0, image_height)

    x1 = np.clip(
        boxes[:, 0] + boxes[:, 2],
        0, image_width,
    )
    y1 = np.clip(
        boxes[:, 1] + boxes[:, 3],
        0, image_height,
    )

    require(
        np.all(x1 > x0) and np.all(y1 > y0),
        "GT bounding box has no in-image area",
    )

    return np.column_stack([x0, y0, x1, y1])


def box_overlap_features(boxes):
    """All pairs in one image; no occlusion direction inferred."""
    n = len(boxes)

    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    x0, y0, x1, y1 = boxes.T

    areas = (x1 - x0) * (y1 - y0)

    intersection_w = np.maximum(
        0,
        np.minimum(x1[:, None], x1[None, :])
        - np.maximum(x0[:, None], x0[None, :]),
    )

    intersection_h = np.maximum(
        0,
        np.minimum(y1[:, None], y1[None, :])
        - np.maximum(y0[:, None], y0[None, :]),
    )

    intersection = intersection_w * intersection_h

    # An annotation must never overlap itself.
    np.fill_diagonal(intersection, 0.0)

    target_coverage = intersection / areas[:, None]

    union = (
        areas[:, None]
        + areas[None, :]
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )

    return (
        np.max(target_coverage, axis=1),
        np.max(iou, axis=1),
        np.sum(intersection > 0, axis=1),
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_pck(rows, group_name):
    """Descriptive source-image cluster bootstrap."""
    by_image = defaultdict(lambda: [0, 0])

    for row in rows:
        stats = by_image[int(row["image_id"])]
        stats[0] += int(row["correct_keypoints"])
        stats[1] += int(row["valid_keypoints"])

    # Do not give pseudo-precise scene intervals
    # for categories represented by very few images.
    if len(by_image) < 5:
        return None, None

    values = np.asarray(
        list(by_image.values()), dtype=np.int64
    )

    digest = hashlib.sha256(
        f"{SEED}|{group_name}".encode()
    ).digest()

    rng = np.random.default_rng(
        int.from_bytes(digest[:8], "big")
    )

    n = len(values)

    draws = rng.integers(
        0, n, size=(BOOTSTRAP_REPS, n)
    )

    sampled = values[draws].sum(axis=1)

    valid = sampled[:, 1] > 0
    if not np.any(valid):
        return None, None

    scores = (
        sampled[valid, 0]
        / sampled[valid, 1]
    )

    lo, hi = np.quantile(
        scores, [0.025, 0.975]
    )
    return float(lo), float(hi)


def aggregate(label, rows):
    """Micro-averaged fixed-index PCK over selected GT objects."""
    n_valid = sum(
        int(r["valid_keypoints"]) for r in rows
    )

    n_correct = sum(
        int(r["correct_keypoints"]) for r in rows
    )

    profiles = dict(
        sorted(Counter(r["profile"] for r in rows).items())
    )

    image_count = len({
        int(r["image_id"]) for r in rows
    })

    low, high = bootstrap_pck(rows, label) if rows else (
        None, None
    )

    return {
        "group": label,
        "instances": len(rows),
        "source_images": image_count,
        "valid_keypoints": n_valid,
        "correct_keypoints": n_correct,
        "fixed_pck": (
            n_correct / n_valid if n_valid else None
        ),
        "cluster_ci_low": low,
        "cluster_ci_high": high,
        "mean_bbox_area_fraction": (
            float(np.mean([
                r["bbox_area_fraction"]
                for r in rows
            ])) if rows else None
        ),
        "mean_max_bbox_overlap_fraction": (
            float(np.mean([
                r["max_bbox_overlap_fraction"]
                for r in rows
            ])) if rows else None
        ),
        "profile_counts_json": json.dumps(profiles),
    }


def main():
    require(COCO_PATH.is_file(), COCO_PATH)
    require(not OUT.exists(), f"Output exists: {OUT}")

    coco = json.loads(COCO_PATH.read_text())

    images = {
        int(im["id"]): im
        for im in coco["images"]
    }

    anns = sorted(
        coco["annotations"],
        key=identity,
    )

    require(len(images) == 74, "Unexpected test image count")
    require(len(anns) == 1626, "Unexpected GT object count")

    annotation_keys = [identity(a) for a in anns]

    require(
        len(annotation_keys) == len(set(annotation_keys)),
        "Duplicate composite GT annotation identity",
    )

    category_names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    actual_profile_counts = Counter(
        category_names[int(a["category_id"])]
        for a in anns
    )

    require(
        dict(actual_profile_counts) == EXPECTED_COUNTS,
        "Profile distribution differs from frozen RO2 test set",
    )

    run_report_path = PRED_DIR / "metrics.json"
    require(run_report_path.is_file(), run_report_path)

    run_report = json.loads(
        run_report_path.read_text()
    )

    spec = run_report["run_spec"]

    require(spec["split"] == "test", "Wrong inference split")
    require(spec["level"] == 0, "Not clean inference")
    require(spec["replicate"] == 0, "Wrong replicate")
    require(spec["max_instances"] is None, "Partial inference")
    require(
        spec["selected_instances"] == 1626,
        "Incomplete predictions",
    )
    require(
        run_report["reference_comparison"] is not None,
        "Missing original E03 prediction reference check",
    )

    pred_file = PRED_DIR / "vitpose_predictions.npy"
    image_id_file = PRED_DIR / "vitpose_image_ids.npy"
    ann_id_file = PRED_DIR / "vitpose_annotation_ids.npy"

    pred = np.load(
        pred_file, allow_pickle=False
    )
    image_ids = np.load(
        image_id_file, allow_pickle=False
    )
    ann_ids = np.load(
        ann_id_file, allow_pickle=False
    )

    require(
        pred.shape == (1626, 6, 2),
        "Incorrect clean prediction shape",
    )

    require(
        np.isfinite(pred).all(),
        "Non-finite clean predictions",
    )

    predictions = {}

    for i in range(len(pred)):
        key = (int(image_ids[i]), int(ann_ids[i]))
        require(key not in predictions, "Duplicate prediction")
        predictions[key] = pred[i]

    require(
        set(predictions) == set(annotation_keys),
        "Prediction/GT identity mismatch",
    )

    anns_by_image = defaultdict(list)

    for ann in anns:
        anns_by_image[int(ann["image_id"])].append(ann)

    object_rows = []
    keypoint_rows = []
    image_rows = []

    # Include original negative images in image-level audit.
    for image_id, image in sorted(images.items()):
        scene_anns = anns_by_image.get(image_id, [])
        n = len(scene_anns)

        if n == 0:
            image_rows.append({
                "image_id": image_id,
                "file_name": image["file_name"],
                "gt_object_count": 0,
                "density_group": density_group(0),
                "valid_keypoints": 0,
                "correct_keypoints": 0,
                "fixed_pck": None,
                "objects_with_bbox_overlap": 0,
                "mean_max_bbox_overlap_fraction": None,
            })
            continue

        width = int(image["width"])
        height = int(image["height"])

        original_boxes = np.asarray(
            [a["bbox"] for a in scene_anns],
            dtype=float,
        )

        require(
            np.all(original_boxes[:, 2:] > 0),
            "Invalid original GT bbox",
        )

        clipped = clipped_xyxy(
            original_boxes, width, height
        )

        coverage, max_iou, neighbors = (
            box_overlap_features(clipped)
        )

        start = len(object_rows)

        for i, ann in enumerate(scene_anns):
            key = identity(ann)

            kp = np.asarray(
                ann["keypoints"],
                dtype=np.float32,
            ).reshape(6, 3)

            require(
                set(np.unique(kp[:, 2])).issubset(
                    {0.0, 1.0, 2.0}
                ),
                "Unexpected COCO visibility values",
            )

            bbox = original_boxes[i]
            scale = max(bbox[2], bbox[3])

            distance = (
                np.linalg.norm(
                    predictions[key] - kp[:, :2],
                    axis=1,
                ) / scale
            )

            gt_valid = kp[:, 2] > 0
            correct = (distance <= THRESHOLD) & gt_valid

            n_valid = int(gt_valid.sum())
            n_correct = int(correct.sum())

            profile = category_names[
                int(ann["category_id"])
            ]

            roi_area = (
                (clipped[i, 2] - clipped[i, 0])
                * (clipped[i, 3] - clipped[i, 1])
            )

            overlap_fraction = float(coverage[i])

            obj = {
                "image_id": image_id,
                "annotation_id": int(ann["id"]),
                "file_name": image["file_name"],
                "profile": profile,
                "category_id": int(ann["category_id"]),
                "scene_object_count": n,
                "density_group": density_group(n),
                "gt_bbox_x": float(bbox[0]),
                "gt_bbox_y": float(bbox[1]),
                "gt_bbox_w": float(bbox[2]),
                "gt_bbox_h": float(bbox[3]),
                "bbox_area_fraction": float(
                    roi_area / (width * height)
                ),
                "bbox_scale_px": float(scale),
                "max_bbox_overlap_fraction": overlap_fraction,
                "max_neighbor_bbox_iou": float(max_iou[i]),
                "overlapping_neighbor_count": int(neighbors[i]),
                "overlap_group": overlap_group(
                    overlap_fraction
                ),
                "valid_keypoints": n_valid,
                "original_visible_keypoints": int(
                    np.count_nonzero(kp[:, 2] == 2)
                ),
                "original_occluded_keypoints": int(
                    np.count_nonzero(kp[:, 2] == 1)
                ),
                "correct_keypoints": n_correct,
                "fixed_pck": (
                    n_correct / n_valid if n_valid else None
                ),
            }

            object_rows.append(obj)

            for j in range(6):
                visibility = int(kp[j, 2])

                if visibility == 0:
                    continue

                keypoint_rows.append({
                    "image_id": image_id,
                    "annotation_id": int(ann["id"]),
                    "profile": profile,
                    "keypoint": f"G{j+1}",
                    "original_visibility": visibility,
                    "visibility_group": (
                        "occluded_labeled"
                        if visibility == 1
                        else "visible_labeled"
                    ),
                    "normalized_error": float(distance[j]),
                    "correct_fixed_index": int(correct[j]),
                    "scene_object_count": n,
                    "density_group": density_group(n),
                    "max_bbox_overlap_fraction":
                        overlap_fraction,
                    "overlap_group": overlap_group(
                        overlap_fraction
                    ),
                })

        scene_rows = object_rows[start:]

        valid_total = sum(
            r["valid_keypoints"] for r in scene_rows
        )

        correct_total = sum(
            r["correct_keypoints"] for r in scene_rows
        )

        image_rows.append({
            "image_id": image_id,
            "file_name": image["file_name"],
            "gt_object_count": n,
            "density_group": density_group(n),
            "valid_keypoints": valid_total,
            "correct_keypoints": correct_total,
            "fixed_pck": (
                correct_total / valid_total
                if valid_total else None
            ),
            "objects_with_bbox_overlap": sum(
                r["overlapping_neighbor_count"] > 0
                for r in scene_rows
            ),
            "mean_max_bbox_overlap_fraction": float(
                np.mean(coverage)
            ),
        })

    # Strict reproduction of original RO2 test metric.
    require(len(object_rows) == 1626, "Object rows lost")
    require(len(image_rows) == 74, "Image rows lost")
    require(len(keypoint_rows) == 7049, "Keypoint rows lost")

    require(
        sum(r["valid_keypoints"] for r in object_rows)
        == 7049,
        "Wrong fixed PCK denominator",
    )

    require(
        sum(r["correct_keypoints"] for r in object_rows)
        == 4325,
        "Original E03 PCK numerator not reproduced",
    )

    require(
        sum(r["original_occluded_keypoints"] for r in object_rows)
        == 57,
        "Unexpected original occluded-keypoint count",
    )

    require(
        sum(r["original_visible_keypoints"] for r in object_rows)
        == 6992,
        "Unexpected original visible-keypoint count",
    )

    require(
        sum(
            r["correct_fixed_index"]
            for r in keypoint_rows
            if r["original_visibility"] == 1
        ) == 26,
        "Originally occluded baseline not reproduced",
    )

    require(
        run_report["fixed_index_pck"]["total_correct"] == 4325,
        "Saved run report differs",
    )

    require(
        run_report["fixed_index_pck"]["total_valid"] == 7049,
        "Saved run denominator differs",
    )

    require(
        len(anns_by_image) == 71,
        "Unexpected annotated-image count",
    )

    # ------------------------------------------------
    # Descriptive summaries.
    # ------------------------------------------------
    density_rows = []
    overlap_rows = []
    profile_rows = []
    profile_overlap_rows = []
    profile_density_rows = []

    for group in (
        "1_low_1to9",
        "2_medium_10to49",
        "3_high_50plus",
    ):
        selected = [
            r for r in object_rows
            if r["density_group"] == group
        ]
        density_rows.append(aggregate(group, selected))

    for group in (
        "0_none",
        "1_low_0to10pct",
        "2_medium_10to30pct",
        "3_high_over30pct",
    ):
        selected = [
            r for r in object_rows
            if r["overlap_group"] == group
        ]
        overlap_rows.append(aggregate(group, selected))

    for profile in EXPECTED_COUNTS:
        selected = [
            r for r in object_rows
            if r["profile"] == profile
        ]
        profile_rows.append(aggregate(profile, selected))

        for group in (
            "0_none",
            "1_low_0to10pct",
            "2_medium_10to30pct",
            "3_high_over30pct",
        ):
            selected = [
                r for r in object_rows
                if r["profile"] == profile
                and r["overlap_group"] == group
            ]
            profile_overlap_rows.append({
                "profile": profile,
                **aggregate(f"{profile}|{group}", selected),
            })

        for group in (
            "1_low_1to9",
            "2_medium_10to49",
            "3_high_50plus",
        ):
            selected = [
                r for r in object_rows
                if r["profile"] == profile
                and r["density_group"] == group
            ]
            profile_density_rows.append({
                "profile": profile,
                **aggregate(f"{profile}|{group}", selected),
            })

    visibility_rows = []

    for profile in ("ALL", *EXPECTED_COUNTS):
        for vis, label in (
            (1, "occluded_labeled"),
            (2, "visible_labeled"),
        ):
            selected = [
                r for r in keypoint_rows
                if r["original_visibility"] == vis
                and (
                    profile == "ALL"
                    or r["profile"] == profile
                )
            ]

            correct = sum(
                r["correct_fixed_index"]
                for r in selected
            )

            visibility_rows.append({
                "profile": profile,
                "visibility": label,
                "keypoints": len(selected),
                "correct": correct,
                "fixed_pck": (
                    correct / len(selected)
                    if selected else None
                ),
                "source_images": len({
                    r["image_id"]
                    for r in selected
                }),
            })

    # ------------------------------------------------
    # Save evidence and protocol metadata.
    # ------------------------------------------------
    OUT.mkdir(parents=True)

    outputs = {
        "instance_features.csv": object_rows,
        "keypoint_features.csv": keypoint_rows,
        "image_summary.csv": image_rows,
        "density_summary.csv": density_rows,
        "bbox_overlap_summary.csv": overlap_rows,
        "profile_summary.csv": profile_rows,
        "profile_by_overlap.csv": profile_overlap_rows,
        "profile_by_density.csv": profile_density_rows,
        "visibility_summary.csv": visibility_rows,
    }

    for filename, rows in outputs.items():
        write_csv(OUT / filename, rows)

    metadata = {
        "protocol": "RO3_R1A_NaturalStackingAudit_v1",
        "model": "ViTPose-B E03 frozen clean GT-box",
        "evaluation": "fixed-index PCK@0.05",
        "source_coco": str(COCO_PATH.relative_to(ROOT)),
        "source_coco_sha256": sha256(COCO_PATH),
        "clean_predictions_sha256": sha256(pred_file),
        "image_ids_sha256": sha256(image_id_file),
        "annotation_ids_sha256": sha256(ann_id_file),
        "script_sha256": sha256(Path(__file__)),
        "gt_objects": 1626,
        "total_images": 74,
        "annotated_images": 71,
        "valid_gt_keypoints": 7049,
        "correct_fixed_index": 4325,
        "original_occluded_keypoints": 57,
        "original_visible_keypoints": 6992,
        "definitions": {
            "density": (
                "Number of GT objects in the original "
                "source photograph; bins 1-9, 10-49, 50+."
            ),
            "bbox_overlap_fraction": (
                "Maximum pairwise intersection area of "
                "clipped GT boxes divided by the target "
                "clipped GT-box area."
            ),
            "overlap_bins": (
                "0; (0,0.10]; (0.10,0.30]; (0.30,1]."
            ),
            "visibility": (
                "COCO v=1 labeled but not visible, "
                "v=2 labeled and visible."
            ),
            "pck": (
                "Micro fixed-index PCK at 0.05, "
                "normalized by max(original GT bbox w,h)."
            ),
        },
        "uncertainty": {
            "method": "source-image cluster bootstrap",
            "iterations": BOOTSTRAP_REPS,
            "seed": SEED,
            "minimum_source_images_for_ci": 5,
        },
        "limitations": [
            "Bounding-box overlap is not physical occlusion.",
            "Image object count is not a local overlap measure.",
            "No occlusion depth or object ordering is inferred.",
            "Profile composition and scale may confound associations.",
            "Channel occurs in two annotated test images.",
            "Sheet Pile occurs in one annotated test image.",
            "Negative images have no GT keypoints to evaluate.",
            "Descriptive observational analysis; no causal inference.",
        ],
    }

    (OUT / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nRO3 R1-A — NATURAL STACKING AUDIT")
    print("=" * 66)
    print("PASS: Frozen E03 clean baseline reproduced")
    print("Test images:", len(images))
    print("Annotated images:", len(anns_by_image))
    print("Objects:", len(object_rows))
    print("Valid GT keypoints:", len(keypoint_rows))
    print("Fixed PCK: 4325/7049 = 0.613562")
    print()

    print("IMAGE DENSITY GROUPS")
    for row in density_rows:
        print(
            f"{row['group']:<22} "
            f"objects={row['instances']:>4} "
            f"images={row['source_images']:>3} "
            f"PCK={row['fixed_pck']:.4f}"
            if row["fixed_pck"] is not None
            else f"{row['group']}: no valid points"
        )

    print("\nBBOX OVERLAP GROUPS")
    for row in overlap_rows:
        if row["fixed_pck"] is None:
            print(f"{row['group']}: no valid points")
            continue

        print(
            f"{row['group']:<25} "
            f"objects={row['instances']:>4} "
            f"images={row['source_images']:>3} "
            f"PCK={row['fixed_pck']:.4f}"
        )

    print("\nPROFILE SUMMARY")
    for row in profile_rows:
        print(
            f"{row['group']:<12} "
            f"objects={row['instances']:>4} "
            f"images={row['source_images']:>3} "
            f"PCK={row['fixed_pck']:.4f}"
        )

    print("\nORIGINAL VISIBILITY")
    for row in visibility_rows:
        if row["profile"] == "ALL":
            print(
                f"{row['visibility']:<19} "
                f"{row['correct']}/{row['keypoints']} "
                f"PCK={row['fixed_pck']:.4f}"
            )

    print("\nPASS: Natural-scene audit completed")
    print("Output:", OUT)
    print("No original files modified")


if __name__ == "__main__":
    main()
