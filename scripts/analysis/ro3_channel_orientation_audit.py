#!/usr/bin/env python3
"""RO3 H1-A: read-only Channel annotation/prediction audit."""

import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / (
    "datasets/InnoCount_COCO_Patched_UniqueIDs_v1"
)

PRED_DIR = ROOT / (
    "experiments/RO2_Final_Patched/"
    "ViTPoseB_E03_SharedProposals"
)

OUT = ROOT / (
    "experiments/RO3/H1A_Channel_Orientation_Audit_v1"
)

CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNERS = np.array([
    [0, 0], [1, 0], [0, 1], [1, 1]
], dtype=float)

PERMUTATIONS = list(itertools.permutations(range(4)))


def canonical_signature(points):
    """
    Describe which image-space corner each keypoint
    most closely occupies.

    Diagnostic only. Does not determine physical
    profile orientation or correct semantic labeling.
    """
    points = np.asarray(points, dtype=float)

    low = points.min(axis=0)
    high = points.max(axis=0)
    extent = high - low

    if np.any(extent < 2.0):
        return {
            "signature": "degenerate",
            "fit": None,
            "margin": None,
            "clear": False,
        }

    normalized = (points - low) / extent

    costs = np.sum(
        (
            normalized[:, None, :]
            - CORNERS[None, :, :]
        ) ** 2,
        axis=2,
    )

    ranked = []

    for perm in PERMUTATIONS:
        cost = sum(costs[i, perm[i]] for i in range(4))
        ranked.append((float(cost), perm))

    ranked.sort(key=lambda x: x[0])

    best_cost, best_perm = ranked[0]
    second_cost = ranked[1][0]

    fit = float(np.sqrt(best_cost / 4))
    margin = float(second_cost - best_cost)

    # Heuristic confidence criteria, not GT truth.
    clear = fit <= 0.35 and margin >= 0.15

    return {
        "signature": "-".join(
            CORNER_NAMES[j] for j in best_perm
        ),
        "fit": fit,
        "margin": margin,
        "clear": clear,
    }


def save_csv(path, rows):
    assert rows, f"No rows for {path}"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_name(name):
    return "".join(
        c for c in str(name).lower() if c.isalnum()
    )


def main():
    assert not OUT.exists(), (
        f"Output exists; refusing overwrite: {OUT}"
    )

    annotations_by_split = {}
    gt_rows = []
    gt_lookup = {}
    missing_splits = set()

    # -----------------------------------------------
    # A. Audit GT across all three partitions.
    # -----------------------------------------------
    for split in ("train", "val", "test"):
        path = DATA / "annotations" / f"{split}.json"

        if split == "val" and not path.is_file():
            for alternative in (
                "valid.json",
                "validation.json",
            ):
                candidate = DATA / "annotations" / alternative
                if candidate.is_file():
                    path = candidate
                    break

        if not path.is_file():
            if split != "val":
                raise FileNotFoundError(
                    f"Required annotation file missing: {path}"
                )

            missing_splits.add(split)
            print(
                "NOTE: Validation COCO annotations unavailable; "
                "skipping validation split."
            )
            continue

        coco = json.loads(path.read_text())

        names = {
            int(c["id"]): normalize_name(c["name"])
            for c in coco["categories"]
        }

        image_map = {
            int(im["id"]): im
            for im in coco["images"]
        }

        annotations_by_split[split] = coco

        for ann in coco["annotations"]:
            if names[int(ann["category_id"])] != "channel":
                continue

            key = (
                int(ann["image_id"]),
                int(ann["id"]),
            )

            kp = np.asarray(
                ann["keypoints"], dtype=np.float32
            ).reshape(6, 3)

            bbox = np.asarray(
                ann["bbox"], dtype=float
            )

            assert bbox[2] > 0 and bbox[3] > 0

            visible = kp[:4, 2] > 0

            if visible.all():
                signature = canonical_signature(
                    kp[:4, :2]
                )
            else:
                signature = {
                    "signature": "incomplete",
                    "fit": None,
                    "margin": None,
                    "clear": False,
                }

            row = {
                "split": split,
                "image_id": key[0],
                "annotation_id": key[1],
                "image_file": Path(
                    image_map[key[0]]["file_name"]
                ).name,
                "bbox_width": float(bbox[2]),
                "bbox_height": float(bbox[3]),
                "bbox_aspect_w_over_h": float(
                    bbox[2] / bbox[3]
                ),
                "valid_channel_points": int(
                    visible.sum()
                ),
                "gt_corner_signature":
                    signature["signature"],
                "gt_corner_fit": signature["fit"],
                "gt_corner_margin": signature["margin"],
                "gt_corner_clear": signature["clear"],
            }

            gt_rows.append(row)

            if split == "test":
                assert key not in gt_lookup
                gt_lookup[key] = (ann, row)

    assert len(gt_lookup) == 598

    # -----------------------------------------------
    # B. Load frozen ViTPose matched predictions.
    # -----------------------------------------------
    pred = np.load(
        PRED_DIR / "vitpose_predictions.npy"
    )
    image_ids = np.load(
        PRED_DIR / "vitpose_image_ids.npy"
    )
    annotation_ids = np.load(
        PRED_DIR / "vitpose_annotation_ids.npy"
    )

    assert pred.shape == (1424, 6, 2)

    pred_map = {}

    for i in range(len(pred)):
        key = (
            int(image_ids[i]),
            int(annotation_ids[i]),
        )
        assert key not in pred_map
        pred_map[key] = pred[i]

    assert len(pred_map) == 1424

    # -----------------------------------------------
    # C. Compare matched Channel GT/predictions.
    # -----------------------------------------------
    prediction_rows = []

    fixed_correct = 0
    spatial_correct = 0
    channel_matched = 0

    for key, (ann, gt_info) in sorted(
        gt_lookup.items()
    ):
        if key not in pred_map:
            continue

        kp = np.asarray(
            ann["keypoints"], dtype=np.float32
        ).reshape(6, 3)

        assert np.all(kp[:4, 2] > 0)

        gt = kp[:4, :2]
        predicted = pred_map[key][:4, :2]

        bbox = np.asarray(
            ann["bbox"], dtype=float
        )
        scale = max(bbox[2], bbox[3])

        distances = np.linalg.norm(
            gt[:, None, :]
            - predicted[None, :, :],
            axis=2,
        ) / scale

        rows, cols = linear_sum_assignment(
            distances
        )

        fixed = np.diag(distances)
        spatial = distances[rows, cols]

        assignment = [0] * 4

        for gt_j, pred_j in zip(rows, cols):
            assignment[int(gt_j)] = int(pred_j) + 1

        pred_signature = canonical_signature(
            predicted
        )

        fixed_count = int(
            (fixed <= 0.05).sum()
        )
        spatial_count = int(
            (spatial <= 0.05).sum()
        )

        fixed_correct += fixed_count
        spatial_correct += spatial_count
        channel_matched += 1

        prediction_rows.append({
            "image_file": gt_info["image_file"],
            "image_id": key[0],
            "annotation_id": key[1],
            "bbox_aspect_w_over_h":
                gt_info["bbox_aspect_w_over_h"],
            "gt_corner_signature":
                gt_info["gt_corner_signature"],
            "gt_corner_clear":
                gt_info["gt_corner_clear"],
            "prediction_corner_signature":
                pred_signature["signature"],
            "prediction_corner_clear":
                pred_signature["clear"],
            "same_corner_signature": (
                gt_info["gt_corner_signature"]
                == pred_signature["signature"]
            ),
            "nearest_gt_to_prediction_mapping":
                "-".join(map(str, assignment)),
            "fixed_correct_at_005": fixed_count,
            "spatial_correct_at_005": spatial_count,
            "fixed_mean_normalized_error":
                float(fixed.mean()),
            "spatial_mean_normalized_error":
                float(spatial.mean()),
            "spatial_mean_error_le_020":
                bool(spatial.mean() <= 0.20),
        })

    assert channel_matched == 504
    assert fixed_correct == 668
    assert spatial_correct == 1287

    # -----------------------------------------------
    # D. Save reports.
    # -----------------------------------------------
    OUT.mkdir(parents=True)

    save_csv(
        OUT / "channel_gt_spatial_signatures.csv",
        gt_rows,
    )
    save_csv(
        OUT / "channel_prediction_correspondence.csv",
        prediction_rows,
    )

    def top_patterns(rows, field, top=6):
        return Counter(
            r[field] for r in rows
        ).most_common(top)

    summary = {
        "matched_channel_objects": channel_matched,
        "fixed_PCK_005": fixed_correct / 2016,
        "spatial_PCK_005": spatial_correct / 2016,
        "gt_counts_by_split": {
            split: (
                None if split in missing_splits
                else sum(
                    r["split"] == split for r in gt_rows
                )
            )
            for split in ("train", "val", "test")
        },
        "test_channel_counts_by_image": dict(
            Counter(
                r["image_file"]
                for r in gt_rows
                if r["split"] == "test"
            )
        ),
        "method_note": (
            "Image-axis corner assignments are diagnostic "
            "and may be ambiguous for rotated or "
            "nonrectangular objects. Spatial matching "
            "uses GT points and is not deployable."
        ),
    }

    (
        OUT / "audit_summary.json"
    ).write_text(json.dumps(summary, indent=2))

    print("\nRO3 H1-A — CHANNEL ORIENTATION AUDIT")
    print("=" * 65)

    for split in ("train", "val", "test"):
        if split in missing_splits:
            print(
                f"\\n{split.upper()} GT: "
                "unavailable (annotation file missing)"
            )
            continue

        rows = [
            r for r in gt_rows
            if r["split"] == split
        ]

        clear_rows = [
            r for r in rows
            if r["gt_corner_clear"]
        ]

        print(f"\n{split.upper()} GT:")
        print("  Channel objects:", len(rows))
        print("  Clear corner assignments:", len(clear_rows))
        print("  Most common GT signatures:")

        for sig, count in top_patterns(
            clear_rows,
            "gt_corner_signature"
        ):
            print(f"    {sig}: {count}")

    print("\nTEST CHANNEL IMAGE DISTRIBUTION:")

    for image, count in sorted(
        summary["test_channel_counts_by_image"].items()
    ):
        print(f"  {image}: {count} GT objects")

    for image in ("000137.jpg", "000477.jpg"):
        rows = [
            r for r in prediction_rows
            if r["image_file"] == image
        ]

        plausible = [
            r for r in rows
            if (
                r["spatial_mean_error_le_020"]
                and r["gt_corner_clear"]
                and r["prediction_corner_clear"]
            )
        ]

        print(f"\n{image}:")
        print("  Matched Channel objects:", len(rows))
        print(
            "  Geometrically plausible and "
            "clear-corner cases:", len(plausible)
        )

        print("  GT corner signatures:")
        for sig, count in top_patterns(
            plausible, "gt_corner_signature"
        ):
            print(f"    {sig}: {count}")

        print("  Predicted corner signatures:")
        for sig, count in top_patterns(
            plausible,
            "prediction_corner_signature"
        ):
            print(f"    {sig}: {count}")

        print("  Nearest GT-to-prediction mappings:")
        for mapping, count in top_patterns(
            rows,
            "nearest_gt_to_prediction_mapping"
        ):
            print(f"    {mapping}: {count}")

    print("\nVALIDATION")
    print(
        f"  Fixed PCK: {fixed_correct}/2016 "
        f"= {fixed_correct/2016:.6f}"
    )
    print(
        f"  Spatial PCK: {spatial_correct}/2016 "
        f"= {spatial_correct/2016:.6f}"
    )
    print("  PASS: RO2 Channel baselines reproduced.")
    print("  Outputs:", OUT)


if __name__ == "__main__":
    main()
