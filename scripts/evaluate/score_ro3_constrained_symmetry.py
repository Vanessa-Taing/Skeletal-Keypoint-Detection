#!/usr/bin/env python3
"""RO2 shared-proposal comparison using original COCO GT boxes."""

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/RO2_Final_Patched"

sys.path.insert(0, str(ROOT / "scripts/evaluate"))

from common_pose_metrics_ro3_constrained_symmetry_v1 import (
    calculate_all_metrics,
    make_category_map_from_coco,
)

COCO_PATH = ROOT / (
    "datasets/InnoCount_COCO_Patched_UniqueIDs_v1"
    "/annotations/test.json"
)
MANIFEST = BASE / (
    "YOLO26n_E09_FrozenSharedProposals"
    "/shared_proposals.jsonl"
)
OUTPUT = ROOT / "experiments/RO3/H1D_ConstrainedSymmetry_Sensitivity_v1"

MODEL_FOLDERS = {
    "HRNet-W32": (
        BASE / "HRNet_SharedProposals_Output", "hrnet"
    ),
    "ViTPose-B": (
        BASE / "ViTPoseB_E03_SharedProposals", "vitpose"
    ),
}


def one_file(folder, filename):
    found = list(folder.rglob(filename))
    if len(found) != 1:
        raise RuntimeError(
            f"Expected one {filename} under {folder}; "
            f"found {len(found)}"
        )
    return found[0]


def load_model(folder, prefix, expected, annotations, manifest):
    def load(name):
        return np.load(one_file(folder, f"{prefix}_{name}.npy"))

    predictions = load("predictions")
    image_ids = load("image_ids")
    annotation_ids = load("annotation_ids")
    gt = load("gt")
    visibility = load("gt_visibility")
    category_ids = load("category_ids")
    bboxes = load("bboxes")

    assert predictions.shape == (1424, 6, 2)
    assert gt.shape == (1424, 6, 2)
    assert len(image_ids) == 1424
    assert len(annotation_ids) == 1424
    assert len(category_ids) == 1424
    assert bboxes.shape == (1424, 4)
    assert np.isfinite(predictions).all()

    result = {}

    for i in range(1424):
        key = (
            int(image_ids[i]),
            int(annotation_ids[i]),
        )
        assert key not in result, f"Duplicate {key}"
        assert key in expected, f"Unexpected identity {key}"

        ann = annotations[key]
        kp = np.asarray(
            ann["keypoints"], dtype=np.float32
        ).reshape(6, 3)

        assert int(category_ids[i]) == int(
            ann["category_id"]
        ), f"Category mismatch: {key}"

        assert np.allclose(
            gt[i], kp[:, :2],
            atol=1e-3, rtol=0
        ), f"GT coordinate mismatch: {key}"

        assert np.array_equal(
            np.asarray(visibility[i]).reshape(6) > 0,
            kp[:, 2] > 0,
        ), f"Visibility mismatch: {key}"

        x1, y1, x2, y2 = manifest[key][
            "proposal_bbox_xyxy"
        ]
        proposal_xywh = np.asarray([
            x1, y1, x2-x1, y2-y1
        ], dtype=np.float32)

        assert np.allclose(
            bboxes[i], proposal_xywh,
            atol=1e-3, rtol=0
        ), f"Proposal bbox mismatch: {key}"

        result[key] = predictions[i].astype(
            np.float32
        )

    assert set(result) == expected, (
        f"Prediction identities differ from "
        f"the frozen matched-proposal set"
    )

    print(
        f"PASS: {prefix} — 1424 matched identities, "
        "GT coordinates, visibility and proposals verified"
    )
    return result


def main():
    assert COCO_PATH.is_file()
    assert MANIFEST.is_file()
    assert not OUTPUT.exists(), (
        f"Output already exists: {OUTPUT}"
    )

    coco = json.loads(COCO_PATH.read_text())
    assert len(coco["images"]) == 74
    assert len(coco["annotations"]) == 1626

    annotations = {}
    for ann in coco["annotations"]:
        key = (int(ann["image_id"]), int(ann["id"]))
        assert key not in annotations
        annotations[key] = ann

    manifest_rows = [
        json.loads(line)
        for line in MANIFEST.read_text().splitlines()
        if line.strip()
    ]

    manifest = {}
    for row in manifest_rows:
        key = (
            int(row["image_id"]),
            int(row["annotation_id"]),
        )
        assert key not in manifest
        assert key in annotations

        ann = annotations[key]
        assert int(row["category_id"]) == int(
            ann["category_id"]
        )

        assert np.allclose(
            row["gt_bbox_xywh"], ann["bbox"],
            atol=1e-3, rtol=0
        )

        if row["matched"]:
            assert row["proposal_bbox_xyxy"] is not None
            assert row["yolo_keypoints_xy"] is not None
        else:
            assert row["proposal_bbox_xyxy"] is None

        manifest[key] = row

    assert set(manifest) == set(annotations)

    matched_keys = {
        key for key, row in manifest.items()
        if row["matched"]
    }

    assert len(matched_keys) == 1424
    assert len(annotations) - len(matched_keys) == 202

    print("PASS: Frozen manifest matches original COCO GT")
    print("Matched identities:", len(matched_keys))
    print("Unmatched identities:", 202)

    model_predictions = {}

    model_predictions["YOLO26n-Pose"] = {
        key: np.asarray(
            manifest[key]["yolo_keypoints_xy"],
            dtype=np.float32,
        ).reshape(6, 2)
        for key in matched_keys
    }

    for model, (folder, prefix) in MODEL_FOLDERS.items():
        model_predictions[model] = load_model(
            folder,
            prefix,
            matched_keys,
            annotations,
            manifest,
        )

    category_map = make_category_map_from_coco(coco)
    keys = sorted(annotations)

    results = {}
    rows = []
    profile_rows = []

    for protocol in ["matched_only", "full_test"]:
        selected_keys = (
            [k for k in keys if k in matched_keys]
            if protocol == "matched_only"
            else keys
        )

        gt = []
        visibility = []
        original_boxes = []
        category_ids = []

        for key in selected_keys:
            ann = annotations[key]
            kp = np.asarray(
                ann["keypoints"], dtype=np.float32
            ).reshape(6, 3)

            gt.append(kp[:, :2])
            visibility.append(kp[:, 2])
            original_boxes.append(ann["bbox"])
            category_ids.append(int(ann["category_id"]))

        gt = np.asarray(gt, dtype=np.float32)
        visibility = np.asarray(
            visibility, dtype=np.float32
        )
        original_boxes = np.asarray(
            original_boxes, dtype=np.float32
        )
        category_ids = np.asarray(
            category_ids, dtype=np.int64
        )

        results[protocol] = {}

        for model, predictions_by_id in model_predictions.items():
            pred = []
            valid = []

            for key in selected_keys:
                if key in matched_keys:
                    pred.append(predictions_by_id[key])
                    valid.append([True] * 6)
                else:
                    pred.append(
                        np.zeros((6, 2), dtype=np.float32)
                    )
                    valid.append([False] * 6)

            pred = np.asarray(pred, dtype=np.float32)
            valid = np.asarray(valid, dtype=bool)

            metrics = calculate_all_metrics(
                predictions=pred,
                gt=gt,
                gt_vis=visibility,
                bboxes=original_boxes,
                category_ids=category_ids,
                category_map=category_map,
                threshold=0.05,
                prediction_valid=valid,
            )

            results[protocol][model] = metrics

            pck = metrics["standard_pck"]
            order = metrics["order_invariant_pck"]
            edge = metrics["structural_edges"]

            rows.append({
                "Protocol": protocol,
                "Model": model,
                "Objects": len(selected_keys),
                "PCK@0.05": pck["overall"],
                "Order_Invariant_PCK": order["overall"],
                "Structural_Edge_Accuracy": edge["overall"],
                "Correct_Keypoints": pck["total_correct"],
                "Valid_Keypoints": pck["total_valid"],
            })

            for profile, values in metrics["profile_pck"].items():
                profile_rows.append({
                    "Protocol": protocol,
                    "Model": model,
                    "Profile": profile,
                    "Instances": values["instances"],
                    "PCK@0.05": values["pck"],
                    "Correct": values["correct"],
                    "Valid": values["valid"],
                })

    # Check original clean-test denominator.
    for model in model_predictions:
        assert (
            results["full_test"][model]
            ["standard_pck"]["total_valid"] == 7049
        ), f"Unexpected full-test denominator for {model}"

    # Original YOLO benchmark reproducibility check.
    yolo_full = results["full_test"]["YOLO26n-Pose"]
    expected_pck = 0.12611717974180736
    observed = yolo_full["standard_pck"]["overall"]

    if abs(observed - expected_pck) > 1e-5:
        raise RuntimeError(
            "Frozen YOLO predictions do not reproduce "
            "the original common-metric baseline. "
            f"Expected {expected_pck}, got {observed}. "
            "Investigate before publishing comparison."
        )

    OUTPUT.mkdir(parents=True)

    with (OUTPUT / "shared_proposal_metrics.json").open("w") as f:
        json.dump({
            "protocol_description": (
                "Frozen YOLO E09 detections with common "
                "GT matching and original COCO bbox normalization"
            ),
            "ground_truth_annotations": str(COCO_PATH),
            "frozen_manifest": str(MANIFEST),
            "total_gt": 1626,
            "matched_proposals": 1424,
            "unmatched_proposals": 202,
            "results": results,
        }, f, indent=2)

    for filename, table in [
        ("model_comparison.csv", rows),
        ("profile_comparison.csv", profile_rows),
    ]:
        with (OUTPUT / filename).open(
            "w", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(table[0])
            )
            writer.writeheader()
            writer.writerows(table)

    print("\nRO2 SHARED-PROPOSAL RESULTS")
    print("=" * 76)

    for row in rows:
        print(
            f"{row['Protocol']:<14}"
            f"{row['Model']:<16}"
            f"Objects={row['Objects']:<5}"
            f"PCK={row['PCK@0.05']:.6f}  "
            f"OI={row['Order_Invariant_PCK']:.6f}  "
            f"Edge={row['Structural_Edge_Accuracy']:.6f}"
        )

    print("\nPASS: Full-test YOLO baseline reproduced.")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
