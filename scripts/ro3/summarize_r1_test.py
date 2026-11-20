#!/usr/bin/env python3
"""RO3 R1: Paired frozen-test analysis with three occlusion replicates."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from summarize_r1_validation import (
    paired_row,
    correctness,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / (
    "experiments/RO3/R1_Occlusion_v1/"
    "test_uniform_seed20261119/occlusion_manifest.jsonl"
)

BASE = ROOT / "experiments/RO3/R1_Eval_v1"

OUT = ROOT / (
    "experiments/RO3/R1_Test_PairedSummary_v1"
)

COCO_PATH = ROOT / (
    "datasets/InnoCount_COCO_Patched_UniqueIDs_v1/"
    "annotations/test.json"
)

LEVELS = (0, 10, 20, 30, 40, 50)
REPLICATES = (0, 1, 2)
N = 1626


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return int(row["image_id"]), int(row["annotation_id"])


def selected_replicates(level):
    return (0,) if level == 0 else REPLICATES


def load_manifest():
    require(MANIFEST.is_file(), MANIFEST)

    meta = json.loads(
        (MANIFEST.parent / "protocol_metadata.json").read_text()
    )

    require(meta["split"] == "test", "Wrong split")
    require(meta["mode"] == "uniform", "Wrong mode")
    require(meta["gt_instance_count"] == N, "Wrong GT count")
    require(meta["replicates_nonzero"] == 3, "Wrong replicates")
    require(
        set(meta["levels_pct"]) == set(LEVELS),
        "Unexpected occlusion levels",
    )
    require(
        sha256(COCO_PATH) == meta["source_annotation_sha256"],
        "Original COCO annotations changed",
    )

    groups = defaultdict(list)

    with MANIFEST.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            require(row["split"] == "test", "Wrong row split")
            require(row["placement_mode_requested"] == "uniform",
                    "Unexpected placement mode")
            require(row["scope"] == "instance", "Wrong scope")

            level = int(row["requested_bbox_occlusion_pct"])
            rep = int(row["replicate"])

            require(level in LEVELS, "Unexpected severity")
            require(
                rep in selected_replicates(level),
                "Unexpected replicate",
            )

            groups[(level, rep)].append(row)

    clean = None

    for level in LEVELS:
        for rep in selected_replicates(level):
            rows = groups[(level, rep)]
            rows.sort(key=key)

            require(
                len(rows) == N,
                f"Missing rows at {level}% replicate {rep}",
            )
            require(
                len({key(r) for r in rows}) == N,
                "Duplicate GT identity",
            )

            if clean is None:
                clean = rows
                continue

            for a, b in zip(clean, rows):
                require(key(a) == key(b), "Identity mismatch")
                require(
                    a["gt_keypoints"] == b["gt_keypoints"],
                    "GT keypoints changed",
                )
                require(
                    a["gt_bbox_xywh"] == b["gt_bbox_xywh"],
                    "GT bounding boxes changed",
                )
                require(
                    a["category_id"] == b["category_id"],
                    "GT categories changed",
                )

    return groups, meta


def load_run(level, rep, rows, manifest_hash, clean_spec=None):
    path = BASE / f"test_uniform_p{level:02d}_r{rep}"

    require(
        (path / "metrics.json").is_file(),
        f"Missing completed metrics: {path}",
    )

    report = json.loads(
        (path / "metrics.json").read_text()
    )

    spec = report["run_spec"]

    require(spec["split"] == "test", "Wrong run split")
    require(spec["level"] == level, "Wrong severity")
    require(spec["replicate"] == rep, "Wrong replicate")
    require(spec["max_instances"] is None, "Partial inference run")
    require(
        spec["selected_instances"] == N,
        "Incomplete inference run",
    )
    require(
        spec["manifest_sha256"] == manifest_hash,
        "Inference used a different manifest",
    )

    if clean_spec is not None:
        for field in ("checkpoint_sha256", "config_sha256"):
            require(
                spec[field] == clean_spec[field],
                f"Different model/config at {level}% rep {rep}",
            )

    predictions = np.load(
        path / "vitpose_predictions.npy",
        allow_pickle=False,
    )

    image_ids = np.load(
        path / "vitpose_image_ids.npy",
        allow_pickle=False,
    )

    annotation_ids = np.load(
        path / "vitpose_annotation_ids.npy",
        allow_pickle=False,
    )

    require(
        predictions.shape == (N, 6, 2),
        "Unexpected prediction shape",
    )
    require(
        np.isfinite(predictions).all(),
        "Non-finite predictions",
    )

    lookup = {}

    for i in range(N):
        identity = (
            int(image_ids[i]),
            int(annotation_ids[i]),
        )
        require(
            identity not in lookup,
            "Duplicate prediction identity",
        )
        lookup[identity] = predictions[i]

    require(
        set(lookup) == {key(r) for r in rows},
        "Prediction/manifest identity mismatch",
    )

    ordered = np.stack(
        [lookup[key(row)] for row in rows]
    )

    return ordered, report


def newly_hidden_mask(rows):
    mask = np.zeros((len(rows), 6), dtype=bool)

    for i, row in enumerate(rows):
        for j in row["newly_hidden_keypoint_indices"]:
            j = int(j)
            require(1 <= j <= 6, "Invalid keypoint index")
            mask[i, j-1] = True

    return mask


def add_source_image_count(result, image_ids, mask):
    n_images = len(np.unique(image_ids[mask.any(axis=1)]))
    result["source_images_with_keypoints"] = n_images

    # Extremely small scene counts do not support
    # meaningful image-cluster uncertainty intervals.
    if n_images < 3:
        result["cluster_ci_low_pp"] = None
        result["cluster_ci_high_pp"] = None

    return result


def main():
    require(
        not OUT.exists(),
        f"Refusing overwrite: {OUT}",
    )

    groups, meta = load_manifest()
    manifest_hash = sha256(MANIFEST)

    coco = json.loads(COCO_PATH.read_text())

    category_names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    clean_rows = groups[(0, 0)]

    gt_kpv = np.asarray(
        [r["gt_keypoints"] for r in clean_rows],
        dtype=np.float32,
    ).reshape(N, 6, 3)

    gt = gt_kpv[:, :, :2]
    visibility = gt_kpv[:, :, 2]

    bboxes = np.asarray(
        [r["gt_bbox_xywh"] for r in clean_rows],
        dtype=np.float32,
    )

    category_ids = np.asarray(
        [r["category_id"] for r in clean_rows],
        dtype=np.int64,
    )

    image_ids = np.asarray(
        [r["image_id"] for r in clean_rows],
        dtype=np.int64,
    )

    valid = visibility > 0

    require(
        int(valid.sum()) == 7049,
        "Wrong frozen test GT denominator",
    )

    annotated_image_count = len({
        int(a["image_id"])
        for a in coco["annotations"]
    })

    require(
        len(np.unique(image_ids)) == annotated_image_count,
        "Wrong annotated source-image count",
    )

    print(
        f"Annotated source images: {annotated_image_count}; "
        f"total COCO images: {len(coco['images'])}"
    )

    clean_predictions, clean_report = load_run(
        0, 0, clean_rows, manifest_hash
    )

    clean_correct = correctness(
        clean_predictions, gt, bboxes
    )

    require(
        int((clean_correct & valid).sum()) == 4325,
        "Frozen E03 baseline not reproduced",
    )

    clean_spec = clean_report["run_spec"]

    require(
        clean_report["reference_comparison"] is not None,
        "Missing clean reference verification",
    )

    summary_rows = []
    replicate_rows = []
    subgroup_rows = []
    profile_rows = []

    for level in LEVELS:
        all_rows = []
        all_predictions = []
        all_clean_correct = []
        all_pert_correct = []
        all_visibility = []
        all_categories = []
        all_image_ids = []
        all_coverage = []

        oi_values = []
        edge_values = []

        for rep in selected_replicates(level):
            rows = groups[(level, rep)]

            pred, report = load_run(
                level, rep, rows, manifest_hash,
                clean_spec,
            )

            pert_correct = correctness(
                pred, gt, bboxes
            )

            fixed = report["fixed_index_pck"]

            require(
                fixed["total_valid"] == 7049,
                "Incorrect GT denominator",
            )

            require(
                fixed["total_correct"]
                == int((pert_correct & valid).sum()),
                f"PCK numerator mismatch at {level}% rep {rep}",
            )

            oi = report[
                "constrained_order_invariant_pck"
            ]["overall"]

            edge = report[
                "structural_edge_accuracy"
            ]["overall"]

            oi_values.append(float(oi))
            edge_values.append(float(edge))

            replicate_rows.append({
                "severity_pct": level,
                "replicate": rep,
                "fixed_pck": float(fixed["overall"]),
                "fixed_correct": int(fixed["total_correct"]),
                "fixed_valid": 7049,
                "constrained_oi_pck": float(oi),
                "structural_edge_accuracy": float(edge),
                "mean_actual_bbox_occlusion_pct": float(
                    np.mean([
                        r["actual_bbox_occlusion_pct"]
                        for r in rows
                    ])
                ),
            })

            all_rows.extend(rows)
            all_predictions.append(pred)
            all_clean_correct.append(clean_correct)
            all_pert_correct.append(pert_correct)
            all_visibility.append(visibility)
            all_categories.append(category_ids)
            all_image_ids.append(image_ids)
            all_coverage.extend(
                r["actual_bbox_occlusion_pct"]
                for r in rows
            )

        clean_good = np.concatenate(
            all_clean_correct, axis=0
        )
        pert_good = np.concatenate(
            all_pert_correct, axis=0
        )
        vis = np.concatenate(
            all_visibility, axis=0
        )
        cats = np.concatenate(
            all_categories, axis=0
        )
        images = np.concatenate(
            all_image_ids, axis=0
        )

        hidden = newly_hidden_mask(all_rows)

        require(
            not np.any(hidden & (vis != 2)),
            "Newly hidden point was not originally visible",
        )

        masks = {
            "all_valid": vis > 0,
            "newly_hidden": hidden,
            "originally_visible_uncovered":
                (vis == 2) & ~hidden,
            "originally_occluded": vis == 1,
        }

        paired = {}

        for group, mask in masks.items():
            result = paired_row(
                level, group, mask, images,
                clean_good, pert_good,
            )

            result = add_source_image_count(
                result, images, mask
            )

            result["replicates"] = len(
                selected_replicates(level)
            )

            subgroup_rows.append(result)
            paired[group] = result

        # Per-profile paired PCK analysis.
        for cid, profile in sorted(category_names.items()):
            mask = (
                (vis > 0)
                & (cats == cid)[:, None]
            )

            result = paired_row(
                level, profile, mask, images,
                clean_good, pert_good,
            )

            result = add_source_image_count(
                result, images, mask
            )

            profile_rows.append({
                "profile": profile,
                "replicates": len(
                    selected_replicates(level)
                ),
                **result,
            })

        overall = paired["all_valid"]

        summary_rows.append({
            "severity_pct": level,
            "replicates": len(selected_replicates(level)),
            "original_instances": N,
            "original_source_images": annotated_image_count,
            "unique_valid_gt_keypoints": 7049,
            "repeated_gt_keypoint_evaluations":
                overall["valid_keypoints"],
            "mean_actual_bbox_occlusion_pct":
                float(np.mean(all_coverage)),
            "fixed_pck": overall["occluded_pck"],
            "fixed_delta_pp": overall["delta_pp"],
            "fixed_ci_low_pp":
                overall["cluster_ci_low_pp"],
            "fixed_ci_high_pp":
                overall["cluster_ci_high_pp"],
            "constrained_oi_pck":
                float(np.mean(oi_values)),
            "structural_edge_accuracy":
                float(np.mean(edge_values)),
            "newly_hidden_keypoint_evaluations":
                int(hidden.sum()),
        })

        print(
            f"{level:>2}% | "
            f"reps={len(selected_replicates(level))} | "
            f"Fixed={overall['occluded_pck']:.4%} | "
            f"Delta={overall['delta_pp']:+.2f} pp | "
            f"OI={np.mean(oi_values):.4%} | "
            f"Edge={np.mean(edge_values):.4%}"
        )

    OUT.mkdir(parents=True)

    write_csv(
        OUT / "test_summary.csv",
        summary_rows,
    )

    write_csv(
        OUT / "replicate_results.csv",
        replicate_rows,
    )

    write_csv(
        OUT / "paired_keypoint_subgroups.csv",
        subgroup_rows,
    )

    write_csv(
        OUT / "paired_profile_results.csv",
        profile_rows,
    )

    report = {
        "protocol": "RO3_R1_test_paired_v1",
        "model": "ViTPose-B E03",
        "split": "test",
        "manifest_sha256": manifest_hash,
        "checkpoint_sha256":
            clean_spec["checkpoint_sha256"],
        "config_sha256":
            clean_spec["config_sha256"],
        "source_gt_instances": N,
        "source_images": annotated_image_count,
        "unique_valid_gt_keypoints": 7049,
        "nonzero_severity_replicates": 3,
        "bootstrap_unit": "original source image",
        "bootstrap_repetitions": 2000,
        "uncertainty_note": (
            "Repeated occlusions of the same instance "
            "are kept together within the source-image "
            "bootstrap cluster. Groups represented by "
            "fewer than three source images have no "
            "reported cluster confidence interval."
        ),
        "denominator_note": (
            "At nonzero severity, pooled percentages "
            "include three perturbations of the same "
            "7049 GT keypoints. These are repeated "
            "measurements, not new independent objects."
        ),
        "metrics_note": (
            "Primary fixed-index PCK@0.05. "
            "Constrained order-invariant PCK and "
            "structural edge accuracy are secondary. "
            "All changes are relative to the same "
            "frozen clean E03 predictions."
        ),
        "results": summary_rows,
    }

    (OUT / "analysis_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nPASS: Frozen test paired analysis completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
