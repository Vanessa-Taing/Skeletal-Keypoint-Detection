#!/usr/bin/env python3
"""RO3 R1: Paired validation occlusion robustness analysis."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / (
    "experiments/RO3/R1_Occlusion_v1/"
    "valid_uniform_seed20261119/occlusion_manifest.jsonl"
)

BASE = ROOT / "experiments/RO3/R1_Eval_v1"

OUT = ROOT / (
    "experiments/RO3/"
    "R1_Validation_PairedSummary_v1"
)

LEVELS = (0, 10, 20, 30, 40, 50)
SEED = 20261119
BOOTSTRAP_REPS = 2000


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return (
        int(row["image_id"]),
        int(row["annotation_id"]),
    )


def load_manifest():
    meta = json.loads(
        (MANIFEST.parent / "protocol_metadata.json").read_text()
    )

    require(meta["split"] == "valid", "Wrong split")
    require(meta["mode"] == "uniform", "Wrong placement mode")
    require(meta["gt_instance_count"] == 1551, "Wrong GT count")
    require(
        set(LEVELS).issubset(set(meta["levels_pct"])),
        "Severity levels are missing",
    )

    groups = defaultdict(list)

    with MANIFEST.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            if int(row["replicate"]) == 0:
                groups[
                    int(row["requested_bbox_occlusion_pct"])
                ].append(row)

    for level in LEVELS:
        groups[level].sort(key=key)

        require(
            len(groups[level]) == 1551,
            f"Incomplete manifest at {level}%",
        )

        identities = [key(r) for r in groups[level]]
        require(
            len(identities) == len(set(identities)),
            f"Duplicate identity at {level}%",
        )

    clean = groups[0]

    for level in LEVELS[1:]:
        for a, b in zip(clean, groups[level]):
            require(key(a) == key(b), "Identity mismatch")
            require(
                a["gt_keypoints"] == b["gt_keypoints"],
                "GT keypoints changed",
            )
            require(
                a["gt_bbox_xywh"] == b["gt_bbox_xywh"],
                "GT bbox changed",
            )
            require(
                a["category_id"] == b["category_id"],
                "Category changed",
            )

    return groups, meta


def load_run(level, rows, manifest_hash):
    path = BASE / f"valid_uniform_p{level:02d}_r0"

    report = json.loads(
        (path / "metrics.json").read_text()
    )

    spec = report["run_spec"]

    require(spec["split"] == "valid", "Wrong run split")
    require(spec["level"] == level, "Wrong run severity")
    require(spec["replicate"] == 0, "Wrong replicate")
    require(
        spec["max_instances"] is None,
        "Partial run cannot be summarized",
    )
    require(
        spec["selected_instances"] == 1551,
        "Incomplete prediction run",
    )
    require(
        spec["manifest_sha256"] == manifest_hash,
        "Manifest has changed since inference",
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
        predictions.shape == (1551, 6, 2),
        f"Unexpected prediction shape for {level}%",
    )

    prediction_map = {}

    for i, (im_id, ann_id) in enumerate(
        zip(image_ids, annotation_ids)
    ):
        identity = (int(im_id), int(ann_id))
        require(
            identity not in prediction_map,
            "Duplicate prediction identity",
        )
        prediction_map[identity] = predictions[i]

    require(
        set(prediction_map) == {key(r) for r in rows},
        "Prediction identities do not match manifest",
    )

    ordered = np.stack(
        [prediction_map[key(r)] for r in rows]
    )

    require(
        np.isfinite(ordered).all(),
        "Non-finite prediction encountered",
    )

    return ordered, report


def correctness(pred, gt, bbox):
    scales = np.maximum(
        bbox[:, 2], bbox[:, 3]
    ).astype(np.float32)

    require(np.all(scales > 0), "Invalid GT bbox")

    errors = (
        np.linalg.norm(pred - gt, axis=2)
        / scales[:, None]
    )

    return errors <= 0.05


def bootstrap_delta(
    image_ids, mask, clean_correct, pert_correct,
    level, group,
):
    """Paired micro-PCK difference, resampling source images."""

    if not np.any(mask):
        return (None, None)

    images, inverse = np.unique(
        image_ids, return_inverse=True
    )

    m = len(images)

    denominator = np.bincount(
        inverse,
        weights=mask.sum(axis=1),
        minlength=m,
    )

    clean_counts = np.bincount(
        inverse,
        weights=(clean_correct & mask).sum(axis=1),
        minlength=m,
    )

    pert_counts = np.bincount(
        inverse,
        weights=(pert_correct & mask).sum(axis=1),
        minlength=m,
    )

    seed_text = f"{SEED}|{level}|{group}"
    seed = int.from_bytes(
        hashlib.sha256(
            seed_text.encode()
        ).digest()[:8],
        "big",
    )

    rng = np.random.default_rng(seed)

    draws = rng.integers(
        0, m, size=(BOOTSTRAP_REPS, m)
    )

    totals = denominator[draws].sum(axis=1)
    old = clean_counts[draws].sum(axis=1)
    new = pert_counts[draws].sum(axis=1)

    valid = totals > 0

    if not np.any(valid):
        return (None, None)

    differences = (
        100.0
        * (new[valid] - old[valid])
        / totals[valid]
    )

    low, high = np.quantile(
        differences, [0.025, 0.975]
    )

    return float(low), float(high)


def paired_row(
    level, group, mask, images,
    clean_correct, pert_correct,
):
    denominator = int(mask.sum())

    if denominator == 0:
        return {
            "severity_pct": level,
            "group": group,
            "valid_keypoints": 0,
            "clean_correct": 0,
            "occluded_correct": 0,
            "clean_pck": None,
            "occluded_pck": None,
            "delta_pp": None,
            "cluster_ci_low_pp": None,
            "cluster_ci_high_pp": None,
        }

    old_n = int((clean_correct & mask).sum())
    new_n = int((pert_correct & mask).sum())

    low, high = bootstrap_delta(
        images, mask, clean_correct, pert_correct,
        level, group,
    )

    return {
        "severity_pct": level,
        "group": group,
        "valid_keypoints": denominator,
        "clean_correct": old_n,
        "occluded_correct": new_n,
        "clean_pck": old_n / denominator,
        "occluded_pck": new_n / denominator,
        "delta_pp": 100.0 * (new_n - old_n) / denominator,
        "cluster_ci_low_pp": low,
        "cluster_ci_high_pp": high,
    }


def write_csv(path, rows):
    require(rows, f"No rows for {path}")

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    require(
        not OUT.exists(),
        f"Output already exists: {OUT}",
    )

    groups, metadata = load_manifest()
    manifest_hash = sha256(MANIFEST)

    rows = groups[0]
    n = len(rows)

    gt_kpv = np.asarray(
        [r["gt_keypoints"] for r in rows],
        dtype=np.float32,
    ).reshape(n, 6, 3)

    gt = gt_kpv[:, :, :2]
    visibility = gt_kpv[:, :, 2]

    bbox = np.asarray(
        [r["gt_bbox_xywh"] for r in rows],
        dtype=np.float32,
    )

    categories = np.asarray(
        [r["category_id"] for r in rows]
    )

    image_ids = np.asarray(
        [r["image_id"] for r in rows]
    )

    valid = visibility > 0

    require(
        int(valid.sum()) == 6720,
        "Unexpected validation GT denominator",
    )

    clean_pred, clean_report = load_run(
        0, rows, manifest_hash
    )

    clean_correct = correctness(
        clean_pred, gt, bbox
    )

    require(
        int((clean_correct & valid).sum()) == 5196,
        "Clean validation PCK not reproduced",
    )

    reference_spec = clean_report["run_spec"]

    summary = []
    subgroup_results = []
    profile_results = []

    # Load category names directly from the frozen
    # annotation JSON used to create the manifest.
    source_coco = json.loads(
        Path(metadata["source_annotation"]).read_text()
    )

    category_names = {
        int(c["id"]): c["name"]
        for c in source_coco["categories"]
    }

    for level in LEVELS:
        current_rows = groups[level]

        pred, report = load_run(
            level, current_rows, manifest_hash
        )

        spec = report["run_spec"]

        for field in (
            "checkpoint_sha256",
            "config_sha256",
        ):
            require(
                spec[field] == reference_spec[field],
                f"Model or config changed at {level}%",
            )

        correct = correctness(pred, gt, bbox)

        total_correct = int((correct & valid).sum())

        require(
            total_correct
            == report["fixed_index_pck"]["total_correct"],
            f"Fixed-index numerator mismatch at {level}%",
        )

        require(
            report["fixed_index_pck"]["total_valid"] == 6720,
            f"Fixed-index denominator mismatch at {level}%",
        )

        if level == 10:
            require(
                total_correct == 3863,
                "Existing 10% validation result changed",
            )

        newly_hidden = np.zeros(
            (n, 6), dtype=bool
        )

        for i, row in enumerate(current_rows):
            for j in row["newly_hidden_keypoint_indices"]:
                newly_hidden[i, int(j) - 1] = True

        require(
            not np.any(
                newly_hidden & (visibility != 2)
            ),
            "Newly hidden index is not originally visible",
        )

        masks = {
            "all_valid": valid,
            "newly_hidden": newly_hidden,
            "originally_visible_uncovered":
                (visibility == 2) & ~newly_hidden,
            "originally_occluded":
                visibility == 1,
        }

        paired = {}

        for name, mask in masks.items():
            result = paired_row(
                level, name, mask, image_ids,
                clean_correct, correct,
            )
            subgroup_results.append(result)
            paired[name] = result

        all_row = paired["all_valid"]

        summary.append({
            "severity_pct": level,
            "original_instances": n,
            "mean_actual_bbox_coverage_pct":
                float(np.mean([
                    r["actual_bbox_occlusion_pct"]
                    for r in current_rows
                ])),
            "fixed_pck": all_row["occluded_pck"],
            "fixed_delta_pp": all_row["delta_pp"],
            "fixed_ci_low_pp":
                all_row["cluster_ci_low_pp"],
            "fixed_ci_high_pp":
                all_row["cluster_ci_high_pp"],
            "constrained_oi_pck":
                report[
                    "constrained_order_invariant_pck"
                ]["overall"],
            "structural_edge_accuracy":
                report[
                    "structural_edge_accuracy"
                ]["overall"],
            "newly_hidden_keypoints":
                int(newly_hidden.sum()),
        })

        # Profile-specific paired fixed-index scores.
        for cid, profile in sorted(category_names.items()):
            mask = (
                valid
                & (categories == cid)[:, None]
            )

            result = paired_row(
                level, profile, mask, image_ids,
                clean_correct, correct,
            )

            profile_results.append({
                "profile": profile,
                **result,
            })

        print(
            f"{level:>2}% | "
            f"Fixed={all_row['occluded_pck']:.4%} | "
            f"Delta={all_row['delta_pp']:+.2f} pp | "
            f"OI={summary[-1]['constrained_oi_pck']:.4%} | "
            f"Edge={summary[-1]['structural_edge_accuracy']:.4%}"
        )

    OUT.mkdir(parents=True)

    write_csv(
        OUT / "validation_summary.csv",
        summary,
    )

    write_csv(
        OUT / "paired_keypoint_subgroups.csv",
        subgroup_results,
    )

    write_csv(
        OUT / "paired_profile_results.csv",
        profile_results,
    )

    report = {
        "protocol": "RO3_R1_validation_paired_v1",
        "manifest_sha256": manifest_hash,
        "original_gt_instances": n,
        "original_source_images": int(
            len(np.unique(image_ids))
        ),
        "valid_gt_keypoints": int(valid.sum()),
        "bootstrap_unit": "original source image",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "confidence_level": 0.95,
        "primary_metric": "fixed-index PCK@0.05",
        "note": (
            "For each severity and subgroup, clean and "
            "occluded predictions are compared using "
            "the same GT keypoint identities. "
            "Subgroup definitions are conditioned on "
            "the actual synthetic occlusion at that "
            "severity. Bootstrap resamples original "
            "source images, not individual keypoints."
        ),
        "results": summary,
    }

    (OUT / "analysis_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nPASS: Paired validation analysis completed")
    print("Source images:", len(np.unique(image_ids)))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
