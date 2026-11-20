#!/usr/bin/env python3
"""RO3 R2: Paired validation analysis for exposure and contrast."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_r2_lighting import (
    LEVELS, NEUTRAL, condition_id
)

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "experiments/RO3/R2_Eval_v1"
MANIFEST_BASE = ROOT / "experiments/RO3/R2_Lighting_v1"

CLEAN_DIR = ROOT / (
    "experiments/RO3/R1_Eval_v1/"
    "valid_uniform_p00_r0"
)

OUT = ROOT / (
    "experiments/RO3/"
    "R2_Validation_PairedSummary_v1"
)

BOOTSTRAP_REPS = 2000
SEED = 20261120


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(ann):
    return int(ann["image_id"]), int(ann["id"])


def write_csv(path, rows):
    require(rows, f"No records for {path}")

    with path.open("x", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def load_arrays(folder, keys):
    predictions = np.load(
        folder / "vitpose_predictions.npy",
        allow_pickle=False
    )
    image_ids = np.load(
        folder / "vitpose_image_ids.npy",
        allow_pickle=False
    )
    annotation_ids = np.load(
        folder / "vitpose_annotation_ids.npy",
        allow_pickle=False
    )

    require(
        predictions.shape == (1551, 6, 2),
        f"Unexpected predictions in {folder}",
    )
    require(
        len(image_ids) == len(annotation_ids) == 1551,
        "Identity-array size mismatch",
    )
    require(
        np.isfinite(predictions).all(),
        "Non-finite prediction coordinates",
    )

    lookup = {}

    for i in range(1551):
        key = (
            int(image_ids[i]),
            int(annotation_ids[i])
        )
        require(key not in lookup, "Duplicate prediction identity")
        lookup[key] = predictions[i]

    require(set(lookup) == set(keys), "GT identities differ")

    return np.stack([lookup[k] for k in keys])


def correctness(predictions, gt, boxes):
    scale = np.maximum(
        boxes[:, 2], boxes[:, 3]
    )

    require(np.all(scale > 0), "Invalid GT box")

    distance = (
        np.linalg.norm(predictions - gt, axis=2)
        / scale[:, None]
    )

    return distance <= 0.05


def paired_stats(
    image_ids, mask, old_correct, new_correct,
    seed_label
):
    """Micro PCK with source-image clustered bootstrap."""
    valid_n = int(mask.sum())

    if valid_n == 0:
        return {
            "valid_keypoints": 0,
            "clean_pck": None,
            "perturbed_pck": None,
            "delta_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "source_images": 0,
        }

    old_n = int(np.count_nonzero(old_correct & mask))
    new_n = int(np.count_nonzero(new_correct & mask))

    included = np.any(mask, axis=1)

    images, inverse = np.unique(
        image_ids[included],
        return_inverse=True
    )

    totals = np.bincount(
        inverse,
        weights=mask[included].sum(axis=1),
        minlength=len(images)
    )

    old_by_image = np.bincount(
        inverse,
        weights=(old_correct[included] & mask[included]).sum(axis=1),
        minlength=len(images)
    )

    new_by_image = np.bincount(
        inverse,
        weights=(new_correct[included] & mask[included]).sum(axis=1),
        minlength=len(images)
    )

    low = high = None

    # Sparse profile groups cannot support meaningful
    # image-level uncertainty intervals.
    if len(images) >= 5:
        seed_bytes = hashlib.sha256(
            f"{SEED}|{seed_label}".encode()
        ).digest()

        rng = np.random.default_rng(
            int.from_bytes(seed_bytes[:8], "big")
        )

        draws = rng.integers(
            0,
            len(images),
            size=(BOOTSTRAP_REPS, len(images))
        )

        den = totals[draws].sum(axis=1)
        a = old_by_image[draws].sum(axis=1)
        b = new_by_image[draws].sum(axis=1)

        valid_draws = den > 0

        delta = (
            100.0
            * (b[valid_draws] - a[valid_draws])
            / den[valid_draws]
        )

        low, high = (
            float(x)
            for x in np.quantile(delta, [0.025, 0.975])
        )

    return {
        "valid_keypoints": valid_n,
        "clean_pck": old_n / valid_n,
        "perturbed_pck": new_n / valid_n,
        "delta_pp": 100.0 * (new_n - old_n) / valid_n,
        "ci_low_pp": low,
        "ci_high_pp": high,
        "source_images": len(images),
    }


def read_protocol(mode):
    folder = MANIFEST_BASE / f"valid_{mode}"

    metadata = json.loads(
        (folder / "protocol_metadata.json").read_text()
    )

    manifest = folder / "lighting_manifest.jsonl"

    require(metadata["split"] == "valid", "Wrong split")
    require(metadata["mode"] == mode, "Wrong lighting axis")
    require(metadata["levels"] == LEVELS[mode], "Wrong levels")
    require(metadata["source_images"] == 74, "Wrong image count")
    require(metadata["source_gt_instances"] == 1551, "Wrong GT count")

    generator = ROOT / "scripts/ro3/prepare_r2_lighting.py"

    require(
        sha256(generator) == metadata["generator_sha256"],
        "Lighting generator differs from frozen manifest",
    )

    annotation = ROOT / metadata["annotation_path"]

    require(
        sha256(annotation) == metadata["annotation_sha256"],
        "COCO annotation differs from R2 manifest",
    )

    # Verify completeness of all transformation conditions.
    counts = {}

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            cid = row["condition_id"]

            counts.setdefault(cid, set())

            image_id = int(row["image_id"])

            require(
                image_id not in counts[cid],
                f"Duplicate image in {cid}",
            )

            counts[cid].add(image_id)

    expected_ids = {
        condition_id(mode, v)
        for v in LEVELS[mode]
    }

    require(
        set(counts) == expected_ids,
        "Unexpected/missing lighting conditions",
    )
    require(
        all(len(ids) == 74 for ids in counts.values()),
        "Incomplete source-image manifest",
    )

    return manifest, metadata, json.loads(annotation.read_text())


def main():
    require(
        not OUT.exists(),
        f"Refusing to overwrite existing report: {OUT}",
    )

    protocols = {
        mode: read_protocol(mode)
        for mode in ("exposure", "contrast")
    }

    # Use the verified frozen R1 validation clean predictions
    # as a common reference for both lighting axes.
    coco = protocols["exposure"][2]
    coco_contrast = protocols["contrast"][2]

    require(
        protocols["exposure"][1]["annotation_sha256"]
        == protocols["contrast"][1]["annotation_sha256"],
        "Exposure and contrast have different GT source data",
    )

    require(
        len(coco_contrast["annotations"]) == 1551,
        "Contrast GT count mismatch",
    )

    anns = sorted(coco["annotations"], key=identity)
    keys = [identity(a) for a in anns]

    require(
        len(keys) == len(set(keys)) == 1551,
        "Invalid GT identity set",
    )

    gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32
    ).reshape(1551, 6, 3)

    boxes = np.asarray(
        [a["bbox"] for a in anns],
        dtype=np.float32
    )

    categories = np.asarray(
        [a["category_id"] for a in anns],
        dtype=np.int64
    )

    image_ids = np.asarray(
        [a["image_id"] for a in anns],
        dtype=np.int64
    )

    valid = gt[:, :, 2] > 0

    require(
        int(valid.sum()) == 6720,
        "Wrong validation denominator",
    )

    require(
        len(np.unique(image_ids)) == 72,
        "Wrong annotated source-image count",
    )

    clean_pred = load_arrays(CLEAN_DIR, keys)

    clean_correct = correctness(
        clean_pred, gt[:, :, :2], boxes
    )

    require(
        int(np.count_nonzero(clean_correct & valid)) == 5196,
        "Original E03 clean PCK not reproduced",
    )

    clean_report = json.loads(
        (CLEAN_DIR / "metrics.json").read_text()
    )
    clean_spec = clean_report["run_spec"]

    category_names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    summary_rows = []
    profile_rows = []

    for mode in ("exposure", "contrast"):
        manifest, metadata, _ = protocols[mode]
        manifest_hash = sha256(manifest)

        for value in LEVELS[mode]:
            cid = condition_id(mode, value)

            folder = BASE / f"valid_{mode}_{cid}"

            report = json.loads(
                (folder / "metrics.json").read_text()
            )

            spec = report["run_spec"]

            require(
                spec["split"] == "valid"
                and spec["mode"] == mode
                and spec["condition_id"] == cid,
                "Wrong evaluation condition",
            )

            require(
                abs(float(spec["value"]) - value) < 1e-9,
                "Wrong lighting setting",
            )

            require(
                spec["max_instances"] is None
                and spec["selected_instances"] == 1551,
                "Incomplete full validation inference",
            )

            require(
                spec["manifest_sha256"] == manifest_hash,
                "Manifest hash mismatch",
            )

            for name in ("checkpoint_sha256", "config_sha256"):
                require(
                    spec[name] == clean_spec[name],
                    f"Frozen model changed in {cid}",
                )

            predictions = load_arrays(folder, keys)

            correct = correctness(
                predictions, gt[:, :, :2], boxes
            )

            measured_correct = int(
                np.count_nonzero(correct & valid)
            )

            fixed = report["fixed_index_pck"]

            require(
                fixed["total_valid"] == 6720
                and fixed["total_correct"] == measured_correct,
                f"Fixed PCK scorer mismatch in {cid}",
            )

            if abs(value - NEUTRAL[mode]) < 1e-9:
                require(
                    report["reference_comparison"] is not None,
                    "Missing neutral reference verification",
                )
                require(
                    measured_correct == 5196,
                    "Neutral baseline not reproduced",
                )

                max_difference = float(
                    np.abs(predictions - clean_pred).max()
                )
                require(
                    max_difference <= 0.25,
                    "Neutral predictions differ from frozen baseline",
                )

            total = paired_stats(
                image_ids,
                valid,
                clean_correct,
                correct,
                f"{mode}:{cid}:overall",
            )

            summary_rows.append({
                "mode": mode,
                "value": value,
                "condition_id": cid,
                "instances": 1551,
                "valid_keypoints": 6720,
                "correct_keypoints": measured_correct,
                **total,
                "constrained_oi_pck": report[
                    "constrained_order_invariant_pck"
                ]["overall"],
                "original_oi_pck": report[
                    "original_order_invariant_pck"
                ]["overall"],
                "structural_edge_accuracy": report[
                    "structural_edge_accuracy"
                ]["overall"],
            })

            for category_id, name in sorted(
                category_names.items()
            ):
                mask = (
                    valid
                    & (categories == category_id)[:, None]
                )

                profile_rows.append({
                    "mode": mode,
                    "value": value,
                    "condition_id": cid,
                    "profile": name,
                    **paired_stats(
                        image_ids,
                        mask,
                        clean_correct,
                        correct,
                        f"{mode}:{cid}:{name}",
                    ),
                })

            print(
                f"{mode:<9} {value:>+5.2f} | "
                f"Fixed={fixed['overall']:.4%} | "
                f"Delta={total['delta_pp']:+.2f} pp | "
                f"OI={summary_rows[-1]['constrained_oi_pck']:.4%} | "
                f"Edge={summary_rows[-1]['structural_edge_accuracy']:.4%}"
            )

    OUT.mkdir(parents=True)

    write_csv(
        OUT / "validation_lighting_summary.csv",
        summary_rows
    )
    write_csv(
        OUT / "validation_profile_results.csv",
        profile_rows
    )

    metadata = {
        "protocol": "RO3_R2_Validation_Paired_v1",
        "model": "ViTPose-B E03, frozen",
        "baseline": "R1 clean validation",
        "gt_instances": 1551,
        "valid_gt_keypoints": 6720,
        "annotated_source_images": 72,
        "primary_metric": "Fixed-index PCK@0.05",
        "bootstrap": {
            "unit": "Original source photograph",
            "replications": BOOTSTRAP_REPS,
            "seed": SEED,
            "minimum_source_images_for_ci": 5,
        },
        "interpretation": (
            "Separate exposure and contrast axes. "
            "All paired differences use the same original "
            "GT keypoints and clean predictions. "
            "No object counting or detector performance "
            "is evaluated."
        ),
        "results": summary_rows,
    }

    (OUT / "analysis_report.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nPASS: R2 paired validation analysis completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
