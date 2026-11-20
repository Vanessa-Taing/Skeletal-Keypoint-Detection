#!/usr/bin/env python3
"""R3-A: Verify and summarize paired orientation validation."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_r3_orientation import (
    ROOT, ANGLES, BORDER, sha256, geometry,
    condition_id, transform_points, transform_bbox
)

SPLIT = "valid"
MANIFEST_DIR = ROOT / "experiments/RO3/R3_Orientation_v1/valid"
EVAL = ROOT / "experiments/RO3/R3_Eval_v1"

CLEAN = ROOT / (
    "experiments/RO3/R1_Eval_v1/valid_uniform_p00_r0"
)

OUT = ROOT / (
    "experiments/RO3/R3_Orientation_ValidationSummary_v1"
)

BOOTSTRAPS = 2000
SEED = 20261121


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def identity(ann):
    return int(ann["image_id"]), int(ann["id"])


def load_predictions(folder, keys):
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
        f"Wrong prediction shape: {folder}"
    )
    require(
        len(image_ids) == len(annotation_ids) == 1551,
        "Wrong identity-array length"
    )
    require(
        np.isfinite(predictions).all(),
        "Non-finite prediction coordinates"
    )

    lookup = {}

    for i in range(1551):
        key = int(image_ids[i]), int(annotation_ids[i])
        require(key not in lookup, "Duplicate prediction identity")
        lookup[key] = predictions[i]

    require(
        set(lookup) == set(keys),
        f"Prediction identity mismatch: {folder}"
    )

    return np.stack([lookup[key] for key in keys])


def make_correct(predictions, ground_truth, scales, valid):
    require(
        np.all(scales > 0),
        "Invalid bounding-box normalization scale"
    )

    errors = np.linalg.norm(
        predictions - ground_truth,
        axis=2
    )

    return (
        (errors / scales[:, None] <= 0.05)
        & valid
    )


def paired_stats(image_ids, mask, before, after, label):
    total = int(mask.sum())

    require(total > 0, "Empty subgroup")

    clean_n = int((before & mask).sum())
    changed_n = int((after & mask).sum())

    instance_mask = mask.any(axis=1)

    scenes, inverse = np.unique(
        image_ids[instance_mask],
        return_inverse=True
    )

    per_scene_valid = np.bincount(
        inverse,
        weights=mask[instance_mask].sum(axis=1),
        minlength=len(scenes)
    )

    per_scene_clean = np.bincount(
        inverse,
        weights=(
            before[instance_mask] & mask[instance_mask]
        ).sum(axis=1),
        minlength=len(scenes)
    )

    per_scene_changed = np.bincount(
        inverse,
        weights=(
            after[instance_mask] & mask[instance_mask]
        ).sum(axis=1),
        minlength=len(scenes)
    )

    ci_low = ci_high = None

    if len(scenes) >= 5:
        seed_digest = hashlib.sha256(
            f"{SEED}|{label}".encode("utf-8")
        ).digest()

        rng = np.random.default_rng(
            int.from_bytes(seed_digest[:8], "big")
        )

        draws = rng.integers(
            0,
            len(scenes),
            size=(BOOTSTRAPS, len(scenes))
        )

        denominators = per_scene_valid[draws].sum(axis=1)

        require(
            np.all(denominators > 0),
            "Invalid bootstrap draw"
        )

        differences = 100.0 * (
            per_scene_changed[draws].sum(axis=1)
            - per_scene_clean[draws].sum(axis=1)
        ) / denominators

        ci_low, ci_high = (
            float(x)
            for x in np.quantile(
                differences, [0.025, 0.975]
            )
        )

    return {
        "valid_keypoints": total,
        "clean_correct": clean_n,
        "rotated_correct": changed_n,
        "clean_pck": clean_n / total,
        "rotated_pck": changed_n / total,
        "delta_pp": 100.0 * (
            changed_n - clean_n
        ) / total,
        "ci_low_pp": ci_low,
        "ci_high_pp": ci_high,
        "source_images": len(scenes)
    }


def save_csv(path, rows):
    require(rows, "Cannot write empty table")

    with path.open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def load_manifest():
    meta_file = MANIFEST_DIR / "protocol_metadata.json"
    manifest = MANIFEST_DIR / "orientation_manifest.jsonl"

    meta = json.loads(meta_file.read_text())

    require(
        meta["protocol"] == "RO3_R3A_InPlaneRotation_v1",
        "Wrong R3 protocol"
    )
    require(meta["split"] == SPLIT, "Wrong split")
    require(
        meta["angles_degrees"] == list(ANGLES),
        "Rotation levels changed"
    )
    require(
        meta["border_bgr"] == list(BORDER),
        "Border setting changed"
    )
    require(
        meta["source_images"] == 74
        and meta["annotated_source_images"] == 72
        and meta["gt_instances"] == 1551,
        "Wrong sample counts"
    )

    generator = ROOT / "scripts/ro3/prepare_r3_orientation.py"

    require(
        sha256(generator) == meta["generator_sha256"],
        "Generator changed after manifest creation"
    )

    annotation = ROOT / meta["annotation_path"]

    require(
        sha256(annotation) == meta["annotation_sha256"],
        "COCO annotations changed"
    )

    coco = json.loads(annotation.read_text())
    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == 1551,
        "Unexpected COCO content"
    )

    rows = {angle: {} for angle in ANGLES}
    count = 0

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            count += 1

            angle = int(record["angle_degrees"])
            require(angle in rows, "Unknown rotation angle")

            iid = int(record["image_id"])

            require(
                iid not in rows[angle],
                "Duplicate image in a rotation condition"
            )
            require(
                record["split"] == SPLIT
                and record["condition_id"]
                == condition_id(angle),
                "Incorrect manifest condition"
            )
            require(
                record["source_sha256"]
                == meta["source_image_hashes"][str(iid)],
                "Source fingerprint mismatch"
            )

            matrix, width, height = geometry(
                int(record["width"]),
                int(record["height"]),
                angle
            )

            require(
                np.allclose(
                    np.asarray(record["matrix"]),
                    matrix,
                    atol=1e-8,
                    rtol=0
                ),
                "Rotation matrix differs"
            )
            require(
                (
                    int(record["output_width"]),
                    int(record["output_height"])
                ) == (width, height),
                "Canvas dimensions differ"
            )

            rows[angle][iid] = record

    require(
        count == 518
        and all(len(values) == 74 for values in rows.values()),
        "Incomplete rotation manifest"
    )

    coco_ids = {int(im["id"]) for im in coco["images"]}

    for angle in ANGLES:
        require(
            set(rows[angle]) == coco_ids,
            "COCO/manifest image set mismatch"
        )

    # Source fingerprints are checked once per
    # original photograph, rather than seven times.
    for iid, record in rows[0].items():
        file = ROOT / record["source_image"]
        require(
            file.is_file()
            and sha256(file) == record["source_sha256"],
            f"Original image changed: {iid}"
        )

    return coco, rows, manifest, meta_file


def main():
    require(
        not OUT.exists(),
        f"Refusing overwrite of existing summary: {OUT}"
    )

    coco, records, manifest, meta_file = load_manifest()

    anns = sorted(
        coco["annotations"],
        key=identity
    )

    keys = [identity(a) for a in anns]

    require(
        len(keys) == len(set(keys)) == 1551,
        "Invalid composite GT identities"
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
        "Incorrect validation keypoint denominator"
    )
    require(
        len(np.unique(image_ids)) == 72,
        "Incorrect annotated image count"
    )

    original_scales = np.maximum(
        boxes[:, 2],
        boxes[:, 3]
    )

    clean_pred = load_predictions(CLEAN, keys)

    clean_correct = make_correct(
        clean_pred,
        gt[:, :, :2],
        original_scales,
        valid
    )

    require(
        int(clean_correct.sum()) == 5196,
        "Frozen E03 clean PCK differs"
    )

    clean_report = json.loads(
        (CLEAN / "metrics.json").read_text()
    )
    clean_spec = clean_report["run_spec"]

    category_names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    summary = []
    profiles = []
    manifest_hash = sha256(manifest)
    metadata_hash = sha256(meta_file)

    for angle in ANGLES:
        cid = condition_id(angle)
        folder = EVAL / f"valid_{cid}"

        metrics = json.loads(
            (folder / "metrics.json").read_text()
        )
        spec = metrics["run_spec"]

        require(
            spec["protocol"]
            == "RO3_R3A_InPlaneRotation_Eval_v1",
            "Incorrect evaluation protocol"
        )
        require(
            spec["split"] == SPLIT
            and spec["angle_degrees"] == angle
            and spec["condition_id"] == cid,
            "Evaluation angle differs"
        )
        require(
            spec["selected_instances"] == 1551
            and spec["max_instances"] is None,
            "Partial evaluation cannot enter summary"
        )
        require(
            spec["manifest_sha256"] == manifest_hash
            and spec["protocol_metadata_sha256"]
            == metadata_hash,
            "Evaluation uses different manifest"
        )

        for field in ("checkpoint_sha256", "config_sha256"):
            require(
                spec[field] == clean_spec[field],
                "Model or configuration changed"
            )

        predictions = load_predictions(folder, keys)

        transformed_gt = np.empty(
            (1551, 6, 2),
            dtype=np.float32
        )
        transformed_scales = np.empty(
            1551,
            dtype=np.float32
        )

        for i, ann in enumerate(anns):
            record = records[angle][int(ann["image_id"])]

            matrix = np.asarray(
                record["matrix"],
                dtype=np.float64
            )

            transformed_gt[i] = transform_points(
                gt[i, :, :2], matrix
            ).astype(np.float32)

            box = transform_bbox(
                ann["bbox"], matrix
            )
            transformed_scales[i] = max(
                box[2], box[3]
            )

        primary = make_correct(
            predictions,
            transformed_gt,
            original_scales,
            valid
        )

        secondary = make_correct(
            predictions,
            transformed_gt,
            transformed_scales,
            valid
        )

        n_primary = int(primary.sum())
        n_secondary = int(secondary.sum())

        primary_saved = metrics["primary_original_box_pck"]
        secondary_saved = metrics["secondary_rotated_box_pck"]

        require(
            (
                primary_saved["total_correct"],
                primary_saved["total_valid"]
            ) == (n_primary, 6720),
            f"Primary PCK mismatch at {angle} degrees"
        )
        require(
            (
                secondary_saved["total_correct"],
                secondary_saved["total_valid"]
            ) == (n_secondary, 6720),
            f"Secondary PCK mismatch at {angle} degrees"
        )

        ratio = float(
            np.mean(transformed_scales / original_scales)
        )

        require(
            abs(
                ratio
                - metrics["mean_rotated_bbox_scale_ratio"]
            ) < 1e-6,
            "Bounding-box scale ratio mismatch"
        )

        if angle == 0:
            require(
                n_primary == n_secondary == 5196,
                "0-degree baseline mismatch"
            )

            check = metrics["clean_reference_check"]
            require(
                check is not None
                and check["maximum_coordinate_difference_px"]
                <= 0.25,
                "0-degree coordinate parity failed"
            )

        stats = paired_stats(
            image_ids, valid,
            clean_correct, primary,
            f"{cid}:overall"
        )

        summary.append({
            "angle_degrees": angle,
            "condition_id": cid,
            "instances": 1551,
            **stats,
            "secondary_rotated_box_pck":
                n_secondary / 6720,
            "secondary_minus_primary_pp":
                100 * (n_secondary - n_primary) / 6720,
            "mean_rotated_bbox_scale_ratio": ratio,
            "secondary_constrained_oi_pck":
                metrics["secondary_constrained_oi_pck"][
                    "overall"
                ],
            "secondary_structural_edge_accuracy":
                metrics["secondary_structural_edges"][
                    "overall"
                ]
        })

        for category_id, name in sorted(
            category_names.items()
        ):
            mask = (
                valid
                & (categories == category_id)[:, None]
            )

            profile_stats = paired_stats(
                image_ids, mask,
                clean_correct, primary,
                f"{cid}:{name}"
            )

            saved_profile = metrics[
                "profile_primary_pck"
            ][name]

            require(
                saved_profile["valid_keypoints"]
                == profile_stats["valid_keypoints"]
                and saved_profile["correct"]
                == profile_stats["rotated_correct"],
                f"Profile score mismatch: {name}, {angle}"
            )

            profiles.append({
                "angle_degrees": angle,
                "condition_id": cid,
                "profile": name,
                **profile_stats
            })

        print(
            f"Angle {angle:+3d}° | "
            f"Primary={stats['rotated_pck']:.4%} | "
            f"Δ={stats['delta_pp']:+.2f} pp | "
            f"95% CI=[{stats['ci_low_pp']:+.2f}, "
            f"{stats['ci_high_pp']:+.2f}] pp | "
            f"Secondary={n_secondary / 6720:.4%} | "
            f"Scale={ratio:.4f}x"
        )

    require(
        len(summary) == 7
        and len(profiles) == 35,
        "Incomplete orientation summary"
    )

    OUT.mkdir(parents=True)

    save_csv(
        OUT / "validation_orientation_summary.csv",
        summary
    )

    save_csv(
        OUT / "validation_profile_results.csv",
        profiles
    )

    analysis = {
        "protocol": "RO3_R3A_OrientationValidation_Paired_v1",
        "split": SPLIT,
        "model": "Frozen ViTPose-B E03 epoch 90",
        "angles_degrees": list(ANGLES),
        "source_images": 74,
        "annotated_source_image_clusters": 72,
        "instances": 1551,
        "valid_gt_keypoints": 6720,
        "clean_correct": 5196,
        "primary_metric": (
            "Fixed-index PCK@0.05 using original "
            "GT-box maximum dimension"
        ),
        "secondary_metric": (
            "Fixed-index PCK@0.05 using transformed "
            "axis-aligned GT-box maximum dimension"
        ),
        "bootstrap": {
            "unit": "Original annotated source photograph",
            "replications": BOOTSTRAPS,
            "seed": SEED,
            "minimum_images_for_ci": 5
        },
        "manifest_sha256": manifest_hash,
        "protocol_metadata_sha256": metadata_hash,
        "evaluator_sha256": sha256(
            ROOT / "scripts/ro3/evaluate_r3_orientation.py"
        ),
        "generator_sha256": sha256(
            ROOT / "scripts/ro3/prepare_r3_orientation.py"
        ),
        "summarizer_sha256": sha256(Path(__file__)),
        "limitations": [
            "Only in-plane orientation; not 3D viewpoint",
            "Rotation resampling and artificial borders",
            "Transformed axis-aligned GT-box proposals",
            "Top-down GT-box inference, not object detection",
            "Image-level clustering, not independent keypoints",
            "Sparse independent scenes for some profiles"
        ],
        "results": summary
    }

    with (
        OUT / "analysis_report.json"
    ).open("x") as f:
        json.dump(analysis, f, indent=2)

    print("\nPASS: R3-A paired validation analysis completed")
    print("Output:", OUT)
    print("Original R1 and R2 evidence unchanged")


if __name__ == "__main__":
    main()
