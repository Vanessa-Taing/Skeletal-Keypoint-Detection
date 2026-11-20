#!/usr/bin/env python3
"""R3-A frozen-test paired orientation analysis."""

import csv
import json
from pathlib import Path

import numpy as np

from prepare_r3_orientation import (
    ROOT, ANGLES, BORDER, sha256, geometry,
    condition_id, transform_points, transform_bbox,
)
from summarize_r3_orientation_validation import (
    paired_stats, identity, make_correct, save_csv,
)

SPLIT = "test"
N_OBJECTS = 1626
N_VALID = 7049
N_CLEAN = 4325
N_IMAGES = 71

MANIFEST_DIR = ROOT / "experiments/RO3/R3_Orientation_v1/test"
EVAL = ROOT / "experiments/RO3/R3_Eval_v1"
CLEAN = ROOT / (
    "experiments/RO3/R1_Eval_v1/test_uniform_p00_r0"
)
OUT = ROOT / (
    "experiments/RO3/R3_Orientation_TestSummary_v1"
)
LOCK = ROOT / (
    "experiments/RO3/R3A_TestProtocolLock_v1.json"
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_predictions(folder, keys):
    p = np.load(
        folder / "vitpose_predictions.npy",
        allow_pickle=False
    )
    im = np.load(
        folder / "vitpose_image_ids.npy",
        allow_pickle=False
    )
    aid = np.load(
        folder / "vitpose_annotation_ids.npy",
        allow_pickle=False
    )

    require(
        p.shape == (N_OBJECTS, 6, 2),
        f"Wrong prediction shape: {folder}"
    )
    require(
        len(im) == len(aid) == N_OBJECTS,
        "Wrong prediction ID array lengths"
    )
    require(np.isfinite(p).all(), "Nonfinite predictions")

    lookup = {}
    for i in range(N_OBJECTS):
        key = int(im[i]), int(aid[i])
        require(key not in lookup, "Duplicate prediction ID")
        lookup[key] = p[i]

    require(
        set(lookup) == set(keys),
        f"Prediction identities differ: {folder}"
    )

    return np.stack([lookup[key] for key in keys])


def verify_lock():
    require(LOCK.is_file(), "Missing test protocol lock")
    lock = json.loads(LOCK.read_text())

    require(
        lock["protocol"] == "RO3_R3A_FrozenTest_v1",
        "Incorrect test protocol lock"
    )

    for name, expected in lock["files_sha256"].items():
        path = ROOT / name
        require(
            path.is_file() and sha256(path) == expected,
            f"Locked input changed: {name}"
        )

    return lock


def load_manifest():
    manifest = MANIFEST_DIR / "orientation_manifest.jsonl"
    meta_path = MANIFEST_DIR / "protocol_metadata.json"
    meta = json.loads(meta_path.read_text())

    require(
        meta["protocol"] == "RO3_R3A_InPlaneRotation_v1",
        "Wrong rotation protocol"
    )
    require(
        meta["split"] == SPLIT
        and meta["angles_degrees"] == list(ANGLES)
        and meta["border_bgr"] == list(BORDER),
        "Test transformation definition differs"
    )
    require(
        meta["source_images"] == 74
        and meta["annotated_source_images"] == N_IMAGES
        and meta["gt_instances"] == N_OBJECTS,
        "Unexpected test sample count"
    )
    require(
        sha256(ROOT / "scripts/ro3/prepare_r3_orientation.py")
        == meta["generator_sha256"],
        "Generator differs from saved manifest"
    )

    coco_file = ROOT / meta["annotation_path"]
    require(
        sha256(coco_file) == meta["annotation_sha256"],
        "COCO annotation changed"
    )
    coco = json.loads(coco_file.read_text())

    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == N_OBJECTS,
        "Unexpected test COCO content"
    )

    records = {a: {} for a in ANGLES}
    count = 0

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            count += 1
            angle = int(row["angle_degrees"])
            iid = int(row["image_id"])

            require(angle in records, "Unknown angle")
            require(
                iid not in records[angle],
                "Duplicate image and angle"
            )
            require(
                row["split"] == SPLIT
                and row["condition_id"] == condition_id(angle),
                "Wrong manifest condition"
            )
            require(
                row["source_sha256"]
                == meta["source_image_hashes"][str(iid)],
                "Incorrect source hash record"
            )

            matrix, width, height = geometry(
                int(row["width"]),
                int(row["height"]),
                angle
            )
            require(
                np.allclose(
                    np.asarray(row["matrix"]),
                    matrix,
                    atol=1e-8, rtol=0
                ),
                "Affine transformation changed"
            )
            require(
                (
                    int(row["output_width"]),
                    int(row["output_height"])
                ) == (width, height),
                "Expanded canvas changed"
            )

            records[angle][iid] = row

    require(
        count == 518
        and all(len(v) == 74 for v in records.values()),
        "Incomplete test manifest"
    )

    image_ids = {int(x["id"]) for x in coco["images"]}

    for angle in ANGLES:
        require(
            set(records[angle]) == image_ids,
            "Incorrect test image set"
        )

    # Verify the bytes of all 74 original photographs.
    for iid, row in records[0].items():
        source = ROOT / row["source_image"]
        require(
            source.is_file()
            and sha256(source) == row["source_sha256"],
            f"Original test image differs: {iid}"
        )

    return coco, records, manifest, meta_path


def main():
    require(
        not OUT.exists(),
        f"Refusing to overwrite existing summary: {OUT}"
    )

    verify_lock()
    coco, records, manifest, meta_path = load_manifest()

    anns = sorted(coco["annotations"], key=identity)
    keys = [identity(a) for a in anns]

    require(
        len(keys) == len(set(keys)) == N_OBJECTS,
        "Duplicate composite GT identity"
    )

    gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32
    ).reshape(N_OBJECTS, 6, 3)

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
        int(valid.sum()) == N_VALID
        and len(np.unique(image_ids)) == N_IMAGES,
        "Test GT totals differ"
    )

    original_scales = np.maximum(
        boxes[:, 2], boxes[:, 3]
    )

    clean_pred = read_predictions(CLEAN, keys)

    clean_correct = make_correct(
        clean_pred,
        gt[:, :, :2],
        original_scales,
        valid
    )

    require(
        int(clean_correct.sum()) == N_CLEAN,
        "Frozen E03 clean test baseline differs"
    )

    clean_report = json.loads(
        (CLEAN / "metrics.json").read_text()
    )
    clean_spec = clean_report["run_spec"]

    profiles = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    manifest_hash = sha256(manifest)
    metadata_hash = sha256(meta_path)
    summary = []
    profile_rows = []

    for angle in ANGLES:
        cid = condition_id(angle)
        folder = EVAL / f"test_{cid}"

        metrics = json.loads(
            (folder / "metrics.json").read_text()
        )
        spec = metrics["run_spec"]

        require(
            spec["protocol"]
            == "RO3_R3A_InPlaneRotation_Eval_v1",
            f"Wrong evaluation protocol: {cid}"
        )
        require(
            spec["split"] == SPLIT
            and spec["angle_degrees"] == angle
            and spec["condition_id"] == cid,
            f"Wrong evaluation angle: {cid}"
        )
        require(
            spec["selected_instances"] == N_OBJECTS
            and spec["max_instances"] is None,
            "Incomplete evaluation"
        )
        require(
            spec["manifest_sha256"] == manifest_hash
            and spec["protocol_metadata_sha256"]
            == metadata_hash,
            "Evaluation used another manifest"
        )

        for name in ("checkpoint_sha256", "config_sha256"):
            require(
                spec[name] == clean_spec[name],
                f"Model/config changed: {cid}"
            )

        pred = read_predictions(folder, keys)

        rotated_gt = np.empty(
            (N_OBJECTS, 6, 2), dtype=np.float32
        )
        rotated_scales = np.empty(
            N_OBJECTS, dtype=np.float32
        )

        for i, ann in enumerate(anns):
            row = records[angle][int(ann["image_id"])]
            matrix = np.asarray(
                row["matrix"], dtype=np.float64
            )

            rotated_gt[i] = transform_points(
                gt[i, :, :2], matrix
            ).astype(np.float32)

            bbox = transform_bbox(
                ann["bbox"], matrix
            )
            rotated_scales[i] = max(bbox[2], bbox[3])

        primary = make_correct(
            pred, rotated_gt, original_scales, valid
        )
        secondary = make_correct(
            pred, rotated_gt, rotated_scales, valid
        )

        n_primary = int(primary.sum())
        n_secondary = int(secondary.sum())

        stored_primary = metrics["primary_original_box_pck"]
        stored_secondary = metrics["secondary_rotated_box_pck"]

        require(
            (
                stored_primary["total_correct"],
                stored_primary["total_valid"]
            ) == (n_primary, N_VALID),
            f"Primary metric mismatch: {cid}"
        )
        require(
            (
                stored_secondary["total_correct"],
                stored_secondary["total_valid"]
            ) == (n_secondary, N_VALID),
            f"Secondary metric mismatch: {cid}"
        )

        ratio = float(
            np.mean(rotated_scales / original_scales)
        )
        require(
            abs(
                ratio
                - metrics["mean_rotated_bbox_scale_ratio"]
            ) < 1e-6,
            "Rotated-box scale ratio mismatch"
        )

        if angle == 0:
            require(
                n_primary == n_secondary == N_CLEAN,
                "Test clean baseline not reproduced"
            )

            reference = metrics["clean_reference_check"]
            require(
                reference is not None
                and reference[
                    "maximum_coordinate_difference_px"
                ] <= 0.25,
                "Clean test coordinate parity failed"
            )

        stats = paired_stats(
            image_ids,
            valid,
            clean_correct,
            primary,
            f"{cid}:overall"
        )

        summary.append({
            "angle_degrees": angle,
            "condition_id": cid,
            "instances": N_OBJECTS,
            **stats,
            "secondary_rotated_box_pck":
                n_secondary / N_VALID,
            "secondary_minus_primary_pp":
                100 * (n_secondary - n_primary) / N_VALID,
            "mean_rotated_bbox_scale_ratio": ratio,
            "secondary_constrained_oi_pck":
                metrics["secondary_constrained_oi_pck"][
                    "overall"
                ],
            "secondary_structural_edge_accuracy":
                metrics["secondary_structural_edges"][
                    "overall"
                ],
        })

        for category_id, name in sorted(profiles.items()):
            mask = (
                valid
                & (categories == category_id)[:, None]
            )
            group = paired_stats(
                image_ids,
                mask,
                clean_correct,
                primary,
                f"{cid}:{name}"
            )

            saved = metrics["profile_primary_pck"][name]
            require(
                saved["valid_keypoints"]
                == group["valid_keypoints"]
                and saved["correct"]
                == group["rotated_correct"],
                f"Profile mismatch: {name}, {cid}"
            )

            profile_rows.append({
                "angle_degrees": angle,
                "condition_id": cid,
                "profile": name,
                **group,
            })

        print(
            f"{angle:+3d}° | "
            f"PCK={stats['rotated_pck']:.4%} | "
            f"Δ={stats['delta_pp']:+.2f} pp | "
            f"95% CI=[{stats['ci_low_pp']:+.2f}, "
            f"{stats['ci_high_pp']:+.2f}] pp | "
            f"secondary={n_secondary / N_VALID:.4%}"
        )

    require(
        len(summary) == 7
        and len(profile_rows) == 35,
        "Incomplete orientation results"
    )

    OUT.mkdir(parents=True)

    save_csv(
        OUT / "test_orientation_summary.csv",
        summary
    )
    save_csv(
        OUT / "test_profile_results.csv",
        profile_rows
    )

    report = {
        "protocol": "RO3_R3A_OrientationTest_Paired_v1",
        "test_protocol_lock_sha256": sha256(LOCK),
        "split": SPLIT,
        "source_images": 74,
        "annotated_image_clusters": N_IMAGES,
        "gt_instances": N_OBJECTS,
        "valid_gt_keypoints": N_VALID,
        "baseline_correct": N_CLEAN,
        "angles": list(ANGLES),
        "primary": "Fixed-index PCK@0.05, original GT-box scale",
        "secondary": "Fixed-index PCK@0.05, rotated GT-box scale",
        "bootstrap": {
            "cluster_unit": "Original annotated photograph",
            "replications": 2000,
            "seed": 20261121,
            "minimum_scenes_for_profile_ci": 5,
        },
        "manifest_sha256": manifest_hash,
        "metadata_sha256": metadata_hash,
        "limitations": [
            "In-plane rotation, not 3D camera viewpoint",
            "Bilinear resampling and artificial canvas borders",
            "Transformed axis-aligned GT boxes",
            "GT-box-conditioned pose, not independent detection",
            "Sparse test-scene support for some profiles",
        ],
        "results": summary,
    }

    with (
        OUT / "analysis_report.json"
    ).open("x") as f:
        json.dump(report, f, indent=2)

    print("\nPASS: R3-A frozen test paired analysis completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
