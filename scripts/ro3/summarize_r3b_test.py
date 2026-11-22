#!/usr/bin/env python3
"""R3-B: Paired validation analysis for nine projective conditions."""

import json
from pathlib import Path

import numpy as np

from prepare_r3_viewpoint import (
    ROOT, CONDITIONS, sha256,
    make_homography, transform_points, transform_bbox
)
from summarize_r3_orientation_validation import (
    identity, make_correct, paired_stats, save_csv,
    BOOTSTRAPS, SEED
)

BASE = ROOT / "experiments/RO3"
SOURCE = BASE / "R3_Viewpoint_v1/test"
EVAL = BASE / "R3B_Eval_v1"
CLEAN = BASE / "R1_Eval_v1/test_uniform_p00_r0"
OUT = BASE / "R3B_TestSummary_v1"
LOCK = BASE / "R3B_TestProtocolLock_v1.json"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)



def load_predictions(folder, keys):
    """Load full test predictions using composite GT identity."""
    n = 1626

    p = np.load(
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
        len(keys) == n and p.shape == (n, 6, 2),
        f"Unexpected test prediction shape: {folder}"
    )
    require(
        len(image_ids) == len(annotation_ids) == n,
        "Incorrect test identity-array lengths"
    )
    require(
        np.isfinite(p).all(),
        "Nonfinite test predictions"
    )

    lookup = {}

    for i in range(n):
        key = int(image_ids[i]), int(annotation_ids[i])
        require(key not in lookup, "Duplicate prediction identity")
        lookup[key] = p[i]

    require(
        set(lookup) == set(keys),
        f"Prediction identities differ: {folder}"
    )

    return np.stack([lookup[key] for key in keys])



def main():
    require(not OUT.exists(), f"Refusing overwrite: {OUT}")
    require(LOCK.is_file(), "Missing validation lock")

    lock = json.loads(LOCK.read_text())
    require(
        lock["protocol"] == "RO3_R3B_FrozenTest_v1",
        "Wrong validation protocol lock"
    )


    require(
        BOOTSTRAPS == 2000 and SEED == 20261121,
        "Bootstrap specification differs from R3-B lock"
    )
    require(
        lock["paired_analysis"]["bootstrap_replications"]
        == BOOTSTRAPS
        and lock["paired_analysis"]["bootstrap_seed"] == SEED
        and lock["paired_analysis"]["test_source_clusters"] == 71,
        "Test lock statistical parameters differ"
    )
    require(
        lock["conditions"] == [
            {
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset
            }
            for cid, axis, edge, inset in CONDITIONS
        ],
        "Test lock condition definitions differ"
    )

    for relative, fingerprint in lock["files_sha256"].items():
        path = ROOT / relative
        require(
            path.is_file() and sha256(path) == fingerprint,
            f"Locked input changed: {relative}"
        )

    manifest = SOURCE / "viewpoint_manifest.jsonl"
    metadata_path = SOURCE / "protocol_metadata.json"
    meta = json.loads(metadata_path.read_text())

    require(
        meta["protocol"] == "RO3_R3B_ProjectiveViewpoint_v1"
        and meta["split"] == "test"
        and meta["source_images"] == 74
        and meta["annotated_source_images"] == 71
        and meta["gt_instances"] == 1626
        and meta["valid_gt_keypoints"] == 7049,
        "R3-B validation metadata differs"
    )

    require(
        sha256(ROOT / "scripts/ro3/prepare_r3_viewpoint.py")
        == meta["generator_sha256"],
        "Generator differs from manifest"
    )

    expected = {
        cid: (axis, edge, inset)
        for cid, axis, edge, inset in CONDITIONS
    }

    require(
        meta["conditions"] == [
            {
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset
            }
            for cid, axis, edge, inset in CONDITIONS
        ],
        "Conditions differ from generator"
    )

    annotation = ROOT / meta["annotation_path"]
    require(
        sha256(annotation) == meta["annotation_sha256"],
        "Original annotations changed"
    )

    coco = json.loads(annotation.read_text())
    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == 1626,
        "Wrong COCO sample counts"
    )

    records = {cid: {} for cid in expected}
    total_rows = 0

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            total_rows += 1

            cid = row["condition_id"]
            iid = int(row["image_id"])

            require(
                cid in expected and iid not in records[cid],
                "Duplicate or unknown manifest condition"
            )

            axis, edge, inset = expected[cid]

            require(
                row["split"] == "test"
                and row["axis"] == axis
                and row["compressed_edge"] == edge
                and float(row["inset_fraction"]) == inset,
                "Manifest condition differs"
            )

            require(
                row["source_sha256"]
                == meta["source_image_hashes"][str(iid)],
                "Source-image fingerprint mismatch"
            )

            H, corners, coverage = make_homography(
                int(row["width"]), int(row["height"]),
                axis, edge, inset
            )

            require(
                np.allclose(
                    H, row["homography"],
                    atol=1e-8, rtol=0
                ),
                "Homography matrix differs"
            )

            require(
                np.allclose(
                    corners, row["destination_corners"],
                    atol=1e-8, rtol=0
                ),
                "Destination corners differ"
            )

            require(
                abs(
                    coverage
                    - float(row["analytic_source_coverage"])
                ) < 1e-8,
                "Coverage fraction differs"
            )

            records[cid][iid] = row

    coco_image_ids = {
        int(image["id"])
        for image in coco["images"]
    }

    require(
        total_rows == 666
        and all(
            set(group) == coco_image_ids
            for group in records.values()
        ),
        "Incomplete R3-B validation manifest"
    )

    anns = sorted(coco["annotations"], key=identity)
    keys = [identity(a) for a in anns]

    require(
        len(keys) == len(set(keys)) == 1626,
        "Duplicate annotation identity"
    )

    gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32
    ).reshape(1626, 6, 3)

    boxes = np.asarray(
        [a["bbox"] for a in anns],
        dtype=np.float32
    )

    image_ids = np.asarray(
        [a["image_id"] for a in anns],
        dtype=np.int64
    )

    category_ids = np.asarray(
        [a["category_id"] for a in anns],
        dtype=np.int64
    )

    valid = gt[:, :, 2] > 0

    require(
        int(valid.sum()) == 7049
        and len(np.unique(image_ids)) == 71,
        "Validation GT totals differ"
    )

    original_scale = np.maximum(
        boxes[:, 2], boxes[:, 3]
    )

    clean_predictions = load_predictions(CLEAN, keys)

    clean_correct = make_correct(
        clean_predictions,
        gt[:, :, :2],
        original_scale,
        valid
    )

    require(
        int(clean_correct.sum()) == 4325,
        "Frozen E03 validation baseline differs"
    )

    clean_spec = json.loads(
        (CLEAN / "metrics.json").read_text()
    )["run_spec"]

    categories = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    overall_rows = []
    profile_rows = []

    for cid, axis, edge, inset in CONDITIONS:
        folder = EVAL / f"test_{cid}"
        metrics = json.loads(
            (folder / "metrics.json").read_text()
        )

        spec = metrics["run_spec"]

        require(
            spec["protocol"]
            == "RO3_R3B_ProjectiveViewpoint_Eval_v1"
            and spec["split"] == "test"
            and spec["condition_id"] == cid
            and spec["max_instances"] is None
            and spec["selected_instances"] == 1626,
            f"Wrong or partial evaluation: {cid}"
        )

        require(
            spec["manifest_sha256"] == sha256(manifest)
            and spec["protocol_metadata_sha256"]
            == sha256(metadata_path),
            f"Manifest fingerprint differs: {cid}"
        )

        for field in ("checkpoint_sha256", "config_sha256"):
            require(
                spec[field] == clean_spec[field],
                f"Frozen E03 model differs: {cid}"
            )

        predictions = load_predictions(folder, keys)

        transformed_gt = np.empty(
            (1626, 6, 2), dtype=np.float32
        )

        transformed_scales = np.empty(
            1626, dtype=np.float32
        )

        for i, ann in enumerate(anns):
            H = np.asarray(
                records[cid][int(ann["image_id"])]["homography"],
                dtype=np.float64
            )

            transformed_gt[i] = transform_points(
                gt[i, :, :2], H
            ).astype(np.float32)

            transformed_box = transform_bbox(
                ann["bbox"], H
            )

            transformed_scales[i] = max(
                transformed_box[2],
                transformed_box[3]
            )

        primary = make_correct(
            predictions, transformed_gt,
            original_scale, valid
        )

        secondary = make_correct(
            predictions, transformed_gt,
            transformed_scales, valid
        )

        n_primary = int(primary.sum())
        n_secondary = int(secondary.sum())

        stored_primary = metrics["primary_original_box_pck"]
        stored_secondary = metrics["secondary_transformed_box_pck"]

        require(
            (
                stored_primary["total_correct"],
                stored_primary["total_valid"]
            ) == (n_primary, 7049),
            f"Primary PCK mismatch: {cid}"
        )

        require(
            (
                stored_secondary["total_correct"],
                stored_secondary["total_valid"]
            ) == (n_secondary, 7049),
            f"Secondary PCK mismatch: {cid}"
        )

        ratios = transformed_scales / original_scale

        require(
            abs(
                float(ratios.mean())
                - metrics["mean_transformed_bbox_scale_ratio"]
            ) < 1e-6,
            f"GT-box scale mismatch: {cid}"
        )

        if cid == "vp_neutral":
            reference = metrics["clean_reference_check"]

            require(
                n_primary == n_secondary == 4325
                and reference is not None
                and reference[
                    "maximum_coordinate_difference_px"
                ] <= 0.25,
                "Neutral parity failed"
            )

        stats = paired_stats(
            image_ids, valid,
            clean_correct, primary,
            f"R3B:{cid}:overall"
        )

        overall_rows.append({
            "condition_id": cid,
            "axis": axis,
            "compressed_edge": edge,
            "inset_fraction": inset,
            "instances": 1626,
            **stats,
            "secondary_transformed_box_pck":
                n_secondary / 7049,
            "secondary_minus_primary_pp":
                100 * (n_secondary - n_primary) / 7049,
            "mean_bbox_scale_ratio": float(ratios.mean()),
            "p05_bbox_scale_ratio":
                float(np.quantile(ratios, 0.05)),
            "median_bbox_scale_ratio":
                float(np.median(ratios)),
            "p95_bbox_scale_ratio":
                float(np.quantile(ratios, 0.95)),
            "secondary_constrained_oi_pck":
                metrics["secondary_constrained_oi_pck"]["overall"],
            "secondary_structural_edge_accuracy":
                metrics["secondary_structural_edges"]["overall"]
        })

        for category_id, name in sorted(categories.items()):
            mask = (
                valid
                & (category_ids == category_id)[:, None]
            )

            group = paired_stats(
                image_ids, mask,
                clean_correct, primary,
                f"R3B:{cid}:{name}"
            )

            saved = metrics["profile_primary_pck"][name]

            require(
                saved["correct"] == group["rotated_correct"]
                and saved["valid_keypoints"]
                == group["valid_keypoints"],
                f"Profile mismatch: {cid}/{name}"
            )

            profile_rows.append({
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset,
                "profile": name,
                **group
            })

        print(
            f"{cid:<20} "
            f"PCK={stats['rotated_pck']:.4%} "
            f"Delta={stats['delta_pp']:+.2f} pp "
            f"95% CI=[{stats['ci_low_pp']:+.2f}, "
            f"{stats['ci_high_pp']:+.2f}] pp "
            f"Secondary={n_secondary/7049:.4%} "
            f"Scale={np.median(ratios):.4f}x"
        )

    require(
        len(overall_rows) == 9
        and len(profile_rows) == 45,
        "Incomplete R3-B paired analysis"
    )

    OUT.mkdir(parents=True)

    save_csv(
        OUT / "test_viewpoint_summary.csv",
        overall_rows
    )

    save_csv(
        OUT / "test_profile_results.csv",
        profile_rows
    )

    with (OUT / "analysis_report.json").open("x") as f:
        json.dump({
            "protocol": "RO3_R3B_TestPaired_v1",
            "model": "Frozen ViTPose-B E03",
            "source_images": 74,
            "annotated_source_clusters": 71,
            "gt_instances": 1626,
            "valid_gt_keypoints": 7049,
            "clean_correct": 4325,
            "primary_metric":
                "Fixed-index PCK@0.05, original GT-box scale",
            "secondary_metric":
                "Fixed-index PCK@0.05, transformed GT-box scale",
            "bootstrap_iterations": 2000,
            "bootstrap_unit": "Original annotated source image",
            "minimum_source_images_for_CI": 5,
            "protocol_lock_sha256": sha256(LOCK),
            "manifest_sha256": sha256(manifest),
            "summarizer_sha256": sha256(Path(__file__)),
            "limitations": [
                "Synthetic 2D projective distortion, not actual 3D camera viewpoint",
                "Warp interpolation and background borders",
                "Perspective changes GT-box dimensions nonuniformly",
                "Original-box normalization is not projectively invariant",
                "GT-box-conditioned pose only, not object detection",
                "Sparse independent images for certain profiles"
            ],
            "results": overall_rows
        }, f, indent=2)

    print("\nPASS: R3-B frozen test paired analysis complete")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
