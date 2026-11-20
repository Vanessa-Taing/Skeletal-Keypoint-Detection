#!/usr/bin/env python3
"""RO3 R2 frozen-test paired analysis. Does not modify prior results."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_r2_lighting import (
    LEVELS, NEUTRAL, condition_id, sha256
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/RO3/R2_Eval_v1"
MB = ROOT / "experiments/RO3/R2_Lighting_v1"

CLEAN = ROOT / (
    "experiments/RO3/R1_Eval_v1/"
    "test_uniform_p00_r0"
)

OUT = ROOT / (
    "experiments/RO3/R2_Test_PairedSummary_v1"
)

SEED = 20261120
N_BOOT = 2000


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_array_set(folder, keys):
    points = np.load(
        folder / "vitpose_predictions.npy",
        allow_pickle=False
    )
    im = np.load(
        folder / "vitpose_image_ids.npy",
        allow_pickle=False
    )
    ann = np.load(
        folder / "vitpose_annotation_ids.npy",
        allow_pickle=False
    )

    check(
        points.shape == (1626, 6, 2),
        f"Wrong prediction shape: {folder}"
    )
    check(
        len(im) == len(ann) == 1626,
        f"Wrong prediction identities: {folder}"
    )
    check(
        np.isfinite(points).all(),
        f"Nonfinite predictions: {folder}"
    )

    lookup = {}

    for i in range(1626):
        key = (int(im[i]), int(ann[i]))
        check(key not in lookup, f"Duplicate identity: {key}")
        lookup[key] = points[i]

    check(
        set(lookup) == set(keys),
        f"GT identity mismatch: {folder}"
    )

    return np.stack([lookup[k] for k in keys])


def paired_stats(image_ids, mask, before, after, label):
    n = int(mask.sum())

    if not n:
        return {
            "valid_keypoints": 0,
            "clean_correct": 0,
            "perturbed_correct": 0,
            "clean_pck": None,
            "perturbed_pck": None,
            "delta_pp": None,
            "ci_low_pp": None,
            "ci_high_pp": None,
            "source_images": 0,
        }

    old = int((before & mask).sum())
    new = int((after & mask).sum())

    selected = mask.any(axis=1)

    images, inverse = np.unique(
        image_ids[selected],
        return_inverse=True
    )

    count = np.bincount(
        inverse,
        weights=mask[selected].sum(axis=1),
        minlength=len(images)
    )

    a = np.bincount(
        inverse,
        weights=(before[selected] & mask[selected]).sum(axis=1),
        minlength=len(images)
    )

    b = np.bincount(
        inverse,
        weights=(after[selected] & mask[selected]).sum(axis=1),
        minlength=len(images)
    )

    lo = hi = None

    if len(images) >= 5:
        digest = hashlib.sha256(
            f"{SEED}|{label}".encode()
        ).digest()

        rng = np.random.default_rng(
            int.from_bytes(digest[:8], "big")
        )

        draws = rng.integers(
            0, len(images),
            size=(N_BOOT, len(images))
        )

        denominator = count[draws].sum(axis=1)

        deltas = 100.0 * (
            b[draws].sum(axis=1)
            - a[draws].sum(axis=1)
        ) / denominator

        lo, hi = [
            float(x)
            for x in np.quantile(
                deltas, [0.025, 0.975]
            )
        ]

    return {
        "valid_keypoints": n,
        "clean_correct": old,
        "perturbed_correct": new,
        "clean_pck": old / n,
        "perturbed_pck": new / n,
        "delta_pp": 100.0 * (new - old) / n,
        "ci_low_pp": lo,
        "ci_high_pp": hi,
        "source_images": len(images),
    }


def write_csv(path, rows):
    check(rows, f"Empty output: {path}")

    with path.open("x", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    check(
        not OUT.exists(),
        f"Refusing to overwrite: {OUT}"
    )

    lock_path = ROOT / (
        "experiments/RO3/R2_ProtocolLock_v1.json"
    )

    check(lock_path.is_file(), "Missing R2 protocol lock")

    lock = json.loads(lock_path.read_text())

    check(
        lock["protocol"]
        == "RO3_R2_GlobalLighting_FrozenTest_v1",
        "Wrong protocol lock"
    )

    for relative_path, expected_hash in (
        lock["files_sha256"].items()
    ):
        path = ROOT / relative_path

        check(
            path.is_file() and sha256(path) == expected_hash,
            f"Locked input changed: {relative_path}"
        )

    protocols = {}

    for mode in ("exposure", "contrast"):
        folder = MB / f"test_{mode}"

        metadata = json.loads(
            (folder / "protocol_metadata.json").read_text()
        )

        check(
            metadata["split"] == "test"
            and metadata["mode"] == mode,
            "Wrong test metadata"
        )

        check(
            metadata["levels"] == LEVELS[mode],
            "Lighting levels changed"
        )

        check(
            metadata["source_images"] == 74
            and metadata["source_gt_instances"] == 1626,
            "Unexpected test sample size"
        )

        check(
            sha256(ROOT / metadata["annotation_path"])
            == metadata["annotation_sha256"],
            "Test annotations changed"
        )

        check(
            sha256(
                ROOT / "scripts/ro3/prepare_r2_lighting.py"
            ) == metadata["generator_sha256"],
            "Generator differs from manifest"
        )

        coco = json.loads(
            (ROOT / metadata["annotation_path"]).read_text()
        )

        protocols[mode] = (
            folder / "lighting_manifest.jsonl",
            coco,
            metadata,
        )

    exp_coco = protocols["exposure"][1]
    con_coco = protocols["contrast"][1]

    check(
        protocols["exposure"][2]["annotation_sha256"]
        == protocols["contrast"][2]["annotation_sha256"],
        "Exposure and contrast GT sources differ"
    )

    def sort_anns(coco):
        return sorted(
            coco["annotations"],
            key=lambda a: (
                int(a["image_id"]),
                int(a["id"])
            )
        )

    anns = sort_anns(exp_coco)
    other_anns = sort_anns(con_coco)

    keys = [
        (int(a["image_id"]), int(a["id"]))
        for a in anns
    ]

    check(
        len(keys) == len(set(keys)) == 1626,
        "Invalid test identities"
    )

    check(
        [
            (int(a["image_id"]), int(a["id"]))
            for a in other_anns
        ] == keys,
        "Exposure/contrast instance identities differ"
    )

    check(
        [a["keypoints"] for a in anns]
        == [a["keypoints"] for a in other_anns],
        "Exposure/contrast keypoints differ"
    )

    gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32
    ).reshape(-1, 6, 3)

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

    check(
        int(valid.sum()) == 7049
        and len(np.unique(image_ids)) == 71,
        "Frozen test totals changed"
    )

    scale = np.maximum(boxes[:, 2], boxes[:, 3])
    check(np.all(scale > 0), "Invalid bbox normalization")

    def correctness(pred):
        error = (
            np.linalg.norm(
                pred - gt[:, :, :2],
                axis=2
            ) / scale[:, None]
        )
        return error <= 0.05

    clean_predictions = read_array_set(CLEAN, keys)
    clean_correct = correctness(clean_predictions)

    check(
        int((clean_correct & valid).sum()) == 4325,
        "Original E03 test PCK not reproduced"
    )

    clean_report = json.loads(
        (CLEAN / "metrics.json").read_text()
    )

    frozen_spec = clean_report["run_spec"]

    category_names = {
        int(c["id"]): c["name"]
        for c in exp_coco["categories"]
    }

    totals = []
    profiles = []

    for mode in ("exposure", "contrast"):
        manifest = protocols[mode][0]
        manifest_hash = sha256(manifest)

        for value in LEVELS[mode]:
            cid = condition_id(mode, value)
            folder = BASE / f"test_{mode}_{cid}"

            report = json.loads(
                (folder / "metrics.json").read_text()
            )

            spec = report["run_spec"]

            check(
                spec["split"] == "test"
                and spec["mode"] == mode
                and spec["condition_id"] == cid
                and abs(float(spec["value"]) - value) < 1e-9,
                f"Wrong condition metadata: {folder}"
            )

            check(
                spec["max_instances"] is None
                and spec["selected_instances"] == 1626,
                f"Incomplete inference: {folder}"
            )

            check(
                spec["manifest_sha256"] == manifest_hash,
                f"Manifest differs: {folder}"
            )

            for name in (
                "checkpoint_sha256",
                "config_sha256",
            ):
                check(
                    spec[name] == frozen_spec[name],
                    f"Frozen model changed: {folder}"
                )

            predictions = read_array_set(folder, keys)
            changed_correct = correctness(predictions)

            n_correct = int(
                (changed_correct & valid).sum()
            )

            fixed = report["fixed_index_pck"]

            check(
                (
                    fixed["total_correct"],
                    fixed["total_valid"]
                ) == (n_correct, 7049),
                f"PCK metric mismatch: {folder}"
            )

            if abs(value - NEUTRAL[mode]) < 1e-9:
                reference = report["reference_comparison"]

                check(
                    reference is not None
                    and reference[
                        "maximum_absolute_coordinate_difference_px"
                    ] <= 0.25,
                    f"Neutral reference failed: {folder}"
                )

                check(
                    n_correct == 4325
                    and float(
                        np.abs(
                            predictions - clean_predictions
                        ).max()
                    ) <= 0.25,
                    f"Neutral baseline failed: {folder}"
                )

            stats = paired_stats(
                image_ids,
                valid,
                clean_correct,
                changed_correct,
                f"{mode}:{cid}:all"
            )

            totals.append({
                "mode": mode,
                "value": value,
                "condition_id": cid,
                "original_instances": 1626,
                "correct_keypoints": n_correct,
                **stats,
                "constrained_oi_pck": report[
                    "constrained_order_invariant_pck"
                ]["overall"],
                "structural_edge_accuracy": report[
                    "structural_edge_accuracy"
                ]["overall"],
            })

            for category_id, profile_name in sorted(
                category_names.items()
            ):
                mask = (
                    valid
                    & (categories == category_id)[:, None]
                )

                profiles.append({
                    "mode": mode,
                    "value": value,
                    "condition_id": cid,
                    "profile": profile_name,
                    **paired_stats(
                        image_ids,
                        mask,
                        clean_correct,
                        changed_correct,
                        f"{mode}:{cid}:{profile_name}",
                    ),
                })

            print(
                f"{mode:<9} {value:+.2f} | "
                f"PCK={stats['perturbed_pck']:.4%} | "
                f"Delta={stats['delta_pp']:+.2f} pp | "
                f"95% CI=[{stats['ci_low_pp']:+.2f}, "
                f"{stats['ci_high_pp']:+.2f}] pp"
            )

    check(
        len(totals) == 10
        and len(profiles) == 50,
        "Missing conditions or profiles"
    )

    OUT.mkdir(parents=True)

    write_csv(
        OUT / "test_lighting_summary.csv",
        totals
    )

    write_csv(
        OUT / "test_profile_results.csv",
        profiles
    )

    metadata = {
        "protocol": "RO3_R2_Test_Paired_v1",
        "source_image_clusters": 71,
        "ground_truth_instances": 1626,
        "valid_gt_keypoints": 7049,
        "baseline_correct_keypoints": 4325,
        "bootstrap_iterations": N_BOOT,
        "bootstrap_seed": SEED,
        "ci_unit": "Original annotated source image",
        "minimum_source_images_for_ci": 5,
        "protocol_lock_sha256": sha256(lock_path),
        "results": totals,
        "limitations": [
            "GT-box-conditioned pose only, not counting.",
            "High contrast includes substantial saturation.",
            "Channel contains two test source images.",
            "Sheet Pile contains one test source image.",
            "Constrained OI is a secondary sensitivity metric.",
        ],
    }

    with (
        OUT / "analysis_report.json"
    ).open("x") as f:
        json.dump(metadata, f, indent=2)

    print("\nPASS: R2 frozen test paired analysis completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
