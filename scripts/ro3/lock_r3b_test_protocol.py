#!/usr/bin/env python3
"""Lock R3-B test inputs before frozen inference; no overwrite."""

import csv
import json
from pathlib import Path

from prepare_r3_viewpoint import (
    ROOT, CONDITIONS, sha256, make_homography
)

RO3 = ROOT / "experiments/RO3"
LOCK = RO3 / "R3B_TestProtocolLock_v1.json"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    validation_lock = (
        RO3 / "R3B_ValidationLock_v1.json"
    )
    previous = json.loads(validation_lock.read_text())

    require(
        previous["protocol"]
        == "RO3_R3B_ValidationLock_v1",
        "Validation lock differs"
    )

    for name, digest in previous["files_sha256"].items():
        path = ROOT / name
        require(
            path.is_file() and sha256(path) == digest,
            f"Validation evidence changed: {name}"
        )

    source = RO3 / "R3_Viewpoint_v1/test"
    metadata_file = source / "protocol_metadata.json"
    manifest_file = source / "viewpoint_manifest.jsonl"

    meta = json.loads(metadata_file.read_text())

    require(
        meta["protocol"]
        == "RO3_R3B_ProjectiveViewpoint_v1"
        and meta["split"] == "test",
        "Incorrect R3-B test protocol"
    )

    require(
        meta["source_images"] == 74
        and meta["annotated_source_images"] == 71
        and meta["gt_instances"] == 1626
        and meta["valid_gt_keypoints"] == 7049,
        "Test counts differ"
    )

    expected_conditions = [
        {
            "condition_id": cid,
            "axis": axis,
            "compressed_edge": edge,
            "inset_fraction": inset
        }
        for cid, axis, edge, inset in CONDITIONS
    ]

    require(
        meta["conditions"] == expected_conditions,
        "Test conditions differ from validation design"
    )

    generator = (
        ROOT / "scripts/ro3/prepare_r3_viewpoint.py"
    )
    require(
        sha256(generator) == meta["generator_sha256"],
        "Generator differs"
    )

    coco_file = ROOT / meta["annotation_path"]

    require(
        sha256(coco_file) == meta["annotation_sha256"],
        "Frozen test annotations changed"
    )

    coco = json.loads(coco_file.read_text())

    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == 1626,
        "Incorrect test COCO content"
    )

    coco_ids = {int(x["id"]) for x in coco["images"]}
    observed = {item["condition_id"]: {} for item in expected_conditions}

    count = 0
    for line in manifest_file.read_text().splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        count += 1

        cid = row["condition_id"]
        iid = int(row["image_id"])

        require(
            cid in observed and iid not in observed[cid],
            "Duplicate or unknown test manifest record"
        )

        spec = next(
            s for s in expected_conditions
            if s["condition_id"] == cid
        )

        require(
            row["split"] == "test"
            and row["axis"] == spec["axis"]
            and row["compressed_edge"]
            == spec["compressed_edge"]
            and float(row["inset_fraction"])
            == spec["inset_fraction"],
            "Test transformation differs"
        )

        H, corners, coverage = make_homography(
            int(row["width"]),
            int(row["height"]),
            spec["axis"],
            spec["compressed_edge"],
            spec["inset_fraction"]
        )

        import numpy as np

        require(
            np.allclose(
                H, row["homography"],
                rtol=0, atol=1e-8
            ),
            "Test homography differs"
        )

        require(
            row["source_sha256"]
            == meta["source_image_hashes"][str(iid)],
            "Original source hash differs"
        )

        observed[cid][iid] = row

    require(
        count == 666
        and all(set(v) == coco_ids for v in observed.values()),
        "Incomplete test conditions"
    )

    # Verify all 74 original test photographs.
    for iid, record in observed["vp_neutral"].items():
        path = (ROOT / record["source_image"]).resolve()
        path.relative_to(ROOT)

        require(
            path.is_file()
            and sha256(path) == record["source_sha256"],
            f"Original test image changed: {iid}"
        )

    summary = (
        RO3 / "R3B_ValidationSummary_v1"
        / "validation_viewpoint_summary.csv"
    )

    with summary.open(newline="") as f:
        rows = list(csv.DictReader(f))

    require(
        len(rows) == 9
        and {r["condition_id"] for r in rows}
        == set(observed),
        "Incomplete validation summary"
    )

    for row in rows:
        require(
            int(row["instances"]) == 1551
            and int(row["valid_keypoints"]) == 6720
            and int(row["clean_correct"]) == 5196,
            "Validation result counts changed"
        )

    neutral = next(
        r for r in rows
        if r["condition_id"] == "vp_neutral"
    )

    require(
        int(neutral["rotated_correct"]) == 5196,
        "Validation clean baseline differs"
    )

    files = [
        "scripts/ro3/prepare_r3_viewpoint.py",
        "scripts/ro3/evaluate_r3_viewpoint.py",
        "scripts/ro3/summarize_r3b_validation.py",
        "scripts/ro3/run_r3b_validation.sh",
        "scripts/ro3/summarize_r3_orientation_validation.py",
        "scripts/ro3/summarize_r3_orientation_test.py",
        "scripts/ro3/lock_r3b_test_protocol.py",
        "experiments/RO3/R3B_ValidationLock_v1.json",
        "experiments/RO3/R3B_ValidationSummary_v1/"
        "validation_viewpoint_summary.csv",
        "experiments/RO3/R3B_ValidationSummary_v1/"
        "validation_profile_results.csv",
        "experiments/RO3/R3B_ValidationSummary_v1/"
        "analysis_report.json",
        "experiments/RO3/R3_Viewpoint_v1/valid/"
        "viewpoint_manifest.jsonl",
        "experiments/RO3/R3_Viewpoint_v1/valid/"
        "protocol_metadata.json",
        rel(manifest_file),
        rel(metadata_file),
        meta["annotation_path"],
        "experiments/RO3/R1_Eval_v1/"
        "test_uniform_p00_r0/metrics.json",
        "experiments/RO3/R1_Eval_v1/"
        "test_uniform_p00_r0/vitpose_predictions.npy",
        "experiments/RO3/R1_Eval_v1/"
        "test_uniform_p00_r0/vitpose_image_ids.npy",
        "experiments/RO3/R1_Eval_v1/"
        "test_uniform_p00_r0/vitpose_annotation_ids.npy",
        "experiments/ViTPose_Runs/"
        "E03_ViTPoseB_Patched_v1/epoch_90.pth",
        "experiments/ViTPose_Runs/"
        "E03_ViTPoseB_Patched_v1/"
        "td-hm_ViTPose-base_innocount_patched_v1.py",
    ]

    for cid in observed:
        folder = f"experiments/RO3/R3B_Eval_v1/valid_{cid}"

        for name in (
            "metrics.json",
            "vitpose_predictions.npy",
            "vitpose_image_ids.npy",
            "vitpose_annotation_ids.npy",
        ):
            files.append(f"{folder}/{name}")

    fingerprints = {}

    for name in sorted(set(files)):
        path = ROOT / name
        require(path.is_file(), f"Missing input: {name}")
        fingerprints[name] = sha256(path)

    payload = {
        "protocol": "RO3_R3B_FrozenTest_v1",
        "conditions": expected_conditions,
        "model": "Frozen ViTPose-B E03 epoch 90",
        "validation_baseline": "5196/6720",
        "test_baseline_expected": "4325/7049",
        "primary_metric":
            "Fixed-index PCK@0.05, original GT-box scale",
        "secondary_metric":
            "Fixed-index PCK@0.05, transformed GT-box scale",
        "paired_analysis": {
            "unit": "Original annotated source photograph",
            "test_source_clusters": 71,
            "bootstrap_replications": 2000,
            "bootstrap_seed": 20261121,
            "confidence_interval": "Percentile 95%",
            "comparisons": "All eight nonneutral conditions",
            "profile_categories": [
                "Angle Bar", "Channel", "T-beam",
                "HI-Beam", "Sheet Pile"
            ]
        },
        "files_sha256": fingerprints,
    }

    if LOCK.exists():
        require(
            json.loads(LOCK.read_text()) == payload,
            "Existing test lock differs; refusing changes"
        )
        print("PASS: Existing R3-B test lock verified")
    else:
        with LOCK.open("x") as f:
            json.dump(payload, f, indent=2)
        print("PASS: R3-B frozen test protocol locked")

    print("PASS: Nine complete test conditions verified")
    print("PASS: Validation evidence preserved")
    print("PASS: All 74 test source images verified")
    print("PASS: Frozen checkpoint/config fingerprinted")
    print("Test inference is now authorized under the lock")


if __name__ == "__main__":
    main()
