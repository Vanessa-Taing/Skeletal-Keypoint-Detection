#!/usr/bin/env python3
"""Freeze R2-A/B evidence; refuse overwrites and verify archived bytes."""

import csv
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RO3 = ROOT / "experiments/RO3"
EV = ROOT / "docs/RO3/evidence"

REPORT = ROOT / "docs/RO3/R2_Methodological_Report_v1.md"
LOCK = RO3 / "R2_ProtocolLock_v1.json"

ARCHIVE = RO3 / "R2_FrozenEvidence_v1.tar.gz"
PARTIAL = RO3 / "R2_FrozenEvidence_v1.tar.gz.partial"

CONTENTS = EV / "R2_contents_v1.sha256"
EXTERNAL = EV / "R2_external_inputs_v1.sha256"
INDEX = EV / "R2_freeze_index_v1.json"
LOCK_COPY = EV / "R2_ProtocolLock_v1.json"
ARCHIVE_HASH = EV / "R2_archive_v1.sha256"

SCRIPTS = [
    "scripts/ro3/prepare_r2_lighting.py",
    "scripts/ro3/evaluate_r2_vitpose.py",
    "scripts/ro3/summarize_r2_validation.py",
    "scripts/ro3/summarize_r2_test.py",
    "scripts/ro3/run_r2_validation_remaining.sh",
    "scripts/ro3/run_r2_test_global.sh",
    "scripts/ro3/audit_r2_lighting_pixels.py",
    "scripts/ro3/audit_r2_lighting_pixels_test.py",
    "scripts/ro3/audit_r2_scene_attribution.py",
    "scripts/ro3/freeze_r2_evidence.py",
    "scripts/evaluate/common_pose_metrics.py",
    "scripts/evaluate/"
    "common_pose_metrics_ro3_constrained_symmetry_v1.py",
]

RESULT_DIRS = [
    "experiments/RO3/R2_Lighting_v1",
    "experiments/RO3/R2_Eval_v1",
    "experiments/RO3/R2_Validation_PairedSummary_v1",
    "experiments/RO3/R2_Test_PairedSummary_v1",
    "experiments/RO3/R2_LightingDiagnostics_v1",
    "experiments/RO3/R2_SceneAttribution_v1",
]

LOGS = [
    "experiments/RO3/ro3_r2_validation_complete.log",
    "experiments/RO3/ro3_r2_test_inference.log",
    "experiments/RO3/ro3_r2_lighting_pixel_audit.log",
    "experiments/RO3/ro3_r2_test_pixel_audit.log",
    "experiments/RO3/ro3_r2_scene_attribution.log",
]

MODEL_INPUTS = [
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "epoch_90.pth",
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "td-hm_ViTPose-base_innocount_patched_v1.py",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(relative):
    p = (ROOT / relative).resolve()
    p.relative_to(ROOT)
    return p


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_results():
    require(REPORT.is_file(), "Save the R2 report first")
    require(REPORT.stat().st_size > 3000, "R2 report incomplete")
    require(LOCK.is_file(), "Missing R2 frozen protocol lock")

    for path in (
        ARCHIVE, PARTIAL, CONTENTS, EXTERNAL,
        INDEX, LOCK_COPY, ARCHIVE_HASH,
    ):
        require(
            not path.exists(),
            f"Existing freeze artifact; refusing overwrite: {path}",
        )

    for item in SCRIPTS + RESULT_DIRS:
        require(safe_path(item).exists(), f"Missing: {item}")

    lock = json.loads(LOCK.read_text())
    require(
        lock["protocol"] == "RO3_R2_GlobalLighting_FrozenTest_v1",
        "Incorrect R2 protocol lock",
    )

    # Verify every pre-test fingerprint still matches.
    for name, expected_hash in lock["files_sha256"].items():
        p = safe_path(name)
        require(p.is_file(), f"Missing locked file: {name}")
        require(
            sha256(p) == expected_hash,
            f"R2 protocol input changed: {name}",
        )

    expected = {
        "valid": (1551, 6720, 5196, 3369, 72),
        "test": (1626, 7049, 4325, 3974, 71),
    }

    source_paths = {}
    external_paths = set(MODEL_INPUTS)

    for split, (n_obj, n_kp, n_clean, n_c140, n_images) in (
        expected.items()
    ):
        summary_name = (
            "validation_lighting_summary.csv"
            if split == "valid"
            else "test_lighting_summary.csv"
        )
        summary_dir = (
            "R2_Validation_PairedSummary_v1"
            if split == "valid"
            else "R2_Test_PairedSummary_v1"
        )

        summary = read_csv(
            RO3 / summary_dir / summary_name
        )
        require(len(summary) == 10, "Incomplete R2 summary")

        pairs = [
            (r["mode"], float(r["value"]))
            for r in summary
        ]
        require(len(set(pairs)) == 10, "Duplicate condition")

        for row in summary:
            require(
                int(row["valid_keypoints"]) == n_kp,
                "Unexpected GT keypoint denominator",
            )
            require(
                int(
                    row.get(
                        "original_instances",
                        row.get("instances"),
                    )
                ) == n_obj,
                "Unexpected GT instance count",
            )

            if (
                (row["mode"] == "exposure" and float(row["value"]) == 0)
                or (
                    row["mode"] == "contrast"
                    and float(row["value"]) == 1
                )
            ):
                require(
                    int(row["correct_keypoints"]) == n_clean,
                    "Neutral E03 baseline differs",
                )

            if (
                row["mode"] == "contrast"
                and float(row["value"]) == 1.4
            ):
                require(
                    int(row["correct_keypoints"]) == n_c140,
                    "Frozen 1.40x contrast result differs",
                )

        profile_name = (
            "validation_profile_results.csv"
            if split == "valid"
            else "test_profile_results.csv"
        )
        require(
            len(read_csv(RO3 / summary_dir / profile_name)) == 50,
            "Incomplete R2 profile analysis",
        )

        diagnostics = read_csv(
            RO3
            / "R2_LightingDiagnostics_v1"
            / split
            / "lighting_clipping_summary.csv"
        )
        require(len(diagnostics) == 10, "Incomplete clipping audit")

        for row in diagnostics:
            require(
                int(row["source_images"]) == 74,
                "Incomplete clipping source-image audit",
            )

        for mode in ("exposure", "contrast"):
            folder = (
                RO3 / "R2_Lighting_v1"
                / f"{split}_{mode}"
            )

            meta = json.loads(
                (folder / "protocol_metadata.json").read_text()
            )
            require(
                meta["split"] == split
                and meta["mode"] == mode
                and meta["source_images"] == 74
                and meta["source_gt_instances"] == n_obj,
                "Lighting manifest metadata mismatch",
            )

            require(
                sha256(safe_path(SCRIPTS[0]))
                == meta["generator_sha256"],
                "R2 generator differs from stored metadata",
            )

            annotation = safe_path(meta["annotation_path"])
            require(
                sha256(annotation) == meta["annotation_sha256"],
                "COCO annotations differ",
            )
            external_paths.add(relative(annotation))

            manifest = folder / "lighting_manifest.jsonl"
            lines = [
                json.loads(line)
                for line in manifest.read_text().splitlines()
                if line.strip()
            ]
            require(
                len(lines) == 370,
                "Incomplete R2 lighting manifest",
            )

            ids_by_condition = {}
            for row in lines:
                ids_by_condition.setdefault(
                    row["condition_id"], set()
                )
                image_id = int(row["image_id"])
                require(
                    image_id
                    not in ids_by_condition[row["condition_id"]],
                    "Duplicate image in lighting condition",
                )
                ids_by_condition[row["condition_id"]].add(
                    image_id
                )

                # All repeated scene records must agree
                # with the source-image hash inventory.
                require(
                    row["source_sha256"]
                    == meta["source_image_hashes"][str(image_id)],
                    "Image hash mismatch in manifest",
                )
                source_paths.setdefault(
                    (split, image_id),
                    (
                        row["source_image"],
                        row["source_sha256"],
                    ),
                )
                require(
                    source_paths[(split, image_id)]
                    == (
                        row["source_image"],
                        row["source_sha256"],
                    ),
                    "Source path/hash differs between conditions",
                )

            require(
                len(ids_by_condition) == 5
                and all(
                    len(v) == 74
                    for v in ids_by_condition.values()
                ),
                "Incomplete R2 scene conditions",
            )

            for condition in ids_by_condition:
                directory = (
                    RO3 / "R2_Eval_v1"
                    / f"{split}_{mode}_{condition}"
                )

                for name in (
                    "metrics.json",
                    "vitpose_predictions.npy",
                    "vitpose_image_ids.npy",
                    "vitpose_annotation_ids.npy",
                ):
                    require(
                        (directory / name).is_file(),
                        f"Missing completed evaluation: {directory / name}",
                    )

                result = json.loads(
                    (directory / "metrics.json").read_text()
                )
                spec = result["run_spec"]

                require(
                    spec["split"] == split
                    and spec["mode"] == mode
                    and spec["condition_id"] == condition
                    and spec["selected_instances"] == n_obj
                    and spec["max_instances"] is None,
                    "Wrong or partial R2 evaluation",
                )

                require(
                    result["fixed_index_pck"]["total_valid"] == n_kp,
                    "Evaluation denominator mismatch",
                )
                require(
                    spec["manifest_sha256"] == sha256(manifest),
                    "Evaluation used a different manifest",
                )
                require(
                    spec["checkpoint_sha256"]
                    == sha256(safe_path(MODEL_INPUTS[0])),
                    "Checkpoint hash differs",
                )
                require(
                    spec["config_sha256"]
                    == sha256(safe_path(MODEL_INPUTS[1])),
                    "Config hash differs",
                )

        baseline = (
            RO3 / "R1_Eval_v1"
            / f"{split}_uniform_p00_r0"
        )

        for name in (
            "metrics.json",
            "vitpose_predictions.npy",
            "vitpose_image_ids.npy",
            "vitpose_annotation_ids.npy",
        ):
            external_paths.add(relative(baseline / name))

    require(
        len(source_paths) == 148,
        "Expected 74 original images per split",
    )

    for (split, image_id), (name, expected_hash) in (
        sorted(source_paths.items())
    ):
        path = safe_path(name)
        require(path.is_file(), f"Missing source image: {name}")
        require(
            sha256(path) == expected_hash,
            f"Original image altered: {split}/{image_id}",
        )
        external_paths.add(relative(path))

    attribution = read_csv(
        RO3 / "R2_SceneAttribution_v1"
        / "source_image_contributions.csv"
    )
    require(
        len(attribution) == (72 + 71) * 3,
        "Incomplete scene attribution table",
    )

    return lock, external_paths


def collect_archive_files():
    paths = [safe_path(p) for p in SCRIPTS]
    paths.append(REPORT)
    paths.append(LOCK)

    for dirname in RESULT_DIRS:
        folder = safe_path(dirname)

        for p in folder.rglob("*"):
            require(
                not p.is_symlink(),
                f"Symlinks excluded from evidence: {p}",
            )
            if p.is_file():
                paths.append(p)

    for name in LOGS:
        p = safe_path(name)
        if p.is_file():
            paths.append(p)

    return sorted(set(paths), key=relative)


def create_ledgers(files, external_paths, lock):
    EV.mkdir(parents=True, exist_ok=True)

    expected = {}
    with CONTENTS.open("x") as f:
        for p in files:
            name = relative(p)
            value = sha256(p)
            expected[name] = value
            f.write(f"{value}  {name}\n")

    with EXTERNAL.open("x") as f:
        for name in sorted(external_paths):
            p = safe_path(name)
            require(p.is_file(), f"Missing external file: {name}")
            f.write(f"{sha256(p)}  {name}\n")

    with LOCK_COPY.open("xb") as f:
        f.write(LOCK.read_bytes())

    require(
        sha256(LOCK_COPY) == sha256(LOCK),
        "Protocol-lock copy mismatch",
    )

    index = {
        "protocol": "RO3_R2_FrozenEvidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_scope": "R2-A exposure and R2-B contrast",
        "checkpoint": MODEL_INPUTS[0],
        "validation_baseline": "5196/6720",
        "test_baseline": "4325/7049",
        "file_count_before_ledgers": len(files),
        "source_image_count": 148,
        "result_directories": RESULT_DIRS,
        "contents_ledger": relative(CONTENTS),
        "external_inputs_ledger": relative(EXTERNAL),
        "protocol_lock_sha256": sha256(LOCK),
        "archive_path": relative(ARCHIVE),
        "archive_excludes": [
            "Original full-resolution dataset photographs",
            "Frozen model checkpoint",
            "Original R1 and RO2 evidence archives",
        ],
    }

    with INDEX.open("x") as f:
        json.dump(index, f, indent=2)

    for p in (CONTENTS, EXTERNAL, LOCK_COPY, INDEX):
        expected[relative(p)] = sha256(p)

    return expected


def create_archive(expected):
    require(
        not ARCHIVE.exists() and not PARTIAL.exists(),
        "Archive or partial archive already exists",
    )

    with tarfile.open(PARTIAL, "w:gz") as tar:
        for name in sorted(expected):
            tar.add(
                safe_path(name),
                arcname=name,
                recursive=False,
            )

    seen = set()

    with tarfile.open(PARTIAL, "r:gz") as tar:
        for member in tar:
            require(member.isfile(), "Unexpected archive entry")
            name = member.name

            require(name in expected, "Unlisted archive entry")
            require(name not in seen, "Duplicate archive entry")

            stream = tar.extractfile(member)
            require(stream is not None, "Unreadable archive member")

            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)

            require(
                h.hexdigest() == expected[name],
                f"Archived bytes changed: {name}",
            )
            seen.add(name)

    require(
        seen == set(expected),
        "Archived file list incomplete",
    )

    PARTIAL.rename(ARCHIVE)

    digest = sha256(ARCHIVE)
    with ARCHIVE_HASH.open("x") as f:
        f.write(f"{digest}  {relative(ARCHIVE)}\n")

    os.chmod(ARCHIVE, 0o444)
    return digest


def main():
    print("RO3 R2 — EVIDENCE FREEZE")
    print("=" * 64)

    lock, external_paths = verify_results()
    print("PASS: Protocol lock and all R2 results verified")

    files = collect_archive_files()
    print("Derived files and scripts:", len(files))

    expected = create_ledgers(files, external_paths, lock)
    print("PASS: SHA-256 evidence ledgers written")

    archive_sha = create_archive(expected)
    print("PASS: Archived file bytes independently verified")

    print("\nR2 evidence archive:", ARCHIVE)
    print("Archive SHA-256:", archive_sha)
    print("Contents ledger:", CONTENTS)
    print("External inputs ledger:", EXTERNAL)
    print("Protocol-lock copy:", LOCK_COPY)
    print("Freeze index:", INDEX)
    print("No existing R1 or RO2 evidence modified")


if __name__ == "__main__":
    main()
