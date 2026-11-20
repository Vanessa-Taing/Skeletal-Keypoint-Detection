#!/usr/bin/env python3
"""Freeze RO3 R1 evidence without modifying existing experiment results."""

import csv
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RO3 = ROOT / "experiments/RO3"
DOCS = ROOT / "docs/RO3"
EVIDENCE = DOCS / "evidence"

ARCHIVE = RO3 / "R1_FrozenEvidence_v1.tar.gz"
TEMP_ARCHIVE = RO3 / "R1_FrozenEvidence_v1.tar.gz.partial"

CONTENTS = EVIDENCE / "R1_contents_v1.sha256"
INPUTS = EVIDENCE / "R1_external_inputs_v1.sha256"
IMAGES = EVIDENCE / "R1_source_images_v1.csv"
INDEX = EVIDENCE / "R1_freeze_index_v1.json"
ARCHIVE_HASH = EVIDENCE / "R1_archive_v1.sha256"

REPORT = DOCS / "R1_Methodological_Report_v1.md"

SCRIPTS = [
    "scripts/ro3/prepare_r1_occlusion.py",
    "scripts/ro3/evaluate_r1_vitpose.py",
    "scripts/ro3/summarize_r1_validation.py",
    "scripts/ro3/summarize_r1_test.py",
    "scripts/ro3/audit_r1_natural_stacking.py",
    "scripts/ro3/run_r1_validation_remaining.sh",
    "scripts/ro3/run_r1_test_uniform.sh",
    "scripts/ro3/freeze_r1_evidence.py",
    "scripts/evaluate/common_pose_metrics.py",
    "scripts/evaluate/common_pose_metrics_ro3_constrained_symmetry_v1.py",
]

RESULT_DIRS = [
    "experiments/RO3/R1_Occlusion_v1",
    "experiments/RO3/R1_Eval_v1",
    "experiments/RO3/R1_Validation_PairedSummary_v1",
    "experiments/RO3/R1_Test_PairedSummary_v1",
    "experiments/RO3/R1_NaturalStackingAudit_v1",
]

OPTIONAL_LOGS = [
    "experiments/RO3/ro3_r1_validation_complete.log",
    "experiments/RO3/ro3_r1_test_inference.log",
    "experiments/RO3/ro3_r1_test_summary.log",
    "experiments/RO3/ro3_r1_natural_stacking_audit.log",
]

MODEL_INPUTS = [
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/epoch_90.pth",
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "td-hm_ViTPose-base_innocount_patched_v1.py",
]

REFERENCE_INPUTS = [
    "experiments/RO2_Final_Patched/"
    "ViTPoseB_E03_CommonTest/vitpose_predictions.npy",
    "experiments/RO2_Final_Patched/"
    "ViTPoseB_E03_CommonTest/vitpose_image_ids.npy",
    "experiments/RO2_Final_Patched/"
    "ViTPoseB_E03_CommonTest/vitpose_annotation_ids.npy",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix()


def json_read(path):
    return json.loads(Path(path).read_text())


def check_input():
    require(REPORT.is_file(), f"Save report first: {REPORT}")
    require(REPORT.stat().st_size > 1000, "Report appears incomplete")

    for item in SCRIPTS + RESULT_DIRS:
        require((ROOT / item).exists(), f"Missing: {item}")

    for target in (
        ARCHIVE, TEMP_ARCHIVE, CONTENTS,
        INPUTS, IMAGES, INDEX, ARCHIVE_HASH,
    ):
        require(
            not target.exists(),
            f"Refusing to overwrite existing evidence: {target}",
        )

    base = RO3 / "R1_Eval_v1"

    for level in (0, 10, 20, 30, 40, 50):
        valid = base / f"valid_uniform_p{level:02d}_r0"
        require(
            (valid / "metrics.json").is_file(),
            f"Missing validation evaluation: {valid}",
        )

        repetitions = (0,) if level == 0 else (0, 1, 2)

        for rep in repetitions:
            test = base / f"test_uniform_p{level:02d}_r{rep}"

            for name in (
                "metrics.json",
                "vitpose_predictions.npy",
                "vitpose_image_ids.npy",
                "vitpose_annotation_ids.npy",
            ):
                require(
                    (test / name).is_file(),
                    f"Missing frozen test evidence: {test / name}",
                )

    clean_test = json_read(
        base / "test_uniform_p00_r0/metrics.json"
    )

    clean_valid = json_read(
        base / "valid_uniform_p00_r0/metrics.json"
    )

    test_fixed = clean_test["fixed_index_pck"]
    valid_fixed = clean_valid["fixed_index_pck"]

    require(
        (test_fixed["total_correct"], test_fixed["total_valid"])
        == (4325, 7049),
        "Frozen E03 test baseline differs",
    )

    require(
        (valid_fixed["total_correct"], valid_fixed["total_valid"])
        == (5196, 6720),
        "Frozen validation baseline differs",
    )

    reference = clean_test["reference_comparison"]
    require(reference is not None, "Missing reference comparison")
    require(
        reference["max_abs_error_px"] <= 0.25,
        "E03 reference prediction check failed",
    )

    test_summary = RO3 / (
        "R1_Test_PairedSummary_v1/test_summary.csv"
    )

    with test_summary.open(newline="") as f:
        summary = list(csv.DictReader(f))

    require(
        [int(r["severity_pct"]) for r in summary]
        == [0, 10, 20, 30, 40, 50],
        "Incomplete test severity summary",
    )

    require(
        abs(float(summary[-1]["fixed_pck"])
            - 0.37877713150801534) < 1e-10,
        "50% test PCK differs from verified R1 result",
    )

    audit = json_read(
        RO3 / "R1_NaturalStackingAudit_v1/audit_metadata.json"
    )

    require(
        audit["correct_fixed_index"] == 4325
        and audit["valid_gt_keypoints"] == 7049,
        "Natural audit baseline differs",
    )

    manifests = {}

    for split, mode in (
        ("valid", "uniform"),
        ("valid", "keypoint"),
        ("test", "uniform"),
    ):
        folder = (
            RO3 / "R1_Occlusion_v1"
            / f"{split}_{mode}_seed20261119"
        )
        metadata = json_read(folder / "protocol_metadata.json")
        manifest = folder / "occlusion_manifest.jsonl"

        require(manifest.is_file(), f"Missing: {manifest}")
        require(metadata["split"] == split, "Wrong manifest split")
        require(metadata["mode"] == mode, "Wrong placement mode")

        expected_count = 1626 if split == "test" else 1551

        require(
            metadata["gt_instance_count"] == expected_count,
            "Unexpected instance count in generator metadata",
        )

        if split == "test":
            require(
                metadata["manifest_record_count"] == 26016,
                "Frozen test manifest count differs",
            )
        else:
            require(
                metadata["manifest_record_count"] == 9306,
                "Validation manifest count differs",
            )

        require(
            metadata["generator_sha256"]
            == digest(ROOT / "scripts/ro3/prepare_r1_occlusion.py"),
            "Generator has changed since manifest generation",
        )

        annotation = Path(metadata["source_annotation"])

        if not annotation.is_absolute():
            annotation = ROOT / annotation

        require(annotation.is_file(), f"Missing: {annotation}")
        require(
            digest(annotation)
            == metadata["source_annotation_sha256"],
            "Source annotation hash mismatch",
        )

        manifests[(split, mode)] = metadata

    # Verify model and configuration against recorded E03 fingerprints.
    clean_spec = clean_test["run_spec"]

    require(
        digest(ROOT / MODEL_INPUTS[0])
        == clean_spec["checkpoint_sha256"],
        "Frozen checkpoint differs",
    )

    require(
        digest(ROOT / MODEL_INPUTS[1])
        == clean_spec["config_sha256"],
        "Frozen E03 config differs",
    )

    return manifests, clean_spec


def gather_files():
    paths = [ROOT / p for p in SCRIPTS]
    paths.append(REPORT)

    for dirname in RESULT_DIRS:
        folder = ROOT / dirname

        for file in folder.rglob("*"):
            require(
                not file.is_symlink(),
                f"Symlink not permitted in archive: {file}",
            )
            if file.is_file():
                paths.append(file)

    for item in OPTIONAL_LOGS:
        file = ROOT / item
        if file.is_file():
            paths.append(file)

    paths = sorted(set(paths), key=rel)

    for p in paths:
        require(p.is_file(), f"Missing evidence file: {p}")

    return paths


def prepare_ledgers(files, manifests, clean_spec):
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    entries = []

    with CONTENTS.open("x") as f:
        for p in files:
            item = {
                "path": rel(p),
                "sha256": digest(p),
                "size_bytes": p.stat().st_size,
            }
            entries.append(item)

            # Standard sha256sum-compatible format.
            f.write(f"{item['sha256']}  {item['path']}\n")

    external = set(MODEL_INPUTS + REFERENCE_INPUTS)

    for metadata in manifests.values():
        p = Path(metadata["source_annotation"])
        if not p.is_absolute():
            p = ROOT / p
        external.add(rel(p))

    # Record the existing RO2 evidence archive when available.
    old_ro2 = (
        "experiments/RO2_Final_Patched/"
        "RO2_Reproducibility_20261113_160940.tar.gz"
    )
    if (ROOT / old_ro2).is_file():
        external.add(old_ro2)

    with INPUTS.open("x") as f:
        for item in sorted(external):
            p = ROOT / item
            require(p.is_file(), f"Missing external input: {p}")
            f.write(f"{digest(p)}  {item}\n")

    with IMAGES.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "image_id", "source_image_sha256"
        ])

        # One original-image hash per image and split.
        for split in ("valid", "test"):
            metadata = manifests[(split, "uniform")]
            image_hashes = metadata[
                "source_image_sha256_by_image_id"
            ]

            for image_id, hash_value in sorted(
                image_hashes.items(),
                key=lambda pair: int(pair[0]),
            ):
                writer.writerow([split, image_id, hash_value])

    index = {
        "protocol": "RO3_R1_FrozenEvidence_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "ViTPose-B E03 epoch 90",
        "test_gt_instances": 1626,
        "test_source_images_with_annotations": 71,
        "test_gt_valid_keypoints": 7049,
        "clean_correct_keypoints": 4325,
        "clean_pck": 4325 / 7049,
        "test_50pct_pck": 0.37877713150801534,
        "test_50pct_delta_pp": -23.47850758972904,
        "evaluation": "GT-box-conditioned fixed-index PCK@0.05",
        "model_checkpoint_sha256":
            clean_spec["checkpoint_sha256"],
        "model_config_sha256":
            clean_spec["config_sha256"],
        "result_directories": RESULT_DIRS,
        "included_file_count": len(entries),
        "included_original_bytes": sum(
            x["size_bytes"] for x in entries
        ),
        "contents_ledger": rel(CONTENTS),
        "external_input_ledger": rel(INPUTS),
        "source_image_hash_file": rel(IMAGES),
        "archive_path": rel(ARCHIVE),
        "archive_policy": (
            "Derived evidence and source scripts only. "
            "Original dataset images and model weights "
            "are external and fingerprinted, not archived."
        ),
        "immutability_policy": (
            "Versioned output names; refuse overwrite; "
            "verified SHA-256 archive; Git commit and tag."
        ),
    }

    with INDEX.open("x") as f:
        json.dump(index, f, indent=2)

    return entries


def create_and_verify_archive(entries):
    expected = {
        item["path"]: item["sha256"]
        for item in entries
    }

    for p in (CONTENTS, INPUTS, IMAGES, INDEX):
        expected[rel(p)] = digest(p)

    with tarfile.open(TEMP_ARCHIVE, "w:gz") as tar:
        for item in entries:
            p = ROOT / item["path"]
            tar.add(
                p,
                arcname=item["path"],
                recursive=False,
            )

        for p in (CONTENTS, INPUTS, IMAGES, INDEX):
            tar.add(p, arcname=rel(p), recursive=False)

    # Verify the actual content INSIDE the compressed archive,
    # not only the archive file's existence.
    seen = set()

    with tarfile.open(TEMP_ARCHIVE, "r:gz") as tar:
        for member in tar:
            require(
                member.isfile(),
                f"Unexpected archive entry: {member.name}",
            )
            require(
                member.name in expected,
                f"Unlisted archive entry: {member.name}",
            )
            require(
                member.name not in seen,
                f"Duplicate archive entry: {member.name}",
            )

            stream = tar.extractfile(member)
            require(stream is not None, "Unreadable archive entry")

            h = hashlib.sha256()
            while True:
                block = stream.read(1 << 20)
                if not block:
                    break
                h.update(block)

            require(
                h.hexdigest() == expected[member.name],
                f"Archived file checksum mismatch: {member.name}",
            )
            seen.add(member.name)

    require(
        seen == set(expected),
        "Some files are missing from the archive",
    )

    TEMP_ARCHIVE.rename(ARCHIVE)

    archive_sha = digest(ARCHIVE)

    ARCHIVE_HASH.write_text(
        f"{archive_sha}  {rel(ARCHIVE)}\n"
    )

    # Read-only is an additional accidental-edit safeguard.
    # SHA-256 verification remains the real integrity check.
    os.chmod(ARCHIVE, 0o444)

    return archive_sha


def main():
    print("RO3 R1 — FROZEN EVIDENCE PRESERVATION")
    print("=" * 65)

    manifests, clean_spec = check_input()
    files = gather_files()

    print("Source/evidence files:", len(files))
    print("Creating SHA-256 ledgers...")

    entries = prepare_ledgers(
        files, manifests, clean_spec
    )

    print("Creating and verifying frozen archive...")
    archive_sha = create_and_verify_archive(entries)

    print("\nPASS: R1 evidence frozen and verified")
    print("Files archived:", len(entries) + 4)
    print("Archive:", ARCHIVE)
    print("SHA-256:", archive_sha)
    print("Contents ledger:", CONTENTS)
    print("External input ledger:", INPUTS)
    print("Source image hashes:", IMAGES)
    print("Index:", INDEX)
    print()
    print("Original datasets and previous evidence unchanged.")


if __name__ == "__main__":
    main()
