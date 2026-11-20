#!/usr/bin/env python3
"""Verify and archive frozen R3-A validation/test evidence."""

import csv
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RO3 = ROOT / "experiments/RO3"
EVIDENCE = ROOT / "docs/RO3/evidence"

ARCHIVE = RO3 / "R3A_FrozenEvidence_v1.tar.gz"
PARTIAL = RO3 / "R3A_FrozenEvidence_v1.tar.gz.partial"

LEDGER = EVIDENCE / "R3A_contents_v1.sha256"
ARCHIVE_LEDGER = EVIDENCE / "R3A_archive_v1.sha256"

LOCK = RO3 / "R3A_TestProtocolLock_v1.json"
REPORT = ROOT / "docs/RO3/R3A_Methodological_Conclusion_v1.md"

ANGLES = {
    -45: "rot_m045", -30: "rot_m030",
    -15: "rot_m015",   0: "rot_p000",
     15: "rot_p015",  30: "rot_p030",
     45: "rot_p045",
}

EXPECTED = {
    "valid": {
        "objects": 1551,
        "keypoints": 6720,
        "counts": {
            -45: 4651, -30: 4654, -15: 4718,
              0: 5196,  15: 4806,  30: 4673, 45: 4580
        }
    },
    "test": {
        "objects": 1626,
        "keypoints": 7049,
        "counts": {
            -45: 4018, -30: 4100, -15: 4206,
              0: 4325,  15: 4169,  30: 4112, 45: 4065
        }
    },
}

SCRIPTS = [
    "scripts/ro3/prepare_r3_orientation.py",
    "scripts/ro3/evaluate_r3_orientation.py",
    "scripts/ro3/summarize_r3_orientation_validation.py",
    "scripts/ro3/summarize_r3_orientation_test.py",
    "scripts/ro3/run_r3_orientation_validation.sh",
    "scripts/ro3/run_r3_orientation_test.sh",
    "scripts/ro3/freeze_r3a_evidence.py",
]

DIRECTORIES = [
    "experiments/RO3/R3_Orientation_v1",
    "experiments/RO3/R3_Eval_v1",
    "experiments/RO3/R3_Orientation_ValidationSummary_v1",
    "experiments/RO3/R3_Orientation_TestSummary_v1",
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify():
    for p in (ARCHIVE, PARTIAL, LEDGER, ARCHIVE_LEDGER):
        require(not p.exists(), f"Refusing overwrite: {p}")

    require(REPORT.is_file(), "Save the R3-A report first")
    require(LOCK.is_file(), "Missing R3-A protocol lock")

    lock = json.loads(LOCK.read_text())
    require(
        lock["protocol"] == "RO3_R3A_FrozenTest_v1",
        "Wrong frozen protocol"
    )

    for name, digest in lock["files_sha256"].items():
        path = ROOT / name
        require(
            path.is_file() and sha256(path) == digest,
            f"Locked file changed: {name}"
        )

    for split, expected in EXPECTED.items():
        summary_dir = (
            "R3_Orientation_ValidationSummary_v1"
            if split == "valid"
            else "R3_Orientation_TestSummary_v1"
        )
        summary_name = (
            "validation_orientation_summary.csv"
            if split == "valid"
            else "test_orientation_summary.csv"
        )
        profile_name = (
            "validation_profile_results.csv"
            if split == "valid"
            else "test_profile_results.csv"
        )

        directory = RO3 / summary_dir
        rows = read_csv(directory / summary_name)
        profiles = read_csv(directory / profile_name)

        require(len(rows) == 7, "Incomplete overall summary")
        require(len(profiles) == 35, "Incomplete profile summary")

        observed = set()

        clean = json.loads(
            (
                RO3 / "R1_Eval_v1"
                / f"{split}_uniform_p00_r0"
                / "metrics.json"
            ).read_text()
        )["run_spec"]

        manifest = (
            RO3 / "R3_Orientation_v1"
            / split / "orientation_manifest.jsonl"
        )
        manifest_hash = sha256(manifest)

        for row in rows:
            angle = int(row["angle_degrees"])
            require(angle in ANGLES, "Unexpected angle")
            require(angle not in observed, "Duplicate angle")
            observed.add(angle)

            require(
                int(row["instances"]) == expected["objects"]
                and int(row["valid_keypoints"])
                == expected["keypoints"]
                and int(row["rotated_correct"])
                == expected["counts"][angle],
                f"Unexpected PCK count: {split}/{angle}"
            )

            condition = ANGLES[angle]
            require(
                row["condition_id"] == condition,
                "Condition ID mismatch"
            )

            folder = (
                RO3 / "R3_Eval_v1"
                / f"{split}_{condition}"
            )

            result = json.loads(
                (folder / "metrics.json").read_text()
            )
            spec = result["run_spec"]
            primary = result["primary_original_box_pck"]

            require(
                spec["split"] == split
                and spec["angle_degrees"] == angle
                and spec["selected_instances"]
                == expected["objects"]
                and spec["max_instances"] is None,
                "Wrong or incomplete inference"
            )
            require(
                spec["manifest_sha256"] == manifest_hash,
                "Evaluation manifest mismatch"
            )
            require(
                primary["total_correct"]
                == expected["counts"][angle]
                and primary["total_valid"]
                == expected["keypoints"],
                "Stored metric mismatch"
            )

            for key in ("checkpoint_sha256", "config_sha256"):
                require(
                    spec[key] == clean[key],
                    "Frozen model differs"
                )

            for filename in (
                "vitpose_predictions.npy",
                "vitpose_image_ids.npy",
                "vitpose_annotation_ids.npy",
            ):
                require(
                    (folder / filename).is_file(),
                    f"Missing prediction array: {folder / filename}"
                )

        require(set(ANGLES) == observed, "Missing rotation angle")

    print("PASS: Frozen protocol and 14 evaluation runs verified")


def main():
    verify()

    files = [ROOT / name for name in SCRIPTS]
    files.extend([LOCK, REPORT])

    for name in DIRECTORIES:
        directory = ROOT / name
        require(directory.is_dir(), f"Missing directory: {name}")

        for path in directory.rglob("*"):
            require(
                not path.is_symlink(),
                f"Symlink in evidence directory: {path}"
            )
            if path.is_file():
                files.append(path)

    for name in (
        "ro3_r3a_validation_complete.log",
        "ro3_r3a_test_complete.log",
    ):
        path = RO3 / name
        if path.is_file():
            files.append(path)

    files = sorted(set(files), key=relative)

    for path in files:
        require(path.is_file(), f"Missing evidence: {path}")

    EVIDENCE.mkdir(parents=True, exist_ok=True)

    expected_hashes = {
        relative(path): sha256(path)
        for path in files
    }

    with LEDGER.open("x") as f:
        for name, digest in sorted(expected_hashes.items()):
            f.write(f"{digest}  {name}\n")

    expected_hashes[relative(LEDGER)] = sha256(LEDGER)

    with tarfile.open(PARTIAL, "x:gz") as archive:
        for name in sorted(expected_hashes):
            archive.add(
                ROOT / name,
                arcname=name,
                recursive=False
            )

    observed = set()

    with tarfile.open(PARTIAL, "r:gz") as archive:
        for member in archive:
            require(member.isfile(), "Unexpected archive entry")
            require(
                member.name in expected_hashes
                and member.name not in observed,
                "Unexpected or duplicate archived file"
            )

            stream = archive.extractfile(member)
            require(stream is not None, "Unreadable archive file")

            h = hashlib.sha256()
            for block in iter(
                lambda: stream.read(1 << 20), b""
            ):
                h.update(block)

            require(
                h.hexdigest() == expected_hashes[member.name],
                f"Archive hash mismatch: {member.name}"
            )
            observed.add(member.name)

    require(
        observed == set(expected_hashes),
        "Archive missing required evidence"
    )

    PARTIAL.rename(ARCHIVE)

    with ARCHIVE_LEDGER.open("x") as f:
        f.write(
            f"{sha256(ARCHIVE)}  {relative(ARCHIVE)}\n"
        )

    print("PASS: R3-A archive independently verified")
    print("Evidence archive:", ARCHIVE)
    print("Contents ledger:", LEDGER)
    print("Archive SHA-256:", ARCHIVE_LEDGER)
    print("Original R1/R2 evidence unchanged")


if __name__ == "__main__":
    main()
