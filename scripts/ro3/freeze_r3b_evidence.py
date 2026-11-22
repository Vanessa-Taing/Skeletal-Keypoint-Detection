#!/usr/bin/env python3
"""Verify and archive R3-B evidence without modifying frozen inputs."""

import csv
import hashlib
import json
import tarfile
from pathlib import Path

from prepare_r3_viewpoint import ROOT, CONDITIONS, sha256

RO3 = ROOT / "experiments/RO3"
EVIDENCE = ROOT / "docs/RO3/evidence"

ARCHIVE = RO3 / "R3B_FrozenEvidence_v1.tar.gz"
PARTIAL = RO3 / "R3B_FrozenEvidence_v1.tar.gz.partial"

FILELIST = EVIDENCE / "R3B_files_v1.txt"
LEDGER = EVIDENCE / "R3B_contents_v1.sha256"
ARCHIVE_SHA = EVIDENCE / "R3B_archive_v1.sha256"

IDS = [row[0] for row in CONDITIONS]

EXPECTED = {
    "valid": {
        "objects": 1551,
        "keypoints": 6720,
        "clean": 5196,
        "scenes": 72,
        "counts": [
            5196, 5084, 5111, 5142, 5212,
            5013, 5036, 5184, 5183
        ]
    },
    "test": {
        "objects": 1626,
        "keypoints": 7049,
        "clean": 4325,
        "scenes": 71,
        "counts": [
            4325, 4375, 4361, 4358, 4342,
            4363, 4298, 4344, 4266
        ]
    }
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verify_locks():
    for name, protocol in [
        ("R3B_ValidationLock_v1.json",
         "RO3_R3B_ValidationLock_v1"),
        ("R3B_TestProtocolLock_v1.json",
         "RO3_R3B_FrozenTest_v1")
    ]:
        path = RO3 / name
        lock = json.loads(path.read_text())

        require(
            lock["protocol"] == protocol,
            f"Incorrect protocol: {name}"
        )

        for filename, digest in lock["files_sha256"].items():
            source = ROOT / filename
            require(
                source.is_file()
                and sha256(source) == digest,
                f"Locked input changed: {filename}"
            )

    print("PASS: Both frozen protocol locks verified")


def verify_evaluations():
    for split, expected in EXPECTED.items():
        summary_dir = (
            "R3B_ValidationSummary_v1"
            if split == "valid"
            else "R3B_TestSummary_v1"
        )

        summary_name = (
            "validation_viewpoint_summary.csv"
            if split == "valid"
            else "test_viewpoint_summary.csv"
        )

        profile_name = (
            "validation_profile_results.csv"
            if split == "valid"
            else "test_profile_results.csv"
        )

        folder = RO3 / summary_dir
        overall = csv_rows(folder / summary_name)
        profiles = csv_rows(folder / profile_name)

        require(
            len(overall) == 9
            and len(profiles) == 45,
            f"Incomplete {split} summaries"
        )

        by_id = {row["condition_id"]: row for row in overall}

        require(
            len(by_id) == 9
            and set(by_id) == set(IDS),
            f"Missing or duplicate {split} conditions"
        )

        profile_ids = {
            (r["profile"], r["condition_id"])
            for r in profiles
        }

        require(
            len(profile_ids) == 45,
            f"Duplicate {split} profile rows"
        )

        clean_spec = json.loads(
            (
                RO3 / "R1_Eval_v1"
                / f"{split}_uniform_p00_r0"
                / "metrics.json"
            ).read_text()
        )["run_spec"]

        for index, cid in enumerate(IDS):
            row = by_id[cid]

            require(
                int(row["instances"]) == expected["objects"]
                and int(row["valid_keypoints"])
                == expected["keypoints"]
                and int(row["clean_correct"])
                == expected["clean"]
                and int(row["rotated_correct"])
                == expected["counts"][index]
                and int(row["source_images"])
                == expected["scenes"],
                f"Incorrect summary counts: {split}/{cid}"
            )

            result_dir = (
                RO3 / "R3B_Eval_v1"
                / f"{split}_{cid}"
            )

            metrics = json.loads(
                (result_dir / "metrics.json").read_text()
            )

            spec = metrics["run_spec"]
            primary = metrics["primary_original_box_pck"]

            require(
                spec["protocol"]
                == "RO3_R3B_ProjectiveViewpoint_Eval_v1"
                and spec["split"] == split
                and spec["condition_id"] == cid
                and spec["selected_instances"]
                == expected["objects"]
                and spec["max_instances"] is None,
                f"Wrong or partial run: {split}/{cid}"
            )

            require(
                primary["total_correct"]
                == expected["counts"][index]
                and primary["total_valid"]
                == expected["keypoints"],
                f"Stored PCK differs: {split}/{cid}"
            )

            for field in (
                "checkpoint_sha256",
                "config_sha256"
            ):
                require(
                    spec[field] == clean_spec[field],
                    f"Frozen model changed: {split}/{cid}"
                )

            for filename in (
                "vitpose_predictions.npy",
                "vitpose_image_ids.npy",
                "vitpose_annotation_ids.npy"
            ):
                require(
                    (result_dir / filename).is_file(),
                    f"Missing prediction array: {split}/{cid}"
                )

            if cid == "vp_neutral":
                check = metrics["clean_reference_check"]
                require(
                    check is not None
                    and check[
                        "maximum_coordinate_difference_px"
                    ] <= 0.25,
                    f"Neutral coordinate parity failed: {split}"
                )

            else:
                low = float(row["ci_low_pp"])
                high = float(row["ci_high_pp"])

                require(
                    low <= 0 <= high,
                    f"Unexpected interval: {split}/{cid}"
                )

        print(
            f"PASS: {split} — 9 conditions, "
            f"{expected['objects']} objects per condition"
        )


def evidence_files():
    named_files = [
        "scripts/ro3/prepare_r3_viewpoint.py",
        "scripts/ro3/evaluate_r3_viewpoint.py",
        "scripts/ro3/summarize_r3b_validation.py",
        "scripts/ro3/summarize_r3b_test.py",
        "scripts/ro3/run_r3b_validation.sh",
        "scripts/ro3/run_r3b_test.sh",
        "scripts/ro3/lock_r3b_test_protocol.py",
        "scripts/ro3/freeze_r3b_evidence.py",
        "docs/RO3/R3B_Methodological_Conclusion_v1.md",
        "experiments/RO3/R3B_ValidationLock_v1.json",
        "experiments/RO3/R3B_TestProtocolLock_v1.json",
    ]

    directories = [
        "experiments/RO3/R3_Viewpoint_v1",
        "experiments/RO3/R3B_Eval_v1",
        "experiments/RO3/R3B_ValidationSummary_v1",
        "experiments/RO3/R3B_TestSummary_v1",
    ]

    files = [ROOT / name for name in named_files]

    for name in directories:
        directory = ROOT / name
        require(directory.is_dir(), f"Missing: {directory}")

        for path in directory.rglob("*"):
            require(
                not path.is_symlink(),
                f"Symlink in evidence: {path}"
            )
            if path.is_file():
                files.append(path)

    files = sorted(set(files), key=rel)

    for path in files:
        require(
            path.is_file() and not path.is_symlink(),
            f"Invalid evidence file: {path}"
        )

    return files


def main():
    for target in (
        ARCHIVE, PARTIAL, FILELIST,
        LEDGER, ARCHIVE_SHA
    ):
        require(
            not target.exists(),
            f"Refusing overwrite: {target}"
        )

    verify_locks()
    verify_evaluations()

    # Confirm the post-lock correction is preserved.
    summarizer = (
        ROOT / "scripts/ro3/summarize_r3b_test.py"
    ).read_text()

    require(
        'folder = EVAL / f"test_{cid}"' in summarizer,
        "Corrected test directory selector missing"
    )

    files = evidence_files()
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    with FILELIST.open("x") as f:
        for path in files:
            f.write(rel(path) + "\n")

    with LEDGER.open("x") as f:
        for path in files:
            f.write(f"{sha256(path)}  {rel(path)}\n")

    archive_files = files + [FILELIST, LEDGER]
    hashes = {
        rel(path): sha256(path)
        for path in archive_files
    }

    with tarfile.open(PARTIAL, "x:gz") as tar:
        for path in sorted(archive_files, key=rel):
            tar.add(
                path,
                arcname=rel(path),
                recursive=False
            )

    # Read each archived member and verify its content hash.
    found = set()

    with tarfile.open(PARTIAL, "r:gz") as tar:
        for member in tar:
            require(
                member.isfile()
                and member.name in hashes
                and member.name not in found,
                "Unexpected archive entry"
            )

            stream = tar.extractfile(member)
            require(stream is not None, "Unreadable member")

            digest = hashlib.sha256()

            for chunk in iter(
                lambda: stream.read(1 << 20), b""
            ):
                digest.update(chunk)

            require(
                digest.hexdigest() == hashes[member.name],
                f"Archive checksum failed: {member.name}"
            )

            found.add(member.name)

    require(
        found == set(hashes),
        "Archive evidence inventory incomplete"
    )

    PARTIAL.rename(ARCHIVE)

    with ARCHIVE_SHA.open("x") as f:
        f.write(
            f"{sha256(ARCHIVE)}  {rel(ARCHIVE)}\n"
        )

    print("\nPASS: R3-B evidence freeze completed")
    print("Archived evidence files:", len(archive_files))
    print("Archive:", ARCHIVE)
    print("Contents ledger:", LEDGER)
    print("Archive checksum:", ARCHIVE_SHA)
    print("Frozen R1, R2, R3-A evidence unchanged")


if __name__ == "__main__":
    main()
