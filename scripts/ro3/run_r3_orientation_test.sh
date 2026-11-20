#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER="scripts/ro3/evaluate_r3_orientation.py"
MANIFEST="experiments/RO3/R3_Orientation_v1/test/orientation_manifest.jsonl"
BASE="experiments/RO3/R3_Eval_v1"

python -m py_compile \
  scripts/ro3/prepare_r3_orientation.py \
  "$RUNNER" \
  scripts/ro3/summarize_r3_orientation_validation.py \
  scripts/ro3/summarize_r3_orientation_test.py

# Validate and fingerprint the entire pre-test protocol.
python - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

root = Path.cwd()

angles = [
    (-45, "rot_m045"),
    (-30, "rot_m030"),
    (-15, "rot_m015"),
    (0, "rot_p000"),
    (15, "rot_p015"),
    (30, "rot_p030"),
    (45, "rot_p045"),
]

paths = [
    "scripts/ro3/prepare_r3_orientation.py",
    "scripts/ro3/evaluate_r3_orientation.py",
    "scripts/ro3/summarize_r3_orientation_validation.py",
    "scripts/ro3/summarize_r3_orientation_test.py",
    "scripts/ro3/run_r3_orientation_validation.sh",
    "scripts/ro3/run_r3_orientation_test.sh",
    "experiments/RO3/R3_Orientation_v1/valid/"
    "orientation_manifest.jsonl",
    "experiments/RO3/R3_Orientation_v1/valid/"
    "protocol_metadata.json",
    "experiments/RO3/R3_Orientation_v1/test/"
    "orientation_manifest.jsonl",
    "experiments/RO3/R3_Orientation_v1/test/"
    "protocol_metadata.json",
    "experiments/RO3/R3_Orientation_ValidationSummary_v1/"
    "validation_orientation_summary.csv",
    "experiments/RO3/R3_Orientation_ValidationSummary_v1/"
    "validation_profile_results.csv",
    "experiments/RO3/R3_Orientation_ValidationSummary_v1/"
    "analysis_report.json",
]

for angle, cid in angles:
    folder = f"experiments/RO3/R3_Eval_v1/valid_{cid}"
    paths.extend([
        f"{folder}/metrics.json",
        f"{folder}/vitpose_predictions.npy",
        f"{folder}/vitpose_image_ids.npy",
        f"{folder}/vitpose_annotation_ids.npy",
    ])

summary_path = root / (
    "experiments/RO3/"
    "R3_Orientation_ValidationSummary_v1/"
    "validation_orientation_summary.csv"
)

with summary_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 7, "Incomplete validation summary"

by_angle = {
    int(row["angle_degrees"]): row
    for row in rows
}

assert set(by_angle) == {angle for angle, _ in angles}

for angle, cid in angles:
    row = by_angle[angle]
    assert row["condition_id"] == cid
    assert int(row["instances"]) == 1551
    assert int(row["valid_keypoints"]) == 6720
    assert int(row["clean_correct"]) == 5196

    if angle == 0:
        assert int(row["rotated_correct"]) == 5196
        assert float(row["delta_pp"]) == 0.0
    else:
        assert float(row["ci_high_pp"]) < 0.0

assert int(by_angle[15]["rotated_correct"]) == 4806
assert int(by_angle[-45]["rotated_correct"]) == 4651
assert int(by_angle[45]["rotated_correct"]) == 4580

test_meta_path = root / (
    "experiments/RO3/R3_Orientation_v1/test/"
    "protocol_metadata.json"
)

test_meta = json.loads(test_meta_path.read_text())

assert test_meta["protocol"] == "RO3_R3A_InPlaneRotation_v1"
assert test_meta["split"] == "test"
assert test_meta["angles_degrees"] == [
    angle for angle, _ in angles
]
assert test_meta["source_images"] == 74
assert test_meta["annotated_source_images"] == 71
assert test_meta["gt_instances"] == 1626

test_manifest = root / (
    "experiments/RO3/R3_Orientation_v1/test/"
    "orientation_manifest.jsonl"
)

counts = {}

with test_manifest.open() as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        counts.setdefault(row["condition_id"], set())
        image_id = int(row["image_id"])
        assert image_id not in counts[row["condition_id"]]
        counts[row["condition_id"]].add(image_id)

assert set(counts) == {cid for _, cid in angles}
assert all(len(ids) == 74 for ids in counts.values())

hashes = {}

for name in paths:
    p = root / name
    assert p.is_file(), f"Missing protocol file: {name}"

    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)

    hashes[name] = h.hexdigest()

payload = {
    "protocol": "RO3_R3A_FrozenTest_v1",
    "angles_degrees": [angle for angle, _ in angles],
    "validation_baseline": "5196/6720",
    "test_baseline_expected": "4325/7049",
    "primary_metric": (
        "Fixed-index PCK@0.05, "
        "original GT bbox max dimension"
    ),
    "files_sha256": hashes,
}

lock = root / (
    "experiments/RO3/R3A_TestProtocolLock_v1.json"
)

if lock.exists():
    assert json.loads(lock.read_text()) == payload, (
        "Existing R3-A protocol lock differs. "
        "Refusing to continue."
    )
    print("PASS: Existing R3-A test protocol lock verified")
else:
    with lock.open("x") as f:
        json.dump(payload, f, indent=2)
    print("PASS: R3-A test protocol locked")

print("PASS: All seven validation conditions verified")
print("PASS: Test manifest contains seven complete conditions")
PY

run_angle() {
    local angle="$1"
    local condition="$2"
    local output="$BASE/test_${condition}"

    echo
    echo "============================================"
    echo "R3-A TEST: $angle degrees"
    echo "============================================"

    if [[ -f "$output/metrics.json" ]]; then
        echo "Existing result; final analysis will verify."
        return
    fi

    local args=(
        --manifest "$MANIFEST"
        --split test
        --angle "$angle"
        --batch-size 8
        --device cuda:0
        --output "$output"
    )

    if [[ -d "$output" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
}

# Neutral baseline first. Stop immediately if parity fails.
run_angle 0 rot_p000

python - <<'PY'
import json
from pathlib import Path

path = Path(
    "experiments/RO3/R3_Eval_v1/"
    "test_rot_p000/metrics.json"
)

assert path.is_file(), "Missing neutral test result"

data = json.loads(path.read_text())

primary = data["primary_original_box_pck"]
secondary = data["secondary_rotated_box_pck"]
reference = data["clean_reference_check"]

assert (
    primary["total_correct"], primary["total_valid"]
) == (4325, 7049)

assert secondary == primary

assert reference is not None
assert (
    reference["maximum_coordinate_difference_px"]
    <= 0.25
)

print("PASS: R3-A neutral test baseline 4325/7049")
PY

run_angle -45 rot_m045
run_angle -30 rot_m030
run_angle -15 rot_m015
run_angle  15 rot_p015
run_angle  30 rot_p030
run_angle  45 rot_p045

python scripts/ro3/summarize_r3_orientation_test.py

echo
echo "PASS: R3-A frozen test completed"
