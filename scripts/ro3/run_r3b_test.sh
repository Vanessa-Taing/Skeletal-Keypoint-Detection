#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER="scripts/ro3/evaluate_r3_viewpoint.py"
SUMMARY="scripts/ro3/summarize_r3b_test.py"
BASE="experiments/RO3/R3B_Eval_v1"

python -m py_compile \
    scripts/ro3/prepare_r3_viewpoint.py \
    "$RUNNER" \
    "$SUMMARY"

# Reverify the existing pre-test lock.
python scripts/ro3/lock_r3b_test_protocol.py

# Independent neutral test verification.
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
ro3 = root / "experiments/RO3"

lock_path = ro3 / "R3B_TestProtocolLock_v1.json"
assert lock_path.is_file(), "STOP: Missing R3-B test lock"

lock = json.loads(lock_path.read_text())

assert lock["protocol"] == "RO3_R3B_FrozenTest_v1"
assert lock["test_baseline_expected"] == "4325/7049"

for name, expected in lock["files_sha256"].items():
    path = root / name
    assert path.is_file(), f"Missing locked input: {name}"

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    assert h.hexdigest() == expected, (
        f"Locked input changed: {name}"
    )

neutral_path = (
    ro3 / "R3B_Eval_v1/test_vp_neutral/metrics.json"
)
assert neutral_path.is_file(), "Missing neutral test result"

neutral = json.loads(neutral_path.read_text())
spec = neutral["run_spec"]

assert spec["split"] == "test"
assert spec["condition_id"] == "vp_neutral"
assert spec["selected_instances"] == 1626
assert spec["max_instances"] is None

primary = neutral["primary_original_box_pck"]
secondary = neutral["secondary_transformed_box_pck"]

assert (
    primary["total_correct"],
    primary["total_valid"]
) == (4325, 7049)

assert (
    secondary["total_correct"],
    secondary["total_valid"]
) == (4325, 7049)

assert abs(
    neutral["mean_transformed_bbox_scale_ratio"] - 1.0
) < 1e-8

reference = neutral["clean_reference_check"]
assert reference is not None

assert reference["maximum_coordinate_difference_px"] <= 0.25

for name in (
    "checkpoint_sha256",
    "config_sha256",
):
    clean = json.loads(
        (
            ro3 / "R1_Eval_v1/test_uniform_p00_r0/"
            "metrics.json"
        ).read_text()
    )
    assert spec[name] == clean["run_spec"][name]

summary_dir = ro3 / "R3B_TestSummary_v1"
assert not summary_dir.exists(), (
    "A test summary already exists; refusing overwrite"
)

print("PASS: Frozen R3-B protocol fingerprints")
print("PASS: Neutral test baseline 4325/7049")
print("PASS: Clean-coordinate compatibility")
print("PASS: Test analysis parameters unchanged")
PY

run_condition() {
    local condition="$1"
    local output="$BASE/test_${condition}"

    echo
    echo "=============================================="
    echo "R3-B FROZEN TEST: ${condition}"
    echo "=============================================="

    if [[ -f "$output/metrics.json" ]]; then
        echo "Existing completed evaluation; will be verified."
        return
    fi

    local args=(
        --split test
        --condition "$condition"
        --batch-size 8
        --device cuda:0
        --output "$output"
    )

    if [[ -d "$output" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
}

run_condition vp_neutral

run_condition vp_yaw_left_06
run_condition vp_yaw_right_06
run_condition vp_pitch_top_06
run_condition vp_pitch_bottom_06

run_condition vp_yaw_left_12
run_condition vp_yaw_right_12
run_condition vp_pitch_top_12
run_condition vp_pitch_bottom_12

python "$SUMMARY"

echo
echo "PASS: All nine R3-B frozen test conditions verified"
echo "PASS: Paired test summary generated"
