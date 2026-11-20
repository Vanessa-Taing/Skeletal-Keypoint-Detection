#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER="scripts/ro3/evaluate_r3_orientation.py"
MANIFEST="experiments/RO3/R3_Orientation_v1/valid/orientation_manifest.jsonl"
BASE="experiments/RO3/R3_Eval_v1"

python -m py_compile \
  scripts/ro3/prepare_r3_orientation.py \
  "$RUNNER" \
  scripts/ro3/summarize_r3_orientation_validation.py

python - <<'PY'
import json
from pathlib import Path

base = Path("experiments/RO3/R3_Eval_v1")

checks = {
    "valid_rot_p000": (0, 5196, 5196),
    "valid_rot_p015": (15, 4806, 4964),
}

for name, (angle, primary, secondary) in checks.items():
    path = base / name / "metrics.json"
    assert path.is_file(), f"Missing completed result: {path}"

    result = json.loads(path.read_text())
    assert result["run_spec"]["angle_degrees"] == angle
    assert result["run_spec"]["selected_instances"] == 1551
    assert result["run_spec"]["max_instances"] is None

    a = result["primary_original_box_pck"]
    b = result["secondary_rotated_box_pck"]

    assert (a["total_correct"], a["total_valid"]) == (
        primary, 6720
    )
    assert (b["total_correct"], b["total_valid"]) == (
        secondary, 6720
    )

assert json.loads(
    (base / "valid_rot_p000/metrics.json").read_text()
)["clean_reference_check"] is not None

print("PASS: Existing 0° and +15° validation runs verified")
PY

run_angle() {
    local angle="$1"
    local condition="$2"
    local out="$BASE/valid_${condition}"

    echo
    echo "============================================"
    echo "R3-A VALIDATION ANGLE: $angle degrees"
    echo "============================================"

    if [[ -f "$out/metrics.json" ]]; then
        echo "Existing completed result; summarizer will reverify."
        return
    fi

    local args=(
        --manifest "$MANIFEST"
        --split valid
        --angle "$angle"
        --batch-size 8
        --device cuda:0
        --output "$out"
    )

    if [[ -d "$out" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
}

run_angle -45 rot_m045
run_angle -30 rot_m030
run_angle -15 rot_m015
run_angle   0 rot_p000
run_angle  15 rot_p015
run_angle  30 rot_p030
run_angle  45 rot_p045

python scripts/ro3/summarize_r3_orientation_validation.py

echo
echo "PASS: All seven R3-A validation conditions verified"
