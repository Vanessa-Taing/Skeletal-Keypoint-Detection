#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER="scripts/ro3/evaluate_r2_vitpose.py"
BASE="experiments/RO3/R2_Eval_v1"
MANIFEST_BASE="experiments/RO3/R2_Lighting_v1"

python -m py_compile \
  scripts/ro3/prepare_r2_lighting.py \
  scripts/ro3/evaluate_r2_vitpose.py \
  scripts/ro3/summarize_r2_validation.py

# Require the already verified clean exposure baseline.
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "experiments/RO3/R2_Eval_v1/"
    "valid_exposure_ev_p000/metrics.json"
)

assert path.is_file(), "Missing R2 clean validation baseline"

data = json.loads(path.read_text())
fixed = data["fixed_index_pck"]

assert (fixed["total_correct"], fixed["total_valid"]) == (
    5196, 6720
)

ref = data["reference_comparison"]
assert ref is not None
assert ref["maximum_absolute_coordinate_difference_px"] <= 0.25

print("PASS: R2 clean baseline 5196/6720")
PY

run_condition() {
    local mode="$1"
    local value="$2"
    local condition="$3"

    local manifest="$MANIFEST_BASE/valid_${mode}/lighting_manifest.jsonl"
    local out="$BASE/valid_${mode}_${condition}"

    echo
    echo "================================================="
    echo "R2 validation: $mode = $value"
    echo "================================================="

    if [[ -f "$out/metrics.json" ]]; then
        echo "Completed results exist; analysis will verify them."
        return
    fi

    local args=(
        --manifest "$manifest"
        --split valid
        --mode "$mode"
        --value "$value"
        --device cuda:0
        --batch-size 8
        --output "$out"
    )

    if [[ -d "$out" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
}

# Exposure axis. 0 EV and -1 EV are already completed,
# but existing runs are checked rather than overwritten.
run_condition exposure -2 ev_m200
run_condition exposure -1 ev_m100
run_condition exposure  0 ev_p000
run_condition exposure  1 ev_p100
run_condition exposure  2 ev_p200

# Independent contrast axis.
run_condition contrast 0.65 ct_065
run_condition contrast 0.80 ct_080
run_condition contrast 1.00 ct_100
run_condition contrast 1.20 ct_120
run_condition contrast 1.40 ct_140

echo
echo "================================================="
echo "Generating paired validation report"
echo "================================================="

python scripts/ro3/summarize_r2_validation.py

echo
echo "PASS: R2 exposure and contrast validation complete"
