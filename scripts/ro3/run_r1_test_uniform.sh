#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

MANIFEST="experiments/RO3/R1_Occlusion_v1/test_uniform_seed20261119/occlusion_manifest.jsonl"
BASE="experiments/RO3/R1_Eval_v1"
RUNNER="scripts/ro3/evaluate_r1_vitpose.py"

echo "======================================================"
echo "RO3 R1 — FROZEN TEST UNIFORM OCCLUSION"
echo "======================================================"

# Check the existing clean baseline before any
# additional test inference.

python - <<'PY'
import json
from pathlib import Path

p = Path(
    "experiments/RO3/R1_Eval_v1/"
    "test_uniform_p00_r0/metrics.json"
)

assert p.is_file(), "Missing frozen test baseline"

r = json.loads(p.read_text())

fixed = r["fixed_index_pck"]
ref = r["reference_comparison"]
spec = r["run_spec"]

assert spec["split"] == "test"
assert spec["level"] == 0
assert spec["replicate"] == 0
assert spec["max_instances"] is None
assert spec["selected_instances"] == 1626

assert fixed["total_correct"] == 4325
assert fixed["total_valid"] == 7049

assert ref is not None
assert ref["max_abs_error_px"] <= 0.25

print("PASS: Frozen test E03 baseline = 4325/7049")
print(
    "PASS: Reference prediction maximum difference =",
    ref["max_abs_error_px"],
    "px",
)
PY

# Never overwrite or rerun the original clean baseline.
# Each severity and replicate gets a separate directory.

for PCT in 10 20 30 40 50; do
    for REP in 0 1 2; do

        OUT="$BASE/test_uniform_p${PCT}_r${REP}"

        echo
        echo "======================================================"
        echo "Occlusion: ${PCT}% | Replicate: ${REP}"
        echo "======================================================"

        if [[ -f "$OUT/metrics.json" ]]; then
            echo "Completed output exists; skipping."
            continue
        fi

        ARGS=(
            --manifest "$MANIFEST"
            --split test
            --level "$PCT"
            --replicate "$REP"
            --device cuda:0
            --batch-size 8
            --output "$OUT"
        )

        if [[ -d "$OUT" ]]; then
            ARGS+=(--resume)
        fi

        python "$RUNNER" "${ARGS[@]}"

    done
done

echo
echo "PASS: All frozen test inference conditions completed."
