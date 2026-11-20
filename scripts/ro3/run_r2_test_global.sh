#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER=scripts/ro3/evaluate_r2_vitpose.py
BASE=experiments/RO3/R2_Eval_v1
MB=experiments/RO3/R2_Lighting_v1

python -m py_compile \
  scripts/ro3/prepare_r2_lighting.py \
  "$RUNNER" \
  scripts/ro3/summarize_r2_test.py

# Record exact experimental inputs before test inference.
# On resume, reject any differences from the original lock.
python - <<'PY'
import csv
import hashlib
import json
from pathlib import Path

root = Path.cwd()

required = [
    "scripts/ro3/prepare_r2_lighting.py",
    "scripts/ro3/evaluate_r2_vitpose.py",
    "scripts/ro3/summarize_r2_test.py",
    "experiments/RO3/R2_Validation_PairedSummary_v1/"
    "validation_lighting_summary.csv",
    "experiments/RO3/R2_LightingDiagnostics_v1/valid/"
    "lighting_clipping_summary.csv",
]

for split in ("valid", "test"):
    for mode in ("exposure", "contrast"):
        base = (
            f"experiments/RO3/R2_Lighting_v1/"
            f"{split}_{mode}"
        )
        required.extend([
            f"{base}/lighting_manifest.jsonl",
            f"{base}/protocol_metadata.json",
        ])

hashes = {}

for name in required:
    path = root / name
    assert path.is_file(), f"Missing protocol input: {path}"

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    hashes[name] = h.hexdigest()

with (root / required[3]).open(newline="") as f:
    validation = list(csv.DictReader(f))

expected = (
    {("exposure", v) for v in (-2., -1., 0., 1., 2.)}
    | {("contrast", v) for v in (.65, .8, 1., 1.2, 1.4)}
)

assert len(validation) == 10
assert {
    (row["mode"], float(row["value"]))
    for row in validation
} == expected

for row in validation:
    assert int(row["instances"]) == 1551
    assert int(row["valid_keypoints"]) == 6720

with (root / required[4]).open(newline="") as f:
    clipping = list(csv.DictReader(f))

assert len(clipping) == 10
assert {
    (row["mode"], float(row["value"]))
    for row in clipping
} == expected

for mode in ("exposure", "contrast"):
    path = root / (
        f"experiments/RO3/R2_Lighting_v1/"
        f"test_{mode}/protocol_metadata.json"
    )
    meta = json.loads(path.read_text())

    assert meta["split"] == "test"
    assert meta["mode"] == mode
    assert meta["source_images"] == 74
    assert meta["source_gt_instances"] == 1626

    manifest = path.with_name("lighting_manifest.jsonl")
    count = sum(1 for line in manifest.open() if line.strip())
    assert count == 370, (manifest, count)

lock = root / "experiments/RO3/R2_ProtocolLock_v1.json"

payload = {
    "protocol": "RO3_R2_GlobalLighting_FrozenTest_v1",
    "validation_fixed_baseline": "5196/6720",
    "test_fixed_baseline_expected": "4325/7049",
    "files_sha256": hashes,
}

if lock.exists():
    assert json.loads(lock.read_text()) == payload, (
        "R2 protocol lock differs. Refusing to continue."
    )
    print("PASS: Existing R2 protocol lock unchanged")
else:
    with lock.open("x") as f:
        json.dump(payload, f, indent=2)
    print("PASS: Created R2 protocol lock:", lock)
PY

run_condition() {
    local mode="$1"
    local value="$2"
    local condition="$3"

    local manifest="$MB/test_${mode}/lighting_manifest.jsonl"
    local out="$BASE/test_${mode}_${condition}"

    echo
    echo "===== Test: ${mode} ${value} ====="

    if [[ -f "$out/metrics.json" ]]; then
        echo "Existing result; final summarizer will verify it."
        return
    fi

    local args=(
        --manifest "$manifest"
        --split test
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

run_condition exposure -2 ev_m200
run_condition exposure -1 ev_m100
run_condition exposure  0 ev_p000
run_condition exposure  1 ev_p100
run_condition exposure  2 ev_p200

run_condition contrast 0.65 ct_065
run_condition contrast 0.80 ct_080
run_condition contrast 1.00 ct_100
run_condition contrast 1.20 ct_120
run_condition contrast 1.40 ct_140

python scripts/ro3/summarize_r2_test.py

echo "PASS: R2 frozen exposure/contrast test completed"
