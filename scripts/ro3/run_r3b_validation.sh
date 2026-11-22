#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

RUNNER="scripts/ro3/evaluate_r3_viewpoint.py"
BASE="experiments/RO3/R3B_Eval_v1"

python -m py_compile \
  scripts/ro3/prepare_r3_viewpoint.py \
  "$RUNNER" \
  scripts/ro3/summarize_r3b_validation.py

python - <<'PY'
import hashlib
import json
from collections import defaultdict
from pathlib import Path

root = Path.cwd()

conditions = [
    "vp_neutral",
    "vp_yaw_left_06",
    "vp_yaw_right_06",
    "vp_pitch_top_06",
    "vp_pitch_bottom_06",
    "vp_yaw_left_12",
    "vp_yaw_right_12",
    "vp_pitch_top_12",
    "vp_pitch_bottom_12",
]

meta_path = root / (
    "experiments/RO3/R3_Viewpoint_v1/valid/"
    "protocol_metadata.json"
)

manifest_path = meta_path.with_name(
    "viewpoint_manifest.jsonl"
)

meta = json.loads(meta_path.read_text())

assert meta["protocol"] == "RO3_R3B_ProjectiveViewpoint_v1"
assert meta["split"] == "valid"
assert meta["source_images"] == 74
assert meta["annotated_source_images"] == 72
assert meta["gt_instances"] == 1551
assert meta["valid_gt_keypoints"] == 6720

assert [
    item["condition_id"]
    for item in meta["conditions"]
] == conditions

seen = defaultdict(set)
count = 0

with manifest_path.open() as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        cid = row["condition_id"]
        iid = int(row["image_id"])

        assert cid in conditions
        assert iid not in seen[cid]

        seen[cid].add(iid)
        count += 1

assert count == 666
assert set(seen) == set(conditions)
assert all(len(values) == 74 for values in seen.values())

neutral_path = root / (
    "experiments/RO3/R3B_Eval_v1/"
    "valid_vp_neutral/metrics.json"
)

neutral = json.loads(neutral_path.read_text())

assert neutral["run_spec"]["condition_id"] == "vp_neutral"
assert neutral["run_spec"]["split"] == "valid"
assert neutral["run_spec"]["max_instances"] is None
assert neutral["instances"] == 1551

primary = neutral["primary_original_box_pck"]
secondary = neutral["secondary_transformed_box_pck"]

assert (
    primary["total_correct"],
    primary["total_valid"]
) == (5196, 6720)

assert (
    secondary["total_correct"],
    secondary["total_valid"]
) == (5196, 6720)

assert abs(
    neutral["mean_transformed_bbox_scale_ratio"] - 1.0
) < 1e-8

reference = neutral["clean_reference_check"]
assert reference is not None
assert reference["maximum_coordinate_difference_px"] <= 0.25

paths = [
    "scripts/ro3/prepare_r3_viewpoint.py",
    "scripts/ro3/evaluate_r3_viewpoint.py",
    "scripts/ro3/summarize_r3b_validation.py",
    "scripts/ro3/run_r3b_validation.sh",
    "experiments/RO3/R3_Viewpoint_v1/valid/"
    "viewpoint_manifest.jsonl",
    "experiments/RO3/R3_Viewpoint_v1/valid/"
    "protocol_metadata.json",
    "experiments/RO3/R1_Eval_v1/"
    "valid_uniform_p00_r0/metrics.json",
    "experiments/RO3/R3B_Eval_v1/"
    "valid_vp_neutral/metrics.json",
    "experiments/RO3/R3B_Eval_v1/"
    "valid_vp_neutral/vitpose_predictions.npy",
    "experiments/RO3/R3B_Eval_v1/"
    "valid_vp_neutral/vitpose_image_ids.npy",
    "experiments/RO3/R3B_Eval_v1/"
    "valid_vp_neutral/vitpose_annotation_ids.npy",
]

hashes = {}

for name in paths:
    path = root / name
    assert path.is_file(), f"Missing protocol input: {name}"

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)

    hashes[name] = h.hexdigest()

assert (
    hashes["scripts/ro3/prepare_r3_viewpoint.py"]
    == meta["generator_sha256"]
)

payload = {
    "protocol": "RO3_R3B_ValidationLock_v1",
    "conditions": conditions,
    "validation_baseline": "5196/6720",
    "primary_metric":
        "Fixed-index PCK@0.05 using original GT-box scale",
    "files_sha256": hashes,
}

lock_path = root / (
    "experiments/RO3/R3B_ValidationLock_v1.json"
)

if lock_path.exists():
    assert json.loads(lock_path.read_text()) == payload, (
        "Existing R3-B validation lock differs. "
        "Refusing to continue."
    )
    print("PASS: Existing R3-B validation lock verified")
else:
    with lock_path.open("x") as f:
        json.dump(payload, f, indent=2)

    print("PASS: R3-B validation protocol locked")

print("PASS: Neutral baseline 5196/6720")
print("PASS: Nine complete geometric conditions")
PY

run_condition() {
    local condition="$1"
    local output="$BASE/valid_${condition}"

    echo
    echo "=============================================="
    echo "R3-B VALIDATION: ${condition}"
    echo "=============================================="

    if [[ -f "$output/metrics.json" ]]; then
        echo "Existing result; final summarizer will verify."
        return
    fi

    local args=(
        --split valid
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

python scripts/ro3/summarize_r3b_validation.py

echo
echo "PASS: All nine R3-B validation conditions verified"
