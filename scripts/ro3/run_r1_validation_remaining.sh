#!/usr/bin/env bash
set -euo pipefail

cd ~/skeletal_keypoint_detection

MANIFEST="experiments/RO3/R1_Occlusion_v1/valid_uniform_seed20261119/occlusion_manifest.jsonl"
BASE="experiments/RO3/R1_Eval_v1"
RUNNER="scripts/ro3/evaluate_r1_vitpose.py"

echo "================================================="
echo "RO3 R1 — VALIDATION EVALUATION"
echo "================================================="

# --------------------------------------------------
# A. Require completed full validation baselines.
# --------------------------------------------------

test -f "$BASE/valid_uniform_p00_r0/metrics.json"
test -f "$BASE/valid_uniform_p10_r0/metrics.json"

echo "PASS: Clean and 10% validation runs exist."

# --------------------------------------------------
# B. Clean single-box inference check (12 objects).
# --------------------------------------------------

SINGLE="$BASE/valid_uniform_p00_single_n12"

if [[ ! -f "$SINGLE/metrics.json" ]]; then
    args=(
        --manifest "$MANIFEST"
        --split valid
        --level 0
        --replicate 0
        --batch-size 1
        --max-instances 12
        --device cuda:0
        --output "$SINGLE"
    )

    if [[ -d "$SINGLE" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
fi

echo
echo "Checking batched versus single-box predictions..."

python - <<'PY'
import json
from pathlib import Path
import numpy as np

base = Path("experiments/RO3/R1_Eval_v1")

def read(name):
    d = base / name
    predictions = np.load(
        d / "vitpose_predictions.npy",
        allow_pickle=False,
    )
    images = np.load(
        d / "vitpose_image_ids.npy",
        allow_pickle=False,
    )
    annotations = np.load(
        d / "vitpose_annotation_ids.npy",
        allow_pickle=False,
    )

    return {
        (int(im), int(ann)): predictions[i]
        for i, (im, ann) in enumerate(
            zip(images, annotations)
        )
    }

batched = read("valid_uniform_p00_r0")
single = read("valid_uniform_p00_single_n12")

assert len(single) == 12
assert set(single).issubset(batched)

manifest = Path(
    "experiments/RO3/R1_Occlusion_v1/"
    "valid_uniform_seed20261119/occlusion_manifest.jsonl"
)

clean_gt = {}

with manifest.open() as f:
    for line in f:
        row = json.loads(line)

        if row["requested_bbox_occlusion_pct"] != 0:
            continue

        identity = (
            int(row["image_id"]),
            int(row["annotation_id"]),
        )

        clean_gt[identity] = row

max_difference = 0.0
decision_changes = 0
valid_points = 0

for identity, pred_single in single.items():
    pred_batch = batched[identity]

    max_difference = max(
        max_difference,
        float(
            np.max(np.abs(pred_batch - pred_single))
        ),
    )

    row = clean_gt[identity]

    gt = np.asarray(
        row["gt_keypoints"], dtype=np.float32
    ).reshape(6, 3)

    scale = max(
        float(row["gt_bbox_xywh"][2]),
        float(row["gt_bbox_xywh"][3]),
    )

    mask = gt[:, 2] > 0

    good_batch = (
        np.linalg.norm(
            pred_batch - gt[:, :2], axis=1
        ) / scale <= 0.05
    )

    good_single = (
        np.linalg.norm(
            pred_single - gt[:, :2], axis=1
        ) / scale <= 0.05
    )

    valid_points += int(mask.sum())

    decision_changes += int(
        np.count_nonzero(
            (good_batch != good_single) & mask
        )
    )

print("Compared objects:", len(single))
print("Compared valid GT keypoints:", valid_points)
print(
    "Maximum coordinate difference:",
    f"{max_difference:.6f} px",
)
print("PCK decision changes:", decision_changes)

assert max_difference <= 0.25, (
    "Batching changes coordinates beyond tolerance"
)

assert decision_changes == 0, (
    "Batching changes PCK classification. "
    "Investigate before evaluating robustness."
)

print("PASS: Clean single-box/batched inference compatible")
PY

# --------------------------------------------------
# C. Run remaining validation severities.
# --------------------------------------------------

for PCT in 20 30 40 50; do
    OUT="$BASE/valid_uniform_p${PCT}_r0"

    echo
    echo "================================================="
    echo "Evaluating uniform occlusion: ${PCT}%"
    echo "================================================="

    if [[ -f "$OUT/metrics.json" ]]; then
        echo "Completed run found; skipping."
        continue
    fi

    args=(
        --manifest "$MANIFEST"
        --split valid
        --level "$PCT"
        --replicate 0
        --device cuda:0
        --output "$OUT"
    )

    if [[ -d "$OUT" ]]; then
        args+=(--resume)
    fi

    python "$RUNNER" "${args[@]}"
done

# --------------------------------------------------
# D. Generate paired robustness results.
# --------------------------------------------------

echo
echo "================================================="
echo "Generating paired validation report"
echo "================================================="

python scripts/ro3/summarize_r1_validation.py

echo
echo "PASS: RO3 R1 validation evaluation completed"
