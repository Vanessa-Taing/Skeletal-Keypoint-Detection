#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/RO3"
SOURCE = BASE / "H1A_Channel_Orientation_Audit_v1"
OUT = BASE / "H1B_Channel_Signature_Crosstab_v1"

assert not OUT.exists(), f"Output exists: {OUT}"

def read_csv(name):
    with (SOURCE / name).open(newline="") as f:
        return list(csv.DictReader(f))

def is_true(value):
    return str(value).strip().lower() == "true"

def write_csv(name, rows):
    assert rows
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

gt_rows = read_csv("channel_gt_spatial_signatures.csv")
pred_rows = read_csv("channel_prediction_correspondence.csv")

assert len(pred_rows) == 504
assert sum(
    int(r["fixed_correct_at_005"]) for r in pred_rows
) == 668
assert sum(
    int(r["spatial_correct_at_005"]) for r in pred_rows
) == 1287

# Both GT and prediction corner assignments must be
# clear, and the predicted geometry must be reasonably
# close to the GT under oracle-assisted assignment.
plausible = [
    r for r in pred_rows
    if is_true(r["gt_corner_clear"])
    and is_true(r["prediction_corner_clear"])
    and is_true(r["spatial_mean_error_le_020"])
]

assert len(plausible) == 394, len(plausible)

OUT.mkdir(parents=True)

# A. Pairwise GT-to-predicted signature counts.
grouped = defaultdict(list)

for r in plausible:
    key = (
        r["image_file"],
        r["gt_corner_signature"],
        r["prediction_corner_signature"],
    )
    grouped[key].append(r)

pair_rows = []

for (image, gt_sig, pred_sig), items in grouped.items():
    n = len(items)
    fixed = sum(
        int(r["fixed_correct_at_005"]) for r in items
    )
    spatial = sum(
        int(r["spatial_correct_at_005"]) for r in items
    )

    pair_rows.append({
        "Image": image,
        "GT_signature": gt_sig,
        "Predicted_signature": pred_sig,
        "Same_signature": gt_sig == pred_sig,
        "Instances": n,
        "Fixed_PCK_005": fixed / (4*n),
        "Spatial_PCK_005": spatial / (4*n),
    })

pair_rows.sort(
    key=lambda r: (r["Image"], -r["Instances"])
)

write_csv("gt_vs_prediction_signatures.csv", pair_rows)

# B. Compare clear GT corner signatures across splits.
split_counts = Counter(
    (r["split"], r["gt_corner_signature"])
    for r in gt_rows
    if is_true(r["gt_corner_clear"])
)

split_totals = Counter(
    r["split"] for r in gt_rows
    if is_true(r["gt_corner_clear"])
)

distribution = []

for (split, signature), count in sorted(
    split_counts.items()
):
    distribution.append({
        "Split": split,
        "Signature": signature,
        "Objects": count,
        "Share_of_clear_GT": count / split_totals[split],
    })

write_csv("gt_signature_distribution.csv", distribution)

# C. Select representative mismatched examples for
# manual visual inspection, without changing predictions.
review = []

for image in ("000137.jpg", "000477.jpg"):
    scene = [
        r for r in plausible if r["image_file"] == image
    ]

    mismatch_count = sum(
        r["gt_corner_signature"]
        != r["prediction_corner_signature"]
        for r in scene
    )

    print(f"\n{image}")
    print("Plausible clear-corner cases:", len(scene))
    print("Signature mismatches:", mismatch_count)

    print("Most common GT -> predicted patterns:")

    pairs = Counter(
        (
            r["gt_corner_signature"],
            r["prediction_corner_signature"],
        )
        for r in scene
    )

    for (gt_sig, pred_sig), count in pairs.most_common(10):
        print(
            f"  {gt_sig} -> {pred_sig}: "
            f"{count} instances"
        )

    selected_pairs = [
        pair for pair, count in pairs.most_common()
        if pair[0] != pair[1] and count >= 3
    ][:5]

    for gt_sig, pred_sig in selected_pairs:
        candidates = [
            r for r in scene
            if r["gt_corner_signature"] == gt_sig
            and r["prediction_corner_signature"] == pred_sig
        ]

        candidates.sort(
            key=lambda r: (
                -int(r["spatial_correct_at_005"]),
                int(r["fixed_correct_at_005"]),
                float(r["spatial_mean_normalized_error"]),
            )
        )

        for r in candidates[:2]:
            review.append({
                "Image": image,
                "Annotation_ID": r["annotation_id"],
                "GT_signature": gt_sig,
                "Predicted_signature": pred_sig,
                "Fixed_correct_005":
                    r["fixed_correct_at_005"],
                "Spatial_correct_005":
                    r["spatial_correct_at_005"],
                "Spatial_mean_error":
                    r["spatial_mean_normalized_error"],
                "Nearest_mapping":
                    r["nearest_gt_to_prediction_mapping"],
            })

if review:
    write_csv("visual_review_candidates.csv", review)

summary = {
    "matched_channel_predictions": len(pred_rows),
    "clear_and_geometrically_plausible": len(plausible),
    "fixed_correct_all_matched": 668,
    "spatial_correct_all_matched": 1287,
    "analysis_note": (
        "Image-space signatures are heuristic. "
        "Pairwise mismatches do not independently prove "
        "incorrect annotation semantics."
    ),
}

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\nPASS: H1-B completed")
print("Output:", OUT)
