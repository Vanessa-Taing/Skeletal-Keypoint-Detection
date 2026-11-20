#!/usr/bin/env python3
"""Read-only R2 scene-level attribution for validation/test."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/RO3/R2_SceneAttribution_v1"

CONDITIONS = [
    ("contrast", 1.20, "ct_120"),
    ("contrast", 1.40, "ct_140"),
    ("exposure", 2.00, "ev_p200"),
]

SPLITS = {
    "valid": {"objects": 1551, "keypoints": 6720,
              "correct": 5196, "scenes": 72},
    "test": {"objects": 1626, "keypoints": 7049,
             "correct": 4325, "scenes": 71},
}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    require(rows, f"No rows: {path}")
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_predictions(folder, keys, expected):
    p = np.load(folder / "vitpose_predictions.npy",
                allow_pickle=False)
    im = np.load(folder / "vitpose_image_ids.npy",
                 allow_pickle=False)
    ann = np.load(folder / "vitpose_annotation_ids.npy",
                  allow_pickle=False)

    require(p.shape == (expected, 6, 2),
            f"Incorrect prediction shape: {folder}")
    require(len(im) == len(ann) == expected,
            "Identity-array length mismatch")
    require(np.isfinite(p).all(),
            "Nonfinite predictions")

    lookup = {}
    for i in range(expected):
        key = (int(im[i]), int(ann[i]))
        require(key not in lookup, "Duplicate composite ID")
        lookup[key] = p[i]

    require(set(lookup) == set(keys),
            f"Prediction identities differ: {folder}")

    return np.stack([lookup[k] for k in keys])


def main():
    require(not OUT.exists(),
            f"Refusing to overwrite: {OUT}")

    all_rows = []
    provenance = {}

    for split, exp in SPLITS.items():
        ann_path = ROOT / (
            "datasets/InnoCount_COCO_Patched_UniqueIDs_v1/"
            f"annotations/{split}.json"
        )
        coco = json.loads(ann_path.read_text())
        require(len(coco["images"]) == 74,
                "Unexpected image count")

        annotations = sorted(
            coco["annotations"],
            key=lambda a: (int(a["image_id"]), int(a["id"]))
        )
        require(len(annotations) == exp["objects"],
                "Unexpected instance count")

        keys = [
            (int(a["image_id"]), int(a["id"]))
            for a in annotations
        ]
        require(len(set(keys)) == len(keys),
                "Duplicate GT identity")

        gt = np.asarray(
            [a["keypoints"] for a in annotations],
            dtype=np.float32
        ).reshape(-1, 6, 3)

        bbox = np.asarray(
            [a["bbox"] for a in annotations],
            dtype=np.float32
        )

        scale = np.maximum(bbox[:, 2], bbox[:, 3])
        require(np.all(scale > 0), "Invalid GT box")

        valid = gt[:, :, 2] > 0
        require(int(valid.sum()) == exp["keypoints"],
                "Wrong keypoint denominator")

        ids = np.asarray(
            [a["image_id"] for a in annotations],
            dtype=np.int64
        )

        require(
            len(np.unique(ids)) == exp["scenes"],
            "Unexpected annotated image count"
        )

        categories = {
            int(c["id"]): c["name"]
            for c in coco["categories"]
        }
        image_names = {
            int(im["id"]): im["file_name"]
            for im in coco["images"]
        }

        base = ROOT / (
            f"experiments/RO3/R1_Eval_v1/"
            f"{split}_uniform_p00_r0"
        )
        original = read_predictions(
            base, keys, exp["objects"]
        )

        def correct(pred):
            dist = np.linalg.norm(
                pred - gt[:, :, :2], axis=2
            ) / scale[:, None]
            return (dist <= .05) & valid

        clean = correct(original)
        require(int(clean.sum()) == exp["correct"],
                f"Frozen {split} baseline changed")

        provenance[str(ann_path.relative_to(ROOT))] = (
            sha256(ann_path)
        )

        for mode, value, cid in CONDITIONS:
            folder = ROOT / (
                f"experiments/RO3/R2_Eval_v1/"
                f"{split}_{mode}_{cid}"
            )

            report_file = folder / "metrics.json"
            report = json.loads(report_file.read_text())

            spec = report["run_spec"]
            require(
                spec["split"] == split
                and spec["mode"] == mode
                and spec["condition_id"] == cid
                and spec["selected_instances"] == exp["objects"]
                and spec["max_instances"] is None,
                "Incomplete or incorrect evaluation run"
            )

            changed = correct(
                read_predictions(folder, keys, exp["objects"])
            )

            require(
                (report["fixed_index_pck"]["total_correct"],
                 report["fixed_index_pck"]["total_valid"])
                == (int(changed.sum()), exp["keypoints"]),
                "Saved metric mismatch"
            )

            provenance[str(report_file.relative_to(ROOT))] = (
                sha256(report_file)
            )

            diagnostics_file = ROOT / (
                "experiments/RO3/R2_LightingDiagnostics_v1/"
                f"{split}/image_pixel_diagnostics.csv"
            )
            diagnostics = {
                int(r["image_id"]): r
                for r in read_csv(diagnostics_file)
                if r["condition_id"] == cid
            }
            require(len(diagnostics) == 74,
                    "Incomplete image clipping audit")

            provenance[str(diagnostics_file.relative_to(ROOT))] = (
                sha256(diagnostics_file)
            )

            condition_rows = []

            for image_id in np.unique(ids):
                indices = np.where(ids == image_id)[0]
                n_points = int(valid[indices].sum())

                old_n = int(clean[indices].sum())
                new_n = int(changed[indices].sum())

                profiles = Counter(
                    categories[int(annotations[i]["category_id"])]
                    for i in indices
                )

                dominant = profiles.most_common(1)[0][0]

                diagnostic = diagnostics[int(image_id)]
                box_n = int(
                    diagnostic["gt_box_weighted_pixels"]
                )
                box_clipped = int(
                    diagnostic[
                        "gt_box_weighted_new_clipped_pixels"
                    ]
                )
                require(box_n > 0, "Missing GT-box pixels")

                row = {
                    "split": split,
                    "mode": mode,
                    "value": value,
                    "condition_id": cid,
                    "image_id": int(image_id),
                    "file_name": image_names[int(image_id)],
                    "gt_objects": len(indices),
                    "valid_keypoints": n_points,
                    "clean_correct": old_n,
                    "perturbed_correct": new_n,
                    "within_image_delta_pp":
                        100 * (new_n - old_n) / n_points,
                    "overall_delta_contribution_pp":
                        100 * (new_n - old_n)
                        / exp["keypoints"],
                    "gt_box_clipped_pct":
                        100 * box_clipped / box_n,
                    "dominant_profile": dominant,
                    "profile_counts_json": json.dumps(
                        dict(sorted(profiles.items()))
                    ),
                }

                condition_rows.append(row)
                all_rows.append(row)

            require(
                len(condition_rows) == exp["scenes"],
                "Incorrect condition scene count"
            )
            require(
                sum(r["valid_keypoints"]
                    for r in condition_rows)
                == exp["keypoints"],
                "Scene denominator mismatch"
            )

            delta = 100 * (
                int(changed.sum()) - exp["correct"]
            ) / exp["keypoints"]

            require(
                abs(sum(
                    r["overall_delta_contribution_pp"]
                    for r in condition_rows
                ) - delta) < 1e-9,
                "Scene contributions do not reconcile"
            )

            print(
                f"\n{split.upper()} {cid}: "
                f"overall Δ = {delta:+.2f} pp"
            )
            print("Largest absolute scene contributions:")

            largest = sorted(
                condition_rows,
                key=lambda r: abs(
                    r["overall_delta_contribution_pp"]
                ),
                reverse=True,
            )[:5]

            for r in largest:
                print(
                    f"  image={r['image_id']:>4} "
                    f"objects={r['gt_objects']:>3} "
                    f"scene Δ={r['within_image_delta_pp']:+7.2f} pp "
                    f"overall contribution="
                    f"{r['overall_delta_contribution_pp']:+7.2f} pp "
                    f"box clip={r['gt_box_clipped_pct']:5.1f}% "
                    f"main={r['dominant_profile']}"
                )

    require(
        len(all_rows) == (72 + 71) * len(CONDITIONS),
        "Wrong complete audit row count"
    )

    OUT.mkdir(parents=True)

    save_csv(
        OUT / "source_image_contributions.csv",
        all_rows
    )

    (OUT / "audit_metadata.json").write_text(
        json.dumps({
            "protocol": "RO3_R2_SceneAttribution_v1",
            "method": (
                "Exact paired fixed-index PCK@0.05, "
                "decomposed by original annotated image. "
                "Signed overall contributions sum to the "
                "corresponding condition's global PCK change."
            ),
            "conditions": [
                cid for _, _, cid in CONDITIONS
            ],
            "source_file_sha256": provenance,
            "script_sha256": sha256(Path(__file__)),
            "limitations": [
                "Descriptive decomposition, not causal attribution.",
                "Large scenes contribute more to micro PCK.",
                "Clipping proportions are GT-box weighted.",
                "Dominant profile is an object-count descriptor.",
                "No prediction or dataset files were modified.",
            ],
        }, indent=2)
    )

    print("\nPASS: R2 scene-attribution audit completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
