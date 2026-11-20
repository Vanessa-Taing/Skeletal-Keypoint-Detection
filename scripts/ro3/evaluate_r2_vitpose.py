#!/usr/bin/env python3
"""RO3 R2: Frozen ViTPose-B scene-level lighting evaluation."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "scripts/evaluate"))

from common_pose_metrics import (
    calculate_all_metrics as score_original,
    make_category_map_from_coco,
)
from common_pose_metrics_ro3_constrained_symmetry_v1 import (
    calculate_all_metrics as score_revised,
)

from prepare_r2_lighting import (
    LEVELS,
    NEUTRAL,
    condition_id,
    render_lighting,
    sha256,
)

CONFIG = ROOT / (
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "td-hm_ViTPose-base_innocount_patched_v1.py"
)
CHECKPOINT = ROOT / (
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "epoch_90.pth"
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def identity(ann):
    return int(ann["image_id"]), int(ann["id"])


def from_sample(sample):
    pred = np.asarray(
        sample.pred_instances.keypoints,
        dtype=np.float32,
    ).reshape(-1, 6, 2)

    require(pred.shape == (1, 6, 2), "Unexpected prediction shape")
    require(np.isfinite(pred).all(), "Non-finite prediction")
    return pred[0]


def load_manifest(path, split, mode, value):
    meta_path = path.parent / "protocol_metadata.json"
    require(meta_path.is_file(), f"Missing: {meta_path}")

    meta = json.loads(meta_path.read_text())

    require(meta["split"] == split, "Manifest split mismatch")
    require(meta["mode"] == mode, "Manifest mode mismatch")
    require(
        any(abs(float(value) - v) < 1e-9 for v in LEVELS[mode]),
        "Unrecognized lighting level",
    )

    require(
        meta["levels"] == LEVELS[mode],
        "Transformation levels differ from protocol",
    )

    require(
        sha256(ROOT / "scripts/ro3/prepare_r2_lighting.py")
        == meta["generator_sha256"],
        "Generator differs from manifest-producing version",
    )

    annotation_path = ROOT / meta["annotation_path"]
    require(
        sha256(annotation_path) == meta["annotation_sha256"],
        "COCO annotation hash mismatch",
    )

    selected = []
    all_ids = set()

    wanted = condition_id(mode, value)

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            if row["condition_id"] != wanted:
                continue

            require(row["split"] == split, "Row split mismatch")
            require(row["mode"] == mode, "Row mode mismatch")
            require(
                abs(float(row["value"]) - value) < 1e-9,
                "Row lighting value mismatch",
            )
            require(
                row["geometry_changed"] is False,
                "Unexpected geometric modification",
            )

            image_id = int(row["image_id"])

            require(image_id not in all_ids, "Duplicate image ID")
            all_ids.add(image_id)

            require(
                row["source_sha256"]
                == meta["source_image_hashes"][str(image_id)],
                "Source image hash record mismatch",
            )

            selected.append(row)

    selected.sort(key=lambda r: int(r["image_id"]))

    require(
        len(selected) == 74,
        "Expected every original source image",
    )

    coco = json.loads(annotation_path.read_text())

    require(
        {int(x["id"]) for x in coco["images"]} == all_ids,
        "COCO/manifest image IDs differ",
    )

    expected_objects = 1551 if split == "valid" else 1626

    require(
        len(coco["annotations"]) == expected_objects,
        "Unexpected GT instance count",
    )

    return selected, meta, coco


def make_predictions(
    model, inference_topdown,
    rows_by_image, image_records, completed,
    progress_file, batch_size,
):
    """Evaluate one photometrically transformed image per scene."""
    with progress_file.open("a") as progress:
        for image in image_records:
            image_id = int(image["image_id"])

            anns = [
                a for a in rows_by_image.get(image_id, [])
                if f"{image_id}:{int(a['id'])}" not in completed
            ]

            if not anns:
                continue

            source_path = ROOT / image["source_image"]

            require(
                source_path.is_file(),
                f"Missing source image: {source_path}",
            )

            require(
                sha256(source_path) == image["source_sha256"],
                f"Source image changed: {source_path}",
            )

            source = cv2.imread(
                str(source_path),
                cv2.IMREAD_COLOR,
            )

            require(source is not None, f"Cannot decode: {source_path}")
            require(
                source.shape[:2] == (
                    int(image["height"]),
                    int(image["width"]),
                ),
                "Input dimensions differ from COCO",
            )

            changed = render_lighting(image, source)
            require(changed.shape == source.shape, "Geometry changed")

            if abs(float(image["value"]) - NEUTRAL[image["mode"]]) < 1e-9:
                require(
                    np.array_equal(changed, source),
                    "Neutral transform is not pixel-identical",
                )

            for start in range(0, len(anns), batch_size):
                chunk = anns[start:start + batch_size]

                boxes = np.asarray(
                    [
                        [
                            float(a["bbox"][0]),
                            float(a["bbox"][1]),
                            float(a["bbox"][0]) + float(a["bbox"][2]),
                            float(a["bbox"][1]) + float(a["bbox"][3]),
                        ]
                        for a in chunk
                    ],
                    dtype=np.float32,
                )

                require(
                    np.all(boxes[:, 2] > boxes[:, 0])
                    and np.all(boxes[:, 3] > boxes[:, 1]),
                    "Invalid GT bounding box",
                )

                samples = inference_topdown(
                    model,
                    changed,
                    bboxes=boxes,
                    bbox_format="xyxy",
                )

                require(
                    len(samples) == len(chunk),
                    "Inference count does not match supplied GT boxes",
                )

                for ann, sample in zip(chunk, samples):
                    rid = f"{image_id}:{int(ann['id'])}"
                    require(rid not in completed, "Duplicate prediction")

                    record = {
                        "record_id": rid,
                        "prediction_xy": from_sample(sample).tolist(),
                    }

                    progress.write(json.dumps(record) + "\n")
                    progress.flush()
                    completed[rid] = record

            print(
                f"Image {image_id}: "
                f"{len(completed)} predictions complete"
            )


def compare_reference(rows, predictions, split):
    """Compare neutral R2 predictions with frozen R1 E03 clean arrays."""
    ref = ROOT / (
        f"experiments/RO3/R1_Eval_v1/"
        f"{split}_uniform_p00_r0"
    )

    old_pred = np.load(
        ref / "vitpose_predictions.npy",
        allow_pickle=False,
    )
    old_im = np.load(
        ref / "vitpose_image_ids.npy",
        allow_pickle=False,
    )
    old_ann = np.load(
        ref / "vitpose_annotation_ids.npy",
        allow_pickle=False,
    )

    require(
        old_pred.shape == predictions.shape,
        "Reference prediction shape mismatch",
    )

    old_map = {
        (int(old_im[i]), int(old_ann[i])): old_pred[i]
        for i in range(len(old_pred))
    }

    require(len(old_map) == len(rows), "Duplicate reference identities")
    require(
        set(old_map) == {identity(a) for a in rows},
        "Reference identity set mismatch",
    )

    old_ordered = np.stack(
        [old_map[identity(a)] for a in rows]
    )

    maximum = float(
        np.abs(predictions - old_ordered).max()
    )

    require(
        maximum <= 0.25,
        f"Neutral R2 inference differs from E03: {maximum:.6f}px",
    )

    return {
        "reference": str(ref.relative_to(ROOT)),
        "maximum_absolute_coordinate_difference_px": maximum,
        "tolerance_px": 0.25,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", choices=["valid", "test"], required=True)
    ap.add_argument("--mode", choices=["exposure", "contrast"], required=True)
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    require(args.batch_size > 0, "Invalid batch size")
    require(
        args.max_instances is None or args.max_instances > 0,
        "Invalid max-instances",
    )

    manifest = (ROOT / args.manifest).resolve()
    output = (ROOT / args.output).resolve()

    image_records, metadata, coco = load_manifest(
        manifest,
        args.split,
        args.mode,
        args.value,
    )

    anns = sorted(coco["annotations"], key=identity)

    require(
        len({identity(a) for a in anns}) == len(anns),
        "Duplicate composite GT identities",
    )

    if args.max_instances is not None:
        anns = anns[:args.max_instances]

    require(anns, "No annotations selected")

    r1_base = ROOT / (
        f"experiments/RO3/R1_Eval_v1/"
        f"{args.split}_uniform_p00_r0/metrics.json"
    )

    require(r1_base.is_file(), "Missing frozen R1 clean evaluation")
    r1 = json.loads(r1_base.read_text())

    require(CHECKPOINT.is_file(), "Missing E03 checkpoint")
    require(CONFIG.is_file(), "Missing E03 config")

    require(
        sha256(CHECKPOINT)
        == r1["run_spec"]["checkpoint_sha256"],
        "Checkpoint differs from frozen E03",
    )
    require(
        sha256(CONFIG)
        == r1["run_spec"]["config_sha256"],
        "Config differs from frozen E03",
    )

    neutral = abs(args.value - NEUTRAL[args.mode]) < 1e-9

    spec = {
        "protocol": "RO3_R2_ViTPose_GlobalLighting_v1",
        "split": args.split,
        "mode": args.mode,
        "value": args.value,
        "condition_id": condition_id(args.mode, args.value),
        "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "batch_size": args.batch_size,
        "max_instances": args.max_instances,
        "selected_instances": len(anns),
        "geometry_changed": False,
    }

    if output.exists():
        require(args.resume, "Output exists; use --resume")
        require(
            json.loads((output / "run_spec.json").read_text()) == spec,
            "Resume configuration mismatch",
        )
    else:
        require(not args.resume, "Cannot resume nonexistent output")
        output.mkdir(parents=True)
        (output / "run_spec.json").write_text(
            json.dumps(spec, indent=2)
        )

    progress_path = output / "predictions_progress.jsonl"

    completed = {}

    if progress_path.is_file():
        with progress_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                rid = record["record_id"]
                require(rid not in completed, "Duplicate progress identity")
                completed[rid] = record

    expected_ids = {
        f"{int(a['image_id'])}:{int(a['id'])}"
        for a in anns
    }

    require(
        set(completed).issubset(expected_ids),
        "Progress contains unrelated identities",
    )

    pending = len(anns) - len(completed)

    print("\nRO3 R2 FROZEN ViTPose-B E03")
    print("=" * 60)
    print("Split:", args.split)
    print("Mode:", args.mode)
    print("Value:", args.value)
    print("Selected GT objects:", len(anns))
    print("Pending:", pending)

    if pending:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules

        register_all_modules()

        model = init_model(
            str(CONFIG),
            str(CHECKPOINT),
            device=args.device,
        )
        model.eval()

        by_image = defaultdict(list)
        for ann in anns:
            by_image[int(ann["image_id"])].append(ann)

        with torch.no_grad():
            make_predictions(
                model,
                inference_topdown,
                by_image,
                image_records,
                completed,
                progress_path,
                args.batch_size,
            )

    require(
        set(completed) == expected_ids,
        "Incomplete R2 prediction set",
    )

    predictions = np.asarray(
        [
            completed[
                f"{int(a['image_id'])}:{int(a['id'])}"
            ]["prediction_xy"]
            for a in anns
        ],
        dtype=np.float32,
    )

    require(
        predictions.shape == (len(anns), 6, 2),
        "Incorrect prediction array shape",
    )
    require(np.isfinite(predictions).all(), "Non-finite prediction")

    gt_kpv = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32,
    ).reshape(-1, 6, 3)

    bboxes = np.asarray(
        [a["bbox"] for a in anns],
        dtype=np.float32,
    )

    categories = np.asarray(
        [a["category_id"] for a in anns],
        dtype=np.int64,
    )

    kwargs = {
        "predictions": predictions,
        "gt": gt_kpv[:, :, :2],
        "gt_vis": gt_kpv[:, :, 2],
        "bboxes": bboxes,
        "category_ids": categories,
        "category_map": make_category_map_from_coco(coco),
        "threshold": 0.05,
    }

    original = score_original(**kwargs)
    revised = score_revised(**kwargs)

    for name in (
        "standard_pck",
        "profile_pck",
        "semantic_roles",
        "structural_edges",
    ):
        require(
            original[name] == revised[name],
            f"Unintended metric difference: {name}",
        )

    reference = None

    if neutral and args.max_instances is None:
        reference = compare_reference(
            anns, predictions, args.split
        )

        expected = (
            (5196, 6720) if args.split == "valid"
            else (4325, 7049)
        )

        fixed = revised["standard_pck"]

        require(
            (fixed["total_correct"], fixed["total_valid"])
            == expected,
            "Clean R2 baseline does not reproduce E03",
        )

    if args.max_instances is None:
        expected_n = 1551 if args.split == "valid" else 1626
        require(
            len(anns) == expected_n,
            "Incomplete full-split evaluation",
        )

    report = {
        "run_spec": spec,
        "instances": len(anns),
        "valid_gt_keypoints":
            int((gt_kpv[:, :, 2] > 0).sum()),
        "fixed_index_pck": revised["standard_pck"],
        "constrained_order_invariant_pck":
            revised["order_invariant_pck"],
        "original_order_invariant_pck":
            original["order_invariant_pck"],
        "structural_edge_accuracy":
            revised["structural_edges"],
        "profile_pck": revised["profile_pck"],
        "reference_comparison": reference,
    }

    np.save(output / "vitpose_predictions.npy", predictions)
    np.save(
        output / "vitpose_image_ids.npy",
        np.asarray([a["image_id"] for a in anns]),
    )
    np.save(
        output / "vitpose_annotation_ids.npy",
        np.asarray([a["id"] for a in anns]),
    )

    (output / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    fixed = report["fixed_index_pck"]

    print("\nPASS: R2 evaluation completed")
    print(
        f"Fixed-index PCK: {fixed['overall']:.6f} "
        f"({fixed['total_correct']}/{fixed['total_valid']})"
    )
    print(
        "Constrained OI:",
        report["constrained_order_invariant_pck"]["overall"],
    )
    print(
        "Structural edge:",
        report["structural_edge_accuracy"]["overall"],
    )

    if reference:
        print("Clean reference check:", reference)

    print("Output:", output)


if __name__ == "__main__":
    main()
