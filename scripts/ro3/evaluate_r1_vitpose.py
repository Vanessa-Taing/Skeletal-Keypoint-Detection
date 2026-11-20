#!/usr/bin/env python3
"""RO3 R1: Frozen ViTPose-B occlusion robustness evaluation."""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "scripts/ro3"))
sys.path.insert(0, str(ROOT / "scripts/evaluate"))

from prepare_r1_occlusion import (
    render_record,
    find_annotations,
    sha256_file,
)

import common_pose_metrics as original_metrics
import common_pose_metrics_ro3_constrained_symmetry_v1 as revised_metrics


DEFAULT_CONFIG = (
    "experiments/ViTPose_Runs/"
    "E03_ViTPoseB_Patched_v1/"
    "td-hm_ViTPose-base_innocount_patched_v1.py"
)

DEFAULT_CHECKPOINT = (
    "experiments/ViTPose_Runs/"
    "E03_ViTPoseB_Patched_v1/epoch_90.pth"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def resolve(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def digest(path):
    return sha256_file(path)


def bbox_xyxy(row):
    x, y, w, h = map(float, row["gt_bbox_xywh"])
    require(w > 0 and h > 0, "Invalid GT bbox")
    return [x, y, x + w, y + h]


def prediction_from_sample(sample):
    pred = np.asarray(
        sample.pred_instances.keypoints,
        dtype=np.float32,
    ).reshape(-1, 6, 2)

    require(pred.shape == (1, 6, 2), "Unexpected keypoint shape")
    require(np.isfinite(pred).all(), "Non-finite keypoints")

    return pred[0].copy()


def read_manifest(path, level, replicate, split):
    metadata_path = path.parent / "protocol_metadata.json"
    require(metadata_path.is_file(), metadata_path)

    meta = json.loads(metadata_path.read_text())

    require(meta["split"] == split, "Manifest split mismatch")
    require(level in meta["levels_pct"], "Severity not in manifest")

    if level == 0:
        require(replicate == 0, "Clean baseline has only replicate 0")
    else:
        require(
            0 <= replicate < meta["replicates_nonzero"],
            "Replicate outside manifest range",
        )

    annotation_path = find_annotations(split)

    require(
        digest(annotation_path) == meta["source_annotation_sha256"],
        "Source COCO annotation hash mismatch",
    )

    rows = []

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if (
                int(row["requested_bbox_occlusion_pct"]) != level
                or int(row["replicate"]) != replicate
            ):
                continue

            require(row["split"] == split, "Row split mismatch")
            require(row["scope"] == "instance", "Unexpected augmentation scope")
            require(row["transform_type"] == "opaque_stripe", "Unexpected transform")

            if level == 0:
                require(row["occluder_xyxy"] is None, "Nonempty clean occluder")
            else:
                require(row["occluder_xyxy"] is not None, "Missing occluder")

            require(
                len(row["gt_keypoints"]) == 18,
                "Expected six COCO keypoints",
            )

            rows.append(row)

    rows.sort(
        key=lambda r: (
            int(r["image_id"]),
            int(r["annotation_id"]),
        )
    )

    keys = [
        (int(r["image_id"]), int(r["annotation_id"]))
        for r in rows
    ]

    require(len(keys) == len(set(keys)), "Duplicate GT identity")
    require(
        len(rows) == meta["gt_instance_count"],
        "Incomplete selected severity/replicate",
    )

    return rows, meta, annotation_path


def check_parity(model, inference_topdown, row, tolerance):
    """Verify path and BGR-array inputs give equivalent clean predictions."""
    source = resolve(row["source_image"])
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)

    require(image is not None, f"Cannot read {source}")

    box = np.asarray([bbox_xyxy(row)], dtype=np.float32)

    pred_path = prediction_from_sample(
        inference_topdown(
            model, str(source), bboxes=box, bbox_format="xyxy"
        )[0]
    )

    pred_array = prediction_from_sample(
        inference_topdown(
            model, image, bboxes=box, bbox_format="xyxy"
        )[0]
    )

    difference = float(
        np.max(np.abs(pred_path - pred_array))
    )

    print(f"Image path vs BGR-array maximum difference: {difference:.6f} px")

    require(
        difference <= tolerance,
        "Path/array preprocessing differs. Stop before occlusion inference.",
    )


def load_source(row, cache, hashes):
    image_id = int(row["image_id"])
    source = resolve(row["source_image"])

    require(source.is_file(), f"Missing source image: {source}")

    if image_id not in cache:
        expected = hashes.get(str(image_id))
        require(expected is not None, "Missing source image hash")

        require(
            digest(source) == expected,
            f"Source image SHA-256 mismatch: {source}",
        )

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        require(image is not None, f"Cannot decode {source}")

        require(
            image.shape[:2] == (
                int(row["image_height"]),
                int(row["image_width"]),
            ),
            f"Image dimensions mismatch: {source}",
        )

        cache.clear()
        cache[image_id] = image

    return cache[image_id]


def inference_rows(model, inference_topdown, pending, level, batch_size, hashes):
    """Yield (manifest_row, predicted_xy) in stable GT identity order."""
    cache = {}

    if level == 0:
        # Clean samples use source paths, as in the RO2 evaluator.
        # Chunked calls limit memory for densely populated images.
        for _, group in itertools.groupby(
            pending, key=lambda r: int(r["image_id"])
        ):
            group = list(group)

            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                source = resolve(chunk[0]["source_image"])

                # Verify image hash once for the source image.
                load_source(chunk[0], cache, hashes)

                boxes = np.asarray(
                    [bbox_xyxy(row) for row in chunk],
                    dtype=np.float32,
                )

                samples = inference_topdown(
                    model,
                    str(source),
                    bboxes=boxes,
                    bbox_format="xyxy",
                )

                require(
                    len(samples) == len(chunk),
                    "Clean inference row-count mismatch",
                )

                for row, sample in zip(chunk, samples):
                    yield row, prediction_from_sample(sample)

    else:
        for row in pending:
            source_image = load_source(row, cache, hashes)

            # Input image is copied and modified in memory.
            # GT coordinates are never drawn on this image.
            occluded = render_record(row, source_image)

            boxes = np.asarray(
                [bbox_xyxy(row)], dtype=np.float32
            )

            samples = inference_topdown(
                model,
                occluded,
                bboxes=boxes,
                bbox_format="xyxy",
            )

            require(len(samples) == 1, "Expected one pose")

            yield row, prediction_from_sample(samples[0])


def score_subset(predictions, gt, visibility, bboxes, mask):
    """Micro PCK over a specified set of GT keypoints."""
    errors = (
        np.linalg.norm(predictions - gt, axis=2)
        / np.maximum(bboxes[:, 2], bboxes[:, 3])[:, None]
    )

    valid = mask & (visibility > 0)
    denominator = int(valid.sum())

    numerator = int(
        ((errors <= 0.05) & valid).sum()
    )

    return {
        "correct": numerator,
        "valid": denominator,
        "pck": (
            numerator / denominator
            if denominator else None
        ),
    }


def score_results(rows, predictions, category_map):
    n = len(rows)

    gt_kpv = np.asarray(
        [r["gt_keypoints"] for r in rows],
        dtype=np.float32,
    ).reshape(n, 6, 3)

    gt = gt_kpv[:, :, :2]
    visibility = gt_kpv[:, :, 2]

    bboxes = np.asarray(
        [r["gt_bbox_xywh"] for r in rows],
        dtype=np.float32,
    )

    category_ids = np.asarray(
        [r["category_id"] for r in rows],
        dtype=np.int64,
    )

    kwargs = dict(
        predictions=predictions,
        gt=gt,
        gt_vis=visibility,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=0.05,
    )

    original = original_metrics.calculate_all_metrics(**kwargs)
    revised = revised_metrics.calculate_all_metrics(**kwargs)

    # The symmetry revision must not affect primary metrics.
    for field in (
        "standard_pck",
        "profile_pck",
        "semantic_roles",
        "structural_edges",
    ):
        require(
            original[field] == revised[field],
            f"Unexpected metric change: {field}",
        )

    newly_hidden = np.zeros((n, 6), dtype=bool)

    for i, row in enumerate(rows):
        for j in row["newly_hidden_keypoint_indices"]:
            require(1 <= int(j) <= 6, "Invalid keypoint index")
            newly_hidden[i, int(j) - 1] = True

    require(
        not np.any(newly_hidden & (visibility != 2)),
        "Newly hidden keypoint must have original visibility v=2",
    )

    original_visible = visibility == 2
    original_occluded = visibility == 1

    subset_scores = {
        "newly_hidden": score_subset(
            predictions, gt, visibility, bboxes,
            newly_hidden,
        ),
        "originally_visible_not_newly_hidden": score_subset(
            predictions, gt, visibility, bboxes,
            original_visible & ~newly_hidden,
        ),
        "originally_occluded": score_subset(
            predictions, gt, visibility, bboxes,
            original_occluded,
        ),
    }

    report = {
        "instances": n,
        "valid_gt_keypoints": int((visibility > 0).sum()),
        "fixed_index_pck": revised["standard_pck"],
        "constrained_order_invariant_pck":
            revised["order_invariant_pck"],
        "original_order_invariant_pck":
            original["order_invariant_pck"],
        "structural_edge_accuracy":
            revised["structural_edges"],
        "profile_pck": revised["profile_pck"],
        "keypoint_subgroups": subset_scores,
    }

    return report


def check_reference(rows, predictions, ref_dir, tolerance):
    ref_predictions = np.load(
        ref_dir / "vitpose_predictions.npy",
        allow_pickle=False,
    )
    image_ids = np.load(
        ref_dir / "vitpose_image_ids.npy",
        allow_pickle=False,
    )
    annotation_ids = np.load(
        ref_dir / "vitpose_annotation_ids.npy",
        allow_pickle=False,
    )

    require(
        ref_predictions.shape == (len(rows), 6, 2),
        "Reference prediction shape mismatch",
    )

    ref_map = {
        (int(image_ids[i]), int(annotation_ids[i])): ref_predictions[i]
        for i in range(len(ref_predictions))
    }

    require(len(ref_map) == len(rows), "Reference identities mismatch")

    deltas = []

    for row, prediction in zip(rows, predictions):
        key = (
            int(row["image_id"]),
            int(row["annotation_id"]),
        )
        require(key in ref_map, f"Missing reference object {key}")
        deltas.append(
            np.abs(prediction - ref_map[key]).max()
        )

    maximum = float(np.max(deltas))
    mean = float(np.mean(deltas))

    print(
        f"Reference prediction difference: "
        f"mean={mean:.6f}px, max={maximum:.6f}px"
    )

    require(
        maximum <= tolerance,
        "Clean predictions differ from E03 reference beyond tolerance",
    )

    return {"mean_max_abs_error_px": mean, "max_abs_error_px": maximum}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--parity-tolerance-px", type=float, default=0.1)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--reference-tolerance-px", type=float, default=0.25)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    require(args.batch_size >= 1, "Invalid batch size")
    require(
        args.max_instances is None or args.max_instances > 0,
        "Invalid max-instances",
    )

    manifest = resolve(args.manifest)
    config = resolve(args.config)
    checkpoint = resolve(args.checkpoint)
    out = resolve(args.output)

    for p in (manifest, config, checkpoint):
        require(p.is_file(), f"Missing: {p}")

    selected, metadata, coco_path = read_manifest(
        manifest, args.level, args.replicate, args.split
    )

    if args.max_instances is not None:
        selected = selected[:args.max_instances]

    require(selected, "No selected samples")

    coco = json.loads(coco_path.read_text())
    category_map = original_metrics.make_category_map_from_coco(coco)

    reference_dir = (
        resolve(args.reference_dir) if args.reference_dir else None
    )

    if reference_dir:
        require(args.level == 0, "Reference is for clean baseline only")
        require(
            args.max_instances is None,
            "Reference comparison requires complete split",
        )

    spec = {
        "protocol": "RO3_R1_ViTPose_E03_v1",
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "split": args.split,
        "level": args.level,
        "replicate": args.replicate,
        "config": str(config),
        "config_sha256": digest(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "batch_size": args.batch_size,
        "max_instances": args.max_instances,
        "reference_dir": str(reference_dir) if reference_dir else None,
        "reference_tolerance_px": args.reference_tolerance_px,
        "parity_tolerance_px": args.parity_tolerance_px,
        "selected_instances": len(selected),
    }

    if out.exists():
        require(args.resume, f"Output exists: {out}; use --resume")
        require(
            json.loads((out / "run_spec.json").read_text()) == spec,
            "Resume configuration does not match original run",
        )
    else:
        require(not args.resume, "Cannot resume a nonexistent run")
        out.mkdir(parents=True)
        (out / "run_spec.json").write_text(
            json.dumps(spec, indent=2)
        )

    progress_file = out / "predictions_progress.jsonl"

    completed = {}

    if progress_file.is_file():
        with progress_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record["record_id"]
                require(key not in completed, "Duplicate progress record")
                completed[key] = record

    selected_ids = {r["record_id"] for r in selected}

    require(
        set(completed).issubset(selected_ids),
        "Progress contains unrelated records",
    )

    pending = [
        r for r in selected
        if r["record_id"] not in completed
    ]

    print("=" * 65)
    print("RO3 R1 — ViTPose E03")
    print("Split:", args.split)
    print("Severity:", args.level)
    print("Replicate:", args.replicate)
    print("Selected:", len(selected))
    print("Previously completed:", len(completed))
    print("Pending:", len(pending))
    print("Output:", out)

    if pending:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules

        register_all_modules()

        model = init_model(
            str(config),
            str(checkpoint),
            device=args.device,
        )
        model.eval()

        with torch.no_grad():
            # Always check the unmodified image pipeline first.
            check_parity(
                model,
                inference_topdown,
                selected[0],
                args.parity_tolerance_px,
            )

            with progress_file.open("a") as f:
                for index, (row, pred) in enumerate(
                    inference_rows(
                        model,
                        inference_topdown,
                        pending,
                        args.level,
                        args.batch_size,
                        metadata["source_image_sha256_by_image_id"],
                    ),
                    start=1,
                ):
                    record = {
                        "record_id": row["record_id"],
                        "prediction_xy": pred.tolist(),
                    }

                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    completed[row["record_id"]] = record

                    if index % 100 == 0:
                        print(
                            f"Completed {len(completed)}/{len(selected)}"
                        )

    require(
        len(completed) == len(selected),
        "Inference is incomplete",
    )

    predictions = np.asarray(
        [
            completed[r["record_id"]]["prediction_xy"]
            for r in selected
        ],
        dtype=np.float32,
    )

    require(
        predictions.shape == (len(selected), 6, 2),
        "Unexpected final prediction shape",
    )

    report = score_results(
        selected, predictions, category_map
    )

    reference_result = None

    if reference_dir:
        reference_result = check_reference(
            selected,
            predictions,
            reference_dir,
            args.reference_tolerance_px,
        )

    # Frozen test baseline is a strict reproducibility gate.
    if (
        args.split == "test"
        and args.level == 0
        and args.max_instances is None
    ):
        fixed = report["fixed_index_pck"]

        require(
            fixed["total_correct"] == 4325
            and fixed["total_valid"] == 7049,
            "ViTPose E03 frozen test baseline not reproduced",
        )

    report["run_spec"] = spec
    report["reference_comparison"] = reference_result

    np.save(out / "vitpose_predictions.npy", predictions)

    np.save(
        out / "vitpose_image_ids.npy",
        np.asarray(
            [r["image_id"] for r in selected],
            dtype=np.int64,
        ),
    )
    np.save(
        out / "vitpose_annotation_ids.npy",
        np.asarray(
            [r["annotation_id"] for r in selected],
            dtype=np.int64,
        ),
    )

    (out / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    fixed = report["fixed_index_pck"]
    oi = report["constrained_order_invariant_pck"]
    edges = report["structural_edge_accuracy"]

    print("\nPASS: Occlusion inference and scoring completed")
    print(
        f"Fixed PCK: {fixed['overall']:.6f} "
        f"({fixed['total_correct']}/{fixed['total_valid']})"
    )
    print(f"Constrained OI PCK: {oi['overall']:.6f}")
    print(f"Structural edge accuracy: {edges['overall']:.6f}")

    print("\nKeypoint subgroups:")
    for name, values in report["keypoint_subgroups"].items():
        print(f"  {name}: {values}")

    print("\nSaved:", out / "metrics.json")


if __name__ == "__main__":
    main()
