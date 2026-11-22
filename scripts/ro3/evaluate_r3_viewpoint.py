#!/usr/bin/env python3
"""R3-B frozen ViTPose-B evaluation under projective scene warps."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from prepare_r3_viewpoint import (
    ROOT, CONDITIONS, BORDER_BGR, sha256,
    make_homography, transform_points,
    transform_bbox, warp_image,
)

sys.path.insert(0, str(ROOT / "scripts/evaluate"))

from common_pose_metrics import (
    calculate_all_metrics as score_original,
    make_category_map_from_coco,
)
from common_pose_metrics_ro3_constrained_symmetry_v1 import (
    calculate_all_metrics as score_constrained,
)

CONFIG = ROOT / (
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "td-hm_ViTPose-base_innocount_patched_v1.py"
)
CHECKPOINT = ROOT / (
    "experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/"
    "epoch_90.pth"
)
OUTPUT_BASE = ROOT / "experiments/RO3/R3B_Eval_v1"

CONDITION_MAP = {
    cid: (axis, edge, inset)
    for cid, axis, edge, inset in CONDITIONS
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def identity(a):
    return int(a["image_id"]), int(a["id"])


def read_predictions(folder, keys):
    points = np.load(
        folder / "vitpose_predictions.npy",
        allow_pickle=False,
    )
    image_ids = np.load(
        folder / "vitpose_image_ids.npy",
        allow_pickle=False,
    )
    annotation_ids = np.load(
        folder / "vitpose_annotation_ids.npy",
        allow_pickle=False,
    )

    require(
        points.shape == (len(keys), 6, 2),
        "Reference prediction shape differs",
    )
    require(
        len(image_ids) == len(annotation_ids) == len(keys),
        "Reference prediction identities differ",
    )
    require(np.isfinite(points).all(), "Nonfinite predictions")

    lookup = {}

    for i in range(len(keys)):
        key = int(image_ids[i]), int(annotation_ids[i])
        require(key not in lookup, "Duplicate reference identity")
        lookup[key] = points[i]

    require(
        set(lookup) == set(keys),
        "Reference prediction identities do not match GT",
    )

    return np.stack([lookup[key] for key in keys])


def load_protocol(split, condition):
    folder = ROOT / (
        f"experiments/RO3/R3_Viewpoint_v1/{split}"
    )
    manifest = folder / "viewpoint_manifest.jsonl"
    metadata_path = folder / "protocol_metadata.json"

    meta = json.loads(metadata_path.read_text())

    require(
        meta["protocol"] == "RO3_R3B_ProjectiveViewpoint_v1",
        "Incorrect viewpoint protocol",
    )
    require(meta["split"] == split, "Wrong split")

    expected_objects = 1551 if split == "valid" else 1626
    expected_points = 6720 if split == "valid" else 7049

    require(
        meta["source_images"] == 74
        and meta["gt_instances"] == expected_objects
        and meta["valid_gt_keypoints"] == expected_points,
        "Unexpected dataset totals",
    )

    require(
        sha256(ROOT / "scripts/ro3/prepare_r3_viewpoint.py")
        == meta["generator_sha256"],
        "Viewpoint generator changed",
    )

    require(
        meta["conditions"] == [
            {
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset,
            }
            for cid, axis, edge, inset in CONDITIONS
        ],
        "Perspective conditions changed",
    )

    r3a = ROOT / (
        f"experiments/RO3/R3_Orientation_v1/{split}"
    )

    require(
        sha256(r3a / "protocol_metadata.json")
        == meta["r3a_source_metadata_sha256"],
        "R3-A source metadata changed",
    )
    require(
        sha256(r3a / "orientation_manifest.jsonl")
        == meta["r3a_source_manifest_sha256"],
        "R3-A source manifest changed",
    )

    coco_path = ROOT / meta["annotation_path"]
    require(
        sha256(coco_path) == meta["annotation_sha256"],
        "Frozen COCO annotations changed",
    )

    coco = json.loads(coco_path.read_text())

    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == expected_objects,
        "Unexpected COCO contents",
    )

    selected = {}
    counts = Counter()
    ids = defaultdict(set)

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            cid = row["condition_id"]
            iid = int(row["image_id"])

            require(cid in CONDITION_MAP, "Unknown condition")
            require(
                iid not in ids[cid],
                "Duplicate image/condition",
            )

            ids[cid].add(iid)
            counts[cid] += 1

            axis, edge, inset = CONDITION_MAP[cid]

            require(
                row["protocol"] == meta["protocol"]
                and row["split"] == split
                and row["axis"] == axis
                and row["compressed_edge"] == edge
                and float(row["inset_fraction"]) == inset,
                "Manifest condition differs",
            )

            require(
                row["source_sha256"]
                == meta["source_image_hashes"][str(iid)],
                "Source hash record differs",
            )

            H, corners, coverage = make_homography(
                int(row["width"]),
                int(row["height"]),
                axis,
                edge,
                inset,
            )

            require(
                np.allclose(
                    H, row["homography"],
                    atol=1e-8, rtol=0,
                ),
                "Homography differs from generator",
            )

            require(
                np.allclose(
                    corners, row["destination_corners"],
                    atol=1e-8, rtol=0,
                ),
                "Destination corners differ",
            )

            require(
                abs(
                    coverage
                    - float(row["analytic_source_coverage"])
                ) < 1e-8,
                "Coverage ratio differs",
            )

            require(
                row["border_bgr"] == list(BORDER_BGR),
                "Border setting changed",
            )

            if cid == condition:
                selected[iid] = row

    expected_ids = {
        int(im["id"]) for im in coco["images"]
    }

    require(
        sum(counts.values()) == 666
        and set(counts) == set(CONDITION_MAP),
        "Incomplete perspective manifest",
    )

    require(
        all(
            counts[cid] == 74
            and ids[cid] == expected_ids
            for cid in CONDITION_MAP
        ),
        "Missing condition photographs",
    )

    require(
        set(selected) == expected_ids,
        "Selected condition is incomplete",
    )

    return meta, coco, selected, manifest, metadata_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["valid", "test"],
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=list(CONDITION_MAP),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    require(args.batch_size > 0, "Invalid batch size")
    require(
        args.max_instances is None
        or args.max_instances > 0,
        "Invalid maximum instance count",
    )

    output = (ROOT / args.output).resolve()
    base = OUTPUT_BASE.resolve()

    require(output != base, "Output must be a subdirectory")
    output.relative_to(base)

    meta, coco, scenes, manifest, meta_path = (
        load_protocol(args.split, args.condition)
    )

    anns = sorted(coco["annotations"], key=identity)

    require(
        len({identity(a) for a in anns}) == len(anns),
        "Duplicate GT identity",
    )

    if args.max_instances is not None:
        anns = anns[:args.max_instances]

    require(len(anns) > 0, "Empty annotation selection")

    clean_dir = ROOT / (
        "experiments/RO3/R1_Eval_v1/"
        f"{args.split}_uniform_p00_r0"
    )

    clean_report = json.loads(
        (clean_dir / "metrics.json").read_text()
    )
    clean_spec = clean_report["run_spec"]

    for key, path in (
        ("checkpoint_sha256", CHECKPOINT),
        ("config_sha256", CONFIG),
    ):
        require(
            path.is_file()
            and sha256(path) == clean_spec[key],
            f"Frozen E03 {key} differs",
        )

    spec = {
        "protocol": "RO3_R3B_ProjectiveViewpoint_Eval_v1",
        "split": args.split,
        "condition_id": args.condition,
        "manifest_sha256": sha256(manifest),
        "protocol_metadata_sha256": sha256(meta_path),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "selected_instances": len(anns),
        "max_instances": args.max_instances,
        "batch_size": args.batch_size,
        "device": args.device,
        "primary_pck_normalizer":
            "original_GT_bbox_max_dimension",
        "secondary_pck_normalizer":
            "transformed_GT_bbox_max_dimension",
    }

    progress = output / "predictions_progress.jsonl"

    if output.exists():
        require(
            args.resume,
            "Output exists; resume explicitly",
        )
        require(
            not (output / "metrics.json").exists(),
            "Completed evaluation cannot be overwritten",
        )
        require(
            json.loads(
                (output / "run_spec.json").read_text()
            ) == spec,
            "Resume parameters differ",
        )
    else:
        require(
            not args.resume,
            "Cannot resume a nonexistent evaluation",
        )
        output.mkdir(parents=True)
        (output / "run_spec.json").write_text(
            json.dumps(spec, indent=2)
        )

    completed = {}

    if progress.is_file():
        with progress.open() as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                rid = record["record_id"]

                require(
                    rid not in completed,
                    "Duplicate progress identity",
                )

                points = np.asarray(
                    record["prediction_xy"],
                    dtype=np.float32,
                )

                require(
                    points.shape == (6, 2)
                    and np.isfinite(points).all(),
                    "Invalid saved prediction",
                )

                completed[rid] = record

    expected_ids = {
        f"{a['image_id']}:{a['id']}"
        for a in anns
    }

    require(
        set(completed).issubset(expected_ids),
        "Progress contains unexpected GT objects",
    )

    by_image = defaultdict(list)

    for a in anns:
        by_image[int(a["image_id"])].append(a)

    print("\nRO3 R3-B — FROZEN ViTPose-B E03")
    print("=" * 60)
    print("Split:", args.split)
    print("Condition:", args.condition)
    print("Selected objects:", len(anns))
    print("Pending:", len(anns) - len(completed))

    if len(completed) < len(anns):
        import torch

        from mmpose.apis import (
            init_model, inference_topdown,
        )
        from mmpose.utils import register_all_modules

        register_all_modules()

        model = init_model(
            str(CONFIG),
            str(CHECKPOINT),
            device=args.device,
        )
        model.eval()

        with torch.no_grad():
            with progress.open("a") as f:
                for iid in sorted(scenes):
                    pending = [
                        a
                        for a in by_image.get(iid, [])
                        if f"{a['image_id']}:{a['id']}"
                        not in completed
                    ]

                    if not pending:
                        continue

                    record = scenes[iid]
                    source_path = ROOT / record["source_image"]

                    require(
                        sha256(source_path)
                        == record["source_sha256"],
                        f"Source image changed: {iid}",
                    )

                    source = cv2.imread(
                        str(source_path),
                        cv2.IMREAD_COLOR,
                    )

                    require(
                        source is not None,
                        f"Unreadable source image: {iid}",
                    )

                    require(
                        source.shape[:2] == (
                            int(record["height"]),
                            int(record["width"]),
                        ),
                        "Image dimensions changed",
                    )

                    H = np.asarray(
                        record["homography"],
                        dtype=np.float64,
                    )

                    transformed = warp_image(
                        source, H, args.condition
                    )

                    if args.condition == "vp_neutral":
                        require(
                            np.array_equal(
                                transformed, source
                            ),
                            "Neutral image differs",
                        )

                    for start in range(
                        0, len(pending), args.batch_size
                    ):
                        batch = pending[
                            start:start + args.batch_size
                        ]

                        boxes = np.asarray(
                            [
                                transform_bbox(a["bbox"], H)
                                for a in batch
                            ],
                            dtype=np.float32,
                        )

                        xyxy = boxes.copy()
                        xyxy[:, 2:] += xyxy[:, :2]

                        samples = inference_topdown(
                            model,
                            transformed,
                            bboxes=xyxy,
                            bbox_format="xyxy",
                        )

                        require(
                            len(samples) == len(batch),
                            "Wrong pose output count",
                        )

                        for a, sample in zip(batch, samples):
                            points = np.asarray(
                                sample.pred_instances.keypoints,
                                dtype=np.float32,
                            ).reshape(-1, 6, 2)

                            require(
                                points.shape == (1, 6, 2)
                                and np.isfinite(points).all(),
                                "Invalid pose output",
                            )

                            rid = f"{a['image_id']}:{a['id']}"

                            require(
                                rid not in completed,
                                "Duplicate prediction",
                            )

                            item = {
                                "record_id": rid,
                                "prediction_xy": points[0].tolist(),
                            }

                            f.write(json.dumps(item) + "\n")
                            f.flush()
                            completed[rid] = item

                    print(
                        f"Image {iid}: "
                        f"{len(completed)}/{len(anns)} complete"
                    )

    require(
        set(completed) == expected_ids,
        "Incomplete prediction set",
    )

    predictions = np.asarray(
        [
            completed[
                f"{a['image_id']}:{a['id']}"
            ]["prediction_xy"]
            for a in anns
        ],
        dtype=np.float32,
    )

    original_gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32,
    ).reshape(-1, 6, 3)

    original_boxes = np.asarray(
        [a["bbox"] for a in anns],
        dtype=np.float32,
    )

    transformed_gt = []
    transformed_boxes = []

    for a, kp in zip(anns, original_gt):
        H = np.asarray(
            scenes[int(a["image_id"])]["homography"],
            dtype=np.float64,
        )

        transformed_gt.append(
            transform_points(kp[:, :2], H)
        )
        transformed_boxes.append(
            transform_bbox(a["bbox"], H)
        )

    transformed_gt = np.asarray(
        transformed_gt,
        dtype=np.float32,
    )
    transformed_boxes = np.asarray(
        transformed_boxes,
        dtype=np.float32,
    )

    valid = original_gt[:, :, 2] > 0

    original_scale = np.maximum(
        original_boxes[:, 2],
        original_boxes[:, 3],
    )

    transformed_scale = np.maximum(
        transformed_boxes[:, 2],
        transformed_boxes[:, 3],
    )

    require(
        np.all(original_scale > 0)
        and np.all(transformed_scale > 0),
        "Invalid normalization dimensions",
    )

    error = np.linalg.norm(
        predictions - transformed_gt,
        axis=2,
    )

    primary = (
        (error / original_scale[:, None] <= 0.05)
        & valid
    )
    secondary = (
        (error / transformed_scale[:, None] <= 0.05)
        & valid
    )

    n_valid = int(valid.sum())
    n_primary = int(primary.sum())
    n_secondary = int(secondary.sum())

    category_ids = np.asarray(
        [a["category_id"] for a in anns],
        dtype=np.int64,
    )

    metric_args = {
        "predictions": predictions,
        "gt": transformed_gt,
        "gt_vis": original_gt[:, :, 2],
        "bboxes": transformed_boxes,
        "category_ids": category_ids,
        "category_map":
            make_category_map_from_coco(coco),
        "threshold": 0.05,
    }

    original_metrics = score_original(**metric_args)
    revised_metrics = score_constrained(**metric_args)

    require(
        revised_metrics["standard_pck"]["total_correct"]
        == n_secondary
        and revised_metrics["standard_pck"]["total_valid"]
        == n_valid,
        "Shared scorer differs from secondary PCK",
    )

    for name in (
        "standard_pck",
        "profile_pck",
        "semantic_roles",
        "structural_edges",
    ):
        require(
            original_metrics[name] == revised_metrics[name],
            f"Metric integrity mismatch: {name}",
        )

    reference = None

    if (
        args.condition == "vp_neutral"
        and args.max_instances is None
    ):
        clean = read_predictions(
            clean_dir, [identity(a) for a in anns]
        )

        difference = float(
            np.max(np.abs(predictions - clean))
        )

        require(
            difference <= 0.25,
            f"Clean E03 parity failed: {difference} px",
        )

        expected = (
            (5196, 6720)
            if args.split == "valid"
            else (4325, 7049)
        )

        require(
            (n_primary, n_valid) == expected
            and n_secondary == n_primary,
            "Neutral PCK differs from frozen E03",
        )

        reference = {
            "directory":
                clean_dir.relative_to(ROOT).as_posix(),
            "maximum_coordinate_difference_px": difference,
            "tolerance_px": 0.25,
        }

    if args.max_instances is None:
        expected = (
            (1551, 6720)
            if args.split == "valid"
            else (1626, 7049)
        )

        require(
            (len(anns), n_valid) == expected,
            "Incomplete full-split evaluation",
        )

    category_names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    profiles = {}

    for cid, name in sorted(category_names.items()):
        mask = (
            valid
            & (category_ids == cid)[:, None]
        )

        count = int(mask.sum())
        correct = int((primary & mask).sum())

        profiles[name] = {
            "valid_keypoints": count,
            "correct": correct,
            "pck": correct / count if count else None,
        }

    report = {
        "run_spec": spec,
        "instances": len(anns),
        "valid_gt_keypoints": n_valid,
        "primary_original_box_pck": {
            "total_correct": n_primary,
            "total_valid": n_valid,
            "overall": n_primary / n_valid,
        },
        "secondary_transformed_box_pck": {
            "total_correct": n_secondary,
            "total_valid": n_valid,
            "overall": n_secondary / n_valid,
        },
        "mean_transformed_bbox_scale_ratio": float(
            np.mean(transformed_scale / original_scale)
        ),
        "profile_primary_pck": profiles,
        "secondary_constrained_oi_pck":
            revised_metrics["order_invariant_pck"],
        "secondary_original_oi_pck":
            original_metrics["order_invariant_pck"],
        "secondary_structural_edges":
            revised_metrics["structural_edges"],
        "clean_reference_check": reference,
        "limitations": [
            "Synthetic 2D projective warp, not physical 3D viewpoint",
            "Newly visible surfaces are not synthesized",
            "Image interpolation, scale, borders and boxes change",
            "GT-box-conditioned pose, not independent detection",
        ],
    }

    np.save(
        output / "vitpose_predictions.npy",
        predictions,
    )
    np.save(
        output / "vitpose_image_ids.npy",
        np.asarray(
            [a["image_id"] for a in anns],
            dtype=np.int64,
        ),
    )
    np.save(
        output / "vitpose_annotation_ids.npy",
        np.asarray(
            [a["id"] for a in anns],
            dtype=np.int64,
        ),
    )

    (output / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nPASS: R3-B evaluation completed")
    print(
        f"Primary original-box PCK: "
        f"{n_primary}/{n_valid} = "
        f"{n_primary/n_valid:.6f}"
    )
    print(
        f"Secondary transformed-box PCK: "
        f"{n_secondary}/{n_valid} = "
        f"{n_secondary/n_valid:.6f}"
    )
    print(
        "Mean transformed-box scale ratio:",
        report["mean_transformed_bbox_scale_ratio"],
    )
    if reference:
        print("Clean E03 parity:", reference)
    print("Output:", output)


if __name__ == "__main__":
    main()
