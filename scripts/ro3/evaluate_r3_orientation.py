#!/usr/bin/env python3
"""R3-A: Frozen ViTPose-B evaluation under in-plane rotation."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from prepare_r3_orientation import (
    ROOT, ANGLES, BORDER, sha256,
    geometry, condition_id,
    transform_points, transform_bbox,
    rotate_image
)

sys.path.insert(0, str(ROOT / "scripts/evaluate"))

from common_pose_metrics import (
    calculate_all_metrics as score_original,
    make_category_map_from_coco
)
from common_pose_metrics_ro3_constrained_symmetry_v1 import (
    calculate_all_metrics as score_constrained
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


def identity(a):
    return int(a["image_id"]), int(a["id"])


def load_protocol(manifest, split, angle):
    meta_path = manifest.parent / "protocol_metadata.json"
    meta = json.loads(meta_path.read_text())

    require(
        meta["protocol"] == "RO3_R3A_InPlaneRotation_v1",
        "Incorrect R3 protocol"
    )
    require(meta["split"] == split, "Wrong split")
    require(
        meta["angles_degrees"] == list(ANGLES),
        "Rotation conditions changed"
    )
    require(
        meta["border_bgr"] == list(BORDER),
        "Border setting changed"
    )
    require(
        sha256(ROOT / "scripts/ro3/prepare_r3_orientation.py")
        == meta["generator_sha256"],
        "Generator differs from saved manifest"
    )

    ann_path = ROOT / meta["annotation_path"]
    require(
        sha256(ann_path) == meta["annotation_sha256"],
        "COCO annotations changed"
    )

    coco = json.loads(ann_path.read_text())
    required_count = 1551 if split == "valid" else 1626

    require(
        len(coco["annotations"]) == required_count,
        "Wrong GT instance count"
    )

    chosen = {}
    total = 0

    with manifest.open() as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            total += 1

            require(
                row["split"] == split,
                "Manifest contains wrong split"
            )
            require(
                row["condition_id"]
                == condition_id(int(row["angle_degrees"])),
                "Incorrect row condition"
            )

            if int(row["angle_degrees"]) != angle:
                continue

            iid = int(row["image_id"])
            require(iid not in chosen, "Duplicate source scene")
            require(
                row["source_sha256"]
                == meta["source_image_hashes"][str(iid)],
                "Source hash record differs"
            )

            m, width, height = geometry(
                int(row["width"]),
                int(row["height"]),
                angle
            )

            require(
                np.allclose(
                    np.asarray(row["matrix"]), m,
                    atol=1e-8, rtol=0
                ),
                "Affine matrix differs from protocol"
            )
            require(
                (int(row["output_width"]),
                 int(row["output_height"]))
                == (width, height),
                "Canvas dimensions changed"
            )

            chosen[iid] = row

    require(
        total == 74 * len(ANGLES),
        "Incomplete orientation manifest"
    )
    require(
        len(chosen) == 74,
        "Incomplete rotation condition"
    )
    require(
        set(chosen) ==
        {int(im["id"]) for im in coco["images"]},
        "COCO and orientation image IDs differ"
    )

    return meta, coco, chosen


def read_saved_predictions(folder, keys):
    p = np.load(
        folder / "vitpose_predictions.npy",
        allow_pickle=False
    )
    im = np.load(
        folder / "vitpose_image_ids.npy",
        allow_pickle=False
    )
    aid = np.load(
        folder / "vitpose_annotation_ids.npy",
        allow_pickle=False
    )

    require(
        len(p) == len(im) == len(aid),
        "Reference array length mismatch"
    )

    lookup = {}
    for i in range(len(p)):
        key = (int(im[i]), int(aid[i]))
        require(key not in lookup, "Duplicate reference ID")
        lookup[key] = p[i]

    require(set(lookup) == set(keys), "Reference IDs differ")
    return np.stack([lookup[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest", required=True
    )
    ap.add_argument(
        "--split", choices=["valid", "test"],
        required=True
    )
    ap.add_argument(
        "--angle", type=int, choices=ANGLES,
        required=True
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-instances", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    require(args.batch_size > 0, "Invalid batch size")
    require(
        args.max_instances is None
        or args.max_instances > 0,
        "Invalid max-instances"
    )

    manifest = (ROOT / args.manifest).resolve()
    output = (ROOT / args.output).resolve()

    meta, coco, scene_records = load_protocol(
        manifest, args.split, args.angle
    )

    anns = sorted(
        coco["annotations"], key=identity
    )
    require(
        len({identity(a) for a in anns}) == len(anns),
        "Duplicate composite annotation ID"
    )

    if args.max_instances is not None:
        anns = anns[:args.max_instances]

    require(bool(anns), "No selected GT objects")

    r1 = ROOT / (
        f"experiments/RO3/R1_Eval_v1/"
        f"{args.split}_uniform_p00_r0"
    )
    original_report = json.loads(
        (r1 / "metrics.json").read_text()
    )

    for name, path in (
        ("checkpoint_sha256", CHECKPOINT),
        ("config_sha256", CONFIG)
    ):
        require(
            path.is_file()
            and sha256(path)
            == original_report["run_spec"][name],
            f"Frozen E03 {name} differs"
        )

    spec = {
        "protocol": "RO3_R3A_InPlaneRotation_Eval_v1",
        "split": args.split,
        "angle_degrees": args.angle,
        "condition_id": condition_id(args.angle),
        "manifest_sha256": sha256(manifest),
        "protocol_metadata_sha256":
            sha256(manifest.parent / "protocol_metadata.json"),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "batch_size": args.batch_size,
        "max_instances": args.max_instances,
        "selected_instances": len(anns),
        "primary_pck_normalizer": "original_GT_bbox_max_dimension",
        "secondary_pck_normalizer":
            "rotated_axis_aligned_GT_bbox_max_dimension"
    }

    progress_path = output / "predictions_progress.jsonl"

    if output.exists():
        require(args.resume, "Output exists; use --resume")
        require(
            not (output / "metrics.json").exists(),
            "Completed R3 evaluation cannot be overwritten"
        )
        require(
            json.loads((output / "run_spec.json").read_text())
            == spec,
            "Resume specification differs"
        )
    else:
        require(
            not args.resume,
            "Cannot resume a nonexistent output directory"
        )
        output.mkdir(parents=True)
        (output / "run_spec.json").write_text(
            json.dumps(spec, indent=2)
        )

    completed = {}

    if progress_path.is_file():
        with progress_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                rid = record["record_id"]
                require(rid not in completed, "Duplicate progress ID")
                completed[rid] = record

    expected_ids = {
        f"{a['image_id']}:{a['id']}"
        for a in anns
    }

    require(
        set(completed).issubset(expected_ids),
        "Progress contains unrelated annotations"
    )

    by_image = defaultdict(list)
    for a in anns:
        by_image[int(a["image_id"])].append(a)

    print("\nRO3 R3-A — FROZEN ViTPose-B E03")
    print("=" * 60)
    print("Split:", args.split)
    print("Rotation:", args.angle, "degrees")
    print("Selected objects:", len(anns))
    print("Pending:", len(anns) - len(completed))

    if len(completed) < len(anns):
        import torch
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules

        register_all_modules()
        model = init_model(
            str(CONFIG), str(CHECKPOINT),
            device=args.device
        )
        model.eval()

        with torch.no_grad():
            with progress_path.open("a") as progress:
                for iid in sorted(scene_records):
                    selected = [
                        a for a in by_image.get(iid, [])
                        if f"{a['image_id']}:{a['id']}"
                        not in completed
                    ]
                    if not selected:
                        continue

                    rec = scene_records[iid]
                    source_path = ROOT / rec["source_image"]

                    require(
                        sha256(source_path) == rec["source_sha256"],
                        f"Source image changed: {source_path}"
                    )

                    source = cv2.imread(
                        str(source_path),
                        cv2.IMREAD_COLOR
                    )
                    require(
                        source is not None,
                        "Source image unreadable"
                    )
                    require(
                        source.shape[:2] ==
                        (rec["height"], rec["width"]),
                        "Source image geometry changed"
                    )

                    matrix = np.asarray(
                        rec["matrix"], dtype=np.float64
                    )
                    transformed_image = rotate_image(
                        source,
                        matrix,
                        int(rec["output_width"]),
                        int(rec["output_height"]),
                        args.angle
                    )

                    if args.angle == 0:
                        require(
                            np.array_equal(
                                transformed_image, source
                            ),
                            "Neutral image changed"
                        )

                    for start in range(
                        0, len(selected), args.batch_size
                    ):
                        chunk = selected[
                            start:start + args.batch_size
                        ]

                        boxes = np.asarray(
                            [
                                transform_bbox(
                                    a["bbox"], matrix
                                )
                                for a in chunk
                            ],
                            dtype=np.float32
                        )

                        xyxy = boxes.copy()
                        xyxy[:, 2] += xyxy[:, 0]
                        xyxy[:, 3] += xyxy[:, 1]

                        samples = inference_topdown(
                            model,
                            transformed_image,
                            bboxes=xyxy,
                            bbox_format="xyxy"
                        )

                        require(
                            len(samples) == len(chunk),
                            "Wrong number of pose outputs"
                        )

                        for a, sample in zip(chunk, samples):
                            points = np.asarray(
                                sample.pred_instances.keypoints,
                                dtype=np.float32
                            ).reshape(-1, 6, 2)

                            require(
                                points.shape == (1, 6, 2)
                                and np.isfinite(points).all(),
                                "Invalid predicted keypoints"
                            )

                            rid = f"{a['image_id']}:{a['id']}"
                            require(
                                rid not in completed,
                                "Duplicate result"
                            )

                            record = {
                                "record_id": rid,
                                "prediction_xy": points[0].tolist()
                            }

                            progress.write(
                                json.dumps(record) + "\n"
                            )
                            progress.flush()
                            completed[rid] = record

                    print(
                        f"Image {iid}: "
                        f"{len(completed)}/{len(anns)} complete"
                    )

    require(
        set(completed) == expected_ids,
        "Incomplete result set"
    )

    predictions = np.asarray(
        [
            completed[f"{a['image_id']}:{a['id']}"][
                "prediction_xy"
            ]
            for a in anns
        ],
        dtype=np.float32
    )
    require(
        predictions.shape == (len(anns), 6, 2),
        "Incorrect prediction shape"
    )

    original_gt = np.asarray(
        [a["keypoints"] for a in anns],
        dtype=np.float32
    ).reshape(-1, 6, 3)

    original_bbox = np.asarray(
        [a["bbox"] for a in anns],
        dtype=np.float32
    )

    gt_transformed = []
    bbox_transformed = []

    for a, kp in zip(anns, original_gt):
        rec = scene_records[int(a["image_id"])]
        matrix = np.asarray(
            rec["matrix"], dtype=np.float64
        )

        gt_transformed.append(
            transform_points(kp[:, :2], matrix)
        )
        bbox_transformed.append(
            transform_bbox(a["bbox"], matrix)
        )

    gt_transformed = np.asarray(
        gt_transformed, dtype=np.float32
    )
    bbox_transformed = np.asarray(
        bbox_transformed, dtype=np.float32
    )

    valid = original_gt[:, :, 2] > 0

    original_scale = np.maximum(
        original_bbox[:, 2],
        original_bbox[:, 3]
    )

    rotated_scale = np.maximum(
        bbox_transformed[:, 2],
        bbox_transformed[:, 3]
    )

    require(
        np.all(original_scale > 0)
        and np.all(rotated_scale > 0),
        "Invalid normalization scale"
    )

    error_px = np.linalg.norm(
        predictions - gt_transformed,
        axis=2
    )

    correct_primary = (
        (error_px / original_scale[:, None] <= 0.05)
        & valid
    )

    correct_rotated = (
        (error_px / rotated_scale[:, None] <= 0.05)
        & valid
    )

    total_valid = int(valid.sum())
    primary_n = int(correct_primary.sum())
    rotated_n = int(correct_rotated.sum())

    categories = np.asarray(
        [a["category_id"] for a in anns],
        dtype=np.int64
    )

    kwargs = {
        "predictions": predictions,
        "gt": gt_transformed,
        "gt_vis": original_gt[:, :, 2],
        "bboxes": bbox_transformed,
        "category_ids": categories,
        "category_map": make_category_map_from_coco(coco),
        "threshold": 0.05
    }

    original_metrics = score_original(**kwargs)
    revised_metrics = score_constrained(**kwargs)

    require(
        revised_metrics["standard_pck"]["total_correct"]
        == rotated_n,
        "Secondary rotated-box PCK differs from scorer"
    )
    require(
        revised_metrics["standard_pck"]["total_valid"]
        == total_valid,
        "Metric denominator mismatch"
    )

    for name in (
        "standard_pck", "profile_pck",
        "semantic_roles", "structural_edges"
    ):
        require(
            original_metrics[name] == revised_metrics[name],
            f"Unexpected metric change: {name}"
        )

    reference = None

    if args.angle == 0 and args.max_instances is None:
        keys = [identity(a) for a in anns]

        old = read_saved_predictions(r1, keys)

        difference = float(
            np.max(np.abs(predictions - old))
        )

        require(
            difference <= 0.25,
            f"Clean E03 coordinate parity failed: {difference}"
        )

        target = (
            (5196, 6720)
            if args.split == "valid"
            else (4325, 7049)
        )

        require(
            (primary_n, total_valid) == target,
            "0-degree primary PCK does not match E03"
        )
        require(
            rotated_n == primary_n,
            "0-degree normalization mismatch"
        )

        reference = {
            "directory": str(r1.relative_to(ROOT)),
            "maximum_coordinate_difference_px": difference,
            "tolerance_px": 0.25
        }

    if args.max_instances is None:
        target_n = (
            1551 if args.split == "valid" else 1626
        )
        target_k = (
            6720 if args.split == "valid" else 7049
        )
        require(
            len(anns) == target_n
            and total_valid == target_k,
            "Full-split evaluation incomplete"
        )

    names = {
        int(c["id"]): c["name"]
        for c in coco["categories"]
    }

    profile_rows = {}
    for cid, name in sorted(names.items()):
        mask = valid & (categories == cid)[:, None]
        count = int(mask.sum())
        profile_rows[name] = {
            "valid_keypoints": count,
            "correct": int((correct_primary & mask).sum()),
            "pck": (
                float((correct_primary & mask).sum() / count)
                if count else None
            )
        }

    report = {
        "run_spec": spec,
        "instances": len(anns),
        "valid_gt_keypoints": total_valid,
        "primary_original_box_pck": {
            "total_correct": primary_n,
            "total_valid": total_valid,
            "overall": primary_n / total_valid
        },
        "secondary_rotated_box_pck": {
            "total_correct": rotated_n,
            "total_valid": total_valid,
            "overall": rotated_n / total_valid
        },
        "mean_rotated_bbox_scale_ratio": float(
            np.mean(rotated_scale / original_scale)
        ),
        "profile_primary_pck": profile_rows,
        "secondary_constrained_oi_pck":
            revised_metrics["order_invariant_pck"],
        "secondary_structural_edges":
            revised_metrics["structural_edges"],
        "secondary_original_oi_pck":
            original_metrics["order_invariant_pck"],
        "clean_reference_check": reference,
        "limitations": [
            "In-plane rotation only, not true 3D viewpoint",
            "Expanded canvas and interpolation are synthetic",
            "GT-box-conditioned pose, not object detection",
            "Secondary metrics use rotated bbox normalization"
        ]
    }

    np.save(
        output / "vitpose_predictions.npy",
        predictions
    )
    np.save(
        output / "vitpose_image_ids.npy",
        np.asarray([a["image_id"] for a in anns])
    )
    np.save(
        output / "vitpose_annotation_ids.npy",
        np.asarray([a["id"] for a in anns])
    )

    (output / "metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nPASS: R3-A evaluation completed")
    print(
        f"Primary original-box PCK: "
        f"{primary_n}/{total_valid} = "
        f"{primary_n/total_valid:.6f}"
    )
    print(
        f"Secondary rotated-box PCK: "
        f"{rotated_n}/{total_valid} = "
        f"{rotated_n/total_valid:.6f}"
    )
    print(
        "Mean rotated-bbox scale ratio:",
        report["mean_rotated_bbox_scale_ratio"]
    )
    if reference:
        print("Clean E03 parity:", reference)
    print("Output:", output)


if __name__ == "__main__":
    main()
