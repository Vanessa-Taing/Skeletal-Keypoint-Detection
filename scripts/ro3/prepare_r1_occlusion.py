#!/usr/bin/env python3
"""
RO3 R1: Reproducible structural-steel occlusion benchmark.

Creates an instance-level, paired synthetic occlusion
manifest for pose-only evaluation.

Properties:
- Original images/COCO annotations are never edited.
- The same object is evaluated at increasing severities.
- Deterministic shape and placement per object/replicate.
- Severity means pixel coverage of the clipped GT bbox.
- Original GT keypoint coordinates/visibility stay intact.
- Newly visually hidden keypoints are logged separately.
- Natural bbox-overlap statistics are diagnostic proxies.
- Images may be rendered lazily with render_record().
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/InnoCount_COCO_Patched_UniqueIDs_v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_rng(*parts):
    text = "|".join(map(str, parts))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:8], "big")
    )


def find_annotations(split):
    names = (
        ("valid.json", "val.json", "validation.json")
        if split == "valid"
        else (f"{split}.json",)
    )
    for name in names:
        path = DATA / "annotations" / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No annotation JSON for {split} in "
        f"{DATA / 'annotations'}"
    )


def find_image(split, image):
    name = Path(image["file_name"])
    candidates = [
        DATA / "images" / split / name,
        DATA / "images" / split / name.name,
        DATA / "images" / (
            "val" if split == "valid" else split
        ) / name.name,
        DATA / name,
        DATA / name.name,
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        f"Image not found for {image['file_name']}"
    )


def clip_box_xywh(box, width, height):
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox: {box}")

    x0 = max(0, min(width, math.floor(x)))
    y0 = max(0, min(height, math.floor(y)))
    x1 = max(0, min(width, math.ceil(x + w)))
    y1 = max(0, min(height, math.ceil(y + h)))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"GT box has no in-image area: {box}"
        )
    return [x0, y0, x1, y1]


def area(box):
    return max(0, box[2] - box[0]) * max(
        0, box[3] - box[1]
    )


def intersection(a, b):
    return [
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        min(a[3], b[3]),
    ]


def point_inside(point, rect):
    x, y = float(point[0]), float(point[1])
    return (
        rect[0] <= x < rect[2]
        and rect[1] <= y < rect[3]
    )


def plan_base(split, ann, roi, mode, replicate, seed):
    """Select a fixed orientation and anchor across severities."""
    rng = stable_rng(
        seed, split, ann["image_id"], ann["id"],
        mode, replicate,
    )

    orientation = (
        "vertical" if int(rng.integers(2)) == 0
        else "horizontal"
    )

    points = np.asarray(
        ann["keypoints"], dtype=float
    ).reshape(-1, 3)

    mode_effective = mode
    anchor_index = None

    if mode == "keypoint":
        candidates = [
            j for j, (x, y, v) in enumerate(points)
            if v == 2 and point_inside((x, y), roi)
        ]
        if candidates:
            anchor_index = int(rng.choice(candidates))
            ax, ay = map(
                float, points[anchor_index, :2]
            )
        else:
            # Keep all instances in the benchmark,
            # but identify those without a visible
            # target point.
            mode_effective = "uniform_fallback"

    if mode_effective != "keypoint":
        ax = float(rng.uniform(roi[0], roi[2]))
        ay = float(rng.uniform(roi[1], roi[3]))

    # Steel-like solid occluder, in OpenCV BGR.
    gray = int(rng.integers(70, 131))
    color = [
        min(255, gray + 16),
        min(255, gray + 8),
        gray,
    ]

    return {
        "orientation": orientation,
        "anchor_xy": [ax, ay],
        "anchor_keypoint_index": anchor_index,
        "mode_effective": mode_effective,
        "color_bgr": color,
    }


def make_occluder(roi, orientation, anchor, pct):
    """Pixel-aligned strip, clipped entirely to GT-box ROI."""
    if pct == 0:
        return None

    x0, y0, x1, y1 = roi
    rw, rh = x1 - x0, y1 - y0

    if orientation == "vertical":
        thickness = min(
            rw, max(1, round(rw * pct / 100))
        )
        start = round(anchor[0] - thickness / 2)
        start = max(x0, min(x1 - thickness, start))
        return [
            int(start), y0,
            int(start + thickness), y1,
        ]

    thickness = min(
        rh, max(1, round(rh * pct / 100))
    )
    start = round(anchor[1] - thickness / 2)
    start = max(y0, min(y1 - thickness, start))

    return [
        x0, int(start),
        x1, int(start + thickness),
    ]


def render_record(record, image=None):
    """Render an evaluation image without drawing GT keypoints.

    This function may be imported by a future ViTPose
    inference runner. Returned coordinates remain in
    the original image frame.
    """
    if image is None:
        source = ROOT / record["source_image"]
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)

    output = image.copy()

    rect = record["occluder_xyxy"]
    if rect is not None:
        x0, y0, x1, y1 = rect
        output[y0:y1, x0:x1] = (
            record["occluder_color_bgr"]
        )

    return output


def natural_overlap(target, neighbors):
    """Maximum other-box intersection / target box area.

    This is a spatial overlap proxy, NOT observed occlusion.
    """
    denom = area(target)
    if denom <= 0:
        return 0.0

    return max(
        (
            area(intersection(target, other)) / denom
            for other in neighbors
        ),
        default=0.0,
    )


def build_record(
    split, ann, image, image_path, roi,
    natural_fraction, scene_count,
    mode, replicate, pct, seed,
):
    base = plan_base(
        split, ann, roi, mode, replicate, seed
    )

    rect = make_occluder(
        roi,
        base["orientation"],
        base["anchor_xy"],
        pct,
    )

    kpts = np.asarray(
        ann["keypoints"], dtype=float
    ).reshape(-1, 3)

    covered = []
    newly_hidden = []
    originally_occluded = []

    for j, (x, y, v) in enumerate(kpts):
        if v == 1:
            originally_occluded.append(j + 1)

        if v > 0 and rect is not None:
            if point_inside((x, y), rect):
                covered.append(j + 1)
                if v == 2:
                    newly_hidden.append(j + 1)

    actual = (
        100.0 * area(rect) / area(roi)
        if rect is not None else 0.0
    )

    return {
        "record_id": (
            f"{split}_im{int(ann['image_id'])}"
            f"_ann{int(ann['id'])}"
            f"_{mode}_r{replicate}_p{pct:02d}"
        ),
        "split": split,
        "scope": "instance",
        "transform_type": "opaque_stripe",
        "image_id": int(ann["image_id"]),
        "annotation_id": int(ann["id"]),
        "category_id": int(ann["category_id"]),
        "source_image": str(image_path.relative_to(ROOT)),
        "image_width": int(image["width"]),
        "image_height": int(image["height"]),
        "gt_bbox_xywh": list(map(float, ann["bbox"])),
        "clipped_bbox_xyxy": roi,
        "gt_keypoints": ann["keypoints"],
        "scene_object_count": scene_count,
        "natural_max_bbox_overlap_fraction":
            natural_fraction,
        "originally_occluded_gt_indices":
            originally_occluded,
        "placement_mode_requested": mode,
        "placement_mode_effective":
            base["mode_effective"],
        "anchor_keypoint_index": (
            base["anchor_keypoint_index"] + 1
            if base["anchor_keypoint_index"] is not None
            else None
        ),
        "replicate": replicate,
        "requested_bbox_occlusion_pct": pct,
        "actual_bbox_occlusion_pct": actual,
        "orientation": base["orientation"],
        "anchor_xy": base["anchor_xy"],
        "occluder_xyxy": rect,
        "occluder_color_bgr": base["color_bgr"],
        "covered_labeled_keypoint_indices": covered,
        "newly_hidden_keypoint_indices": newly_hidden,
    }


def write_preview(row, output):
    source = ROOT / row["source_image"]
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(source)

    aug = render_record(row, image)

    H, W = image.shape[:2]
    bx0, by0, bx1, by1 = row["clipped_bbox_xyxy"]

    pad = max(
        20, round(0.3 * max(bx1 - bx0, by1 - by0))
    )

    x0 = max(0, bx0 - pad)
    y0 = max(0, by0 - pad)
    x1 = min(W, bx1 + pad)
    y1 = min(H, by1 + pad)

    left = image[y0:y1, x0:x1].copy()
    right = aug[y0:y1, x0:x1].copy()

    cv2.rectangle(
        left,
        (bx0-x0, by0-y0),
        (bx1-x0, by1-y0),
        (0, 220, 0), 2,
    )

    points = np.asarray(
        row["gt_keypoints"], dtype=float
    ).reshape(-1, 3)

    for j, (x, y, v) in enumerate(points):
        if v <= 0:
            continue

        px = round(x - x0)
        py = round(y - y0)

        if not (0 <= px < left.shape[1] and
                0 <= py < left.shape[0]):
            continue

        cv2.circle(
            left, (px, py), 3,
            (0, 220, 0), -1,
        )
        cv2.putText(
            left, f"G{j+1}",
            (px+5, py-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 220, 0), 1,
        )

    # These labels are for QA only and are NOT
    # part of the model evaluation image.
    def label_panel(panel, text):
        panel = cv2.copyMakeBorder(
            panel, 34, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )
        cv2.putText(
            panel, text, (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255,255,255), 1,
            cv2.LINE_AA,
        )
        return panel

    left = label_panel(left, "GT / original")
    right = label_panel(
        right,
        "Occluded input (no GT overlay)",
    )

    combined = cv2.hconcat([left, right])
    assert cv2.imwrite(str(output), combined)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split", choices=("valid", "test"),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("uniform", "keypoint"),
        default="uniform",
    )
    parser.add_argument(
        "--levels", nargs="+", type=int,
        default=[0, 10, 20, 30, 40, 50],
    )
    parser.add_argument(
        "--replicates", type=int, default=1,
    )
    parser.add_argument(
        "--seed", type=int, default=20261119,
    )
    parser.add_argument(
        "--qa-per-level", type=int, default=3,
    )
    parser.add_argument(
        "--output", type=Path, default=None,
    )

    args = parser.parse_args()

    assert args.replicates >= 1
    assert args.qa_per_level >= 0
    assert args.levels == sorted(set(args.levels))
    assert args.levels[0] == 0
    assert all(0 <= p < 100 for p in args.levels)

    ann_path = find_annotations(args.split)
    coco = json.loads(ann_path.read_text())

    images = {
        int(im["id"]): im
        for im in coco["images"]
    }

    annotations = sorted(
        coco["annotations"],
        key=lambda a: (
            int(a["image_id"]), int(a["id"])
        ),
    )

    assert len(annotations) == len({
        (int(a["image_id"]), int(a["id"]))
        for a in annotations
    })

    by_image = defaultdict(list)

    for ann in annotations:
        by_image[int(ann["image_id"])].append(ann)

    source_paths = {}
    source_hashes = {}

    for image_id, image in images.items():
        path = find_image(args.split, image)

        # Check source image dimensions once.
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ValueError(f"Cannot read image: {path}")

        assert pixels.shape[:2] == (
            int(image["height"]),
            int(image["width"]),
        ), f"Image dimensions differ from COCO: {path}"

        source_paths[image_id] = path
        source_hashes[str(image_id)] = sha256_file(path)

    if args.output is None:
        out = ROOT / (
            "experiments/RO3/R1_Occlusion_v1/"
            f"{args.split}_{args.mode}_seed{args.seed}"
        )
    else:
        out = args.output.expanduser().resolve()

    assert not out.exists(), (
        f"Output already exists; refusing overwrite: {out}"
    )

    records = []

    for ann in annotations:
        image_id = int(ann["image_id"])
        image = images[image_id]

        W = int(image["width"])
        H = int(image["height"])

        roi = clip_box_xywh(ann["bbox"], W, H)

        neighbors = [
            clip_box_xywh(other["bbox"], W, H)
            for other in by_image[image_id]
            if other is not ann
        ]

        natural_fraction = natural_overlap(
            roi, neighbors
        )

        for pct in args.levels:
            repetitions = (
                [0] if pct == 0
                else range(args.replicates)
            )

            for replicate in repetitions:
                row = build_record(
                    args.split, ann, image,
                    source_paths[image_id],
                    roi, natural_fraction,
                    len(by_image[image_id]),
                    args.mode, replicate,
                    pct, args.seed,
                )
                records.append(row)

    # Sanity checks.
    expected = len(annotations) * (
        1 + (len(args.levels) - 1) * args.replicates
    )
    assert len(records) == expected

    assert len({
        r["record_id"] for r in records
    }) == len(records)

    for row in records:
        pct = row["requested_bbox_occlusion_pct"]

        if pct == 0:
            assert row["occluder_xyxy"] is None
            assert row["actual_bbox_occlusion_pct"] == 0
        else:
            assert row["occluder_xyxy"] is not None

        assert all(
            i in range(1, 7)
            for i in row["covered_labeled_keypoint_indices"]
        )

    out.mkdir(parents=True)
    manifest = out / "occlusion_manifest.jsonl"

    with manifest.open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    summary = []

    for pct in args.levels:
        rows = [
            r for r in records
            if r["requested_bbox_occlusion_pct"] == pct
        ]

        summary.append({
            "severity_pct": pct,
            "instances": len(rows),
            "mean_actual_bbox_coverage_pct": float(
                np.mean([
                    r["actual_bbox_occlusion_pct"]
                    for r in rows
                ])
            ),
            "rows_with_newly_hidden_keypoints": sum(
                bool(r["newly_hidden_keypoint_indices"])
                for r in rows
            ),
            "total_newly_hidden_keypoints": sum(
                len(r["newly_hidden_keypoint_indices"])
                for r in rows
            ),
        })

    metadata = {
        "protocol": "RO3_R1_instance_occlusion_v1",
        "split": args.split,
        "mode": args.mode,
        "levels_pct": args.levels,
        "replicates_nonzero": args.replicates,
        "seed": args.seed,
        "source_annotation": str(ann_path),
        "source_annotation_sha256":
            sha256_file(ann_path),
        "generator_sha256":
            sha256_file(Path(__file__)),
        "source_image_sha256_by_image_id":
            source_hashes,
        "gt_instance_count": len(annotations),
        "manifest_record_count": len(records),
        "denominator_definition": (
            "Clipped raster GT bounding-box area. "
            "This is not true steel foreground area."
        ),
        "gt_policy": (
            "GT keypoint coordinates and COCO "
            "visibility are preserved for scoring. "
            "Synthetic coverage is separate metadata."
        ),
        "interpretation": (
            "One target instance per rendered input. "
            "Natural bbox overlap is a spatial proxy, "
            "not a validated occlusion or depth label."
        ),
        "summary_by_severity": summary,
    }

    (out / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    # Visual QA (not used as evaluation inputs).
    if args.qa_per_level:
        qa_dir = out / "qa_previews"
        qa_dir.mkdir()

        for pct in args.levels:
            options = [
                r for r in records
                if r["requested_bbox_occlusion_pct"] == pct
                and r["replicate"] == 0
            ]

            rng = stable_rng(
                args.seed, args.split, pct, "qa"
            )

            choices = rng.choice(
                len(options),
                size=min(args.qa_per_level, len(options)),
                replace=False,
            )

            for j in choices:
                row = options[int(j)]
                write_preview(
                    row,
                    qa_dir / f"{row['record_id']}.png",
                )

    print("\nRO3 R1 OCCLUSION DATASET")
    print("=" * 65)
    print("Split:", args.split)
    print("Mode:", args.mode)
    print("Original instances:", len(annotations))
    print("Virtual evaluation records:", len(records))
    print("Manifest:", manifest)
    print()

    print(
        f"{'Severity':>10}"
        f"{'Mean actual':>16}"
        f"{'Rows with hidden KP':>22}"
    )

    for item in summary:
        print(
            f"{item['severity_pct']:>9}%"
            f"{item['mean_actual_bbox_coverage_pct']:>15.2f}%"
            f"{item['rows_with_newly_hidden_keypoints']:>22}"
        )

    fallback = sum(
        r["placement_mode_effective"]
        == "uniform_fallback"
        for r in records
    )

    if fallback:
        print(
            "\nNOTE: Keypoint-targeted placement used "
            f"uniform fallback in {fallback} rows."
        )

    print("\nPASS: Manifest and QA prepared")
    print("Original dataset unchanged")


if __name__ == "__main__":
    main()
