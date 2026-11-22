#!/usr/bin/env python3
"""RO3 R3-B: Deterministic perspective-like scene transformations.

No inference, training, or mutation of original datasets.
Uses the existing R3-A source inventory as an integrity anchor.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

BORDER_BGR = (114, 114, 114)

# Fractional corner inset at EACH end of one image edge.
LEVELS = (0.06, 0.12)

CONDITIONS = [
    ("vp_neutral", "neutral", "none", 0.0),

    ("vp_yaw_left_06", "yaw", "left", 0.06),
    ("vp_yaw_right_06", "yaw", "right", 0.06),
    ("vp_pitch_top_06", "pitch", "top", 0.06),
    ("vp_pitch_bottom_06", "pitch", "bottom", 0.06),

    ("vp_yaw_left_12", "yaw", "left", 0.12),
    ("vp_yaw_right_12", "yaw", "right", 0.12),
    ("vp_pitch_top_12", "pitch", "top", 0.12),
    ("vp_pitch_bottom_12", "pitch", "bottom", 0.12),
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)

    return h.hexdigest()


def source_corners(width, height):
    """Pixel-center image corners, clockwise from top-left."""
    return np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [width - 1.0, height - 1.0],
        [0.0, height - 1.0],
    ], dtype=np.float64)


def make_homography(width, height, axis, edge, inset):
    """Create a convex trapezoidal warp inside the source canvas."""

    require(
        width >= 16 and height >= 16,
        "Unexpectedly small image"
    )

    require(
        0.0 <= inset <= 0.12,
        "Unsupported projective severity"
    )

    src = source_corners(width, height)
    dst = src.copy()

    w = width - 1.0
    h = height - 1.0

    if axis == "neutral":
        require(
            edge == "none" and inset == 0.0,
            "Invalid neutral condition"
        )
        H = np.eye(3, dtype=np.float64)

    elif axis == "yaw":

        require(edge in ("left", "right"),
                "Invalid yaw edge")

        dy = inset * h

        if edge == "left":
            # Shrink left vertical edge toward its middle.
            dst[0, 1] += dy
            dst[3, 1] -= dy
        else:
            # Shrink right vertical edge toward its middle.
            dst[1, 1] += dy
            dst[2, 1] -= dy

        H = cv2.getPerspectiveTransform(
            src.astype(np.float32),
            dst.astype(np.float32)
        ).astype(np.float64)

    elif axis == "pitch":

        require(edge in ("top", "bottom"),
                "Invalid pitch edge")

        dx = inset * w

        if edge == "top":
            # Shrink top horizontal edge toward its middle.
            dst[0, 0] += dx
            dst[1, 0] -= dx
        else:
            # Shrink bottom horizontal edge toward its middle.
            dst[3, 0] += dx
            dst[2, 0] -= dx

        H = cv2.getPerspectiveTransform(
            src.astype(np.float32),
            dst.astype(np.float32)
        ).astype(np.float64)

    else:
        raise ValueError(f"Unknown transformation axis: {axis}")

    require(
        np.isfinite(H).all()
        and abs(float(np.linalg.det(H))) > 1e-10,
        "Singular or invalid homography"
    )

    # Homography scale is arbitrary; normalize consistently.
    H /= H[2, 2]

    mapped = transform_points(src, H)

    require(
        np.max(np.abs(mapped - dst)) < 0.02,
        "Homography does not map expected corners"
    )

    require(
        np.all(mapped[:, 0] >= -0.02)
        and np.all(mapped[:, 1] >= -0.02)
        and np.all(mapped[:, 0] <= w + 0.02)
        and np.all(mapped[:, 1] <= h + 0.02),
        "Transformed image corners outside canvas"
    )

    # Inverse mapping must recover the source coordinates.
    inverse = np.linalg.inv(H)
    restored = transform_points(mapped, inverse)

    require(
        np.max(np.abs(restored - src)) < 0.02,
        "Inverse homography check failed"
    )

    # Shoelace formula: coverage of the destination quad.
    x = dst[:, 0]
    y = dst[:, 1]

    area = 0.5 * abs(
        float(
            np.dot(x, np.roll(y, -1))
            - np.dot(y, np.roll(x, -1))
        )
    )

    source_area = w * h

    coverage = area / source_area

    require(
        0.85 < coverage <= 1.000001,
        "Unexpected destination quadrilateral coverage"
    )

    return H, dst, coverage


def transform_points(points, H):
    """Project any Nx2 coordinates through the homography."""

    points = np.asarray(points, dtype=np.float64)

    require(
        points.ndim == 2 and points.shape[1] == 2,
        "Expected Nx2 coordinates"
    )

    require(
        np.isfinite(points).all(),
        "Nonfinite source coordinates"
    )

    homogeneous = np.column_stack([
        points,
        np.ones(len(points), dtype=np.float64)
    ])

    projected = homogeneous @ H.T

    require(
        np.all(projected[:, 2] > 1e-9),
        "Invalid projective denominator"
    )

    result = (
        projected[:, :2]
        / projected[:, 2, None]
    )

    require(
        np.isfinite(result).all(),
        "Nonfinite transformed coordinates"
    )

    return result


def transform_bbox(box, H):
    """Transform all four bbox corners, returning enclosing xywh."""

    x, y, w, h = map(float, box)

    require(
        w > 0 and h > 0,
        "Nonpositive original GT box"
    )

    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ], dtype=np.float64)

    moved = transform_points(corners, H)

    minimum = moved.min(axis=0)
    maximum = moved.max(axis=0)

    result = np.array([
        minimum[0],
        minimum[1],
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
    ])

    require(
        np.all(result[2:] > 0),
        "Degenerate projective GT box"
    )

    return result


def warp_image(image, H, condition_id):
    """Neutral is pixel-identical; others use bilinear warping."""

    if condition_id == "vp_neutral":
        return image.copy()

    height, width = image.shape[:2]

    return cv2.warpPerspective(
        image,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_BGR,
    )


def geometry_self_tests():
    """Independent tests on synthetic image dimensions."""

    for width, height in (
        (640, 480),
        (1920, 1080),
        (1080, 1920),
        (4032, 3024),
    ):
        for cid, axis, edge, inset in CONDITIONS:

            H, dst, coverage = make_homography(
                width, height, axis, edge, inset
            )

            corners = source_corners(width, height)

            mapped = transform_points(corners, H)

            require(
                np.max(np.abs(mapped - dst)) < 0.02,
                f"Corner mismatch: {cid}"
            )

            recovered = transform_points(
                mapped, np.linalg.inv(H)
            )

            require(
                np.max(np.abs(recovered - corners)) < 0.02,
                f"Inverse mismatch: {cid}"
            )

            if cid == "vp_neutral":
                require(
                    np.array_equal(H, np.eye(3)),
                    "Neutral matrix not identity"
                )
                require(
                    abs(coverage - 1.0) < 1e-10,
                    "Neutral coverage differs"
                )
            else:
                expected_coverage = 1.0 - inset

                require(
                    abs(coverage - expected_coverage) < 1e-6,
                    f"Coverage differs: {cid}"
                )

    print("PASS: Synthetic homography geometry tests")


def load_r3a_source_inventory(split):
    """Reuse the existing frozen R3-A image and COCO sources."""

    previous = (
        ROOT / "experiments/RO3/"
        "R3_Orientation_v1" / split
    )

    metadata_path = previous / "protocol_metadata.json"
    manifest_path = previous / "orientation_manifest.jsonl"

    require(
        metadata_path.is_file() and manifest_path.is_file(),
        "R3-A source inventory unavailable"
    )

    meta = json.loads(metadata_path.read_text())

    require(
        meta["protocol"] == "RO3_R3A_InPlaneRotation_v1"
        and meta["split"] == split,
        "Invalid R3-A source protocol"
    )

    expected = {
        "valid": (1551, 6720, 72),
        "test": (1626, 7049, 71),
    }

    expected_objects, expected_kp, expected_images = (
        expected[split]
    )

    require(
        meta["source_images"] == 74
        and meta["annotated_source_images"] == expected_images
        and meta["gt_instances"] == expected_objects,
        "Unexpected R3-A source counts"
    )

    coco_path = ROOT / meta["annotation_path"]

    require(
        sha256(coco_path) == meta["annotation_sha256"],
        "Frozen COCO annotations changed"
    )

    coco = json.loads(coco_path.read_text())

    require(
        len(coco["images"]) == 74
        and len(coco["annotations"]) == expected_objects,
        "Unexpected COCO dataset size"
    )

    visibility_count = sum(
        sum(
            float(kp[2]) > 0
            for kp in np.asarray(
                ann["keypoints"]
            ).reshape(6, 3)
        )
        for ann in coco["annotations"]
    )

    require(
        visibility_count == expected_kp,
        "Frozen valid-keypoint count differs"
    )

    zero_records = {}
    all_records = 0

    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            all_records += 1

            if int(record["angle_degrees"]) != 0:
                continue

            iid = int(record["image_id"])

            require(
                iid not in zero_records,
                "Duplicate R3-A neutral image"
            )

            zero_records[iid] = record

    require(
        all_records == 518
        and len(zero_records) == 74,
        "Incomplete R3-A source inventory"
    )

    require(
        set(zero_records) ==
        {int(im["id"]) for im in coco["images"]},
        "COCO and R3-A image IDs differ"
    )

    for iid, record in zero_records.items():

        expected_hash = meta[
            "source_image_hashes"
        ][str(iid)]

        path = (ROOT / record["source_image"]).resolve()
        path.relative_to(ROOT)

        require(
            path.is_file()
            and sha256(path) == expected_hash
            and record["source_sha256"] == expected_hash,
            f"Original source image changed: {iid}"
        )

        image = cv2.imread(
            str(path), cv2.IMREAD_COLOR
        )

        require(
            image is not None,
            f"Unreadable source image: {iid}"
        )

        require(
            image.shape[:2] == (
                int(record["height"]),
                int(record["width"])
            ),
            f"Image dimensions differ: {iid}"
        )

    print("PASS: Frozen R3-A source inventory verified")
    print("COCO objects:", expected_objects)
    print("Valid GT keypoints:", expected_kp)

    return (
        meta,
        coco,
        zero_records,
        metadata_path,
        manifest_path,
    )


def write_qa(out, coco, sources):
    """Produce labeled previews; inference inputs stay unannotated."""

    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    annotated = sorted(
        (
            (len(anns_by_image[iid]), iid)
            for iid in sources
            if anns_by_image[iid]
        )
    )

    require(
        len(annotated) > 2,
        "Insufficient QA source images"
    )

    # One representative and one dense source photograph.
    selected = sorted(set([
        annotated[len(annotated) // 2][1],
        annotated[-1][1],
    ]))

    qa = out / "qa_previews"
    qa.mkdir()

    for iid in selected:

        record = sources[iid]

        original = cv2.imread(
            str(ROOT / record["source_image"]),
            cv2.IMREAD_COLOR
        )

        height, width = original.shape[:2]

        for cid, axis, edge, inset in CONDITIONS:

            H, dst, coverage = make_homography(
                width, height, axis, edge, inset
            )

            image = warp_image(original, H, cid)

            if cid == "vp_neutral":
                require(
                    np.array_equal(original, image),
                    "Neutral image differs"
                )

            # Measure source-image support after warping.
            support = cv2.warpPerspective(
                np.ones(
                    (height, width), dtype=np.uint8
                ),
                H,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            actual_coverage = float(
                np.mean(support > 0)
            )

            require(
                abs(actual_coverage - coverage) < 0.02,
                f"Raster support differs: {cid}"
            )

            overlay = image.copy()

            for ann in anns_by_image[iid]:

                x, y, w, h = transform_bbox(
                    ann["bbox"], H
                )

                cv2.rectangle(
                    overlay,
                    (round(x), round(y)),
                    (round(x + w), round(y + h)),
                    (255, 190, 40),
                    1
                )

                kp = np.asarray(
                    ann["keypoints"],
                    dtype=np.float64
                ).reshape(6, 3)

                points = transform_points(
                    kp[:, :2], H
                )

                for index in range(6):
                    if kp[index, 2] <= 0:
                        continue

                    p = tuple(
                        int(round(v))
                        for v in points[index]
                    )

                    cv2.circle(
                        overlay, p, 2,
                        (40, 255, 80), -1
                    )

            # Resizing is for QA files only.
            factor = min(
                1.0,
                1100.0 / max(overlay.shape[:2])
            )

            if factor < 1.0:
                overlay = cv2.resize(
                    overlay, None,
                    fx=factor,
                    fy=factor
                )

            target = qa / f"image_{iid:06d}_{cid}.jpg"

            require(
                cv2.imwrite(str(target), overlay),
                f"Unable to save QA preview: {target}"
            )

    print("PASS: Geometric QA previews generated")
    print("QA source image IDs:", selected)
    print("QA files:", len(selected) * len(CONDITIONS))
    print("QA directory:", qa)

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split", required=True,
        choices=("valid", "test")
    )

    args = parser.parse_args()

    split = args.split

    out = ROOT / (
        f"experiments/RO3/R3_Viewpoint_v1/{split}"
    )

    require(
        not out.exists(),
        f"Refusing to overwrite: {out}"
    )

    geometry_self_tests()

    (
        source_meta,
        coco,
        source_records,
        source_meta_path,
        source_manifest_path,
    ) = load_r3a_source_inventory(split)

    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    rows = []

    for cid, axis, edge, inset in CONDITIONS:

        for iid in sorted(source_records):

            original = source_records[iid]

            width = int(original["width"])
            height = int(original["height"])

            H, dst, coverage = make_homography(
                width, height, axis, edge, inset
            )

            # Independently verify all four corners of
            # every GT box are projectively transformable.
            for ann in anns_by_image.get(iid, []):
                transform_bbox(ann["bbox"], H)

                keypoints = np.asarray(
                    ann["keypoints"],
                    dtype=np.float64
                ).reshape(6, 3)

                transform_points(keypoints[:, :2], H)

            rows.append({
                "protocol":
                    "RO3_R3B_ProjectiveViewpoint_v1",
                "split": split,
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset,
                "image_id": iid,
                "source_image": original["source_image"],
                "source_sha256": original["source_sha256"],
                "width": width,
                "height": height,
                "gt_instances":
                    len(anns_by_image.get(iid, [])),
                "homography": H.tolist(),
                "destination_corners": dst.tolist(),
                "output_width": width,
                "output_height": height,
                "analytic_source_coverage": coverage,
                "interpolation": "cv2.INTER_LINEAR",
                "border_bgr": list(BORDER_BGR),
            })

    expected_rows = 74 * len(CONDITIONS)

    require(
        len(rows) == expected_rows,
        "Incorrect R3-B manifest size"
    )

    out.mkdir(parents=True)

    manifest = out / "viewpoint_manifest.jsonl"

    with manifest.open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    qa_images = write_qa(
        out, coco, source_records
    )

    metadata = {
        "protocol": "RO3_R3B_ProjectiveViewpoint_v1",
        "split": split,
        "model": "Frozen ViTPose-B E03",
        "conditions": [
            {
                "condition_id": cid,
                "axis": axis,
                "compressed_edge": edge,
                "inset_fraction": inset,
            }
            for cid, axis, edge, inset in CONDITIONS
        ],
        "source_images": 74,
        "annotated_source_images":
            72 if split == "valid" else 71,
        "gt_instances":
            1551 if split == "valid" else 1626,
        "valid_gt_keypoints":
            6720 if split == "valid" else 7049,
        "virtual_scene_records": expected_rows,
        "annotation_path": source_meta["annotation_path"],
        "annotation_sha256":
            source_meta["annotation_sha256"],
        "source_image_hashes":
            source_meta["source_image_hashes"],
        "r3a_source_metadata_sha256":
            sha256(source_meta_path),
        "r3a_source_manifest_sha256":
            sha256(source_manifest_path),
        "generator_sha256": sha256(Path(__file__)),
        "neutral_pixel_identical": True,
        "original_gt_keypoint_slots_preserved": True,
        "primary_pck_normalizer":
            "original_gt_bbox_max_dimension",
        "secondary_pck_normalizer":
            "transformed_axis_aligned_gt_bbox_max_dimension",
        "image_canvas":
            "original width and height",
        "qa_image_ids": qa_images,
        "limitations": [
            "Synthetic 2D projective warp, not physical 3D view",
            "No hidden surfaces become visible",
            "Artificial border and interpolation effects",
            "Projective distortion is spatially nonuniform",
            "GT-conditioned top-down pose only",
        ]
    }

    metadata_path = out / "protocol_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print()
    print("PASS: R3-B projective viewpoint manifest generated")
    print("Split:", split)
    print("Conditions:", len(CONDITIONS))
    print("Source photographs:", 74)
    print("Virtual scene records:", len(rows))
    print("Manifest:", manifest)
    print("Metadata:", metadata_path)
    print("Original GT and prior RO3 evidence unchanged")


if __name__ == "__main__":
    main()
