#!/usr/bin/env python3
"""R3-A: Deterministic expanded-canvas image rotations."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from prepare_r1_occlusion import find_annotations, find_image

ROOT = Path(__file__).resolve().parents[2]
ANGLES = (-45, -30, -15, 0, 15, 30, 45)
BORDER = (114, 114, 114)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def condition_id(angle):
    sign = "m" if angle < 0 else "p"
    return f"rot_{sign}{abs(int(angle)):03d}"


def geometry(width, height, angle):
    """Map original pixel coordinates to expanded canvas."""
    if angle == 0:
        return np.array(
            [[1., 0., 0.], [0., 1., 0.]],
            dtype=np.float64
        ), width, height

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        float(angle), 1.0
    ).astype(np.float64)

    corners = np.array([
        [0., 0.], [width, 0.],
        [width, height], [0., height]
    ])

    transformed = (
        corners @ matrix[:, :2].T
        + matrix[:, 2]
    )

    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)

    matrix[:, 2] -= minimum

    new_width = int(np.ceil(maximum[0] - minimum[0]))
    new_height = int(np.ceil(maximum[1] - minimum[1]))

    require(
        new_width > 0 and new_height > 0,
        "Invalid expanded canvas"
    )

    return matrix, new_width, new_height


def transform_points(points, matrix):
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def transform_bbox(box, matrix):
    """Original COCO xywh -> transformed enclosing xywh."""
    x, y, w, h = [float(v) for v in box]
    require(w > 0 and h > 0, "Invalid GT bbox")

    corners = np.array([
        [x, y], [x + w, y],
        [x + w, y + h], [x, y + h]
    ])

    new_corners = transform_points(corners, matrix)
    minimum = new_corners.min(axis=0)
    maximum = new_corners.max(axis=0)

    result = np.array([
        minimum[0], minimum[1],
        maximum[0] - minimum[0],
        maximum[1] - minimum[1]
    ], dtype=np.float64)

    require(
        np.all(result[2:] > 0)
        and np.isfinite(result).all(),
        "Invalid transformed bbox"
    )
    return result


def rotate_image(source, matrix, width, height, angle):
    if angle == 0:
        return source.copy()

    return cv2.warpAffine(
        source, matrix, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER
    )


def verify_geometry():
    """Deterministic affine and inverse-coordinate checks."""
    width, height = 641, 477
    corners = np.array([
        [0., 0.], [width, 0.],
        [width, height], [0., height]
    ])

    for angle in ANGLES:
        m, w2, h2 = geometry(width, height, angle)
        moved = transform_points(corners, m)

        require(
            moved.min() >= -1e-5,
            f"Rotation outside canvas: {angle}"
        )
        require(
            np.all(moved[:, 0] <= w2 + 1e-5)
            and np.all(moved[:, 1] <= h2 + 1e-5),
            f"Rotation clipped at canvas: {angle}"
        )

        inverse = cv2.invertAffineTransform(m)
        restored = transform_points(moved, inverse)

        require(
            np.max(np.abs(restored - corners)) < 1e-5,
            f"Affine inverse failed: {angle}"
        )

        if angle == 0:
            require(
                w2 == width and h2 == height,
                "Neutral canvas changed"
            )

    print("PASS: Rotation geometry and inverse tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", required=True,
        choices=["valid", "test"]
    )
    args = parser.parse_args()

    verify_geometry()
    split = args.split

    out = ROOT / (
        f"experiments/RO3/R3_Orientation_v1/{split}"
    )

    require(
        not out.exists(),
        f"Refusing to overwrite: {out}"
    )

    ann_path = find_annotations(split)
    coco = json.loads(ann_path.read_text())

    images = sorted(
        coco["images"], key=lambda x: int(x["id"])
    )
    annotations = coco["annotations"]

    expected = 1551 if split == "valid" else 1626
    annotated_images = 72 if split == "valid" else 71

    require(len(images) == 74, "Unexpected image count")
    require(len(annotations) == expected, "Unexpected GT count")
    require(
        len({int(a["image_id"]) for a in annotations})
        == annotated_images,
        "Unexpected annotated-image count"
    )

    r1_meta_path = ROOT / (
        "experiments/RO3/R1_Occlusion_v1/"
        f"{split}_uniform_seed20261119/"
        "protocol_metadata.json"
    )

    r1 = json.loads(r1_meta_path.read_text())

    require(
        sha256(ann_path) == r1["source_annotation_sha256"],
        "COCO annotation differs from frozen R1"
    )

    source_hashes = r1[
        "source_image_sha256_by_image_id"
    ]

    require(
        len(source_hashes) == 74,
        "R1 source-image hash inventory incomplete"
    )

    ann_by_image = defaultdict(list)
    for a in annotations:
        ann_by_image[int(a["image_id"])].append(a)

    image_rows = []
    for im in images:
        iid = int(im["id"])
        path = find_image(split, im)

        require(
            sha256(path) == source_hashes[str(iid)],
            f"Original image changed: {path}"
        )

        image = cv2.imread(
            str(path), cv2.IMREAD_COLOR
        )
        require(image is not None, f"Unreadable image: {path}")

        height, width = image.shape[:2]
        require(
            (height, width) ==
            (int(im["height"]), int(im["width"])),
            "COCO image dimensions differ"
        )

        image_rows.append({
            "image_id": iid,
            "source_image":
                path.relative_to(ROOT).as_posix(),
            "source_sha256": source_hashes[str(iid)],
            "width": width,
            "height": height,
            "gt_instances": len(ann_by_image.get(iid, []))
        })

    records = []
    for angle in ANGLES:
        for image in image_rows:
            m, w2, h2 = geometry(
                image["width"],
                image["height"],
                angle
            )
            records.append({
                **image,
                "split": split,
                "angle_degrees": angle,
                "condition_id": condition_id(angle),
                "matrix": m.tolist(),
                "output_width": w2,
                "output_height": h2,
                "interpolation": "cv2.INTER_LINEAR",
                "border_bgr": list(BORDER)
            })

    require(
        len(records) == 74 * len(ANGLES),
        "Incorrect R3 manifest record count"
    )

    out.mkdir(parents=True)
    manifest = out / "orientation_manifest.jsonl"

    with manifest.open("x") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Only QA previews contain annotation overlays.
    # Inference images NEVER contain overlays.
    candidates = sorted(
        (
            r for r in image_rows
            if r["gt_instances"] > 0
        ),
        key=lambda r: (r["gt_instances"], r["image_id"])
    )

    chosen = candidates[len(candidates) // 2]
    original = cv2.imread(
        str(ROOT / chosen["source_image"])
    )

    qa = out / "qa_previews"
    qa.mkdir()

    for angle in ANGLES:
        m, w2, h2 = geometry(
            chosen["width"],
            chosen["height"],
            angle
        )

        changed = rotate_image(
            original, m, w2, h2, angle
        )

        if angle == 0:
            require(
                np.array_equal(original, changed),
                "Neutral image differs"
            )

        overlay = changed.copy()

        for ann in ann_by_image[chosen["image_id"]]:
            bbox = transform_bbox(ann["bbox"], m)
            x, y, w, h = bbox

            cv2.rectangle(
                overlay,
                (round(x), round(y)),
                (round(x + w), round(y + h)),
                (255, 190, 40), 1
            )

            kp = np.asarray(
                ann["keypoints"],
                dtype=np.float64
            ).reshape(6, 3)

            transformed = transform_points(kp[:, :2], m)

            for j in range(6):
                if kp[j, 2] <= 0:
                    continue

                point = tuple(
                    int(round(v)) for v in transformed[j]
                )

                cv2.circle(
                    overlay, point, 2,
                    (40, 255, 80), -1
                )

        # Resize QA only; never resize model inputs here.
        factor = min(
            1.0,
            1000.0 / max(overlay.shape[:2])
        )
        if factor < 1.0:
            overlay = cv2.resize(
                overlay, None,
                fx=factor, fy=factor
            )

        require(
            cv2.imwrite(
                str(qa / f"{condition_id(angle)}.jpg"),
                overlay
            ),
            "QA preview write failed"
        )

    metadata = {
        "protocol": "RO3_R3A_InPlaneRotation_v1",
        "split": split,
        "angles_degrees": list(ANGLES),
        "source_images": 74,
        "annotated_source_images": annotated_images,
        "gt_instances": expected,
        "annotation_path":
            ann_path.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(ann_path),
        "r1_source_metadata_sha256":
            sha256(r1_meta_path),
        "source_image_hashes": source_hashes,
        "generator_sha256": sha256(Path(__file__)),
        "canvas": "expanded, no intentional source crop",
        "border_bgr": list(BORDER),
        "interpolation": "cv2.INTER_LINEAR",
        "semantic_keypoint_slots_unchanged": True,
        "limitations": [
            "In-plane image rotation is not 3D viewpoint change",
            "Warp interpolation changes nonneutral image pixels",
            "Expanded canvas introduces artificial border",
            "Transformed GT boxes are axis-aligned envelopes",
            "GT-box-conditioned top-down pose, not detection"
        ]
    }

    (out / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nPASS: R3-A orientation manifest generated")
    print("Split:", split)
    print("Source images:", len(image_rows))
    print("Annotated objects:", expected)
    print("Angle conditions:", ANGLES)
    print("Virtual scene records:", len(records))
    print("Manifest:", manifest)
    print("QA previews:", qa)
    print("Original dataset unchanged")


if __name__ == "__main__":
    main()
