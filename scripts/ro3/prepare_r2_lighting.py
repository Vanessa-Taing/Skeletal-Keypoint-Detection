#!/usr/bin/env python3
"""RO3 R2: Deterministic scene-level exposure/contrast benchmark."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from prepare_r1_occlusion import find_annotations, find_image

ROOT = Path(__file__).resolve().parents[2]

LEVELS = {
    "exposure": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "contrast": [0.65, 0.80, 1.00, 1.20, 1.40],
}

NEUTRAL = {"exposure": 0.0, "contrast": 1.0}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def condition_id(mode, value):
    if mode == "exposure":
        sign = "m" if value < 0 else "p"
        return f"ev_{sign}{round(abs(value)*100):03d}"
    return f"ct_{round(value*100):03d}"


def to_linear(srgb):
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def render_lighting(record, image):
    """Apply photometric change; preserve original geometry.

    Input/output: uint8 BGR, original image dimensions.
    No annotation graphics are added.
    """
    require(
        image.ndim == 3 and image.shape[2] == 3
        and image.dtype == np.uint8,
        "Expected OpenCV BGR uint8 photograph",
    )

    mode = record["mode"]
    value = float(record["value"])

    require(mode in LEVELS, "Unknown photometric mode")
    require(
        any(abs(value - x) < 1e-9 for x in LEVELS[mode]),
        "Unknown transformation level",
    )

    if abs(value - NEUTRAL[mode]) < 1e-9:
        return image.copy()

    srgb = image.astype(np.float32) / 255.0
    linear = to_linear(srgb)

    if mode == "exposure":
        # Simulated exposure adjustment in linear light.
        adjusted = linear * (2.0 ** value)

    elif mode == "contrast":
        # OpenCV stores BGR; linear luminance is RGB-weighted.
        luminance = (
            0.0722 * linear[:, :, 0]
            + 0.7152 * linear[:, :, 1]
            + 0.2126 * linear[:, :, 2]
        )
        pivot = float(luminance.mean())

        # Global contrast around scene mean luminance.
        adjusted = pivot + value * (linear - pivot)

    else:
        raise ValueError(mode)

    result = np.rint(
        to_srgb(adjusted) * 255.0
    ).astype(np.uint8)

    require(result.shape == image.shape, "Geometry changed")
    return result


def make_preview(source, changed, title, path):
    require(source.shape == changed.shape, "QA geometry mismatch")

    height, width = source.shape[:2]
    scale = min(1.0, 520.0 / max(height, width))

    target = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )

    left = cv2.resize(source, target)
    right = cv2.resize(changed, target)

    combined = np.concatenate([left, right], axis=1)

    combined = cv2.copyMakeBorder(
        combined, 34, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )

    cv2.putText(
        combined, "Original",
        (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 1,
    )

    cv2.putText(
        combined, title,
        (left.shape[1] + 8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 1,
    )

    require(cv2.imwrite(str(path), combined), "QA write failed")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--split", choices=["valid", "test"], required=True
    )
    ap.add_argument(
        "--mode", choices=["exposure", "contrast"], required=True
    )
    args = ap.parse_args()

    mode = args.mode
    split = args.split

    out = ROOT / (
        f"experiments/RO3/R2_Lighting_v1/"
        f"{split}_{mode}"
    )

    require(
        not out.exists(),
        f"Output exists; refusing overwrite: {out}",
    )

    annotation_path = find_annotations(split)
    coco = json.loads(annotation_path.read_text())

    images = sorted(
        coco["images"], key=lambda im: int(im["id"])
    )

    object_counts = Counter(
        int(a["image_id"])
        for a in coco["annotations"]
    )

    expected_objects = 1551 if split == "valid" else 1626

    require(len(images) == 74, "Expected 74 original photographs")
    require(
        len(coco["annotations"]) == expected_objects,
        "Unexpected original GT count",
    )

    # R1 source-data provenance must already exist.
    r1_meta_path = ROOT / (
        "experiments/RO3/R1_Occlusion_v1/"
        f"{split}_uniform_seed20261119/"
        "protocol_metadata.json"
    )

    require(
        r1_meta_path.is_file(),
        f"Missing frozen R1 source metadata: {r1_meta_path}",
    )

    r1 = json.loads(r1_meta_path.read_text())

    require(
        sha256(annotation_path)
        == r1["source_annotation_sha256"],
        "COCO annotation file changed since R1",
    )

    hashes = r1["source_image_sha256_by_image_id"]

    require(
        len(hashes) == 74,
        "Frozen R1 image-hash collection incomplete",
    )

    source_rows = []

    for im in images:
        image_id = int(im["id"])
        path = find_image(split, im)

        require(
            sha256(path) == hashes[str(image_id)],
            f"Image hash mismatch: {path}",
        )

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(image is not None, f"Cannot decode {path}")

        require(
            image.shape[:2]
            == (int(im["height"]), int(im["width"])),
            f"Image dimensions differ: {path}",
        )

        source_rows.append({
            "image_id": image_id,
            "file_name": im["file_name"],
            "source_image": path.relative_to(ROOT).as_posix(),
            "source_sha256": hashes[str(image_id)],
            "height": int(im["height"]),
            "width": int(im["width"]),
            "gt_object_count": int(object_counts[image_id]),
        })

    # Baseline once and one deterministic image per
    # nonzero level. Repeating these identical pixel
    # transforms would not add statistical information.
    records = []

    for value in LEVELS[mode]:
        for im in source_rows:
            records.append({
                **im,
                "split": split,
                "mode": mode,
                "value": value,
                "condition_id": condition_id(mode, value),
                "transform_scope": "complete_source_image",
                "geometry_changed": False,
            })

    require(
        len(records) == 74 * len(LEVELS[mode]),
        "Incorrect manifest record count",
    )

    out.mkdir(parents=True)

    manifest = out / "lighting_manifest.jsonl"

    with manifest.open("x") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Preview the same representative annotated scene
    # at every condition. QA labels are never model inputs.
    annotated = sorted(
        (r for r in source_rows if r["gt_object_count"] > 0),
        key=lambda r: (r["gt_object_count"], r["image_id"]),
    )

    preview_source = annotated[len(annotated) // 2]
    preview_path = ROOT / preview_source["source_image"]

    original = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)

    qa = out / "qa_previews"
    qa.mkdir()

    for value in LEVELS[mode]:
        record = {
            "mode": mode,
            "value": value,
        }
        transformed = render_lighting(record, original)

        if value == NEUTRAL[mode]:
            require(
                np.array_equal(original, transformed),
                "Neutral condition must be pixel-identical",
            )
        else:
            require(
                np.any(original != transformed),
                "Nonneutral condition changed no pixels",
            )

        make_preview(
            original,
            transformed,
            f"{mode}: {value:+.2f}",
            qa / f"{condition_id(mode, value)}.png",
        )

    metadata = {
        "protocol": "RO3_R2_GlobalLighting_v1",
        "split": split,
        "mode": mode,
        "levels": LEVELS[mode],
        "neutral_value": NEUTRAL[mode],
        "replicates": 1,
        "source_images": len(images),
        "source_gt_instances": expected_objects,
        "annotation_path":
            annotation_path.relative_to(ROOT).as_posix(),
        "annotation_sha256": sha256(annotation_path),
        "r1_source_metadata_sha256": sha256(r1_meta_path),
        "source_image_hashes": hashes,
        "generator_sha256": sha256(Path(__file__)),
        "description": (
            "Deterministic complete-image photometric "
            "transformations, with GT positions unchanged. "
            "Exposure: 2**EV in linear-light RGB. "
            "Contrast: linear RGB scaling about original "
            "image mean linear luminance."
        ),
        "limitations": [
            "Synthetic changes are not measured real lighting.",
            "Exposure and contrast are separate axes.",
            "Clipping can occur at high exposure/contrast.",
            "No viewpoint, geometry, or GT changes.",
            "No new images or independently sampled scenes.",
        ],
    }

    (out / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nRO3 R2 LIGHTING DATASET")
    print("=" * 60)
    print("Split:", split)
    print("Mode:", mode)
    print("Original images:", len(images))
    print("GT objects:", expected_objects)
    print("Lighting conditions:", len(LEVELS[mode]))
    print("Virtual image records:", len(records))
    print("Levels:", LEVELS[mode])
    print("Manifest:", manifest)
    print("QA:", qa)
    print("PASS: Original data unchanged")


if __name__ == "__main__":
    main()
