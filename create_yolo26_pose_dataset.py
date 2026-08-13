#!/usr/bin/env python3
"""
Convert InnoCount_KeyPoint_ModelReady_v1 into
Ultralytics YOLO26-Pose dataset format.

SOURCE
------
InnoCount_KeyPoint_ModelReady_v1

OUTPUT
------
YOLO26-Pose dataset with:

    K = 6
    dimensions = 3
    [x, y, visibility]

No source data is modified.

The converter:
    1. Reads the model-ready COCO annotations.
    2. Copies images.
    3. Converts bboxes to normalized YOLO format.
    4. Converts K=6 keypoints to normalized YOLO pose format.
    5. Creates empty label files for negative/background images.
    6. Creates dataset.yaml.
    7. Produces conversion and validation reports.

IMPORTANT
---------
The model-ready adapter already masks:
    out-of-image keypoints -> (0, 0, 0)
    unused global slots   -> (0, 0, 0)

This script does NOT reinterpret or correct those values.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("train", "valid", "test")

CLASS_NAMES = [
    "Channel",
    "T-beam",
    "HI-Beam",
    "Angle Bar",
    "Sheet Pile",
]

CLASS_ID_TO_NAME = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}

GLOBAL_K = 6
KPT_DIMS = 3


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def format_float(value: float) -> str:
    """
    Stable compact floating-point representation.
    """
    return f"{value:.10f}".rstrip("0").rstrip(".")


def normalize_bbox(annotation, image):
    """
    COCO bbox:
        [x, y, width, height]

    YOLO bbox:
        [cx/W, cy/H, w/W, h/H]
    """

    x, y, w, h = annotation["bbox"]

    image_w = image["width"]
    image_h = image["height"]

    if image_w <= 0 or image_h <= 0:
        raise ValueError(
            f"Invalid image dimensions: "
            f"{image_w}x{image_h}"
        )

    cx = x + (w / 2.0)
    cy = y + (h / 2.0)

    return (
        cx / image_w,
        cy / image_h,
        w / image_w,
        h / image_h,
    )


def normalize_keypoints(annotation, image):
    """
    Convert model-ready K=6 COCO keypoints:

        [x, y, v] * 6

    into normalized YOLO pose keypoints:

        [x/W, y/H, v] * 6

    Visibility is preserved exactly.

    For visibility=0, the model-ready adapter should already
    contain x=0 and y=0. We retain those values.
    """

    keypoints = annotation.get("keypoints", [])

    expected_length = GLOBAL_K * KPT_DIMS

    if len(keypoints) != expected_length:
        raise ValueError(
            f"Annotation {annotation['id']} has "
            f"{len(keypoints)} keypoint values; "
            f"expected {expected_length}"
        )

    image_w = image["width"]
    image_h = image["height"]

    result = []

    for i in range(GLOBAL_K):

        x = keypoints[i * 3]
        y = keypoints[i * 3 + 1]
        v = keypoints[i * 3 + 2]

        # Visibility is not normalized.
        # It remains the original 0/1/2 value.
        if v == 0:
            # The model-ready adapter guarantees masked
            # points are stored as x=0, y=0.
            result.extend([
                0.0,
                0.0,
                0,
            ])
        else:
            result.extend([
                x / image_w,
                y / image_h,
                v,
            ])

    return result


def convert_annotation(annotation, image):
    """
    Convert one COCO annotation into one YOLO pose row.
    """

    category_id = annotation["category_id"]

    if category_id not in CLASS_ID_TO_NAME:
        raise ValueError(
            f"Unknown category ID: {category_id}"
        )

    class_id = category_id

    bbox = normalize_bbox(
        annotation,
        image,
    )

    keypoints = normalize_keypoints(
        annotation,
        image,
    )

    values = [
        class_id,
        *bbox,
        *keypoints,
    ]

    return " ".join(
        str(v) if isinstance(v, int)
        else format_float(v)
        for v in values
    )


def process_split(
    source_dataset: Path,
    output_dataset: Path,
    split: str,
):
    source_split = (
        source_dataset
        / "splits"
        / split
    )

    source_json = (
        source_split
        / "annotations.json"
    )

    if not source_json.exists():
        raise FileNotFoundError(
            f"Missing annotations: {source_json}"
        )

    data = load_json(source_json)

    images = data["images"]
    annotations = data["annotations"]

    image_by_id = {
        image["id"]: image
        for image in images
    }

    output_images = (
        output_dataset
        / "images"
        / split
    )

    output_labels = (
        output_dataset
        / "labels"
        / split
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations_by_image = {}

    for annotation in annotations:

        image_id = annotation["image_id"]

        annotations_by_image.setdefault(
            image_id,
            []
        ).append(annotation)

    class_counts = Counter()
    total_objects = 0
    negative_images = 0

    for image in images:

        image_id = image["id"]
        filename = image["file_name"]

        source_image = (
            source_split
            / "images"
            / filename
        )

        if not source_image.exists():
            raise FileNotFoundError(
                f"Missing source image: "
                f"{source_image}"
            )

        destination_image = (
            output_images
            / filename
        )

        destination_image.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Exact copy.
        shutil.copy2(
            source_image,
            destination_image,
        )

        image_annotations = annotations_by_image.get(
            image_id,
            []
        )

        label_lines = []

        for annotation in image_annotations:

            line = convert_annotation(
                annotation,
                image,
            )

            label_lines.append(line)

            class_counts[
                CLASS_ID_TO_NAME[
                    annotation["category_id"]
                ]
            ] += 1

            total_objects += 1

        # Every image gets a label file.
        #
        # For a true negative/background image,
        # this file is intentionally empty.
        label_path = (
            output_labels
            / Path(filename).with_suffix(".txt")
        )

        if label_lines:
            write_text(
                label_path,
                "\n".join(label_lines) + "\n",
            )
        else:
            write_text(
                label_path,
                "",
            )
            negative_images += 1

    return {
        "split": split,
        "images": len(images),
        "annotations": total_objects,
        "negative_images": negative_images,
        "class_counts": dict(
            class_counts
        ),
    }


def validate_split(
    output_dataset: Path,
    split: str,
):
    """
    Basic structural validation of the generated YOLO dataset.
    """

    image_dir = (
        output_dataset
        / "images"
        / split
    )

    label_dir = (
        output_dataset
        / "labels"
        / split
    )

    image_files = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    )

    label_files = sorted(
        label_dir.rglob("*.txt")
    )

    image_stems = {
        p.relative_to(image_dir).with_suffix("")
        for p in image_files
    }

    label_stems = {
        p.relative_to(label_dir).with_suffix("")
        for p in label_files
    }

    missing_labels = sorted(
        image_stems - label_stems
    )

    if missing_labels:
        raise ValueError(
            f"{split}: "
            f"{len(missing_labels)} images "
            f"have no label file. "
            f"Example: {missing_labels[:5]}"
        )

    invalid_rows = []

    total_objects = 0

    for label_file in label_files:

        with label_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            parts = line.split()

            expected_values = (
                1
                + 4
                + GLOBAL_K * KPT_DIMS
            )

            if len(parts) != expected_values:
                invalid_rows.append(
                    (
                        label_file,
                        line_number,
                        f"expected "
                        f"{expected_values} values, "
                        f"got {len(parts)}"
                    )
                )
                continue

            try:
                class_id = int(parts[0])

                if not (
                    0 <= class_id < len(CLASS_NAMES)
                ):
                    invalid_rows.append(
                        (
                            label_file,
                            line_number,
                            f"invalid class ID "
                            f"{class_id}"
                        )
                    )
                    continue

                numeric = [
                    float(x)
                    for x in parts[1:]
                ]

            except ValueError as exc:
                invalid_rows.append(
                    (
                        label_file,
                        line_number,
                        f"non-numeric value: {exc}"
                    )
                )
                continue

            # Bbox values.
            bbox = numeric[:4]

            for value in bbox:

                if not (
                    0.0 <= value <= 1.0
                ):
                    invalid_rows.append(
                        (
                            label_file,
                            line_number,
                            f"bbox value outside "
                            f"[0,1]: {value}"
                        )
                    )

            # Keypoints.
            kp_values = numeric[4:]

            for i in range(GLOBAL_K):

                x = kp_values[i * 3]
                y = kp_values[i * 3 + 1]
                v = kp_values[i * 3 + 2]

                if not (
                    0.0 <= x <= 1.0
                ):
                    invalid_rows.append(
                        (
                            label_file,
                            line_number,
                            f"G{i+1} x outside "
                            f"[0,1]: {x}"
                        )
                    )

                if not (
                    0.0 <= y <= 1.0
                ):
                    invalid_rows.append(
                        (
                            label_file,
                            line_number,
                            f"G{i+1} y outside "
                            f"[0,1]: {y}"
                        )
                    )

                if v not in (0.0, 1.0, 2.0):
                    invalid_rows.append(
                        (
                            label_file,
                            line_number,
                            f"G{i+1} visibility "
                            f"is {v}; expected "
                            f"0, 1, or 2"
                        )
                    )

            total_objects += 1

    return {
        "images": len(image_files),
        "labels": len(label_files),
        "objects": total_objects,
        "missing_labels": len(
            missing_labels
        ),
        "invalid_rows": invalid_rows,
    }


def write_dataset_yaml(
    output_dataset: Path,
):
    """
    Write the Ultralytics dataset YAML.

    We intentionally use identity flip_idx because the
    six keypoints are geometric profile landmarks rather
    than left/right anatomical pairs.

    Training-time horizontal flipping will be disabled in
    the baseline training configuration anyway.
    """

    yaml = """# InnoCount KeyPoint - YOLO26-Pose
#
# Derived from:
#   InnoCount_KeyPoint_ModelReady_v1
#
# Classes:
#   0 Channel
#   1 T-beam
#   2 HI-Beam
#   3 Angle Bar
#   4 Sheet Pile
#
# Global pose representation:
#   K = 6
#   dimensions = 3
#   [x, y, visibility]

path: .

train: images/train
val: images/valid
test: images/test

kpt_shape: [6, 3]

# No anatomical left/right pairs exist in this dataset.
# Identity mapping prevents accidental semantic swapping.
flip_idx: [0, 1, 2, 3, 4, 5]

names:
  0: Channel
  1: T-beam
  2: HI-Beam
  3: Angle Bar
  4: Sheet Pile

kpt_names:
  0:
    - G1
    - G2
    - G3
    - G4
    - G5
    - G6
  1:
    - G1
    - G2
    - G3
    - G4
    - G5
    - G6
  2:
    - G1
    - G2
    - G3
    - G4
    - G5
    - G6
  3:
    - G1
    - G2
    - G3
    - G4
    - G5
    - G6
  4:
    - G1
    - G2
    - G3
    - G4
    - G5
    - G6
"""

    write_text(
        output_dataset
        / "dataset.yaml",
        yaml,
    )


def write_report(
    output_dataset: Path,
    split_reports,
    validation_reports,
):
    lines = []

    lines.append(
        "InnoCount KeyPoint -> YOLO26-Pose"
    )
    lines.append(
        "=" * 70
    )
    lines.append("")

    lines.append(
        "SOURCE: "
        "InnoCount_KeyPoint_ModelReady_v1"
    )
    lines.append(
        "GLOBAL KEYPOINT COUNT: 6"
    )
    lines.append(
        "KEYPOINT DIMENSIONS: 3 "
        "(x, y, visibility)"
    )
    lines.append(
        "CLASS COUNT: 5"
    )
    lines.append("")

    lines.append(
        "CLASS MAPPING"
    )
    lines.append("-" * 70)

    for class_id, name in CLASS_ID_TO_NAME.items():
        lines.append(
            f"{class_id}: {name}"
        )

    lines.append("")

    lines.append(
        "SPLITS"
    )
    lines.append("-" * 70)

    for report in split_reports:

        split = report["split"]

        lines.append(
            f"{split}: "
            f"{report['images']} images, "
            f"{report['annotations']} objects, "
            f"{report['negative_images']} negatives"
        )

        for class_name in CLASS_NAMES:
            lines.append(
                f"  {class_name}: "
                f"{report['class_counts'].get(class_name, 0)}"
            )

    lines.append("")

    lines.append(
        "VALIDATION"
    )
    lines.append("-" * 70)

    overall_pass = True

    for split, report in validation_reports.items():

        passed = (
            report["missing_labels"] == 0
            and len(report["invalid_rows"]) == 0
        )

        if not passed:
            overall_pass = False

        lines.append(
            f"{split}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        lines.append(
            f"  images: {report['images']}"
        )

        lines.append(
            f"  labels: {report['labels']}"
        )

        lines.append(
            f"  objects: {report['objects']}"
        )

        lines.append(
            f"  missing labels: "
            f"{report['missing_labels']}"
        )

        lines.append(
            f"  invalid rows: "
            f"{len(report['invalid_rows'])}"
        )

        if report["invalid_rows"]:
            for item in report["invalid_rows"][:20]:
                lines.append(
                    f"    {item}"
                )

    lines.append("")

    lines.append(
        "OVERALL: "
        + ("PASS" if overall_pass else "FAIL")
    )

    lines.append("")

    lines.append(
        "No source ModelReady annotations were modified."
    )

    write_text(
        output_dataset
        / "reports"
        / "yolo26_pose_conversion_report.txt",
        "\n".join(lines) + "\n",
    )

    return overall_pass


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert InnoCount ModelReady "
            "dataset to YOLO26-Pose format."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help=(
            "InnoCount_KeyPoint_ModelReady_v1"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "Output YOLO26-Pose dataset"
        ),
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset does not exist: "
            f"{args.dataset}"
        )

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: "
            f"{args.output}\n"
            "Refusing to overwrite."
        )

    print(
        "InnoCount KeyPoint -> YOLO26-Pose"
    )
    print("=" * 70)

    print(
        "\n[1/5] Creating YOLO dataset..."
    )

    split_reports = []

    for split in SPLITS:

        print(
            f"Processing {split}..."
        )

        split_reports.append(
            process_split(
                args.dataset,
                args.output,
                split,
            )
        )

    print(
        "\n[2/5] Writing dataset YAML..."
    )

    write_dataset_yaml(
        args.output
    )

    print(
        "\n[3/5] Validating YOLO labels..."
    )

    validation_reports = {}

    for split in SPLITS:

        validation_reports[split] = (
            validate_split(
                args.output,
                split,
            )
        )

    print(
        "\n[4/5] Writing reports..."
    )

    passed = write_report(
        args.output,
        split_reports,
        validation_reports,
    )

    print(
        "\n[5/5] Result"
    )
    print("=" * 70)

    if passed:

        print(
            "PASS: YOLO26-Pose dataset "
            "conversion and validation passed."
        )

        print(
            f"\nOutput: {args.output}"
        )

    else:

        print(
            "FAIL: YOLO26-Pose dataset "
            "validation failed."
        )

        print(
            "\nSee:"
        )

        print(
            args.output
            / "reports"
            / "yolo26_pose_conversion_report.txt"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
