#!/usr/bin/env python3
"""
Create InnoCount_KeyPoint_ModelReady_v1 from InnoCount_KeyPoint_Frozen_v2.

Purpose
-------
Create ONE common model-ready representation for:
    - YOLO26-Pose
    - HRNet / MMPose
    - ViTPose-B / MMPose

The frozen dataset is never modified.

Representation
--------------
Global K = 6.

Class-local mapping:
    Channel   K1 K2 K3 K4
    T-beam    K1 K2 K3 K4
    HI-Beam   K1 K2 K3 K4 K5 K6
    Angle Bar K1 K2 K3
    Sheet Pile K1 K2 K3 K4

Unused global slots are:
    visibility = 0
    x = 0
    y = 0

Out-of-image annotated keypoints are also masked in the model-ready
representation:
    visibility = 0
    x = 0
    y = 0

IMPORTANT:
The original frozen coordinate and visibility are preserved in
`original_keypoints` inside every derived annotation. Therefore this is
an adapter, NOT a correction to the frozen ground truth.

Out-of-bbox points are NOT masked unless they are also out-of-image.

The script:
    1. copies images
    2. converts COCO annotations to global K=6
    3. preserves source/provenance metadata
    4. writes per-split COCO JSON
    5. writes a conversion report
    6. validates the resulting adapter

No augmentation, resizing, clipping, class balancing, or split changes occur.

Usage
-----
python create_model_ready_adapter.py \
    --dataset "InnoCount_KeyPoint_Frozen_v2" \
    --output "InnoCount_KeyPoint_ModelReady_v1"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import math

SPLITS = ("train", "valid", "test")

CLASS_ORDER = [
    "Channel",
    "T-beam",
    "HI-Beam",
    "Angle Bar",
    "Sheet Pile",
]

EXPECTED_SOURCE_KEYPOINTS = {
    "Channel": [
        "new-point-0",
        "new-point-1",
        "new-point-2",
        "new-point-3",
    ],
    "T-beam": [
        "new-point-0",
        "new-point-1",
        "new-point-2",
        "new-point-3",
    ],
    "HI-Beam": [
        "new-point-0",
        "new-point-1",
        "new-point-2",
        "new-point-3",
        "new-point-4",
        "new-point-5",
    ],
    "Angle Bar": [
        "new-point-0",
        "new-point-1",
        "new-point-3",
    ],
    "Sheet Pile": [
        "new-point-0",
        "new-point-1",
        "new-point-2",
        "new-point-3",
    ],
}

SEMANTICS = {
    "Channel": [
        "opening corner 1",
        "opening corner 2",
        "opening corner 3",
        "opening corner 4",
    ],
    "T-beam": [
        "flange centre / web-flange intersection",
        "flange end 1",
        "flange end 2",
        "web end opposite the flange",
    ],
    "HI-Beam": [
        "flange 1 centre / web-flange intersection",
        "flange 1 end 1",
        "flange 1 end 2",
        "flange 2 centre / web-flange intersection",
        "flange 2 end 1",
        "flange 2 end 2",
    ],
    "Angle Bar": [
        "apex / heel",
        "leg end 1",
        "leg end 2",
    ],
    "Sheet Pile": [
        "profile corner 1",
        "profile corner 2",
        "profile corner 3",
        "profile corner 4",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def validate_source_schema(data, split):
    cats = {c["id"]: c for c in data["categories"]}

    names = [c["name"] for c in data["categories"]]

    if names != CLASS_ORDER:
        raise ValueError(
            f"{split}: unexpected category order/names: {names}"
        )

    for cat in data["categories"]:
        name = cat["name"]

        expected = EXPECTED_SOURCE_KEYPOINTS[name]

        if cat.get("keypoints") != expected:
            raise ValueError(
                f"{split}: {name}: source keypoint schema mismatch.\n"
                f"Expected: {expected}\n"
                f"Actual:   {cat.get('keypoints')}"
            )

    return cats


def make_categories(source_categories):
    """
    COCO categories in the adapter use a global six-keypoint schema.

    The category-local semantics are retained as metadata.
    """
    result = []

    for src in source_categories:
        name = src["name"]

        result.append(
            {
                "id": src["id"],
                "name": name,
                "supercategory": src.get(
                    "supercategory",
                    "steel_profile",
                ),
                "keypoints": [
                    "G1",
                    "G2",
                    "G3",
                    "G4",
                    "G5",
                    "G6",
                ],
                "skeleton": [
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                ],
                "class_local_keypoints": EXPECTED_SOURCE_KEYPOINTS[name],
                "class_local_semantics": SEMANTICS[name],
                "global_mapping": {
                    f"K{i+1}": f"G{i+1}"
                    for i in range(len(EXPECTED_SOURCE_KEYPOINTS[name]))
                },
            }
        )

    return result


def convert_annotation(
    ann,
    image,
    category,
    counters,
):
    source_kps = ann.get("keypoints", [])

    expected_count = len(
        EXPECTED_SOURCE_KEYPOINTS[
            category["name"]
        ]
    )

    if len(source_kps) != expected_count * 3:
        raise ValueError(
            f"Annotation {ann['id']} has malformed keypoints: "
            f"{len(source_kps)} values; expected {expected_count * 3}"
        )

    global_kps = []

    original_keypoints = []

    out_image_indices = []
    masked_indices = []

    for i in range(expected_count):
        x = source_kps[i * 3]
        y = source_kps[i * 3 + 1]
        v = source_kps[i * 3 + 2]

        original_keypoints.extend(
            [x, y, v]
        )

        is_out_image = (
            v > 0
            and (
                x < 0
                or x > image["width"]
                or y < 0
                or y > image["height"]
            )
        )

        if is_out_image:
            global_kps.extend([0.0, 0.0, 0])
            out_image_indices.append(i + 1)
            masked_indices.append(i + 1)
            counters["out_of_image_keypoints_masked"] += 1
        else:
            global_kps.extend(
                [x, y, v]
            )

    # Pad unused class-specific slots to global K=6.
    unused_count = 6 - expected_count

    for i in range(unused_count):
        global_kps.extend([0.0, 0.0, 0])

        counters["unused_global_slots"] += 1

    derived = dict(ann)

    # Replace standard COCO keypoints with model-ready K=6 representation.
    derived["keypoints"] = global_kps

    # COCO num_keypoints is the number of labeled keypoints with v > 0.
    derived["num_keypoints"] = sum(
        1
        for i in range(0, len(global_kps), 3)
        if global_kps[i + 2] > 0
    )

    # Preserve frozen source exactly.
    derived["original_keypoints"] = original_keypoints
    derived["original_num_keypoints"] = ann.get(
        "num_keypoints"
    )

    derived["model_ready_adapter"] = {
        "global_k": 6,
        "class_local_k": expected_count,
        "masked_out_of_image": out_image_indices,
        "masked_global_slots": [
            expected_count + i + 1
            for i in range(unused_count)
        ],
        "out_of_bbox_not_masked": True,
    }

    if out_image_indices:
        counters["annotations_with_masked_out_of_image"] += 1

    return derived


def process_split(
    dataset,
    output,
    split,
):
    source_json = (
        dataset
        / "splits"
        / split
        / "annotations.json"
    )

    source_data = load_json(source_json)

    source_categories = validate_source_schema(
        source_data,
        split,
    )

    output_split = (
        output
        / "splits"
        / split
    )

    output_images = (
        output_split
        / "images"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy images exactly.
    copied = 0

    source_images = {
        im["id"]: im
        for im in source_data["images"]
    }

    for im in source_data["images"]:
        src = (
            dataset
            / "splits"
            / split
            / "images"
            / im["file_name"]
        )

        if not src.exists():
            raise FileNotFoundError(src)

        dst = output_images / im["file_name"]
        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(src, dst)
        copied += 1

    counters = Counter()

    annotations = []

    for ann in source_data["annotations"]:
        image = source_images[ann["image_id"]]
        category = source_categories[
            ann["category_id"]
        ]

        converted = convert_annotation(
            ann,
            image,
            category,
            counters,
        )

        annotations.append(converted)

    adapter_data = {
        "info": {
            **source_data.get("info", {}),
            "dataset_name": "InnoCount_KeyPoint_ModelReady_v1",
            "source_dataset": "InnoCount_KeyPoint_Frozen_v2",
            "adapter_version": "1.0",
            "global_keypoint_count": 6,
            "coordinate_units": "source-image pixels",
            "out_of_image_policy": (
                "mask to visibility 0 in derived representation; "
                "original coordinates preserved in original_keypoints"
            ),
        },
        "licenses": source_data.get(
            "licenses",
            [],
        ),
        "images": source_data["images"],
        "annotations": annotations,
        "categories": make_categories(
            source_data["categories"]
        ),
    }

    dump_json(
        output_split
        / "annotations.json",
        adapter_data,
    )

    return {
        "split": split,
        "images": len(source_data["images"]),
        "annotations": len(annotations),
        "copied_images": copied,
        "counters": dict(counters),
    }

def keypoints_equal(a, b, tolerance=1e-9):
    """
    Compare two COCO keypoint vectors robustly.

    Handles:
      - int vs float representation
      - -0.0 vs 0.0
      - floating-point serialization differences
      - NaN values, if present

    The function does NOT modify either vector.
    """

    if a is None or b is None:
        return a is b

    if len(a) != len(b):
        return False

    for i, (x, y) in enumerate(zip(a, b)):

        # Handle NaN explicitly.
        if isinstance(x, float) and math.isnan(x):
            if isinstance(y, float) and math.isnan(y):
                continue
            return False

        if isinstance(y, float) and math.isnan(y):
            return False

        # Numeric comparison.
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(
                float(x),
                float(y),
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                return False
        else:
            if x != y:
                return False

    return True

def validate_output(
    dataset,
    output,
):
    report = {
        "passed": True,
        "splits": {},
        "errors": [],
    }

    for split in SPLITS:
        source = load_json(
            dataset
            / "splits"
            / split
            / "annotations.json"
        )

        derived = load_json(
            output
            / "splits"
            / split
            / "annotations.json"
        )

        split_report = {
            "source_images": len(
                source["images"]
            ),
            "derived_images": len(
                derived["images"]
            ),
            "source_annotations": len(
                source["annotations"]
            ),
            "derived_annotations": len(
                derived["annotations"]
            ),
            "image_ids_identical": False,
            "annotation_ids_identical": False,
            "category_ids_identical": False,
            "class_counts_source": {},
            "class_counts_derived": {},
            "masked_out_of_image": 0,
            "unused_slots": 0,
        }

        source_image_ids = [
            x["id"]
            for x in source["images"]
        ]

        derived_image_ids = [
            x["id"]
            for x in derived["images"]
        ]

        split_report["image_ids_identical"] = (
            source_image_ids
            == derived_image_ids
        )

        source_ann_ids = [
            x["id"]
            for x in source["annotations"]
        ]

        derived_ann_ids = [
            x["id"]
            for x in derived["annotations"]
        ]

        split_report["annotation_ids_identical"] = (
            source_ann_ids
            == derived_ann_ids
        )

        source_cat_ids = [
            x["id"]
            for x in source["categories"]
        ]

        derived_cat_ids = [
            x["id"]
            for x in derived["categories"]
        ]

        split_report["category_ids_identical"] = (
            source_cat_ids
            == derived_cat_ids
        )

        source_cats = {
            c["id"]: c["name"]
            for c in source["categories"]
        }

        source_counts = Counter(
            source_cats[a["category_id"]]
            for a in source["annotations"]
        )

        derived_counts = Counter(
            source_cats[a["category_id"]]
            for a in derived["annotations"]
        )

        split_report["class_counts_source"] = dict(
            sorted(source_counts.items())
        )

        split_report["class_counts_derived"] = dict(
            sorted(derived_counts.items())
        )

        if (
            split_report["source_images"]
            != split_report["derived_images"]
        ):
            report["errors"].append(
                f"{split}: image count changed"
            )

        if (
            split_report["source_annotations"]
            != split_report["derived_annotations"]
        ):
            report["errors"].append(
                f"{split}: annotation count changed"
            )

        if not split_report["image_ids_identical"]:
            report["errors"].append(
                f"{split}: image IDs/order changed"
            )

        if not split_report["annotation_ids_identical"]:
            report["errors"].append(
                f"{split}: annotation IDs/order changed"
            )

        if not split_report["category_ids_identical"]:
            report["errors"].append(
                f"{split}: category IDs changed"
            )

        if (
            split_report["class_counts_source"]
            != split_report["class_counts_derived"]
        ):
            report["errors"].append(
                f"{split}: class object counts changed"
            )

        # Validate every derived annotation.
        for ann in derived["annotations"]:
            kps = ann["keypoints"]

            if len(kps) != 18:
                report["errors"].append(
                    f"{split}: annotation {ann['id']} "
                    f"does not have K=6"
                )
                continue

            for i in range(6):
                x = kps[i * 3]
                y = kps[i * 3 + 1]
                v = kps[i * 3 + 2]

                if v == 0:
                    if x != 0 or y != 0:
                        report["errors"].append(
                            f"{split}: annotation {ann['id']} "
                            f"G{i+1} masked but x/y not zero"
                        )

    # Verify source values are preserved in metadata.
    #
    # IMPORTANT:
    # Annotation IDs are NOT assumed to be globally unique in the
    # source dataset. Therefore, do not construct source_by_id = {...}.
    #
    # The derived annotations are generated in the same order as the
    # source annotations, so validate by positional correspondence.

    source_annotations = source["annotations"]
    derived_annotations = derived["annotations"]

    if len(source_annotations) != len(derived_annotations):
        report["errors"].append(
            f"{split}: source/derived annotation count mismatch: "
            f"{len(source_annotations)} vs "
            f"{len(derived_annotations)}"
        )
    else:

        for source_ann, derived_ann in zip(
            source_annotations,
            derived_annotations,
        ):

            # Check that we are comparing the intended annotation.
            if source_ann.get("id") != derived_ann.get("id"):
                report["errors"].append(
                    f"{split}: annotation ID mismatch: "
                    f"source={source_ann.get('id')} "
                    f"derived={derived_ann.get('id')}"
                )
                continue

            if source_ann.get("image_id") != derived_ann.get(
                "image_id"
            ):
                report["errors"].append(
                    f"{split}: annotation "
                    f"{derived_ann.get('id')} image_id mismatch: "
                    f"source={source_ann.get('image_id')} "
                    f"derived={derived_ann.get('image_id')}"
                )
                continue

            if source_ann.get("category_id") != derived_ann.get(
                "category_id"
            ):
                report["errors"].append(
                    f"{split}: annotation "
                    f"{derived_ann.get('id')} category_id mismatch: "
                    f"source={source_ann.get('category_id')} "
                    f"derived={derived_ann.get('category_id')}"
                )
                continue

            source_keypoints = source_ann.get(
                "keypoints"
            )

            preserved_keypoints = derived_ann.get(
                "original_keypoints"
            )

            if not keypoints_equal(
                preserved_keypoints,
                source_keypoints,
            ):
                differences = []

                for i, (x, y) in enumerate(
                    zip(
                        preserved_keypoints or [],
                        source_keypoints or [],
                    )
                ):
                    if (
                        isinstance(x, (int, float))
                        and isinstance(y, (int, float))
                    ):
                        same = False

                        if (
                            isinstance(x, float)
                            and math.isnan(x)
                            and isinstance(y, float)
                            and math.isnan(y)
                        ):
                            same = True
                        else:
                            same = math.isclose(
                                float(x),
                                float(y),
                                rel_tol=1e-9,
                                abs_tol=1e-9,
                            )
                    else:
                        same = x == y

                    if not same:
                        differences.append(
                            {
                                "index": i,
                                "preserved": x,
                                "source": y,
                            }
                        )

                report["errors"].append(
                    f"{split}: annotation "
                    f"{derived_ann.get('id')} "
                    f"original_keypoints do not match source; "
                    f"differences={differences[:5]}"
                )

        report["splits"][split] = split_report

    report["passed"] = not report["errors"]

    return report


def write_summary(
    output,
    split_reports,
    validation,
):
    lines = []

    lines.append(
        "InnoCount KeyPoint ModelReady v1"
    )
    lines.append(
        "Common Model-Ready Adapter Report"
    )
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "SOURCE: InnoCount_KeyPoint_Frozen_v2"
    )
    lines.append(
        "GLOBAL KEYPOINT COUNT: 6"
    )
    lines.append(
        "OUT-OF-IMAGE POLICY: visibility=0, x=0, y=0 in derived adapter"
    )
    lines.append(
        "ORIGINAL VALUES: preserved in annotation.original_keypoints"
    )
    lines.append(
        "OUT-OF-BBOX POLICY: retained unless also out-of-image"
    )
    lines.append("")
    lines.append("SPLITS")
    lines.append("-" * 70)

    for r in split_reports:
        lines.append(
            f"{r['split']}: "
            f"{r['images']} images, "
            f"{r['annotations']} annotations"
        )

        c = r["counters"]

        lines.append(
            f"  out-of-image keypoints masked: "
            f"{c.get('out_of_image_keypoints_masked', 0)}"
        )

        lines.append(
            f"  annotations affected: "
            f"{c.get('annotations_with_masked_out_of_image', 0)}"
        )

        lines.append(
            f"  unused global slots: "
            f"{c.get('unused_global_slots', 0)}"
        )

    lines.append("")
    lines.append("VALIDATION")
    lines.append("-" * 70)

    if validation["passed"]:
        lines.append("PASS")
    else:
        lines.append("FAIL")
        for error in validation["errors"]:
            lines.append(
                f"  - {error}"
            )

    lines.append("")
    lines.append(
        "No source image, source annotation, split assignment, "
        "class ID, bbox, or source keypoint was modified."
    )

    (output / "reports").mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        output
        / "reports"
        / "model_ready_adapter_report.txt"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    dump_json(
        output
        / "reports"
        / "model_ready_adapter_report.json",
        {
            "split_reports": split_reports,
            "validation": validation,
        },
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Refusing to overwrite an existing model-ready dataset."
        )

    print(
        "InnoCount KeyPoint -> ModelReady Adapter v1"
    )
    print("=" * 70)

    print(
        "\n[1/4] Creating derived dataset..."
    )

    split_reports = []

    for split in SPLITS:
        print(
            f"  Processing {split}..."
        )

        split_reports.append(
            process_split(
                args.dataset,
                args.output,
                split,
            )
        )

    print(
        "\n[2/4] Validating adapter..."
    )

    validation = validate_output(
        args.dataset,
        args.output,
    )

    print(
        "\n[3/4] Writing reports..."
    )

    write_summary(
        args.output,
        split_reports,
        validation,
    )

    print(
        "\n[4/4] Result"
    )
    print("=" * 70)

    if validation["passed"]:
        print(
            "PASS: Model-ready adapter validation passed."
        )
    else:
        print(
            "FAIL: Model-ready adapter validation failed."
        )

        for error in validation["errors"]:
            print(
                f"  - {error}"
            )

        raise SystemExit(1)

    print(
        f"\nOutput: {args.output}"
    )

    print(
        "\nNext step: inspect "
        f"{args.output}/reports/model_ready_adapter_report.txt"
    )


if __name__ == "__main__":
    main()
