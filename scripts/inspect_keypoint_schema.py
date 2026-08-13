#!/usr/bin/env python3

"""
Inspect the keypoint schema of the frozen InnoCount COCO dataset.

This script is READ-ONLY.
It does not modify images or annotations.

It reports:
    - COCO categories
    - final class IDs
    - keypoint names
    - number of keypoints per class
    - skeleton connections
    - annotation keypoint lengths
    - visibility distribution
    - missing / malformed keypoints
    - bounding-box/keypoint consistency
    - keypoint coordinate ranges

Usage:

    python inspect_keypoint_schema.py \
        --dataset "InnoCount_KeyPoint_Frozen_v1"

Or directly against a COCO JSON:

    python inspect_keypoint_schema.py \
        --json "InnoCount_KeyPoint_Frozen_v1/splits/train/annotations.json"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_line(char="=", n=80):
    print(char * n)


def get_visibility_name(v):
    return {
        0: "not labeled",
        1: "labeled but not visible / occluded",
        2: "visible",
    }.get(v, f"unknown ({v})")


# ------------------------------------------------------------
# Category inspection
# ------------------------------------------------------------

def inspect_categories(categories):
    print_line()
    print("CATEGORY / KEYPOINT SCHEMA")
    print_line()

    for category in sorted(categories, key=lambda x: x["id"]):
        cid = category["id"]
        name = category["name"]

        keypoints = category.get("keypoints", [])
        skeleton = category.get("skeleton", [])

        print(f"\nClass ID       : {cid}")
        print(f"Class name     : {name}")
        print(f"Keypoint count : {len(keypoints)}")

        print("\nKeypoints:")

        if keypoints:
            for i, kp in enumerate(keypoints, start=1):
                print(f"  K{i:<3} = {kp}")
        else:
            print("  NONE")

        print("\nSkeleton:")

        if skeleton:
            for connection in skeleton:
                if len(connection) == 2:
                    a, b = connection
                    print(f"  K{a} -- K{b}")
                else:
                    print(f"  {connection}")
        else:
            print("  NONE")

        print()


# ------------------------------------------------------------
# Annotation inspection
# ------------------------------------------------------------

def inspect_annotations(data):
    categories = {
        c["id"]: c
        for c in data["categories"]
    }

    images = {
        img["id"]: img
        for img in data["images"]
    }

    annotations = data["annotations"]

    print_line()
    print("ANNOTATION STATISTICS")
    print_line()

    print(f"Images       : {len(images)}")
    print(f"Annotations  : {len(annotations)}")

    per_class = Counter()
    keypoints_per_class = Counter()
    visibility_per_class = defaultdict(Counter)

    malformed = []
    bbox_outside = []
    image_outside = []

    for ann in annotations:

        cid = ann["category_id"]
        category = categories[cid]
        class_name = category["name"]

        per_class[class_name] += 1

        expected_k = len(
            category.get("keypoints", [])
        )

        keypoints = ann.get("keypoints", [])

        actual_k = len(keypoints) // 3

        keypoints_per_class[class_name] += actual_k

        # ----------------------------------------------------
        # Check keypoint vector length
        # ----------------------------------------------------

        if len(keypoints) != expected_k * 3:
            malformed.append({
                "annotation_id": ann.get("id"),
                "class": class_name,
                "expected": expected_k * 3,
                "actual": len(keypoints),
            })

        # ----------------------------------------------------
        # Visibility
        # ----------------------------------------------------

        image = images[ann["image_id"]]
        width = image["width"]
        height = image["height"]

        bbox = ann.get("bbox")

        if bbox:
            bx, by, bw, bh = bbox
        else:
            bx = by = bw = bh = None

        for i in range(actual_k):

            x = keypoints[i * 3]
            y = keypoints[i * 3 + 1]
            v = keypoints[i * 3 + 2]

            visibility_per_class[class_name][v] += 1

            if v > 0:

                # --------------------------------------------
                # Outside image
                # --------------------------------------------

                if (
                    x < 0
                    or x > width
                    or y < 0
                    or y > height
                ):
                    image_outside.append({
                        "annotation_id": ann.get("id"),
                        "class": class_name,
                        "keypoint": i + 1,
                        "x": x,
                        "y": y,
                        "image_width": width,
                        "image_height": height,
                    })

                # --------------------------------------------
                # Outside bbox
                # --------------------------------------------

                if bbox:
                    if not (
                        bx <= x <= bx + bw
                        and
                        by <= y <= by + bh
                    ):
                        bbox_outside.append({
                            "annotation_id": ann.get("id"),
                            "class": class_name,
                            "keypoint": i + 1,
                            "x": x,
                            "y": y,
                            "bbox": bbox,
                        })

    # --------------------------------------------------------
    # Print class statistics
    # --------------------------------------------------------

    print("\nObjects per class:")

    for name in sorted(per_class):
        print(
            f"  {name:<15}"
            f" {per_class[name]:>6} objects"
            f" | {keypoints_per_class[name]:>7} keypoints"
        )

    # --------------------------------------------------------
    # Visibility
    # --------------------------------------------------------

    print("\nVisibility by class:")

    for name in sorted(visibility_per_class):

        counter = visibility_per_class[name]

        print(f"\n  {name}")

        for v in sorted(counter):
            print(
                f"    {v} ({get_visibility_name(v)}): "
                f"{counter[v]}"
            )

    # --------------------------------------------------------
    # Problems
    # --------------------------------------------------------

    print_line()
    print("ANNOTATION QA")
    print_line()

    print(
        f"Malformed keypoint vectors : "
        f"{len(malformed)}"
    )

    print(
        f"Keypoints outside image   : "
        f"{len(image_outside)}"
    )

    print(
        f"Keypoints outside bbox    : "
        f"{len(bbox_outside)}"
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    if malformed:

        print("\nExamples of malformed annotations:")

        for x in malformed[:10]:
            print(
                f"  annotation={x['annotation_id']} "
                f"class={x['class']} "
                f"expected={x['expected']} "
                f"actual={x['actual']}"
            )

    if image_outside:

        print("\nExamples of keypoints outside image:")

        for x in image_outside[:10]:
            print(
                f"  annotation={x['annotation_id']} "
                f"class={x['class']} "
                f"K{x['keypoint']} "
                f"({x['x']}, {x['y']}) "
                f"image={x['image_width']}x{x['image_height']}"
            )

    if bbox_outside:

        print("\nExamples of keypoints outside bbox:")

        for x in bbox_outside[:10]:
            print(
                f"  annotation={x['annotation_id']} "
                f"class={x['class']} "
                f"K{x['keypoint']} "
                f"({x['x']}, {x['y']}) "
                f"bbox={x['bbox']}"
            )


# ------------------------------------------------------------
# Compare train / valid / test schemas
# ------------------------------------------------------------

def inspect_all_splits(dataset_dir):

    print_line()
    print("SPLIT CONSISTENCY CHECK")
    print_line()

    split_categories = {}

    for split in ("train", "valid", "test"):

        json_path = (
            dataset_dir
            / "splits"
            / split
            / "annotations.json"
        )

        if not json_path.exists():
            print(
                f"WARNING: {split} annotation file "
                f"not found: {json_path}"
            )
            continue

        data = load_json(json_path)

        categories = {
            c["id"]: {
                "name": c["name"],
                "keypoints": c.get("keypoints", []),
                "skeleton": c.get("skeleton", []),
            }
            for c in data["categories"]
        }

        split_categories[split] = categories

        print(
            f"{split:<6}: "
            f"{len(data['images'])} images, "
            f"{len(data['annotations'])} annotations"
        )

    print()

    if len(split_categories) < 3:
        print("Not all three splits were found.")
        return

    reference = split_categories["train"]

    consistent = True

    for split in ("valid", "test"):

        if split not in split_categories:
            continue

        current = split_categories[split]

        if current != reference:
            consistent = False

            print(
                f"WARNING: {split} schema differs "
                f"from train."
            )

    if consistent:
        print(
            "PASS: train / valid / test have "
            "identical category/keypoint schemas."
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Frozen dataset directory",
    )

    parser.add_argument(
        "--json",
        type=Path,
        help="Direct path to one COCO annotations.json",
    )

    args = parser.parse_args()

    if not args.dataset and not args.json:
        parser.error(
            "Provide either --dataset or --json"
        )

    if args.json:

        if not args.json.exists():
            raise FileNotFoundError(args.json)

        data = load_json(args.json)

        inspect_categories(
            data["categories"]
        )

        inspect_annotations(data)

        return

    dataset = args.dataset

    if not dataset.exists():
        raise FileNotFoundError(dataset)

    # First compare all three splits.
    inspect_all_splits(dataset)

    # Then inspect train schema/annotations in detail.
    train_json = (
        dataset
        / "splits"
        / "train"
        / "annotations.json"
    )

    if not train_json.exists():
        raise FileNotFoundError(train_json)

    data = load_json(train_json)

    inspect_categories(
        data["categories"]
    )

    inspect_annotations(data)


if __name__ == "__main__":
    main()