#!/usr/bin/env python3
"""
Convert InnoCount KeyPoint ModelReady v2 -> ViTPose COCO format.

This adapter is a format conversion only.

Source:
    datasets/InnoCount_KeyPoint_ModelReady_v2

Destination:
    datasets/ViTPose_COCO

The ModelReady v2 split assignments, annotations, categories,
keypoints, and images are preserved. No augmentation, resizing,
clipping, class balancing, or split changes are performed here.
"""

from pathlib import Path
import json
import shutil


ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "datasets" / "InnoCount_KeyPoint_ModelReady_v2"
DST = ROOT / "datasets" / "ViTPose_COCO"

SPLITS = ["train", "valid", "test"]


def convert(split):

    src_split = SRC / "splits" / split
    dst_images = DST / "images" / split
    dst_annotations = DST / "annotations"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_annotations.mkdir(parents=True, exist_ok=True)

    # Copy images without modification.
    for img in (src_split / "images").glob("*"):
        if img.is_file():
            shutil.copy2(img, dst_images / img.name)

    # Copy the ModelReady COCO annotation structure unchanged.
    with open(
        src_split / "annotations.json",
        "r",
        encoding="utf-8",
    ) as f:
        annotations = json.load(f)

    with open(
        dst_annotations / f"{split}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            annotations,
            f,
            indent=2,
        )

    print(
        f"{split}: "
        f"{len(annotations['images'])} images, "
        f"{len(annotations['annotations'])} annotations"
    )


def main():

    if not SRC.exists():
        raise FileNotFoundError(
            f"ModelReady v2 dataset not found: {SRC}"
        )

    if DST.exists():
        shutil.rmtree(DST)

    for split in SPLITS:
        convert(split)

    print()
    print("ViTPose COCO dataset ready.")
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")


if __name__ == "__main__":
    main()