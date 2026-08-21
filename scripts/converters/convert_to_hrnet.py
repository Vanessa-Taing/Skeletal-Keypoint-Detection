#!/usr/bin/env python3
"""
Convert InnoCount Frozen V2 -> HRNet COCO format

Output:
datasets/HRNet_COCO/
    annotations/
        train.json
        valid.json
        test.json
    images/
        train/
        valid/
        test/
"""

from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "datasets" / "InnoCount_KeyPoint_Frozen_v2"
DST = ROOT / "datasets" / "HRNet_COCO"

SPLITS = ["train", "valid", "test"]


def convert_split(split):

    src_img = SRC / "splits" / split / "images"
    src_json = SRC / "splits" / split / "annotations.json"

    dst_img = DST / "images" / split
    dst_ann = DST / "annotations"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)

    # Copy images
    for img in src_img.glob("*"):
        shutil.copy2(img, dst_img / img.name)

    with open(src_json) as f:
        coco = json.load(f)

    # HRNet accepts standard COCO keypoint format
    with open(dst_ann / f"{split}.json", "w") as f:
        json.dump(coco, f, indent=2)

    print(f"{split}: {len(coco['images'])} images")


def main():

    if DST.exists():
        shutil.rmtree(DST)

    for s in SPLITS:
        convert_split(s)

    print("\nHRNet dataset created:")
    print(DST)


if __name__ == "__main__":
    main()