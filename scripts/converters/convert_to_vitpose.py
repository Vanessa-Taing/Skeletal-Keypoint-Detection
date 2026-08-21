#!/usr/bin/env python3
"""
Convert InnoCount Frozen V2 -> ViTPose COCO format
"""

from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "datasets" / "InnoCount_KeyPoint_Frozen_v2"
DST = ROOT / "datasets" / "ViTPose_COCO"

SPLITS = ["train", "valid", "test"]


def convert(split):

    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "annotations").mkdir(parents=True, exist_ok=True)

    # copy images
    for img in (SRC / "splits" / split / "images").glob("*"):
        shutil.copy2(img, DST / "images" / split / img.name)

    with open(SRC / "splits" / split / "annotations.json") as f:
        ann = json.load(f)

    with open(DST / "annotations" / f"{split}.json", "w") as f:
        json.dump(ann, f, indent=2)

    print(split, "done")


def main():

    if DST.exists():
        shutil.rmtree(DST)

    for s in SPLITS:
        convert(s)

    print("ViTPose dataset ready.")


if __name__ == "__main__":
    main()