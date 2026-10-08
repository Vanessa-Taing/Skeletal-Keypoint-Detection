#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path


ROOT = Path("/home/owner/skeletal_keypoint_detection")

TARGET = "000616.jpg"

FROZEN_ROOT = ROOT / "datasets/InnoCount_KeyPoint_Frozen_v3_Optimized"

V4_ROOT = Path("/tmp/InnoCount_v4")

V4_JSONS = {
    "train": Path("/tmp/v4_train.json"),
    "valid": Path("/tmp/v4_valid.json"),
    "test": Path("/tmp/v4_test.json"),
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


print("=" * 80)
print("000616.jpg PROVENANCE AUDIT")
print("=" * 80)


# ----------------------------------------------------------------------
# Find frozen image
# ----------------------------------------------------------------------

frozen_matches = []

for split in ["train", "valid", "test"]:

    image_path = (
        FROZEN_ROOT
        / "splits"
        / split
        / "images"
        / TARGET
    )

    if image_path.exists():

        coco = load_json(
            FROZEN_ROOT
            / "splits"
            / split
            / "annotations.json"
        )

        image = next(
            img for img in coco["images"]
            if img["file_name"] == TARGET
        )

        annotations = [
            a for a in coco["annotations"]
            if a["image_id"] == image["id"]
        ]

        frozen_matches.append({
            "split": split,
            "path": image_path,
            "image": image,
            "annotations": annotations,
            "md5": md5(image_path),
        })


if not frozen_matches:
    raise RuntimeError("000616.jpg not found in Frozen-v3 Optimized")


# ----------------------------------------------------------------------
# Print frozen information
# ----------------------------------------------------------------------

for frozen in frozen_matches:

    print("\n" + "-" * 80)
    print("FROZEN DATA")
    print("-" * 80)

    print("Split :", frozen["split"])
    print("Image :", frozen["image"])
    print("MD5   :", frozen["md5"])

    for ann in frozen["annotations"]:

        if ann["category_id"] == 3:  # Angle Bar
            print("\nAngle Bar annotation:")
            print(json.dumps(ann, indent=2))


# ----------------------------------------------------------------------
# Find same image in v4 by MD5
# ----------------------------------------------------------------------

target_md5 = frozen_matches[0]["md5"]

print("\n" + "=" * 80)
print("SEARCHING ORIGINAL V4")
print("=" * 80)

for split, json_path in V4_JSONS.items():

    coco = load_json(json_path)

    anns_by_image = {}

    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in coco["images"]:

        filename = Path(img["file_name"]).name
        path = V4_ROOT / split / filename

        if not path.exists():
            continue

        if md5(path) != target_md5:
            continue

        print("\nFOUND V4 MATCH")
        print("Split    :", split)
        print("Image ID :", img["id"])
        print("Filename :", filename)
        print("Size     :", img["width"], "x", img["height"])
        print("MD5      :", target_md5)

        print("\nALL ORIGINAL V4 ANNOTATIONS:")

        for ann in anns_by_image.get(img["id"], []):

            if ann["category_id"] == 3:

                print("\nAngle Bar:")
                print(json.dumps(ann, indent=2))

                kp = ann["keypoints"]

                print("\nDecoded keypoints:")

                for i in range(0, len(kp), 3):
                    print(
                        f"  local K{i//3 + 1}: "
                        f"x={kp[i]}, "
                        f"y={kp[i+1]}, "
                        f"v={kp[i+2]}"
                    )
