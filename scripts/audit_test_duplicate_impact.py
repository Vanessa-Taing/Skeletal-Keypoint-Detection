#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("/home/owner/skeletal_keypoint_detection")

ANN_FILE = (
    ROOT
    / "datasets/InnoCount_KeyPoint_ModelReady_v2"
    / "splits/test/annotations.json"
)


def signature(ann):
    return (
        ann["category_id"],
        tuple(ann.get("bbox", [])),
        tuple(ann.get("keypoints", [])),
    )


with open(ANN_FILE) as f:
    data = json.load(f)


images = {
    x["id"]: x
    for x in data["images"]
}


groups = []

for image_id in sorted(images):

    anns = [
        a for a in data["annotations"]
        if a["image_id"] == image_id
    ]

    by_sig = defaultdict(list)

    for ann in anns:
        by_sig[signature(ann)].append(ann)

    for sig, dupes in by_sig.items():

        if len(dupes) < 2:
            continue

        groups.append({
            "image": images[image_id]["file_name"],
            "image_id": image_id,
            "category_id": dupes[0]["category_id"],
            "annotation_ids": [
                a["id"] for a in dupes
            ],
            "count": len(dupes),
        })


print("=" * 70)
print("FINAL TEST DUPLICATE IMPACT AUDIT")
print("=" * 70)

print()
print(f"Total test annotations : {len(data['annotations'])}")
print(f"Duplicate groups       : {len(groups)}")
print(
    "Redundant rows         : "
    f"{sum(g['count'] - 1 for g in groups)}"
)
print(
    "Unique annotations     : "
    f"{len(data['annotations']) - sum(g['count'] - 1 for g in groups)}"
)

print()
print("DUPLICATE GROUPS")
print("-" * 70)

class_names = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}

for g in groups:

    print(
        f"{g['image']:12s} "
        f"{class_names[g['category_id']]:12s} "
        f"IDs={g['annotation_ids']} "
        f"count={g['count']}"
    )

print()
print("BY PROFILE")
print("-" * 70)

profile_counts = defaultdict(int)

for g in groups:
    profile_counts[
        class_names[g["category_id"]]
    ] += g["count"] - 1

for name in [
    "Channel",
    "T-beam",
    "HI-Beam",
    "Angle Bar",
    "Sheet Pile",
]:

    print(
        f"{name:12s}: "
        f"{profile_counts[name]} redundant rows"
    )
