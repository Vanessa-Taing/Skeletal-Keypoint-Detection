#!/usr/bin/env python3

import json
import hashlib
from pathlib import Path
from collections import defaultdict


ROOT = Path("/home/owner/skeletal_keypoint_detection")

V4_ROOT = Path("/tmp/InnoCount_v4")

V4_JSONS = {
    "train": Path("/tmp/v4_train.json"),
    "valid": Path("/tmp/v4_valid.json"),
    "test": Path("/tmp/v4_test.json"),
}

FROZEN_ROOT = (
    ROOT / "datasets/InnoCount_KeyPoint_Frozen_v3_Optimized"
)


# v4 category ID -> canonical/Frozen category ID
V4_TO_FROZEN = {
    1: 3,  # Angle Bar
    2: 0,  # Channel
    3: 2,  # HI-Beam
    4: 4,  # Sheet Pile
    5: 1,  # T-beam
}


def md5(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def ann_signature_frozen(ann):
    """
    Signature for Frozen annotation.
    Compare only source-semantic annotation content:
      category
      bbox
      keypoints
    """

    return (
        ann["category_id"],
        tuple(ann.get("bbox", [])),
        tuple(ann.get("keypoints", [])),
    )


def ann_signature_v4_as_frozen(ann):
    """
    Convert v4 category ID to Frozen category ID
    before comparison.
    """

    v4_category = ann["category_id"]

    if v4_category not in V4_TO_FROZEN:
        return None

    return (
        V4_TO_FROZEN[v4_category],
        tuple(ann.get("bbox", [])),
        tuple(ann.get("keypoints", [])),
    )


print("=" * 80)
print("CORRECTED DUPLICATE ANNOTATION PROVENANCE AUDIT")
print("=" * 80)


# ============================================================================
# Index v4 images by MD5
# ============================================================================

print("\n[1] Indexing original v4 images...")

v4_by_md5 = defaultdict(list)

for split, json_path in V4_JSONS.items():

    coco = json.load(open(json_path))

    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for img in coco["images"]:

        filename = Path(img["file_name"]).name
        image_path = V4_ROOT / split / filename

        if not image_path.exists():
            continue

        image_hash = md5(image_path)

        v4_by_md5[image_hash].append({
            "split": split,
            "image_id": img["id"],
            "file_name": filename,
            "width": img["width"],
            "height": img["height"],
            "annotations": anns_by_image[img["id"]],
        })

print(f"Unique v4 image hashes: {len(v4_by_md5)}")


# ============================================================================
# Find duplicates in Frozen
# ============================================================================

print("\n[2] Finding exact duplicate annotation groups in Frozen...")

duplicate_groups = []

for split in ["train", "valid", "test"]:

    json_path = (
        FROZEN_ROOT
        / "splits"
        / split
        / "annotations.json"
    )

    image_dir = (
        FROZEN_ROOT
        / "splits"
        / split
        / "images"
    )

    coco = json.load(open(json_path))

    images = {
        img["id"]: img
        for img in coco["images"]
    }

    anns_by_image = defaultdict(list)

    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for image_id, anns in anns_by_image.items():

        groups = defaultdict(list)

        for ann in anns:
            groups[
                ann_signature_frozen(ann)
            ].append(ann)

        for signature, group in groups.items():

            if len(group) < 2:
                continue

            img = images[image_id]
            image_path = image_dir / img["file_name"]

            if not image_path.exists():
                continue

            duplicate_groups.append({
                "frozen_split": split,
                "frozen_image_id": image_id,
                "frozen_file_name": img["file_name"],
                "image_md5": md5(image_path),
                "category_id": group[0]["category_id"],
                "annotation_ids": [
                    a["id"] for a in group
                ],
                "duplicate_count": len(group),
                "annotations": group,
            })


print(
    f"Duplicate groups found: "
    f"{len(duplicate_groups)}"
)

print(
    "Redundant rows: "
    f"{sum(g['duplicate_count'] - 1 for g in duplicate_groups)}"
)


# ============================================================================
# Trace each duplicate to v4
# ============================================================================

print("\n[3] Tracing duplicates to v4...")

source_duplicate = 0
source_not_duplicate = 0
source_missing = 0

for group in duplicate_groups:

    matches = v4_by_md5.get(
        group["image_md5"],
        []
    )

    group["v4_matches"] = []

    if not matches:

        group["provenance"] = "NOT_FOUND_IN_V4"
        source_missing += 1
        continue

    frozen_signature = ann_signature_frozen(
        group["annotations"][0]
    )

    for match in matches:

        matching = []

        for ann in match["annotations"]:

            sig = ann_signature_v4_as_frozen(ann)

            if sig == frozen_signature:
                matching.append(ann)

        group["v4_matches"].append({
            "split": match["split"],
            "image_id": match["image_id"],
            "file_name": match["file_name"],
            "matching_annotation_ids": [
                a["id"] for a in matching
            ],
            "matching_annotation_count": len(matching),
        })

    if any(
        m["matching_annotation_count"]
        >= group["duplicate_count"]
        for m in group["v4_matches"]
    ):

        group["provenance"] = "SOURCE_DUPLICATE"
        source_duplicate += 1

    else:

        group["provenance"] = (
            "NOT_DUPLICATED_IN_SOURCE"
        )
        source_not_duplicate += 1


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"Duplicate groups              : "
    f"{len(duplicate_groups)}"
)

print(
    f"Redundant duplicate rows      : "
    f"{sum(g['duplicate_count'] - 1 for g in duplicate_groups)}"
)

print(
    f"Source duplicates             : "
    f"{source_duplicate}"
)

print(
    f"Not duplicated in source      : "
    f"{source_not_duplicate}"
)

print(
    f"Not found in v4               : "
    f"{source_missing}"
)


# ============================================================================
# Detailed output
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED PROVENANCE")
print("=" * 80)

for i, group in enumerate(duplicate_groups, 1):

    print(f"\n[{i}]")
    print(
        f"{group['frozen_split']:5s} "
        f"{group['frozen_file_name']:12s} "
        f"IDs={group['annotation_ids']} "
        f"category={group['category_id']} "
        f"=> {group['provenance']}"
    )

    for match in group["v4_matches"]:

        print(
            f"    v4/{match['split']} "
            f"image_id={match['image_id']} "
            f"IDs={match['matching_annotation_ids']}"
        )


# ============================================================================
# Save JSON
# ============================================================================

output = (
    ROOT
    / "experiments"
    / "duplicate_provenance_audit_corrected.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output, "w") as f:

    json.dump(
        {
            "summary": {
                "duplicate_groups":
                    len(duplicate_groups),

                "redundant_duplicate_rows":
                    sum(
                        g["duplicate_count"] - 1
                        for g in duplicate_groups
                    ),

                "source_duplicates":
                    source_duplicate,

                "not_duplicated_in_source":
                    source_not_duplicate,

                "not_found_in_v4":
                    source_missing,
            },

            "groups":
                duplicate_groups,
        },
        f,
        indent=2,
    )


print("\nReport:")
print(output)