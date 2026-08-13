#!/usr/bin/env python3
"""
create_frozen_dataset.py

RAW ROBOFLOW ZIP -> PROVENANCE-AWARE FROZEN DATASET

This is the canonical dataset-engineering pipeline for the InnoCount
KeyPoint COCO export.

Input:
    InnoCount KeyPoint.v3i.coco.zip

Output:
    InnoCount_KeyPoint_Frozen_v1/

Pipeline:
    1. Inspect the original COCO export
    2. Merge train/valid/test into one master pool
    3. Verify the five active classes
    4. Analyse image / annotation / keypoint properties
    5. Detect exact duplicates
    6. Group related images using COCO image.extra.name
    7. Canonicalize exact duplicate image files
    8. Create a deterministic source-aware ~80/10/10 split
    9. Validate the split
   10. Freeze/export the dataset
   11. Write reports and a lock manifest

IMPORTANT:
- The original ZIP is never modified.
- The 26 genuine negative/background images are retained.
- Validation/test are never augmented.
- The five final classes are:
      0 Channel
      1 T-beam
      2 HI-Beam
      3 Angle Bar
      4 Sheet Pile

REPRODUCIBILITY:
This script is intentionally self-contained and starts from the original
Roboflow ZIP. Once a frozen output has been produced, its master_manifest.csv
is the authoritative lockfile for that particular frozen split.

The exact historical Frozen-v1 split generated in the earlier interactive
session should be regarded as a released artifact. A fresh run of this
script is deterministic, but because the original session did not preserve
every intermediate candidate split / tie-break state, a fresh run should
not be claimed to reproduce the historical split byte-for-byte unless its
manifest is used as the lockfile.

Dependencies:
    pip install pandas numpy pillow scikit-learn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEED = 19

FINAL_CLASSES = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}

# Original Roboflow COCO category IDs -> final experimental IDs
CATEGORY_MAP = {
    2: 0,  # Channel
    5: 1,  # T-beam
    3: 2,  # HI-Beam
    1: 3,  # Angle Bar
    4: 4,  # Sheet Pile
}

SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------------

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_")


def ensure_clean_output(path: Path):
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists: {path}\n"
            "Refusing to overwrite an existing frozen dataset."
        )


# ---------------------------------------------------------------------------
# 1. EXTRACT + INSPECT
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, work_dir: Path) -> Path:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        z.extractall(work_dir)

    return work_dir


def load_coco_records(root: Path):
    """
    Load all Roboflow image records from train/valid/test.

    The original split is retained only as provenance. It is NOT used as the
    final experimental split.
    """
    records = []
    annotations_by_key = defaultdict(list)

    for split in SPLITS:
        json_path = root / split / "_annotations.coco.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing COCO file: {json_path}"
            )

        data = json.loads(json_path.read_text(encoding="utf-8"))

        categories = {c["id"]: c for c in data["categories"]}

        for ann in data["annotations"]:
            annotations_by_key[(split, ann["image_id"])].append(ann)

        for image in data["images"]:
            image_id = image["id"]
            image_path = root / split / image["file_name"]

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            extra = image.get("extra") or {}
            source_name = extra.get("name")

            records.append(
                {
                    "orig_split": split,
                    "orig_image_id": image_id,
                    "file_name": image["file_name"],
                    "source_name": source_name,
                    "width_json": image["width"],
                    "height_json": image["height"],
                    "annotations": annotations_by_key[
                        (split, image_id)
                    ],
                    "categories": categories,
                    "file_path": image_path,
                }
            )

    return records


def inspect_records(records):
    """
    Basic integrity inspection. Returns a JSON-serializable report.
    """
    report = {
        "original_image_records": len(records),
        "original_split_counts": Counter(
            r["orig_split"] for r in records
        ),
        "missing_source_name": sum(
            r["source_name"] in (None, "") for r in records
        ),
        "images_with_annotations": sum(
            len(r["annotations"]) > 0 for r in records
        ),
        "negative_images": sum(
            len(r["annotations"]) == 0 for r in records
        ),
    }

    dimensions_ok = 0
    dimensions_mismatch = 0

    for r in records:
        try:
            with Image.open(r["file_path"]) as im:
                actual_w, actual_h = im.size
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open image {r['file_path']}: {exc}"
            )

        if (
            actual_w == r["width_json"]
            and actual_h == r["height_json"]
        ):
            dimensions_ok += 1
        else:
            dimensions_mismatch += 1

    report["dimensions_match"] = dimensions_ok
    report["dimensions_mismatch"] = dimensions_mismatch

    return report


# ---------------------------------------------------------------------------
# 2. CATEGORY VERIFICATION
# ---------------------------------------------------------------------------

def verify_categories(records):
    """
    Verify the active exported categories.

    'cargo' is allowed to exist as an unused category because the supplied
    Roboflow export contains it with zero active annotations.
    """
    observed_ids = set()
    observed_names = set()
    annotation_category_counts = Counter()

    for r in records:
        for ann in r["annotations"]:
            cid = ann["category_id"]
            observed_ids.add(cid)

            if cid not in r["categories"]:
                raise RuntimeError(
                    f"Annotation references unknown category ID {cid}"
                )

            name = r["categories"][cid]["name"]
            observed_names.add(name)
            annotation_category_counts[cid] += 1

    active_original_ids = {
        cid for cid, n in annotation_category_counts.items() if n > 0
    }

    if active_original_ids != set(CATEGORY_MAP):
        raise RuntimeError(
            "Unexpected active category IDs.\n"
            f"Expected: {sorted(CATEGORY_MAP)}\n"
            f"Observed: {sorted(active_original_ids)}"
        )

    return {
        "observed_category_ids": sorted(observed_ids),
        "observed_category_names": sorted(observed_names),
        "active_original_category_ids": sorted(active_original_ids),
        "category_counts": dict(annotation_category_counts),
        "final_mapping": {
            str(k): v for k, v in FINAL_CLASSES.items()
        },
        "original_to_final": {
            str(k): v for k, v in CATEGORY_MAP.items()
        },
    }


def remap_annotations(record):
    """
    Convert active Roboflow category IDs to final 0..4 IDs.

    Bboxes, keypoints, visibility values, segmentation and all other COCO
    annotation fields are preserved unchanged.
    """
    output = []

    for ann in record["annotations"]:
        old_id = ann["category_id"]

        # Ignore unused categories such as exported 'cargo'.
        if old_id not in CATEGORY_MAP:
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = CATEGORY_MAP[old_id]
        output.append(new_ann)

    return output


# ---------------------------------------------------------------------------
# 3. ANNOTATION / IMAGE ANALYSIS
# ---------------------------------------------------------------------------

def analyse_annotation(ann, width, height):
    """
    Calculate useful QA information without changing the annotation.
    """
    result = {
        "has_keypoints": "keypoints" in ann,
        "num_keypoints": int(ann.get("num_keypoints", 0)),
        "visible_keypoints": 0,
        "occluded_keypoints": 0,
        "not_labeled_keypoints": 0,
        "out_of_image_keypoints": 0,
        "out_of_bbox_keypoints": 0,
    }

    kp = ann.get("keypoints", [])

    if not kp:
        return result

    bbox = ann.get("bbox")
    bx, by, bw, bh = bbox if bbox else (None, None, None, None)

    for i in range(0, len(kp), 3):
        x, y, v = kp[i:i + 3]

        if v == 2:
            result["visible_keypoints"] += 1
        elif v == 1:
            result["occluded_keypoints"] += 1
        elif v == 0:
            result["not_labeled_keypoints"] += 1

        # Only coordinates with v > 0 are meaningful.
        if v > 0:
            if x < 0 or x > width or y < 0 or y > height:
                result["out_of_image_keypoints"] += 1

            if bbox is not None:
                if not (
                    bx <= x <= bx + bw
                    and by <= y <= by + bh
                ):
                    result["out_of_bbox_keypoints"] += 1

    return result


def build_master_records(records):
    """
    Build a flat master representation before exact-duplicate canonicalization.
    """
    master = []

    for r in records:
        anns = remap_annotations(r)

        class_counts = Counter(
            ann["category_id"] for ann in anns
        )

        keypoint_stats = Counter()

        for ann in anns:
            stats = analyse_annotation(
                ann,
                r["width_json"],
                r["height_json"],
            )

            for key, value in stats.items():
                if isinstance(value, bool):
                    keypoint_stats[key] += int(value)
                else:
                    keypoint_stats[key] += int(value)

        # Confirm actual image dimensions.
        with Image.open(r["file_path"]) as im:
            actual_w, actual_h = im.size

        master.append(
            {
                "orig_split": r["orig_split"],
                "orig_image_id": r["orig_image_id"],
                "file_name": r["file_name"],
                "source_name": r["source_name"],
                "source_group": (
                    r["source_name"]
                    if r["source_name"]
                    else None
                ),
                "width": actual_w,
                "height": actual_h,
                "annotations": anns,
                "annotation_count": len(anns),
                "class_counts": dict(class_counts),
                "negative": len(anns) == 0,
                "has_keypoints": all(
                    "keypoints" in ann for ann in anns
                ) if anns else True,
                "keypoint_stats": dict(keypoint_stats),
                "file_path": r["file_path"],
            }
        )

    return master


# ---------------------------------------------------------------------------
# 4. EXACT DUPLICATES
# ---------------------------------------------------------------------------

def calculate_hashes(master):
    for row in master:
        row["md5"] = md5_file(row["file_path"])
        row["sha256"] = sha256_file(row["file_path"])


def canonicalize_exact_duplicates(master):
    """
    Collapse exact byte-identical image files into one canonical record.

    If duplicate files contain different annotation records, retain the
    record with the largest number of active annotations.

    Every discarded record is preserved in duplicate_groups.csv.
    """
    by_hash = defaultdict(list)

    for row in master:
        by_hash[row["md5"]].append(row)

    canonical = []
    duplicate_rows = []

    split_order = {"train": 0, "valid": 1, "test": 2}

    for md5, group in sorted(by_hash.items()):
        selected = sorted(
            group,
            key=lambda r: (
                -r["annotation_count"],
                split_order[r["orig_split"]],
                r["orig_image_id"],
                r["file_name"],
            ),
        )[0]

        canonical.append(selected)

        for r in group:
            duplicate_rows.append(
                {
                    "md5": md5,
                    "sha256": r["sha256"],
                    "canonical": r is selected,
                    "orig_split": r["orig_split"],
                    "orig_image_id": r["orig_image_id"],
                    "file_name": r["file_name"],
                    "source_name": r["source_name"],
                    "annotation_count": r["annotation_count"],
                }
            )

    return canonical, duplicate_rows


# ---------------------------------------------------------------------------
# 5. SOURCE GROUPS
# ---------------------------------------------------------------------------

def finalize_source_groups(canonical):
    """
    Use COCO image.extra.name as the source/provenance group.

    Exact-MD5 is used as a fallback only when extra.name is unavailable.
    """
    for row in canonical:
        row["source_group"] = (
            row["source_name"]
            if row["source_name"]
            else f"MD5:{row['md5']}"
        )


def source_group_report(canonical):
    groups = defaultdict(list)

    for row in canonical:
        groups[row["source_group"]].append(row)

    rows = []

    for group_name, members in sorted(groups.items()):
        class_counts = Counter()

        for r in members:
            class_counts.update(r["class_counts"])

        splits = sorted(
            set(r["orig_split"] for r in members)
        )

        rows.append(
            {
                "source_group": group_name,
                "n_canonical_images": len(members),
                "original_splits": "|".join(splits),
                "crosses_original_splits": len(splits) > 1,
                "objects_total": sum(class_counts.values()),
                "Channel": class_counts[0],
                "T-beam": class_counts[1],
                "HI-Beam": class_counts[2],
                "Angle Bar": class_counts[3],
                "Sheet Pile": class_counts[4],
                "negative_images": sum(
                    r["negative"] for r in members
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. STRATIFICATION
# ---------------------------------------------------------------------------

def stratification_label(row):
    """
    Deterministic image-level stratification label.

    For the normal pool, use the dominant final class. True negatives are
    represented separately.

    Sheet Pile is deliberately excluded from this stratification stage
    because the entire dataset contains only three Sheet Pile source groups.
    Those three groups are assigned explicitly, one to each final split.
    """
    if row["negative"]:
        return "NEGATIVE"

    counts = row["class_counts"]
    dominant = max(
        counts,
        key=lambda cid: (
            counts[cid],
            -cid,
        ),
    )

    return str(dominant)


def create_source_aware_split(canonical):
    """
    Create a deterministic source-aware ~80/10/10 split.

    The dataset contains only THREE source groups containing Sheet Pile.
    Because those groups cannot be subdivided without causing source leakage,
    they are explicitly assigned one-per-split.

    The remaining images are split with StratifiedGroupKFold:
        n_splits = 10
        random_state = 19
        fold 2 = validation
        fold 6 = test
        all other folds = train

    This produces 592 / 75 / 75 canonical images for the supplied dataset.

    The explicit Sheet Pile assignment is:
        train: PHOTO-2024-10-10-17-33-33.jpg  (83 objects)
        valid: PHOTO-2024-10-10-18-41-20.jpg  (84 objects)
        test : 18cf4515-b854-44c9-9f58-1e94104e3d57.jpg (56 objects)

    This is a dataset-specific constraint, not a general rule for other
    datasets.
    """
    df = pd.DataFrame(canonical)

    # Identify source groups containing Sheet Pile.
    sheet_groups = set()

    for _, row in df.iterrows():
        if row["class_counts"].get(4, 0) > 0:
            sheet_groups.add(row["source_group"])

    if len(sheet_groups) != 3:
        raise RuntimeError(
            "Expected exactly 3 Sheet Pile source groups, found "
            f"{len(sheet_groups)}: {sorted(sheet_groups)}"
        )

    # Verify that each Sheet Pile source group really contains Sheet Pile.
    sheet_group_rows = {
        row["source_group"]: row
        for _, row in df.iterrows()
        if row["source_group"] in sheet_groups
    }

    expected_sheet_assignment = {
        "PHOTO-2024-10-10-17-33-33.jpg": "train",
        "PHOTO-2024-10-10-18-41-20.jpg": "valid",
        "18cf4515-b854-44c9-9f58-1e94104e3d57.jpg": "test",
    }

    if set(expected_sheet_assignment) != sheet_groups:
        raise RuntimeError(
            "The three expected Sheet Pile source names do not match the "
            "source groups found in this dataset.\n"
            f"Expected: {sorted(expected_sheet_assignment)}\n"
            f"Found: {sorted(sheet_groups)}"
        )

    normal = df[
        ~df["source_group"].isin(sheet_groups)
    ].copy()

    y = normal.apply(stratification_label, axis=1)
    groups = normal["source_group"]

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=SEED,
    )

    fold_ids = np.full(len(normal), -1, dtype=int)

    for fold, (_, indices) in enumerate(
        splitter.split(normal, y, groups)
    ):
        fold_ids[indices] = fold

    if np.any(fold_ids < 0):
        raise RuntimeError(
            "Some normal-pool images were not assigned a fold."
        )

    normal["fold"] = fold_ids
    normal["final_split"] = normal["fold"].map(
        lambda fold: (
            "valid" if fold == 2
            else "test" if fold == 6
            else "train"
        )
    )

    # Explicitly place the three Sheet Pile source groups.
    sheet_rows = df[
        df["source_group"].isin(sheet_groups)
    ].copy()

    sheet_rows["fold"] = -1
    sheet_rows["final_split"] = sheet_rows[
        "source_group"
    ].map(expected_sheet_assignment)

    combined = pd.concat(
        [normal, sheet_rows],
        ignore_index=True,
    )

    return combined


# ---------------------------------------------------------------------------
# 7. REPORTING
# ---------------------------------------------------------------------------

def dataframe_manifest(df):
    rows = []

    for _, r in df.sort_values("master_id").iterrows():
        classes = [
            FINAL_CLASSES[c]
            for c in sorted(r["class_counts"])
        ]

        rows.append(
            {
                "master_id": int(r["master_id"]),
                "final_split": r["final_split"],
                "fold": int(r["fold"]),
                "file_name": r["output_name"],
                "orig_split": r["orig_split"],
                "orig_image_id": int(r["orig_image_id"]),
                "source_group": r["source_group"],
                "source_name": r["source_name"],
                "md5": r["md5"],
                "sha256": r["sha256"],
                "width": int(r["width"]),
                "height": int(r["height"]),
                "annotation_count": int(r["annotation_count"]),
                "is_negative": bool(r["negative"]),
                "class_names": ",".join(classes),
                "Channel": int(r["class_counts"].get(0, 0)),
                "T-beam": int(r["class_counts"].get(1, 0)),
                "HI-Beam": int(r["class_counts"].get(2, 0)),
                "Angle Bar": int(r["class_counts"].get(3, 0)),
                "Sheet Pile": int(r["class_counts"].get(4, 0)),
            }
        )

    return pd.DataFrame(rows)


def split_summary(df):
    rows = []

    for split in SPLITS:
        g = df[df["final_split"] == split]

        counts = Counter()
        for d in g["class_counts"]:
            counts.update(d)

        rows.append(
            {
                "split": split,
                "images": len(g),
                "annotations": sum(counts.values()),
                "negative_images": int(
                    g["negative"].sum()
                ),
                "multi_class_images": int(
                    sum(len(d) > 1 for d in g["class_counts"])
                ),
                "Channel": counts[0],
                "T-beam": counts[1],
                "HI-Beam": counts[2],
                "Angle Bar": counts[3],
                "Sheet Pile": counts[4],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. COCO EXPORT
# ---------------------------------------------------------------------------

def build_coco(df, split):
    subset = df[
        df["final_split"] == split
    ].sort_values("master_id")

    images = []
    annotations = []

    # Keypoint metadata is inherited from the original category definitions.
    keypoint_defs = {}

    for _, row in subset.iterrows():
        for ann in row["annotations"]:
            cid = ann["category_id"]

            # Store the original category's keypoint definition.
            if cid not in keypoint_defs:
                original_id = next(
                    old_id
                    for old_id, new_id in CATEGORY_MAP.items()
                    if new_id == cid
                )

                original_cat = row["categories"][original_id]

                keypoint_defs[cid] = {
                    "keypoints": original_cat.get(
                        "keypoints", []
                    ),
                    "skeleton": original_cat.get(
                        "skeleton", []
                    ),
                }

    for _, row in subset.iterrows():
        images.append(
            {
                "id": int(row["master_id"]),
                "file_name": row["output_name"],
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )

        for ann in row["annotations"]:
            new_ann = dict(ann)
            new_ann["image_id"] = int(row["master_id"])
            annotations.append(new_ann)

    categories = []

    for cid, name in FINAL_CLASSES.items():
        categories.append(
            {
                "id": cid,
                "name": name,
                "supercategory": "steel_structure",
                "keypoints": keypoint_defs.get(
                    cid, {}
                ).get("keypoints", []),
                "skeleton": keypoint_defs.get(
                    cid, {}
                ).get("skeleton", []),
            }
        )

    return {
        "info": {
            "description": "InnoCount KeyPoint Frozen Dataset",
            "version": "Frozen-v1",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# 9. VALIDATION
# ---------------------------------------------------------------------------

def validate_split(df):
    errors = []

    # No exact MD5 may occur in multiple final splits.
    hash_splits = defaultdict(set)

    for _, row in df.iterrows():
        hash_splits[row["md5"]].add(row["final_split"])

    crossing_hashes = {
        h: s
        for h, s in hash_splits.items()
        if len(s) > 1
    }

    if crossing_hashes:
        errors.append(
            f"{len(crossing_hashes)} exact image hashes cross splits"
        )

    # No source group may occur in multiple final splits.
    source_splits = defaultdict(set)

    for _, row in df.iterrows():
        source_splits[row["source_group"]].add(
            row["final_split"]
        )

    crossing_sources = {
        s: v
        for s, v in source_splits.items()
        if len(v) > 1
    }

    if crossing_sources:
        errors.append(
            f"{len(crossing_sources)} source groups cross splits"
        )

    # Every final class should be represented in validation and test.
    for split in ("valid", "test"):
        counts = Counter()

        for d in df[
            df["final_split"] == split
        ]["class_counts"]:
            counts.update(d)

        for cid, name in FINAL_CLASSES.items():
            if counts[cid] == 0:
                errors.append(
                    f"{name} absent from {split}"
                )

    # Negative images should be present in every split.
    for split in SPLITS:
        n_negative = int(
            df[df["final_split"] == split]["negative"].sum()
        )

        if n_negative == 0:
            errors.append(
                f"No negative/background images in {split}"
            )

    if errors:
        raise RuntimeError(
            "SPLIT VALIDATION FAILED:\n- "
            + "\n- ".join(errors)
        )

    return {
        "passed": True,
        "exact_hashes_crossing_splits": len(crossing_hashes),
        "source_groups_crossing_splits": len(crossing_sources),
    }


# ---------------------------------------------------------------------------
# 10. FREEZE / EXPORT
# ---------------------------------------------------------------------------

def copy_images(df, out_dir):
    for split in SPLITS:
        image_dir = (
            out_dir
            / "splits"
            / split
            / "images"
        )

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        subset = df[
            df["final_split"] == split
        ].sort_values("master_id")

        for _, row in subset.iterrows():
            dst = image_dir / row["output_name"]

            if dst.exists():
                raise FileExistsError(dst)

            shutil.copy2(
                row["file_path"],
                dst,
            )


def export_coco(df, out_dir):
    for split in SPLITS:
        split_dir = (
            out_dir
            / "splits"
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        coco = build_coco(df, split)

        (split_dir / "annotations.json").write_text(
            json.dumps(
                coco,
                indent=2,
            ),
            encoding="utf-8",
        )


def write_reports(
    df,
    raw_inspection,
    category_report,
    duplicate_rows,
    source_df,
    validation,
    out_dir,
):
    report_dir = out_dir / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = dataframe_manifest(df)

    manifest.to_csv(
        report_dir / "master_manifest.csv",
        index=False,
    )

    source_df.to_csv(
        report_dir / "source_groups.csv",
        index=False,
    )

    pd.DataFrame(duplicate_rows).to_csv(
        report_dir / "duplicate_groups.csv",
        index=False,
    )

    split_summary(df).to_csv(
        report_dir / "split_summary.csv",
        index=False,
    )

    (report_dir / "inspection.json").write_text(
        json.dumps(
            raw_inspection,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    (report_dir / "category_mapping.json").write_text(
        json.dumps(
            category_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    (report_dir / "validation.json").write_text(
        json.dumps(
            validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    return manifest


def write_freeze_metadata(
    out_dir,
    input_zip,
    input_sha256,
    n_raw,
    n_canonical,
):
    freeze = {
        "dataset": "InnoCount KeyPoint",
        "version": "Frozen-v1",
        "input_zip": input_zip.name,
        "input_zip_sha256": input_sha256,
        "seed": SEED,
        "original_image_records": n_raw,
        "canonical_images": n_canonical,
        "exact_duplicate_records_removed": n_raw - n_canonical,
        "final_classes": FINAL_CLASSES,
        "category_map_original_to_final": CATEGORY_MAP,
        "negative_images_retained": True,
        "augmentation_added": False,
        "source_group_field": "images[].extra.name",
        "source_group_fallback": "MD5",
        "split_method": "StratifiedGroupKFold + explicit Sheet Pile source-group assignment",
        "n_splits": 10,
        "validation_fold": 2,
        "test_fold": 6,
        "sheet_pile_source_groups_one_per_split": True,
        "train_folds": [
            0, 1, 3, 4, 5, 7, 8, 9
        ],
        "notes": [
            "Roboflow train/valid/test assignments are provenance only.",
            "Exact duplicate image files are canonicalized before splitting.",
            "Related source groups are kept within one final split.",
            "The 26 genuine negative/background images are retained.",
            "Validation/test receive no additional augmentation.",
            "A fresh run is deterministic but is not guaranteed to reproduce "
            "the historical Frozen-v1 assignment exactly unless the released "
            "master_manifest.csv is used as a lockfile."
        ],
    }

    (out_dir / "FREEZE.json").write_text(
        json.dumps(
            freeze,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_readme(out_dir):
    text = """# InnoCount KeyPoint Frozen-v1

This directory is the frozen research dataset produced from the original
Roboflow COCO ZIP.

## Final classes

| ID | Class |
|---:|---|
| 0 | Channel |
| 1 | T-beam |
| 2 | HI-Beam |
| 3 | Angle Bar |
| 4 | Sheet Pile |

## Rules

- Roboflow train/valid/test are treated as one master pool.
- Exact duplicate JPEGs are detected by MD5.
- Exact duplicates are canonicalized before splitting.
- Source-linked images are grouped using `image.extra.name`.
- 26 genuine negative/background images are retained.
- No new augmentation is added.
- The final target is approximately 80/10/10.
- The split is source-aware.
- Validation/test are not augmented.
- The master manifest is the lockfile for the frozen split.

## Reproduction

Run:

```bash
python create_frozen_dataset.py \
    --input "InnoCount KeyPoint.v3i.coco.zip" \
    --output "InnoCount_KeyPoint_Frozen_v1"
```

For a previously frozen dataset, `reports/master_manifest.csv` should be
kept permanently as the assignment lockfile.
"""
    (out_dir / "README.md").write_text(
        text,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create InnoCount_KeyPoint_Frozen_v1 from "
            "the original Roboflow COCO ZIP."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Original Roboflow COCO ZIP",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New frozen dataset output directory",
    )

    args = parser.parse_args()

    ensure_clean_output(args.output)

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("InnoCount KeyPoint -> Frozen Dataset")
    print("=" * 70)

    # Input hash for provenance.
    input_sha256 = sha256_file(args.input)

    with tempfile.TemporaryDirectory(
        prefix="innocount_freeze_"
    ) as tmp:

        tmp = Path(tmp)

        print("\n[1/10] Extracting original ZIP...")
        root = extract_zip(args.input, tmp)

        print("[2/10] Loading original COCO records...")
        raw_records = load_coco_records(root)

        print(
            f"       Raw image records: {len(raw_records)}"
        )

        print("[3/10] Inspecting dataset...")
        inspection = inspect_records(raw_records)

        print(
            f"       Negatives: "
            f"{inspection['negative_images']}"
        )
        print(
            f"       Dimension mismatches: "
            f"{inspection['dimensions_mismatch']}"
        )

        if inspection["dimensions_mismatch"]:
            raise RuntimeError(
                "Image dimensions disagree with COCO metadata."
            )

        print("[4/10] Verifying category mapping...")
        category_report = verify_categories(
            raw_records
        )

        print(
            "       Active classes:",
            [
                category_report[
                    "final_mapping"
                ][str(i)]
                for i in range(5)
            ],
        )

        print("[5/10] Building master annotation records...")
        master = build_master_records(
            raw_records
        )

        # Keep category definitions available for COCO export.
        # build_master_records intentionally keeps them through this field.
        for row, raw in zip(master, raw_records):
            row["categories"] = raw["categories"]

        print("[6/10] Calculating image hashes...")
        calculate_hashes(master)

        print("[7/10] Canonicalizing exact duplicates...")
        canonical, duplicate_rows = (
            canonicalize_exact_duplicates(master)
        )

        print(
            f"       Canonical images: {len(canonical)}"
        )
        print(
            f"       Removed records: "
            f"{len(master) - len(canonical)}"
        )

        print("[8/10] Building source groups...")
        finalize_source_groups(canonical)

        source_df = source_group_report(
            canonical
        )

        print(
            f"       Source groups: "
            f"{len(source_df)}"
        )

        print("[9/10] Creating source-aware split...")
        df = create_source_aware_split(
            canonical
        )

        # Stable master IDs are assigned after canonicalization.
        # Sorting by MD5 makes IDs deterministic.
        df = df.sort_values(
            ["md5"]
        ).reset_index(drop=True)

        df["master_id"] = np.arange(
            1,
            len(df) + 1,
        )

        df["output_name"] = df[
            "master_id"
        ].map(
            lambda x: f"{int(x):06d}.jpg"
        )

        print(
            split_summary(df).to_string(
                index=False
            )
        )

        print("[10/10] Validating and freezing...")
        validation = validate_split(df)

        args.output.mkdir(
            parents=True,
            exist_ok=False,
        )

        copy_images(
            df,
            args.output,
        )

        export_coco(
            df,
            args.output,
        )

        manifest = write_reports(
            df,
            inspection,
            category_report,
            duplicate_rows,
            source_df,
            validation,
            args.output,
        )

        write_freeze_metadata(
            args.output,
            args.input,
            input_sha256,
            len(raw_records),
            len(canonical),
        )

        write_readme(args.output)

        # Store a copy of the input hash, not the original ZIP itself.
        (args.output / "INPUT_SHA256.txt").write_text(
            input_sha256 + "\n",
            encoding="utf-8",
        )

    print("\n" + "=" * 70)
    print("FREEZE COMPLETE")
    print("=" * 70)
    print(f"Output: {args.output}")
    print(
        "Validation:",
        "PASSED" if validation["passed"] else "FAILED",
    )
    print(
        "Manifest:",
        args.output / "reports" / "master_manifest.csv",
    )


if __name__ == "__main__":
    main()
