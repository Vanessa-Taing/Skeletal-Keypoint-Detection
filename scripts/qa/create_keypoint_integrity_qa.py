#!/usr/bin/env python3
"""
create_keypoint_integrity_qa.py

Quantitative keypoint-integrity QA for the frozen InnoCount dataset.

READ-ONLY:
    This script never modifies the frozen dataset or annotation coordinates.

It reports, by:
    - split
    - class
    - keypoint
    - source group

the following:
    * total keypoints
    * visible keypoints (v=2)
    * occluded keypoints (v=1)
    * unlabeled keypoints (v=0)
    * out-of-image keypoints
    * out-of-bbox keypoints
    * keypoints that are both out-of-image and out-of-bbox

It also produces:
    * overall_summary.csv
    * class_summary.csv
    * keypoint_summary.csv
    * keypoint_summary_by_split.csv
    * source_group_summary.csv
    * suspicious_annotations.csv
    * integrity_report.json
    * integrity_report.txt

Recommended usage:

    python create_keypoint_integrity_qa.py \
        --dataset "datasets/InnoCount_KeyPoint_Frozen_v1" \
        --output "InnoCount_Keypoint_Integrity_QA"

Dependencies:
    Python standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")

FINAL_CLASSES = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows, fieldnames):
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def pct(n, d):
    if d == 0:
        return 0.0
    return 100.0 * n / d


def visibility_name(v):
    return {
        0: "not_labeled",
        1: "occluded",
        2: "visible",
    }.get(v, f"unknown_{v}")


def bbox_contains(x, y, bbox):
    if not bbox:
        return False

    bx, by, bw, bh = bbox

    return (
        bx <= x <= bx + bw
        and
        by <= y <= by + bh
    )


def inside_image(x, y, width, height):
    return (
        0 <= x < width
        and
        0 <= y < height
    )


# ---------------------------------------------------------------------------
# Load frozen dataset
# ---------------------------------------------------------------------------

def load_all_annotations(dataset_dir):
    records = []

    for split in SPLITS:

        json_path = (
            dataset_dir
            / "splits"
            / split
            / "annotations.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing annotation file: {json_path}"
            )

        data = load_json(json_path)

        images = {
            img["id"]: img
            for img in data["images"]
        }

        categories = {
            cat["id"]: cat
            for cat in data["categories"]
        }

        # Load manifest if available so that source groups
        # can be associated with the frozen dataset.
        manifest_path = (
            dataset_dir
            / "reports"
            / "master_manifest.csv"
        )

        manifest_by_filename = {}

        if manifest_path.exists():
            with manifest_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as f:
                reader = csv.DictReader(f)

                for row in reader:
                    manifest_by_filename[
                        row["file_name"]
                    ] = row

        for ann in data["annotations"]:

            image = images[ann["image_id"]]
            category = categories[ann["category_id"]]

            manifest_row = manifest_by_filename.get(
                image["file_name"]
            )

            source_group = (
                manifest_row["source_group"]
                if manifest_row
                else image["file_name"]
            )

            records.append(
                {
                    "split": split,
                    "image": image,
                    "annotation": ann,
                    "category": category,
                    "source_group": source_group,
                }
            )

    return records


# ---------------------------------------------------------------------------
# Per-keypoint analysis
# ---------------------------------------------------------------------------

def analyse_records(records):
    class_stats = defaultdict(Counter)
    keypoint_stats = defaultdict(Counter)
    split_keypoint_stats = defaultdict(Counter)
    source_stats = defaultdict(Counter)

    suspicious = []

    for record in records:

        split = record["split"]
        image = record["image"]
        ann = record["annotation"]
        category = record["category"]
        source_group = record["source_group"]

        cid = category["id"]
        class_name = category["name"]

        keypoint_names = category.get(
            "keypoints",
            [],
        )

        expected_k = len(keypoint_names)

        keypoints = ann.get(
            "keypoints",
            [],
        )

        # ----------------------------------------------------
        # Structural integrity
        # ----------------------------------------------------

        if len(keypoints) != expected_k * 3:

            suspicious.append(
                {
                    "split": split,
                    "image_id": image["id"],
                    "file_name": image["file_name"],
                    "annotation_id": ann["id"],
                    "class_id": cid,
                    "class_name": class_name,
                    "keypoint": "",
                    "keypoint_name": "",
                    "issue": "malformed_keypoint_vector",
                    "x": "",
                    "y": "",
                    "visibility": "",
                    "image_width": image["width"],
                    "image_height": image["height"],
                    "bbox": json.dumps(
                        ann.get("bbox")
                    ),
                    "source_group": source_group,
                }
            )

            continue

        # ----------------------------------------------------
        # Object-level counts
        # ----------------------------------------------------

        class_stats[
            (split, class_name)
        ]["objects"] += 1

        source_stats[
            (source_group,)
        ]["objects"] += 1

        # ----------------------------------------------------
        # Keypoint-level analysis
        # ----------------------------------------------------

        for idx, keypoint_name in enumerate(
            keypoint_names,
            start=1,
        ):

            x = keypoints[
                (idx - 1) * 3
            ]

            y = keypoints[
                (idx - 1) * 3 + 1
            ]

            v = keypoints[
                (idx - 1) * 3 + 2
            ]

            key = (
                split,
                class_name,
                idx,
                keypoint_name,
            )

            stats = keypoint_stats[key]
            split_stats = split_keypoint_stats[key]

            stats["total"] += 1
            split_stats["total"] += 1

            if v == 0:
                stats["not_labeled"] += 1
                split_stats["not_labeled"] += 1

            elif v == 1:
                stats["occluded"] += 1
                split_stats["occluded"] += 1

            elif v == 2:
                stats["visible"] += 1
                split_stats["visible"] += 1

            else:
                stats["unknown_visibility"] += 1
                split_stats["unknown_visibility"] += 1

            in_image = inside_image(
                x,
                y,
                image["width"],
                image["height"],
            )

            in_bbox = bbox_contains(
                x,
                y,
                ann.get("bbox"),
            )

            if not in_image:
                stats["out_of_image"] += 1
                split_stats["out_of_image"] += 1

            if not in_bbox:
                stats["out_of_bbox"] += 1
                split_stats["out_of_bbox"] += 1

            if not in_image and not in_bbox:
                stats["both_out"] += 1
                split_stats["both_out"] += 1

            # Only evaluate geometry for labeled keypoints.
            if v > 0:

                if not in_image:

                    issue = "out_of_image"

                    if not in_bbox:
                        issue = (
                            "out_of_image_and_out_of_bbox"
                        )

                    suspicious.append(
                        {
                            "split": split,
                            "image_id": image["id"],
                            "file_name": image["file_name"],
                            "annotation_id": ann["id"],
                            "class_id": cid,
                            "class_name": class_name,
                            "keypoint": idx,
                            "keypoint_name": keypoint_name,
                            "issue": issue,
                            "x": x,
                            "y": y,
                            "visibility": v,
                            "image_width": image["width"],
                            "image_height": image["height"],
                            "bbox": json.dumps(
                                ann.get("bbox")
                            ),
                            "source_group": source_group,
                        }
                    )

                elif not in_bbox:

                    suspicious.append(
                        {
                            "split": split,
                            "image_id": image["id"],
                            "file_name": image["file_name"],
                            "annotation_id": ann["id"],
                            "class_id": cid,
                            "class_name": class_name,
                            "keypoint": idx,
                            "keypoint_name": keypoint_name,
                            "issue": "out_of_bbox",
                            "x": x,
                            "y": y,
                            "visibility": v,
                            "image_width": image["width"],
                            "image_height": image["height"],
                            "bbox": json.dumps(
                                ann.get("bbox")
                            ),
                            "source_group": source_group,
                        }
                    )

            # Source-group statistics.
            source_key = (
                source_group,
                class_name,
                idx,
                keypoint_name,
            )

            source_counter = source_stats[
                source_key
            ]

            source_counter["total"] += 1

            if v == 1:
                source_counter["occluded"] += 1

            elif v == 2:
                source_counter["visible"] += 1

            if not in_image:
                source_counter["out_of_image"] += 1

            if not in_bbox:
                source_counter["out_of_bbox"] += 1

            if not in_image and not in_bbox:
                source_counter["both_out"] += 1

    return (
        class_stats,
        keypoint_stats,
        split_keypoint_stats,
        source_stats,
        suspicious,
    )


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def build_class_summary(class_stats):
    rows = []

    for (
        split,
        class_name,
    ), stats in sorted(class_stats.items()):

        rows.append(
            {
                "split": split,
                "class": class_name,
                "objects": stats["objects"],
            }
        )

    return rows


def build_keypoint_summary(keypoint_stats):
    rows = []

    for (
        split,
        class_name,
        idx,
        keypoint_name,
    ), stats in sorted(
        keypoint_stats.items()
    ):

        total = stats["total"]

        rows.append(
            {
                "split": split,
                "class": class_name,
                "keypoint_index": idx,
                "keypoint_name": keypoint_name,
                "total": total,
                "visible": stats["visible"],
                "occluded": stats["occluded"],
                "not_labeled": stats["not_labeled"],
                "unknown_visibility": stats[
                    "unknown_visibility"
                ],
                "out_of_image": stats[
                    "out_of_image"
                ],
                "out_of_image_pct": round(
                    pct(
                        stats["out_of_image"],
                        total,
                    ),
                    3,
                ),
                "out_of_bbox": stats[
                    "out_of_bbox"
                ],
                "out_of_bbox_pct": round(
                    pct(
                        stats["out_of_bbox"],
                        total,
                    ),
                    3,
                ),
                "both_out": stats[
                    "both_out"
                ],
                "both_out_pct": round(
                    pct(
                        stats["both_out"],
                        total,
                    ),
                    3,
                ),
            }
        )

    return rows


def build_source_summary(source_stats):
    rows = []

    # Aggregate source-level class totals.
    aggregate = defaultdict(Counter)

    for key, stats in source_stats.items():

        # source_stats also contains an object-level entry keyed as
        # (source_group,), so only process the four-part keypoint entries.
        if len(key) != 4:
            continue

        (
            source_group,
            class_name,
            idx,
            keypoint_name,
        ) = key

        key = (
            source_group,
            class_name,
        )

        for metric in (
            "total",
            "visible",
            "occluded",
            "out_of_image",
            "out_of_bbox",
            "both_out",
        ):
            aggregate[key][metric] += stats[
                metric
            ]

    for (
        source_group,
        class_name,
    ), stats in sorted(
        aggregate.items()
    ):

        total = stats["total"]

        rows.append(
            {
                "source_group": source_group,
                "class": class_name,
                "keypoints": total,
                "visible": stats["visible"],
                "occluded": stats["occluded"],
                "out_of_image": stats[
                    "out_of_image"
                ],
                "out_of_image_pct": round(
                    pct(
                        stats["out_of_image"],
                        total,
                    ),
                    3,
                ),
                "out_of_bbox": stats[
                    "out_of_bbox"
                ],
                "out_of_bbox_pct": round(
                    pct(
                        stats["out_of_bbox"],
                        total,
                    ),
                    3,
                ),
                "both_out": stats[
                    "both_out"
                ],
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Overall report
# ---------------------------------------------------------------------------

def build_overall_summary(records):
    total_annotations = len(records)

    total_keypoints = 0
    visible = 0
    occluded = 0
    not_labeled = 0
    unknown = 0
    out_image = 0
    out_bbox = 0
    both_out = 0

    split_counts = Counter()
    class_counts = Counter()

    for record in records:

        split = record["split"]
        category = record["category"]
        ann = record["annotation"]

        split_counts[split] += 1
        class_counts[
            category["name"]
        ] += 1

        keypoints = ann.get(
            "keypoints",
            [],
        )

        for i in range(
            0,
            len(keypoints),
            3,
        ):

            x = keypoints[i]
            y = keypoints[i + 1]
            v = keypoints[i + 2]

            total_keypoints += 1

            if v == 2:
                visible += 1
            elif v == 1:
                occluded += 1
            elif v == 0:
                not_labeled += 1
            else:
                unknown += 1

            in_image = inside_image(
                x,
                y,
                record["image"]["width"],
                record["image"]["height"],
            )

            in_bbox = bbox_contains(
                x,
                y,
                ann.get("bbox"),
            )

            if not in_image:
                out_image += 1

            if not in_bbox:
                out_bbox += 1

            if not in_image and not in_bbox:
                both_out += 1

    return {
        "images": len({
            (
                r["split"],
                r["image"]["id"],
            )
            for r in records
        }),
        "annotations": total_annotations,
        "keypoints": total_keypoints,
        "visible": visible,
        "occluded": occluded,
        "not_labeled": not_labeled,
        "unknown_visibility": unknown,
        "out_of_image": out_image,
        "out_of_image_pct": round(
            pct(
                out_image,
                total_keypoints,
            ),
            3,
        ),
        "out_of_bbox": out_bbox,
        "out_of_bbox_pct": round(
            pct(
                out_bbox,
                total_keypoints,
            ),
            3,
        ),
        "both_out": both_out,
        "both_out_pct": round(
            pct(
                both_out,
                total_keypoints,
            ),
            3,
        ),
        "annotations_by_split": dict(
            split_counts
        ),
        "annotations_by_class": dict(
            class_counts
        ),
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def write_text_report(
    output,
    overall,
    class_rows,
    keypoint_rows,
):
    path = output / "integrity_report.txt"

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "InnoCount KeyPoint Integrity QA\n"
        )
        f.write(
            "================================\n\n"
        )

        f.write(
            "This report is read-only analysis of the "
            "frozen dataset.\n\n"
        )

        f.write(
            "OVERALL\n"
        )
        f.write(
            "-------\n"
        )

        for key, value in overall.items():
            if isinstance(value, dict):
                continue

            f.write(
                f"{key}: {value}\n"
            )

        f.write("\n")

        f.write(
            "CLASS / SPLIT OBJECT COUNTS\n"
        )
        f.write(
            "---------------------------\n"
        )

        for row in class_rows:
            f.write(
                f"{row['split']:<6} "
                f"{row['class']:<15} "
                f"{row['objects']:>6}\n"
            )

        f.write("\n")

        f.write(
            "KEYPOINT INTEGRITY\n"
        )
        f.write(
            "------------------\n"
        )

        for row in keypoint_rows:

            f.write(
                f"{row['split']:<6} "
                f"{row['class']:<15} "
                f"K{row['keypoint_index']:<2} "
                f"{row['keypoint_name']:<15} "
                f"total={row['total']:>5} "
                f"visible={row['visible']:>5} "
                f"occluded={row['occluded']:>4} "
                f"out_image={row['out_of_image']:>4} "
                f"({row['out_of_image_pct']:>6.2f}%) "
                f"out_bbox={row['out_of_bbox']:>4} "
                f"({row['out_of_bbox_pct']:>6.2f}%)\n"
            )

        f.write("\n")

        f.write(
            "INTERPRETATION NOTES\n"
        )
        f.write(
            "--------------------\n"
        )
        f.write(
            "1. Out-of-image does not automatically mean erroneous.\n"
        )
        f.write(
            "2. Out-of-bbox does not automatically mean erroneous.\n"
        )
        f.write(
            "3. Values are reported exactly as annotated.\n"
        )
        f.write(
            "4. No coordinates are clipped or corrected.\n"
        )
        f.write(
            "5. Use the per-keypoint tables to identify systematic patterns.\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Quantitative keypoint integrity QA "
            "for the frozen InnoCount dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Frozen dataset directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "InnoCount_Keypoint_Integrity_QA"
        ),
        help="QA output directory",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(
            args.dataset
        )

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=False,
    )

    print("=" * 70)
    print("InnoCount KeyPoint Integrity QA")
    print("=" * 70)

    print("\n[1/5] Loading frozen annotations...")

    records = load_all_annotations(
        args.dataset
    )

    print(
        f"       Annotation records: {len(records)}"
    )

    print(
        "\n[2/5] Analysing keypoints..."
    )

    (
        class_stats,
        keypoint_stats,
        split_keypoint_stats,
        source_stats,
        suspicious,
    ) = analyse_records(records)

    print(
        "\n[3/5] Building reports..."
    )

    overall = build_overall_summary(
        records
    )

    class_rows = build_class_summary(
        class_stats
    )

    keypoint_rows = build_keypoint_summary(
        keypoint_stats
    )

    source_rows = build_source_summary(
        source_stats
    )

    write_csv(
        args.output
        / "class_summary.csv",
        class_rows,
        [
            "split",
            "class",
            "objects",
        ],
    )

    write_csv(
        args.output
        / "keypoint_summary.csv",
        keypoint_rows,
        [
            "split",
            "class",
            "keypoint_index",
            "keypoint_name",
            "total",
            "visible",
            "occluded",
            "not_labeled",
            "unknown_visibility",
            "out_of_image",
            "out_of_image_pct",
            "out_of_bbox",
            "out_of_bbox_pct",
            "both_out",
            "both_out_pct",
        ],
    )

    write_csv(
        args.output
        / "source_group_summary.csv",
        source_rows,
        [
            "source_group",
            "class",
            "keypoints",
            "visible",
            "occluded",
            "out_of_image",
            "out_of_image_pct",
            "out_of_bbox",
            "out_of_bbox_pct",
            "both_out",
        ],
    )

    write_csv(
        args.output
        / "suspicious_annotations.csv",
        suspicious,
        [
            "split",
            "image_id",
            "file_name",
            "annotation_id",
            "class_id",
            "class_name",
            "keypoint",
            "keypoint_name",
            "issue",
            "x",
            "y",
            "visibility",
            "image_width",
            "image_height",
            "bbox",
            "source_group",
        ],
    )

    # Overall report.
    (args.output / "integrity_report.json").write_text(
        json.dumps(
            overall,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_text_report(
        args.output,
        overall,
        class_rows,
        keypoint_rows,
    )

    print(
        "\n[4/5] Printing keypoint summary..."
    )

    print()

    print(
        "Overall:"
    )

    print(
        f"  Images             : {overall['images']}"
    )

    print(
        f"  Annotations        : {overall['annotations']}"
    )

    print(
        f"  Keypoints          : {overall['keypoints']}"
    )

    print(
        f"  Visible            : {overall['visible']}"
    )

    print(
        f"  Occluded           : {overall['occluded']}"
    )

    print(
        f"  Out of image       : "
        f"{overall['out_of_image']} "
        f"({overall['out_of_image_pct']}%)"
    )

    print(
        f"  Out of bbox        : "
        f"{overall['out_of_bbox']} "
        f"({overall['out_of_bbox_pct']}%)"
    )

    print(
        f"  Both out           : "
        f"{overall['both_out']} "
        f"({overall['both_out_pct']}%)"
    )

    print()

    for row in keypoint_rows:

        print(
            f"{row['split']:<6} "
            f"{row['class']:<15} "
            f"K{row['keypoint_index']} "
            f"{row['keypoint_name']:<15} "
            f"total={row['total']:>5} "
            f"visible={row['visible']:>5} "
            f"occ={row['occluded']:>4} "
            f"out_img={row['out_of_image']:>4} "
            f"({row['out_of_image_pct']:>6.2f}%) "
            f"out_bbox={row['out_of_bbox']:>4} "
            f"({row['out_of_bbox_pct']:>6.2f}%)"
        )

    print(
        "\n[5/5] QA complete."
    )

    print("=" * 70)
    print(
        f"Output: {args.output}"
    )
    print("=" * 70)

    print(
        "\nImportant:"
    )
    print(
        "No annotation was modified."
    )
    print(
        "No keypoint was clipped or corrected."
    )


if __name__ == "__main__":
    main()
