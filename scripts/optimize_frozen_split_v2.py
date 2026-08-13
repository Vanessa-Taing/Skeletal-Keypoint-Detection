#!/usr/bin/env python3
"""
optimize_frozen_split_v2.py

Create a better source-aware 80/10/10 split from the already-canonicalized
InnoCount_KeyPoint_Frozen_v1 dataset.

IMPORTANT:
    This script does NOT alter annotations, images, class IDs, keypoints,
    bounding boxes, or source groups.

It only reassigns COMPLETE source groups to train/valid/test and creates
a new derived dataset:

    InnoCount_KeyPoint_Frozen_v2

Input:
    InnoCount_KeyPoint_Frozen_v1

Why this script exists:
    Frozen v1 is source-aware, but its validation split contains only
    9 Angle Bar objects. This optimizer explicitly considers class-level
    object distributions in addition to image-level 80/10/10.

Hard constraints:
    - source groups are never split
    - exact duplicate canonical images remain together
    - all 742 canonical images are retained
    - all 26 negative images are retained
    - annotations are copied exactly
    - original v1 is never modified

Optimization objectives:
    1. image count close to 80/10/10
    2. object count for EACH of the 5 classes close to 80/10/10
    3. negative-image count close to 80/10/10
    4. avoid empty class coverage in validation/test
    5. keep source groups intact

The optimizer uses randomized greedy initialization followed by
local-search moves/swaps and multiple restarts.

Usage:

    python optimize_frozen_split_v2.py \
        --input "InnoCount_KeyPoint_Frozen_v1" \
        --output "InnoCount_KeyPoint_Frozen_v2"

Optional:

    --restarts 80
    --iterations 5000
    --seed 20260809

No external Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "valid", "test")

SPLIT_FRACTIONS = {
    "train": 0.80,
    "valid": 0.10,
    "test": 0.10,
}

CLASSES = (
    "Channel",
    "T-beam",
    "HI-Beam",
    "Angle Bar",
    "Sheet Pile",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def pct(value, total):
    if total == 0:
        return 0.0
    return 100.0 * value / total


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_frozen_dataset(input_dir: Path):
    """
    Load every canonical image and annotation from the three v1 splits.

    The resulting pool is independent of the current v1 split assignment.
    """

    images_by_id = {}
    annotations_by_image = defaultdict(list)

    categories = None
    image_original_split = {}

    for split in SPLITS:

        json_path = (
            input_dir
            / "splits"
            / split
            / "annotations.json"
        )

        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing COCO file: {json_path}"
            )

        data = load_json(json_path)

        if categories is None:
            categories = data["categories"]
        else:
            if data["categories"] != categories:
                raise RuntimeError(
                    f"Category schema differs in {split}"
                )

        for image in data["images"]:

            image_id = image["id"]

            if image_id in images_by_id:
                raise RuntimeError(
                    f"Duplicate image ID across splits: {image_id}"
                )

            images_by_id[image_id] = image
            image_original_split[image_id] = split

        for ann in data["annotations"]:
            annotations_by_image[
                ann["image_id"]
            ].append(ann)

    return (
        images_by_id,
        annotations_by_image,
        categories,
        image_original_split,
    )


# ---------------------------------------------------------------------------
# Manifest / source groups
# ---------------------------------------------------------------------------

def load_manifest(input_dir: Path):
    manifest_path = (
        input_dir
        / "reports"
        / "master_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Frozen manifest not found: {manifest_path}"
        )

    rows = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(
                "master_manifest.csv has no columns."
            )

        for row in reader:
            rows.append(row)

    return rows


def find_manifest_column(fieldnames, candidates):
    lower = {
        name.lower(): name
        for name in fieldnames
    }

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    return None


def build_image_metadata(
    images_by_id,
    annotations_by_image,
    categories,
    manifest_rows,
):
    """
    Match frozen images to source groups.

    We prefer image_id if available, otherwise file_name.
    """

    category_by_id = {
        c["id"]: c
        for c in categories
    }

    fieldnames = manifest_rows[0].keys()

    id_col = find_manifest_column(
        fieldnames,
        (
            "image_id",
            "id",
        ),
    )

    file_col = find_manifest_column(
        fieldnames,
        (
            "file_name",
            "filename",
            "image",
        ),
    )

    source_col = find_manifest_column(
        fieldnames,
        (
            "source_group",
            "source_group_id",
            "source",
        ),
    )

    if source_col is None:
        raise RuntimeError(
            "Could not find source_group column in master_manifest.csv"
        )

    by_id = {}
    by_file = {}

    for row in manifest_rows:

        source_group = row[source_col]

        if not source_group:
            raise RuntimeError(
                "Found an image manifest row with an empty source_group."
            )

        if id_col and row.get(id_col):
            try:
                by_id[int(row[id_col])] = row
            except ValueError:
                pass

        if file_col and row.get(file_col):
            by_file[row[file_col]] = row

    metadata = {}

    for image_id, image in images_by_id.items():

        row = None

        if image_id in by_id:
            row = by_id[image_id]

        if row is None:
            row = by_file.get(
                image["file_name"]
            )

        if row is None:
            raise RuntimeError(
                "Could not match image to frozen manifest: "
                f"image_id={image_id}, "
                f"file_name={image['file_name']}"
            )

        source_group = row[source_col]

        class_counts = Counter()

        for ann in annotations_by_image.get(
            image_id,
            [],
        ):
            cid = ann["category_id"]
            name = category_by_id[cid]["name"]

            if name in CLASSES:
                class_counts[name] += 1

        metadata[image_id] = {
            "image_id": image_id,
            "file_name": image["file_name"],
            "source_group": source_group,
            "width": image["width"],
            "height": image["height"],
            "class_counts": class_counts,
            "negative": (
                len(annotations_by_image.get(
                    image_id,
                    [],
                )) == 0
            ),
        }

    return metadata


# ---------------------------------------------------------------------------
# Source group aggregation
# ---------------------------------------------------------------------------

def build_source_groups(metadata):
    groups = defaultdict(
        lambda: {
            "images": [],
            "image_count": 0,
            "negative_images": 0,
            "class_counts": Counter(),
        }
    )

    for image_id, item in metadata.items():

        group = groups[
            item["source_group"]
        ]

        group["images"].append(
            image_id
        )

        group["image_count"] += 1

        if item["negative"]:
            group["negative_images"] += 1

        for class_name, count in item[
            "class_counts"
        ].items():
            group["class_counts"][
                class_name
            ] += count

    return groups


# ---------------------------------------------------------------------------
# Totals and targets
# ---------------------------------------------------------------------------

def calculate_totals(metadata):
    totals = {
        "images": 0,
        "negative_images": 0,
        "classes": Counter(),
    }

    for item in metadata.values():

        totals["images"] += 1

        if item["negative"]:
            totals["negative_images"] += 1

        for class_name, count in item[
            "class_counts"
        ].items():
            totals["classes"][
                class_name
            ] += count

    return totals


def calculate_targets(totals):
    targets = {
        split: {
            "images": totals["images"]
            * SPLIT_FRACTIONS[split],
            "negative_images": totals[
                "negative_images"
            ]
            * SPLIT_FRACTIONS[split],
            "classes": {
                class_name:
                    totals["classes"][class_name]
                    * SPLIT_FRACTIONS[split]
                for class_name in CLASSES
            },
        }
        for split in SPLITS
    }

    return targets


# ---------------------------------------------------------------------------
# Assignment state
# ---------------------------------------------------------------------------

def empty_state():
    return {
        split: {
            "images": 0,
            "negative_images": 0,
            "classes": Counter(),
        }
        for split in SPLITS
    }


def add_group_to_state(
    state,
    split,
    group,
):
    state[split]["images"] += group[
        "image_count"
    ]

    state[split]["negative_images"] += group[
        "negative_images"
    ]

    for class_name, count in group[
        "class_counts"
    ].items():

        state[split]["classes"][
            class_name
        ] += count


def remove_group_from_state(
    state,
    split,
    group,
):
    state[split]["images"] -= group[
        "image_count"
    ]

    state[split]["negative_images"] -= group[
        "negative_images"
    ]

    for class_name, count in group[
        "class_counts"
    ].items():

        state[split]["classes"][
            class_name
        ] -= count


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(
    state,
    targets,
    totals,
):
    """
    Lower is better.

    Main priority:
        class object distributions

    Secondary:
        image counts

    Tertiary:
        negative-image distribution

    A very large penalty is applied if validation/test has zero
    objects for a class that exists in the master pool.
    """

    score = 0.0

    # --------------------------------------------------------
    # Image count error
    # --------------------------------------------------------

    image_total = totals["images"]

    for split in SPLITS:

        target = targets[split]["images"]
        actual = state[split]["images"]

        error = (
            (actual - target)
            / max(1.0, image_total)
        )

        score += 8.0 * error * error

    # --------------------------------------------------------
    # Per-class object error
    # --------------------------------------------------------

    for class_name in CLASSES:

        class_total = totals["classes"][
            class_name
        ]

        if class_total == 0:
            continue

        for split in SPLITS:

            target = targets[split][
                "classes"
            ][class_name]

            actual = state[split][
                "classes"
            ][class_name]

            error = (
                (actual - target)
                / max(1.0, class_total)
            )

            # High weight: class balance is the main
            # reason this v2 optimizer exists.
            score += 70.0 * error * error

            # Strong penalty if validation/test completely
            # lose a class.
            if split in ("valid", "test") and actual == 0:
                score += 1000.0

    # --------------------------------------------------------
    # Negative-image distribution
    # --------------------------------------------------------

    negative_total = totals[
        "negative_images"
    ]

    if negative_total > 0:

        for split in SPLITS:

            target = targets[split][
                "negative_images"
            ]

            actual = state[split][
                "negative_images"
            ]

            error = (
                (actual - target)
                / max(1.0, negative_total)
            )

            score += 20.0 * error * error

    return score


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------

def assignment_state(
    assignment,
    groups,
):
    state = empty_state()

    for group_name, split in assignment.items():
        add_group_to_state(
            state,
            split,
            groups[group_name],
        )

    return state


def random_initial_assignment(
    group_names,
    groups,
    targets,
    totals,
    rng,
):
    """
    Randomized greedy construction.

    Large / rare-class groups are assigned first.
    """

    assignment = {}
    state = empty_state()

    group_order = list(group_names)

    def difficulty(group_name):
        group = groups[group_name]

        rarity = 0.0

        for class_name in CLASSES:
            count = group["class_counts"][
                class_name
            ]

            total = totals["classes"][
                class_name
            ]

            if count > 0 and total > 0:
                rarity += (
                    count / total
                ) * 10.0

        return (
            rarity
            + group["image_count"] / totals[
                "images"
            ]
            + group["negative_images"] / max(
                1,
                totals["negative_images"],
            )
        )

    group_order.sort(
        key=difficulty,
        reverse=True,
    )

    # Add a small random perturbation to avoid identical restarts.
    decorated = []

    for name in group_order:
        decorated.append(
            (
                difficulty(name)
                + rng.random() * 0.05,
                name,
            )
        )

    decorated.sort(
        reverse=True
    )

    for _, group_name in decorated:

        candidates = []

        for split in SPLITS:

            add_group_to_state(
                state,
                split,
                groups[group_name],
            )

            score = objective(
                state,
                targets,
                totals,
            )

            remove_group_from_state(
                state,
                split,
                groups[group_name],
            )

            candidates.append(
                (score, split)
            )

        # Randomized choice among the best few.
        candidates.sort()

        top_n = min(
            2,
            len(candidates),
        )

        _, selected_split = rng.choice(
            candidates[:top_n]
        )

        assignment[
            group_name
        ] = selected_split

        add_group_to_state(
            state,
            selected_split,
            groups[group_name],
        )

    return assignment, state


def improve_by_moves(
    assignment,
    state,
    groups,
    targets,
    totals,
    rng,
    max_iterations,
):
    """
    Repeatedly move one source group between splits whenever
    it improves the objective.

    Random group order prevents deterministic bias.
    """

    current_score = objective(
        state,
        targets,
        totals,
    )

    group_names = list(
        groups.keys()
    )

    for _ in range(max_iterations):

        rng.shuffle(group_names)

        improved = False

        for group_name in group_names:

            current_split = assignment[
                group_name
            ]

            candidate_splits = [
                s
                for s in SPLITS
                if s != current_split
            ]

            rng.shuffle(
                candidate_splits
            )

            best_score = current_score
            best_split = None

            for new_split in candidate_splits:

                remove_group_from_state(
                    state,
                    current_split,
                    groups[group_name],
                )

                add_group_to_state(
                    state,
                    new_split,
                    groups[group_name],
                )

                score = objective(
                    state,
                    targets,
                    totals,
                )

                remove_group_from_state(
                    state,
                    new_split,
                    groups[group_name],
                )

                add_group_to_state(
                    state,
                    current_split,
                    groups[group_name],
                )

                if score + 1e-12 < best_score:
                    best_score = score
                    best_split = new_split

            if best_split is not None:

                remove_group_from_state(
                    state,
                    current_split,
                    groups[group_name],
                )

                add_group_to_state(
                    state,
                    best_split,
                    groups[group_name],
                )

                assignment[
                    group_name
                ] = best_split

                current_score = best_score
                improved = True

        if not improved:
            break

    return assignment, state


def improve_by_swaps(
    assignment,
    state,
    groups,
    targets,
    totals,
    rng,
    max_iterations,
):
    """
    Try pairwise source-group swaps between splits.

    This is slower than single-group moves but can repair situations
    where a large source group has to move together.
    """

    current_score = objective(
        state,
        targets,
        totals,
    )

    group_names = list(
        groups.keys()
    )

    for _ in range(max_iterations):

        rng.shuffle(group_names)

        improved = False

        # Limit pair search per iteration.
        # Full 525^2 search is unnecessary.
        candidates = group_names[:]

        for i in range(
            min(
                len(candidates),
                180,
            )
        ):

            a = candidates[i]
            split_a = assignment[a]

            for j in range(
                i + 1,
                min(
                    len(candidates),
                    i + 100,
                ),
            ):

                b = candidates[j]
                split_b = assignment[b]

                if split_a == split_b:
                    continue

                ga = groups[a]
                gb = groups[b]

                # Apply swap.
                remove_group_from_state(
                    state,
                    split_a,
                    ga,
                )

                remove_group_from_state(
                    state,
                    split_b,
                    gb,
                )

                add_group_to_state(
                    state,
                    split_a,
                    gb,
                )

                add_group_to_state(
                    state,
                    split_b,
                    ga,
                )

                score = objective(
                    state,
                    targets,
                    totals,
                )

                # Undo.
                remove_group_from_state(
                    state,
                    split_a,
                    gb,
                )

                remove_group_from_state(
                    state,
                    split_b,
                    ga,
                )

                add_group_to_state(
                    state,
                    split_a,
                    ga,
                )

                add_group_to_state(
                    state,
                    split_b,
                    gb,
                )

                if score + 1e-12 < current_score:

                    # Apply permanently.
                    remove_group_from_state(
                        state,
                        split_a,
                        ga,
                    )

                    remove_group_from_state(
                        state,
                        split_b,
                        gb,
                    )

                    add_group_to_state(
                        state,
                        split_a,
                        gb,
                    )

                    add_group_to_state(
                        state,
                        split_b,
                        ga,
                    )

                    assignment[a] = split_b
                    assignment[b] = split_a

                    current_score = score
                    improved = True

                    break

            if improved:
                break

        if not improved:
            break

    return assignment, state


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_assignment(
    assignment,
    groups,
    metadata,
    totals,
):
    errors = []

    # Every image assigned exactly once.
    seen_images = set()

    for group_name, split in assignment.items():

        if split not in SPLITS:
            errors.append(
                f"Invalid split: {split}"
            )

        for image_id in groups[
            group_name
        ]["images"]:

            if image_id in seen_images:
                errors.append(
                    f"Image assigned twice: {image_id}"
                )

            seen_images.add(
                image_id
            )

    if len(seen_images) != totals["images"]:
        errors.append(
            "Not all canonical images were assigned."
        )

    # Every source group must have one split.
    for group_name in groups:
        if group_name not in assignment:
            errors.append(
                f"Source group not assigned: {group_name}"
            )

    if errors:
        raise RuntimeError(
            "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def state_to_rows(
    state,
    totals,
    targets,
):
    rows = []

    for split in SPLITS:

        row = {
            "split": split,
            "images": state[split]["images"],
            "image_pct": round(
                pct(
                    state[split]["images"],
                    totals["images"],
                ),
                3,
            ),
            "target_image_pct": (
                SPLIT_FRACTIONS[split]
                * 100.0
            ),
            "negative_images": state[
                split
            ]["negative_images"],
            "target_negative_images": round(
                targets[split][
                    "negative_images"
                ],
                3,
            ),
        }

        for class_name in CLASSES:

            actual = state[split][
                "classes"
            ][class_name]

            total = totals["classes"][
                class_name
            ]

            target = targets[split][
                "classes"
            ][class_name]

            row[
                class_name
            ] = actual

            row[
                f"{class_name}_pct"
            ] = round(
                pct(
                    actual,
                    total,
                ),
                3,
            )

            row[
                f"{class_name}_target"
            ] = round(
                target,
                3,
            )

        rows.append(row)

    return rows


def print_report(
    state,
    totals,
    targets,
    score,
):
    print()
    print("=" * 100)
    print("OPTIMIZED SPLIT")
    print("=" * 100)

    print(
        f"Objective score: {score:.8f}"
    )

    print()

    header = (
        f"{'Split':<8}"
        f"{'Images':>8}"
        f"{'%':>8}"
        f"{'Neg':>8}"
    )

    for class_name in CLASSES:
        header += (
            f"{class_name:>14}"
        )

    print(header)
    print("-" * 100)

    for split in SPLITS:

        line = (
            f"{split:<8}"
            f"{state[split]['images']:>8}"
            f"{pct(state[split]['images'], totals['images']):>7.2f}%"
            f"{state[split]['negative_images']:>8}"
        )

        for class_name in CLASSES:
            line += (
                f"{state[split]['classes'][class_name]:>14}"
            )

        print(line)

    print()

    print(
        "Targets:"
    )

    for split in SPLITS:
        print(
            f"  {split:<6}: "
            f"{targets[split]['images']:.1f} images, "
            f"{targets[split]['negative_images']:.1f} negatives"
        )

        print(
            " " * 10
            + ", ".join(
                f"{c}={targets[split]['classes'][c]:.1f}"
                for c in CLASSES
            )
        )


def write_summary_report(
    output,
    state,
    totals,
    targets,
    score,
):
    rows = state_to_rows(
        state,
        totals,
        targets,
    )

    fieldnames = [
        "split",
        "images",
        "image_pct",
        "target_image_pct",
        "negative_images",
        "target_negative_images",
    ]

    for class_name in CLASSES:
        fieldnames.extend(
            [
                class_name,
                f"{class_name}_pct",
                f"{class_name}_target",
            ]
        )

    write_csv(
        output
        / "reports"
        / "split_summary.csv",
        rows,
        fieldnames,
    )

    report = {
        "objective_score": score,
        "totals": {
            "images": totals["images"],
            "negative_images": totals[
                "negative_images"
            ],
            "classes": dict(
                totals["classes"]
            ),
        },
        "targets": targets,
        "actual": {
            split: {
                "images": state[split]["images"],
                "negative_images": state[
                    split
                ]["negative_images"],
                "classes": dict(
                    state[split]["classes"]
                ),
            }
            for split in SPLITS
        },
    }

    write_json(
        output
        / "reports"
        / "split_optimization_report.json",
        report,
    )

    with (
        output
        / "reports"
        / "split_optimization_report.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "InnoCount Frozen v2 Split Optimization\n"
        )
        f.write(
            "========================================\n\n"
        )

        f.write(
            "The optimizer reassigns complete source groups.\n"
        )
        f.write(
            "No image, annotation, bbox, keypoint, class, or source group "
            "was modified.\n\n"
        )

        f.write(
            f"Objective score: {score:.8f}\n\n"
        )

        for split in SPLITS:

            f.write(
                f"{split.upper()}\n"
            )
            f.write(
                f"  Images: "
                f"{state[split]['images']} "
                f"({pct(state[split]['images'], totals['images']):.2f}%)\n"
            )

            f.write(
                f"  Negative images: "
                f"{state[split]['negative_images']}\n"
            )

            for class_name in CLASSES:

                actual = state[split][
                    "classes"
                ][class_name]

                target = targets[split][
                    "classes"
                ][class_name]

                f.write(
                    f"  {class_name}: "
                    f"{actual} "
                    f"(target {target:.1f}, "
                    f"{pct(actual, totals['classes'][class_name]):.2f}% of class)\n"
                )

            f.write("\n")


# ---------------------------------------------------------------------------
# Build v2 dataset
# ---------------------------------------------------------------------------

def create_v2_dataset(
    input_dir,
    output_dir,
    assignment,
    groups,
    images_by_id,
    annotations_by_image,
    categories,
):
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}"
        )

    # Create structure.
    for split in SPLITS:
        (
            output_dir
            / "splits"
            / split
            / "images"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    (
        output_dir
        / "reports"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # Determine image -> split.
    image_split = {}

    for group_name, split in assignment.items():

        for image_id in groups[
            group_name
        ]["images"]:

            if image_id in image_split:
                raise RuntimeError(
                    f"Image assigned twice: {image_id}"
                )

            image_split[
                image_id
            ] = split

    # --------------------------------------------------------
    # Copy images and create COCO splits.
    # --------------------------------------------------------

    split_data = {}

    # Use categories from frozen v1.
    for split in SPLITS:
        split_data[split] = {
            "info": {
                "description": (
                    "InnoCount KeyPoint Frozen v2"
                ),
                "version": "v2",
            },
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": [],
        }

    for image_id, image in images_by_id.items():

        split = image_split[
            image_id
        ]

        split_data[split][
            "images"
        ].append(image)

        split_data[split][
            "annotations"
        ].extend(
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        # Locate image in original v1.
        # Since v1 has already copied canonical images into splits,
        # search the original split using the old assignment.
        source_found = None

        for old_split in SPLITS:

            candidate = (
                input_dir
                / "splits"
                / old_split
                / "images"
                / image["file_name"]
            )

            if candidate.exists():
                source_found = candidate
                break

        if source_found is None:
            raise FileNotFoundError(
                "Could not locate frozen image: "
                f"{image['file_name']}"
            )

        destination = (
            output_dir
            / "splits"
            / split
            / "images"
            / image["file_name"]
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source_found,
            destination,
        )

    # Sort for deterministic output.
    for split in SPLITS:
        split_data[split]["images"].sort(
            key=lambda x: x["id"]
        )

        split_data[split]["annotations"].sort(
            key=lambda x: x["id"]
        )

        write_json(
            output_dir
            / "splits"
            / split
            / "annotations.json",
            split_data[split],
        )

    # --------------------------------------------------------
    # Write image-level manifest.
    # --------------------------------------------------------

    manifest_rows = []

    for image_id in sorted(
        images_by_id
    ):

        image = images_by_id[
            image_id
        ]

        split = image_split[
            image_id
        ]

        group_name = None

        for group, info in groups.items():
            if image_id in info["images"]:
                group_name = group
                break

        annotation_count = len(
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        class_counts = Counter()

        category_by_id = {
            c["id"]: c
            for c in categories
        }

        for ann in annotations_by_image.get(
            image_id,
            [],
        ):
            name = category_by_id[
                ann["category_id"]
            ]["name"]

            if name in CLASSES:
                class_counts[name] += 1

        manifest_rows.append(
            {
                "image_id": image_id,
                "file_name": image[
                    "file_name"
                ],
                "width": image["width"],
                "height": image["height"],
                "final_split": split,
                "source_group": group_name,
                "annotation_count": annotation_count,
                "negative": annotation_count == 0,
                "Channel": class_counts[
                    "Channel"
                ],
                "T-beam": class_counts[
                    "T-beam"
                ],
                "HI-Beam": class_counts[
                    "HI-Beam"
                ],
                "Angle Bar": class_counts[
                    "Angle Bar"
                ],
                "Sheet Pile": class_counts[
                    "Sheet Pile"
                ],
            }
        )

    write_csv(
        output_dir
        / "reports"
        / "master_manifest.csv",
        manifest_rows,
        [
            "image_id",
            "file_name",
            "width",
            "height",
            "final_split",
            "source_group",
            "annotation_count",
            "negative",
            "Channel",
            "T-beam",
            "HI-Beam",
            "Angle Bar",
            "Sheet Pile",
        ],
    )

    # --------------------------------------------------------
    # Dataset README
    # --------------------------------------------------------

    readme = """# InnoCount KeyPoint Frozen v2

This dataset is derived from `InnoCount_KeyPoint_Frozen_v1`.

The purpose of v2 is to improve split representativeness while preserving
the source-aware split constraint.

## What changed

Only the train/valid/test assignment of complete source groups changed.

The following were NOT changed:

- images
- image pixels
- annotations
- class IDs
- class names
- bounding boxes
- keypoint coordinates
- keypoint visibility
- skeletons
- exact duplicate canonicalization
- negative/background images

## Split policy

Target:

- train ≈ 80%
- valid ≈ 10%
- test ≈ 10%

The optimizer also targets approximately 80/10/10 object distributions
for each of the five classes.

Complete source groups are always kept in a single split.

## Reproducibility

The split optimizer is deterministic for a fixed seed.

The command and seed should be recorded in the research experiment log.
"""

    (
        output_dir
        / "README.md"
    ).write_text(
        readme,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Frozen v1 dataset",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New frozen v2 dataset",
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=60,
        help="Random optimization restarts",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=3000,
        help="Local move iterations per restart",
    )

    parser.add_argument(
        "--swap-iterations",
        type=int,
        default=20,
        help="Pairwise swap passes per restart",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260809,
        help="Random seed",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            args.input
        )

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}"
        )

    print("=" * 80)
    print(
        "InnoCount Frozen v2 Source-Aware Split Optimizer"
    )
    print("=" * 80)

    print(
        "\n[1/8] Loading frozen v1..."
    )

    (
        images_by_id,
        annotations_by_image,
        categories,
        original_split,
    ) = load_frozen_dataset(
        args.input
    )

    print(
        f"Canonical images: {len(images_by_id)}"
    )

    print(
        f"Annotations: "
        f"{sum(len(v) for v in annotations_by_image.values())}"
    )

    print(
        "\n[2/8] Loading master manifest..."
    )

    manifest_rows = load_manifest(
        args.input
    )

    print(
        f"Manifest rows: {len(manifest_rows)}"
    )

    print(
        "\n[3/8] Building image metadata and source groups..."
    )

    metadata = build_image_metadata(
        images_by_id,
        annotations_by_image,
        categories,
        manifest_rows,
    )

    groups = build_source_groups(
        metadata
    )

    totals = calculate_totals(
        metadata
    )

    targets = calculate_targets(
        totals
    )

    print(
        f"Source groups: {len(groups)}"
    )

    print(
        f"Negative images: "
        f"{totals['negative_images']}"
    )

    print(
        "Objects by class:"
    )

    for class_name in CLASSES:
        print(
            f"  {class_name:<12}: "
            f"{totals['classes'][class_name]}"
        )

    print(
        "\n[4/8] Optimizing source-group assignment..."
    )

    rng_master = random.Random(
        args.seed
    )

    best_assignment = None
    best_state = None
    best_score = float("inf")

    group_names = list(
        groups.keys()
    )

    for restart in range(
        args.restarts
    ):

        seed = rng_master.randint(
            0,
            2**32 - 1,
        )

        rng = random.Random(seed)

        assignment, state = (
            random_initial_assignment(
                group_names,
                groups,
                targets,
                totals,
                rng,
            )
        )

        assignment, state = improve_by_moves(
            assignment,
            state,
            groups,
            targets,
            totals,
            rng,
            args.iterations,
        )

        assignment, state = improve_by_swaps(
            assignment,
            state,
            groups,
            targets,
            totals,
            rng,
            args.swap_iterations,
        )

        score = objective(
            state,
            targets,
            totals,
        )

        if score < best_score:

            best_score = score
            best_assignment = dict(
                assignment
            )
            best_state = state

            print(
                f"  Restart "
                f"{restart + 1:>3}/{args.restarts}: "
                f"new best = "
                f"{best_score:.8f}"
            )

    if best_assignment is None:
        raise RuntimeError(
            "Optimizer failed to produce an assignment."
        )

    print(
        "\n[5/8] Validating assignment..."
    )

    validate_assignment(
        best_assignment,
        groups,
        metadata,
        totals,
    )

    print(
        "\n[6/8] Final split statistics..."
    )

    print_report(
        best_state,
        totals,
        targets,
        best_score,
    )

    print(
        "\n[7/8] Creating Frozen v2 dataset..."
    )

    create_v2_dataset(
        args.input,
        args.output,
        best_assignment,
        groups,
        images_by_id,
        annotations_by_image,
        categories,
    )

    # Write group assignment for reproducibility.
    group_rows = []

    for group_name in sorted(
        best_assignment
    ):

        info = groups[
            group_name
        ]

        split = best_assignment[
            group_name
        ]

        row = {
            "source_group": group_name,
            "final_split": split,
            "images": info[
                "image_count"
            ],
            "negative_images": info[
                "negative_images"
            ],
        }

        for class_name in CLASSES:
            row[class_name] = info[
                "class_counts"
            ][class_name]

        group_rows.append(row)

    write_csv(
        args.output
        / "reports"
        / "source_group_assignment.csv",
        group_rows,
        [
            "source_group",
            "final_split",
            "images",
            "negative_images",
            *CLASSES,
        ],
    )

    write_summary_report(
        args.output,
        best_state,
        totals,
        targets,
        best_score,
    )

    # Record optimizer settings.
    write_json(
        args.output
        / "reports"
        / "split_optimizer_config.json",
        {
            "input": str(
                args.input
            ),
            "output": str(
                args.output
            ),
            "seed": args.seed,
            "restarts": args.restarts,
            "iterations": args.iterations,
            "swap_iterations": args.swap_iterations,
            "split_fractions": SPLIT_FRACTIONS,
            "classes": CLASSES,
            "objective_score": best_score,
        },
    )

    print(
        "\n[8/8] COMPLETE"
    )

    print("=" * 80)
    print(
        f"Output: {args.output}"
    )
    print(
        "Original Frozen v1 was not modified."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
