#!/usr/bin/env python3

"""
InnoCount KeyPoint - HRNet Profile-Aware Test Evaluation

Evaluates a trained HRNet-W32 model on the frozen InnoCount
test split.

The evaluation consists of:

1. HRNet native PCK
2. Standard fixed-index PCK
3. Profile-level PCK
4. Semantic-role PCK
5. Structural-edge accuracy
6. Role-aware / order-invariant PCK
7. Per-profile analysis

Important:
HRNet's InnoCountDataset.evaluate() exports the exact GT and
metadata in the same order as HRNet's prediction tensor.

Therefore this script DOES NOT attempt to reconstruct HRNet's
prediction order from COCO annotation IDs.
"""

from pathlib import Path

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np
import yaml


# ============================================================
# PROFILE SCHEMAS
# ============================================================

PROFILE_SCHEMAS = {

    "Angle Bar": {
        "roles": {
            0: "corner",
            1: "tip",
            2: "tip",
            3: "unused",
            4: "unused",
            5: "unused",
        },

        "edges": [
            (1, 0),
            (0, 2),
        ],
    },

    "Channel": {

        "roles": {
            0: "tip",
            1: "corner",
            2: "corner",
            3: "tip",
            4: "unused",
            5: "unused",
        },

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    },

    "T-beam": {

        "roles": {
            0: "flange_center",
            1: "flange_tip",
            2: "flange_tip",
            3: "web_tip",
            4: "unused",
            5: "unused",
        },

        "edges": [
            (1, 0),
            (0, 2),
            (0, 3),
        ],
    },

    "HI-Beam": {

        "roles": {
            0: "upper_web_flange_junction",
            1: "upper_flange_tip",
            2: "upper_flange_tip",
            3: "lower_web_flange_junction",
            4: "lower_flange_tip",
            5: "lower_flange_tip",
        },

        "edges": [
            (2, 0),
            (0, 1),
            (0, 3),
            (5, 3),
            (3, 4),
        ],
    },

    "Sheet Pile": {

        "roles": {
            0: "tip",
            1: "corner",
            2: "corner",
            3: "tip",
            4: "unused",
            5: "unused",
        },

        "edges": [
            (0, 1),
            (1, 2),
            (2, 3),
        ],
    },
}


# ============================================================
# PROFILE ALIASES
# ============================================================

PROFILE_ALIASES = {

    "angle bar": "Angle Bar",
    "angle": "Angle Bar",

    "channel": "Channel",

    "t-beam": "T-beam",
    "t beam": "T-beam",
    "tbeam": "T-beam",

    "hi-beam": "HI-Beam",
    "hi beam": "HI-Beam",
    "h-beam": "HI-Beam",
    "h beam": "HI-Beam",

    "sheet pile": "Sheet Pile",
    "sheetpile": "Sheet Pile",
}


def normalize_profile_name(
    name
):

    if name is None:
        return None

    key = str(
        name
    ).strip().lower()

    return PROFILE_ALIASES.get(
        key,
        str(name).strip(),
    )


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(
    path_arg
):

    return (
        Path(path_arg)
        .expanduser()
        .resolve()
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_paths(
    hrnet_repo,
    weights,
    cfg,
    annotation_file,
):

    if not hrnet_repo.exists():

        raise FileNotFoundError(
            f"HRNet repository not found:\n"
            f"{hrnet_repo}"
        )

    test_script = (
        hrnet_repo
        / "tools"
        / "test.py"
    )

    if not test_script.exists():

        raise FileNotFoundError(
            f"HRNet test script not found:\n"
            f"{test_script}"
        )

    if not weights.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{weights}"
        )

    if not cfg.exists():

        raise FileNotFoundError(
            f"HRNet config not found:\n"
            f"{cfg}"
        )

    if not annotation_file.exists():

        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{annotation_file}"
        )


# ============================================================
# SYNCHRONIZE DATASET LOADER
# ============================================================

def synchronize_innocount_loader(
    hrnet_repo
):

    source = (
        Path(__file__)
        .resolve()
        .parents[2]
        / "hrnet"
        / "innocount.py"
    )

    destination = (
        hrnet_repo
        / "lib"
        / "dataset"
        / "innocount.py"
    )

    if not source.exists():

        raise FileNotFoundError(
            f"Local InnoCount loader not found:\n"
            f"{source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_text = source.read_text(
        encoding="utf-8"
    )

    destination.write_text(
        source_text,
        encoding="utf-8"
    )

    print(
        "✓ Synced dataset loader:"
    )

    print(
        f"  {source}"
    )

    print(
        f"→ {destination}"
    )


# ============================================================
# LOAD COCO INFORMATION
# ============================================================

def load_coco_metadata(
    annotation_file
):

    with open(
        annotation_file,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    category_map = {}

    for category in data.get(
        "categories",
        [],
    ):

        category_map[
            int(category["id"])
        ] = normalize_profile_name(
            category["name"]
        )

    return category_map


# ============================================================
# LOAD HRNET OUTPUT
# ============================================================

def find_required_file(
    output_dir,
    filename,
):

    candidates = list(
        Path(output_dir).rglob(
            filename
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "\nRequired HRNet file was not found:\n\n"
            f"{filename}\n\n"
            f"Searched recursively under:\n"
            f"{output_dir}"
        )

    return candidates[0]


def extract_predictions_from_hrnet_output(
    output_dir
):

    """
    Load the exact arrays exported by
    InnoCountDataset.evaluate().
    """

    output_dir = Path(
        output_dir
    )

    prediction_file = (
        find_required_file(
            output_dir,
            "hrnet_predictions.npy",
        )
    )

    gt_file = (
        find_required_file(
            output_dir,
            "hrnet_gt.npy",
        )
    )

    visibility_file = (
        find_required_file(
            output_dir,
            "hrnet_gt_visibility.npy",
        )
    )

    bbox_file = (
        find_required_file(
            output_dir,
            "hrnet_bboxes.npy",
        )
    )

    image_id_file = (
        find_required_file(
            output_dir,
            "hrnet_image_ids.npy",
        )
    )

    category_id_file = (
        find_required_file(
            output_dir,
            "hrnet_category_ids.npy",
        )
    )

    annotation_id_file = (
        find_required_file(
            output_dir,
            "hrnet_annotation_ids.npy",
        )
    )

    predictions = np.load(
        prediction_file
    )

    gt = np.load(
        gt_file
    )

    gt_vis = np.load(
        visibility_file
    )

    bboxes = np.load(
        bbox_file
    )

    image_ids = np.load(
        image_id_file
    )

    category_ids = np.load(
        category_id_file
    )

    annotation_ids = np.load(
        annotation_id_file
    )

    print(
        "\n✓ Found HRNet predictions:"
    )

    print(
        f"  {prediction_file}"
    )

    print(
        f"✓ Prediction shape: "
        f"{predictions.shape}"
    )

    print(
        f"✓ HRNet GT shape: "
        f"{gt.shape}"
    )

    print(
        f"✓ HRNet visibility shape: "
        f"{gt_vis.shape}"
    )

    print(
        f"✓ HRNet bbox shape: "
        f"{bboxes.shape}"
    )

    # --------------------------------------------------------
    # Length validation
    # --------------------------------------------------------

    n = len(
        predictions
    )

    arrays = {
        "ground truth": gt,
        "visibility": gt_vis,
        "bboxes": bboxes,
        "image IDs": image_ids,
        "category IDs": category_ids,
        "annotation IDs": annotation_ids,
    }

    for name, array in (
        arrays.items()
    ):

        if len(array) != n:

            raise ValueError(
                "\nHRNet evaluation array "
                "length mismatch.\n"
                f"Predictions: {n}\n"
                f"{name}: {len(array)}"
            )

    print(
        f"✓ All HRNet evaluation arrays "
        f"aligned: {n} samples."
    )

    return {
        "predictions": predictions,
        "gt": gt,
        "gt_vis": gt_vis,
        "bboxes": bboxes,
        "image_ids": image_ids,
        "category_ids": category_ids,
        "annotation_ids": annotation_ids,
    }

# ============================================================
# STANDARD PCK
# ============================================================

def calculate_joint_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
):
    """
    Standard fixed-index PCK.

    All visible keypoints are evaluated at their original
    annotation indices.

    Unlike profile-aware PCK, this metric does NOT perform
    keypoint permutation or equivalence matching.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    if (
        gt_vis.ndim == 3
        and gt_vis.shape[-1] == 1
    ):
        gt_vis = gt_vis[:, :, 0]

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for standard PCK evaluation."
        )

    predictions = predictions[:n]
    gt = gt[:n]
    gt_vis = gt_vis[:n]
    bboxes = bboxes[:n]

    pred_xy = predictions[
        :,
        :,
        :2,
    ]

    gt_xy = gt[
        :,
        :,
        :2,
    ]

    valid = (
        gt_vis > 0
    )

    scales = np.maximum(
        bboxes[:, 2],
        bboxes[:, 3],
    ).astype(
        np.float32
    )

    scales[
        scales <= 0
    ] = 1.0

    distances = np.linalg.norm(
        pred_xy - gt_xy,
        axis=2,
    )

    normalized = (
        distances
        / scales[:, None]
    )

    correct = (
        normalized <= threshold
    )

    total_correct = int(
        np.sum(
            correct & valid
        )
    )

    total_valid = int(
        np.sum(valid)
    )

    overall = (
        float(
            total_correct
            / total_valid
        )
        if total_valid > 0
        else 0.0
    )

    joint_pck = {}

    for j in range(
        predictions.shape[1]
    ):

        joint_valid = valid[
            :,
            j,
        ]

        joint_total = int(
            np.sum(
                joint_valid
            )
        )

        joint_correct = int(
            np.sum(
                correct[:, j]
                & joint_valid
            )
        )

        joint_pck[
            f"joint_{j + 1}"
        ] = (
            float(
                joint_correct
                / joint_total
            )
            if joint_total > 0
            else 0.0
        )

    return {
        "overall": overall,

        "joint_pck": joint_pck,

        "total_correct":
            total_correct,

        "total_valid":
            total_valid,
    }

# ============================================================
# PROFILE PCK
# ============================================================

def calculate_profile_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
):
    """
    Calculate standard fixed-index PCK for each structural profile.

    IMPORTANT:
    Only keypoints explicitly defined by the profile schema
    are evaluated. Keypoints marked as "unused" are excluded.

    Returns:
        {
            profile: {
                "instances": int,
                "correct": int,
                "valid": int,
                "pck": float,
            }
        }
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    category_ids = np.asarray(
        category_ids
    )

    # Normalize visibility from [N, 6, 1] -> [N, 6]
    if (
        gt_vis.ndim == 3
        and gt_vis.shape[-1] == 1
    ):
        gt_vis = gt_vis[:, :, 0]

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
        len(category_ids),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for profile PCK evaluation."
        )

    stats = {}

    for profile in PROFILE_SCHEMAS:

        stats[profile] = {
            "instances": 0,
            "correct": 0,
            "valid": 0,
        }

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[
            profile
        ]

        stats[profile][
            "instances"
        ] += 1

        pred = predictions[
            i,
            :,
            :2,
        ]

        gt_i = gt[
            i,
            :,
            :2,
        ]

        valid = (
            gt_vis[i] > 0
        )

        # Only joints explicitly used by this profile
        schema_valid = np.zeros(
            6,
            dtype=bool,
        )

        for joint_id, role in schema[
            "roles"
        ].items():

            if role == "unused":
                continue

            if joint_id >= 6:
                continue

            if valid[joint_id]:
                schema_valid[
                    joint_id
                ] = True

        _, _, w, h = bboxes[i]

        scale = max(
            float(w),
            float(h),
        )

        if scale <= 0:
            scale = 1.0

        distances = np.linalg.norm(
            pred - gt_i,
            axis=1,
        )

        normalized = (
            distances / scale
        )

        correct = (
            normalized <= threshold
        )

        correct_valid = (
            correct
            & schema_valid
        )

        stats[profile][
            "correct"
        ] += int(
            np.sum(
                correct_valid
            )
        )

        stats[profile][
            "valid"
        ] += int(
            np.sum(
                schema_valid
            )
        )

    for profile in stats:

        item = stats[
            profile
        ]

        if item["valid"] > 0:

            item["pck"] = float(
                item["correct"]
                / item["valid"]
            )

        else:

            item["pck"] = 0.0

    return stats

# ============================================================
# SEMANTIC ROLE PCK
# ============================================================

def calculate_role_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
):
    """
    Calculate PCK grouped by semantic keypoint role.

    Only visible and profile-defined keypoints are counted.

    A role is evaluated using its fixed keypoint index within
    the corresponding profile schema.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    category_ids = np.asarray(
        category_ids
    )

    if (
        gt_vis.ndim == 3
        and gt_vis.shape[-1] == 1
    ):
        gt_vis = gt_vis[:, :, 0]

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
        len(category_ids),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for semantic role evaluation."
        )

    role_correct = {}
    role_valid = {}

    profile_role_stats = {}

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    for profile, schema in (
        PROFILE_SCHEMAS.items()
    ):

        profile_role_stats[
            profile
        ] = {}

        for joint_id, role in (
            schema["roles"].items()
        ):

            if role == "unused":
                continue

            profile_role_stats[
                profile
            ][role] = {
                "correct": 0,
                "valid": 0,
            }

            role_correct.setdefault(
                role,
                0,
            )

            role_valid.setdefault(
                role,
                0,
            )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[
            profile
        ]

        pred = predictions[
            i,
            :,
            :2,
        ]

        gt_i = gt[
            i,
            :,
            :2,
        ]

        valid = gt_vis[i] > 0

        _, _, w, h = bboxes[i]

        scale = max(
            float(w),
            float(h),
        )

        if scale <= 0:
            scale = 1.0

        for joint_id, role in (
            schema["roles"].items()
        ):

            if role == "unused":
                continue

            if not valid[joint_id]:
                continue

            distance = np.linalg.norm(
                pred[joint_id]
                - gt_i[joint_id]
            )

            normalized_error = (
                distance / scale
            )

            role_valid[
                role
            ] += 1

            profile_role_stats[
                profile
            ][role][
                "valid"
            ] += 1

            if (
                normalized_error
                <= threshold
            ):

                role_correct[
                    role
                ] += 1

                profile_role_stats[
                    profile
                ][role][
                    "correct"
                ] += 1

    # --------------------------------------------------------
    # Calculate overall role PCK
    # --------------------------------------------------------

    overall_role_pck = {}

    for role in role_valid:

        if role_valid[role] > 0:

            overall_role_pck[
                role
            ] = float(
                role_correct[role]
                / role_valid[role]
            )

        else:

            overall_role_pck[
                role
            ] = 0.0

    # --------------------------------------------------------
    # Calculate profile-role PCK
    # --------------------------------------------------------

    for profile in profile_role_stats:

        for role in profile_role_stats[
            profile
        ]:

            item = profile_role_stats[
                profile
            ][role]

            if item["valid"] > 0:

                item["pck"] = float(
                    item["correct"]
                    / item["valid"]
                )

            else:

                item["pck"] = 0.0

    return {
        "overall_role_pck":
            overall_role_pck,

        "profile_role_pck":
            profile_role_stats,
    }

# ============================================================
# ORDER-INVARIANT / ROLE-AWARE PCK
# ============================================================

def calculate_order_invariant_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
):
    """
    Order-invariant / equivalence-aware PCK.

    The metric allows keypoints within explicitly defined
    equivalent groups to exchange their prediction assignment.

    Equivalent groups:

        Angle Bar:
            (1, 3)

        Sheet Pile:
            (0, 3)
            (1, 2)

    All other joints remain fixed-index.

    Each GT keypoint can be matched to at most one prediction.

    PCK is calculated as:

        number of correctly matched valid GT keypoints
        ------------------------------------------------
        number of valid GT keypoints

    Therefore the result is guaranteed to be in [0, 1].
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    category_ids = np.asarray(
        category_ids
    )

    if (
        gt_vis.ndim == 3
        and gt_vis.shape[-1] == 1
    ):
        gt_vis = gt_vis[:, :, 0]

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
        len(category_ids),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for order-invariant PCK evaluation."
        )

    # --------------------------------------------------------
    # Explicit equivalence definitions
    # --------------------------------------------------------

    equivalent_groups = {

        "Angle Bar": [
            (1, 2),
        ],

        "Channel": [],

        "T-beam": [],

        "HI-Beam": [],

        "Sheet Pile": [
            (0, 3),
            (1, 2),
        ],
    }

    total_correct = 0
    total_valid = 0

    profile_stats = {}

    for profile in PROFILE_SCHEMAS:

        profile_stats[profile] = {
            "instances": 0,
            "correct": 0,
            "valid": 0,
        }

    # --------------------------------------------------------
    # Evaluate each instance
    # --------------------------------------------------------

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[
            profile
        ]

        pred = predictions[
            i,
            :,
            :2,
        ]

        gt_i = gt[
            i,
            :,
            :2,
        ]

        valid = (
            gt_vis[i] > 0
        )

        _, _, w, h = bboxes[i]

        scale = max(
            float(w),
            float(h),
        )

        if scale <= 0:
            scale = 1.0

        # ----------------------------------------------------
        # Only schema-defined joints participate
        # ----------------------------------------------------

        schema_joints = []

        for joint_id, role in schema[
            "roles"
        ].items():

            if role == "unused":
                continue

            if joint_id >= 6:
                continue

            if valid[joint_id]:
                schema_joints.append(
                    joint_id
                )

        valid_count = len(
            schema_joints
        )

        if valid_count == 0:
            continue

        # Track which GT joints were correctly matched.
        matched = np.zeros(
            6,
            dtype=bool,
        )

        # ----------------------------------------------------
        # Fixed-index joints
        # ----------------------------------------------------

        grouped_joints = set()

        for group in equivalent_groups.get(
            profile,
            []
        ):
            grouped_joints.update(
                group
            )

        for joint_id in schema_joints:

            if joint_id in grouped_joints:
                continue

            distance = np.linalg.norm(
                pred[joint_id]
                - gt_i[joint_id]
            )

            normalized_error = (
                distance / scale
            )

            if normalized_error <= threshold:

                matched[
                    joint_id
                ] = True

        # ----------------------------------------------------
        # Equivalent groups
        # ----------------------------------------------------

        for group in equivalent_groups.get(
            profile,
            []
        ):

            # Only use GT joints that are actually
            # part of this profile and visible.
            valid_gt = [
                joint_id
                for joint_id in group
                if (
                    joint_id in schema_joints
                    and valid[joint_id]
                )
            ]

            if not valid_gt:
                continue

            # Predictions corresponding to the same
            # equivalent group.
            valid_pred = [
                joint_id
                for joint_id in group
            ]

            # There must be the same number of
            # predictions and visible GT joints.
            if len(valid_gt) != len(
                valid_pred
            ):
                continue

            import itertools

            best_assignment = None
            best_total = float(
                "inf"
            )

            # Find minimum-cost one-to-one assignment.
            for permutation in itertools.permutations(
                valid_pred
            ):

                total_distance = 0.0

                for gt_idx, pred_idx in zip(
                    valid_gt,
                    permutation,
                ):

                    total_distance += (
                        np.linalg.norm(
                            pred[pred_idx]
                            - gt_i[gt_idx]
                        )
                    )

                if (
                    total_distance
                    < best_total
                ):

                    best_total = (
                        total_distance
                    )

                    best_assignment = (
                        permutation
                    )

            if best_assignment is None:
                continue

            # Evaluate the selected assignment.
            for gt_idx, pred_idx in zip(
                valid_gt,
                best_assignment,
            ):

                error = np.linalg.norm(
                    pred[pred_idx]
                    - gt_i[gt_idx]
                )

                normalized_error = (
                    error / scale
                )

                if (
                    normalized_error
                    <= threshold
                ):

                    matched[
                        gt_idx
                    ] = True

        # ----------------------------------------------------
        # Count this instance exactly once
        # ----------------------------------------------------

        correct_count = int(
            np.sum(
                matched[
                    schema_joints
                ]
            )
        )

        total_correct += (
            correct_count
        )

        total_valid += (
            valid_count
        )

        profile_stats[
            profile
        ][
            "instances"
        ] += 1

        profile_stats[
            profile
        ][
            "correct"
        ] += correct_count

        profile_stats[
            profile
        ][
            "valid"
        ] += valid_count

    # --------------------------------------------------------
    # Calculate PCK
    # --------------------------------------------------------

    overall = (
        float(
            total_correct
            / total_valid
        )
        if total_valid > 0
        else 0.0
    )

    # Safety assertion
    if not (
        0.0
        <= overall
        <= 1.0
    ):
        raise RuntimeError(
            "Order-invariant PCK is outside [0, 1]: "
            f"{overall}. "
            f"correct={total_correct}, "
            f"valid={total_valid}"
        )

    for profile in profile_stats:

        item = profile_stats[
            profile
        ]

        if item["valid"] > 0:

            item["pck"] = float(
                item["correct"]
                / item["valid"]
            )

        else:

            item["pck"] = 0.0

    return {
        "overall": overall,

        "correct": int(
            total_correct
        ),

        "total": int(
            total_valid
        ),

        "profile": profile_stats,
    }

# ============================================================
# STRUCTURAL EDGE ACCURACY
# ============================================================

def calculate_edge_accuracy(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
):
    """
    Evaluate structural edges defined by PROFILE_SCHEMAS.

    An edge is correct when the mean endpoint localization
    error is <= threshold * bbox scale.

    Only edges whose two GT endpoints are visible are counted.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    gt_vis = np.asarray(
        gt_vis,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    category_ids = np.asarray(
        category_ids
    )

    if (
        gt_vis.ndim == 3
        and gt_vis.shape[-1] == 1
    ):
        gt_vis = gt_vis[:, :, 0]

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
        len(category_ids),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for structural edge evaluation."
        )

    total_edges = 0
    correct_edges = 0

    profile_stats = {}

    for profile in PROFILE_SCHEMAS:

        profile_stats[
            profile
        ] = {
            "edges": 0,
            "correct": 0,
        }

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[
            profile
        ]

        pred = predictions[
            i,
            :,
            :2,
        ]

        gt_i = gt[
            i,
            :,
            :2,
        ]

        valid = gt_vis[i] > 0

        _, _, w, h = bboxes[i]

        scale = max(
            float(w),
            float(h),
        )

        if scale <= 0:
            scale = 1.0

        for a, b in schema["edges"]:

            if not valid[a]:
                continue

            if not valid[b]:
                continue

            endpoint_error = (
                np.linalg.norm(
                    pred[a]
                    - gt_i[a]
                )
                +
                np.linalg.norm(
                    pred[b]
                    - gt_i[b]
                )
            ) / 2.0

            normalized_error = (
                endpoint_error
                / scale
            )

            total_edges += 1

            profile_stats[
                profile
            ][
                "edges"
            ] += 1

            if (
                normalized_error
                <= threshold
            ):

                correct_edges += 1

                profile_stats[
                    profile
                ][
                    "correct"
                ] += 1

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    overall = (
        float(
            correct_edges
            / total_edges
        )
        if total_edges > 0
        else 0.0
    )

    for profile in profile_stats:

        item = profile_stats[
            profile
        ]

        item["accuracy"] = (
            float(
                item["correct"]
                / item["edges"]
            )
            if item["edges"] > 0
            else 0.0
        )

    return {
        "overall": overall,

        "correct": int(
            correct_edges
        ),

        "total": int(
            total_edges
        ),

        "profile": profile_stats,
    }

# ============================================================
# NATIVE HRNET METRIC PARSER
# ============================================================

def parse_hrnet_metrics(
    output
):

    metrics = {}

    lines = output.splitlines()

    header = None
    values = None

    for i, line in enumerate(
        lines
    ):

        if "| Arch" in line:

            header = line

            for j in range(
                i + 1,
                min(
                    i + 4,
                    len(lines)
                ),
            ):

                candidate = (
                    lines[j].strip()
                )

                if (
                    candidate.startswith("|")
                    and
                    "pose_hrnet"
                    in candidate
                ):

                    values = candidate
                    break

            if values is not None:
                break

    if (
        header is not None
        and values is not None
    ):

        header_fields = [
            x.strip()
            for x in
            header.strip(
                "|"
            ).split("|")
        ]

        value_fields = [
            x.strip()
            for x in
            values.strip(
                "|"
            ).split("|")
        ]

        for name, value in zip(
            header_fields,
            value_fields,
        ):

            if name == "Arch":
                continue

            try:

                metrics[
                    name
                ] = float(value)

            except ValueError:
                pass

    import re

    matches = re.findall(
        r"Test:\s*\[\d+/\d+\].*?"
        r"Accuracy\s+[0-9.]+\s+"
        r"\(([0-9.]+)\)",
        output,
    )

    if matches:

        metrics[
            "test_accuracy"
        ] = float(
            matches[-1]
        )

    return metrics


# ============================================================
# SAVE CSV
# ============================================================

def save_profile_csv(
    path,
    profile_pck,
    role_pck,
    edge_accuracy,
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Profile",
                "Instances",
                "PCK",
                "Role_PCK",
                "Edge_Accuracy",
            ]
        )

        for profile in (
            PROFILE_SCHEMAS
        ):

            instances = (
                profile_pck[
                    profile
                ][
                    "instances"
                ]
            )

            pck = (
                profile_pck[
                    profile
                ][
                    "pck"
                ]
            )

            roles = (
                role_pck[
                    "profile_role_pck"
                ][profile]
            )

            role_values = [
                item["pck"]
                for item in (
                    roles.values()
                )
                if item["valid"] > 0
            ]

            role_average = (
                float(
                    np.mean(
                        role_values
                    )
                )
                if role_values
                else 0.0
            )

            edge = (
                edge_accuracy[
                    "profile"
                ][profile][
                    "accuracy"
                ]
            )

            writer.writerow(
                [
                    profile,
                    instances,
                    f"{pck:.6f}",
                    f"{role_average:.6f}",
                    f"{edge:.6f}",
                ]
            )


# ============================================================
# REPORT
# ============================================================

def write_report(
    save_dir,
    weights,
    cfg,
    metrics,
    start_time,
    end_time,
    return_code,
):

    report_path = (
        save_dir
        / "evaluation_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HRNET-W32 PROFILE-AWARE TEST EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Start time : "
            f"{start_time}\n"
        )

        f.write(
            f"End time   : "
            f"{end_time}\n"
        )

        f.write(
            f"Checkpoint : "
            f"{weights}\n"
        )

        f.write(
            f"Evaluation Config     : "
            f"{cfg}\n"
        )

        f.write(
            "Test split : test\n\n"
        )

        # ----------------------------------------------------
        # Standard PCK
        # ----------------------------------------------------

        f.write(
            "STANDARD PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        standard = metrics[
            "standard_pck"
        ]

        f.write(
            f"Overall : "
            f"{standard['overall']:.6f}\n"
        )

        for joint, value in (
            standard[
                "joint_pck"
            ].items()
        ):

            f.write(
                f"{joint:<12}: "
                f"{value:.6f}\n"
            )

        # ----------------------------------------------------
        # Order invariant
        # ----------------------------------------------------

        f.write(
            "\nORDER-INVARIANT PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        invariant = metrics[
            "order_invariant_pck"
        ]

        f.write(
            f"Overall : "
            f"{invariant['overall']:.6f}\n"
        )

        for profile, item in (
            invariant[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile:<15} "
                f"{item['pck']:.6f}\n"
            )

        gain = (
            invariant["overall"]
            - standard["overall"]
        )

        f.write(
            "\nOrder-invariance gain : "
            f"{gain:.6f}\n"
        )

        f.write(
            "Percentage points     : "
            f"{gain * 100:.2f}\n"
        )

        # ----------------------------------------------------
        # Profile PCK
        # ----------------------------------------------------

        f.write(
            "\nPROFILE-LEVEL PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for profile, item in (
            metrics[
                "profile_pck"
            ].items()
        ):

            f.write(
                f"{profile:<15} "
                f"instances="
                f"{item['instances']:<5} "
                f"PCK="
                f"{item['pck']:.6f}\n"
            )

        # ----------------------------------------------------
        # Semantic roles
        # ----------------------------------------------------

        f.write(
            "\nSEMANTIC ROLE PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for role, value in (
            metrics[
                "semantic_roles"
            ][
                "overall_role_pck"
            ].items()
        ):

            f.write(
                f"{role:<35}: "
                f"{value:.6f}\n"
            )

        # ----------------------------------------------------
        # Structural edges
        # ----------------------------------------------------

        f.write(
            "\nSTRUCTURAL EDGE ACCURACY\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        edges = metrics[
            "structural_edges"
        ]

        f.write(
            f"Overall : "
            f"{edges['overall']:.6f}\n"
        )

        for profile, item in (
            edges[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile:<15} "
                f"{item['accuracy']:.6f}\n"
            )

        # ----------------------------------------------------
        # Native HRNet accuracy
        # ----------------------------------------------------

        if (
            "hrnet_test_accuracy"
            in metrics
        ):

            f.write(
                "\nHRNET TEST ACCURACY\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            f.write(
                "Final running accuracy: "
                f"{metrics['hrnet_test_accuracy']:.6f}\n"
            )

        f.write(
            "\nHRNet test.py return code: "
            f"{return_code}\n"
        )


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(
    args
):

    # --------------------------------------------------------
    # Project root
    #
    # evaluate_hrnet.py is now:
    #
    # scripts/evaluate/evaluate_hrnet.py
    #
    # Therefore parents[2] = project root.
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    hrnet_repo = resolve_path(
        args.hrnet_repo
    )

    weights = resolve_path(
        args.weights
    )

    cfg = resolve_path(
        args.cfg
    )

    project = resolve_path(
        args.project
    )

    annotation_file = resolve_path(
        args.annotation_file
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    save_dir = (
        project
        / args.name
        / timestamp
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_paths(
        hrnet_repo,
        weights,
        cfg,
        annotation_file,
    )

    # --------------------------------------------------------
    # Synchronize loader
    # --------------------------------------------------------

    synchronize_innocount_loader(
        hrnet_repo
    )

    # --------------------------------------------------------
    # Load category metadata
    # --------------------------------------------------------

    category_map = (
        load_coco_metadata(
            annotation_file
        )
    )

    # --------------------------------------------------------
    # Create evaluation config
    # --------------------------------------------------------

    with open(
        cfg,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(
            f
        )

    config[
        "DATASET"
    ][
        "TEST_SET"
    ] = "test"

    config[
        "OUTPUT_DIR"
    ] = str(
        save_dir
    )

    eval_cfg = (
        save_dir
        / "hrnet_eval_config.yaml"
    )

    with open(
        eval_cfg,
        "w",
        encoding="utf-8",
    ) as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = os.environ.copy()

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        args.device
    )

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    lib_path = (
        hrnet_repo
        / "lib"
    )

    existing = env.get(
        "PYTHONPATH",
        "",
    )

    env[
        "PYTHONPATH"
    ] = (
        f"{lib_path}:{existing}"
        if existing
        else str(lib_path)
    )

    # --------------------------------------------------------
    # Run HRNet
    # --------------------------------------------------------

    cmd = [
        sys.executable,
        "tools/test.py",
        "--cfg",
        str(eval_cfg),
        "TEST.MODEL_FILE",
        str(weights),
    ]

    print(
        "=" * 70
    )

    print(
        "RUNNING HRNET TEST"
    )

    print(
        "=" * 70
    )

    print()

    start_time = (
        datetime.now().isoformat()
    )

    result = subprocess.run(
        cmd,
        cwd=hrnet_repo,
        env=env,
        capture_output=True,
        text=True,
    )

    end_time = (
        datetime.now().isoformat()
    )

    # --------------------------------------------------------
    # Save native output
    # --------------------------------------------------------

    output = (
        result.stdout
    )

    if result.stderr:

        output += (
            "\n\n"
            + "=" * 70
            + "\nSTDERR\n"
            + "=" * 70
            + "\n"
            + result.stderr
        )

    raw_output_path = (
        save_dir
        / "hrnet_test_output.txt"
    )

    with open(
        raw_output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            output
        )

    print(
        output
    )

    # --------------------------------------------------------
    # Parse native metrics
    # --------------------------------------------------------

    native_metrics = (
        parse_hrnet_metrics(
            output
        )
    )

    # --------------------------------------------------------
    # If HRNet itself failed, stop here
    # --------------------------------------------------------

    if result.returncode != 0:

        raise RuntimeError(
            "HRNet tools/test.py failed.\n"
            f"See:\n{raw_output_path}"
        )

    # --------------------------------------------------------
    # Load exact predictions / GT
    # --------------------------------------------------------

    hrnet_data = (
        extract_predictions_from_hrnet_output(
            save_dir
        )
    )

    predictions = (
        hrnet_data[
            "predictions"
        ]
    )

    gt = (
        hrnet_data[
            "gt"
        ]
    )

    gt_vis = (
        hrnet_data[
            "gt_vis"
        ]
    )

    bboxes = (
        hrnet_data[
            "bboxes"
        ]
    )

    category_ids = (
        hrnet_data[
            "category_ids"
        ]
    )

    # --------------------------------------------------------
    # Standard PCK
    # --------------------------------------------------------

    standard = calculate_joint_pck(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Profile PCK
    # --------------------------------------------------------

    profile_pck = calculate_profile_pck(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Semantic role PCK
    # --------------------------------------------------------

    semantic_roles = calculate_role_pck(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Order-invariant PCK
    # --------------------------------------------------------

    order_invariant = calculate_order_invariant_pck(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Structural edges
    # --------------------------------------------------------

    structural_edges = calculate_edge_accuracy(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {

        "model": "HRNet-W32",

        "test_split": "test",

        "threshold": args.threshold,

        "standard_pck": standard,

        "order_invariant_pck": (
            order_invariant
        ),

        "profile_pck": profile_pck,

        "semantic_roles": (
            semantic_roles
        ),

        "structural_edges": (
            structural_edges
        ),

        "profile_schemas": (
            PROFILE_SCHEMAS
        ),
    }

    if (
        "test_accuracy"
        in native_metrics
    ):

        metrics[
            "hrnet_test_accuracy"
        ] = native_metrics[
            "test_accuracy"
        ]

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    report = {

        "timestamp":
            datetime.now().isoformat(),

        "model":
            "HRNet-W32",

        "weights":
            str(weights),

        "config":
            str(cfg),

        "test_split":
            "test",

        "start_time":
            start_time,

        "end_time":
            end_time,

        "metrics":
            metrics,

        "process": {

            "return_code":
                result.returncode,

            "status":
                "completed",
        },
    }

    with open(
        save_dir
        / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    save_profile_csv(
        save_dir
        / "profile_metrics.csv",
        profile_pck,
        semantic_roles,
        structural_edges,
    )

    # --------------------------------------------------------
    # Save TXT
    # --------------------------------------------------------

    write_report(
        save_dir=save_dir,
        weights=weights,
        cfg=cfg,
        metrics=metrics,
        start_time=start_time,
        end_time=end_time,
        return_code=result.returncode,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "RESULTS"
    )
    print(
        "=" * 70
    )

    print(
        "\nNative HRNet PCK"
    )

    if "PCK" in native_metrics:

        print(
            f"  Overall : "
            f"{native_metrics['PCK']:.6f}"
        )

    print(
        "\nExternal Standard PCK"
    )

    print(
        f"  Overall : "
        f"{standard['overall']:.6f}"
    )

    print(
        "\nOrder-Invariant PCK"
    )

    print(
        f"  Overall : "
        f"{order_invariant['overall']:.6f}"
    )

    gain = (
        order_invariant[
            "overall"
        ]
        -
        standard[
            "overall"
        ]
    )

    print(
        "\nOrder-Invariance Gain"
    )

    print(
        f"  Absolute : "
        f"{gain:.6f}"
    )

    print(
        f"  Percentage points : "
        f"{gain * 100:.2f}"
    )

    print(
        "\nProfile PCK"
    )

    for profile, item in (
        profile_pck.items()
    ):

        print(
            f"  {profile:<15} "
            f"{item['pck']:.6f}"
        )

    print(
        "\nSemantic Role PCK"
    )

    for role, value in (
        semantic_roles[
            "overall_role_pck"
        ].items()
    ):

        print(
            f"  {role:<35} "
            f"{value:.6f}"
        )

    print(
        "\nStructural Edge Accuracy"
    )

    print(
        f"  Overall : "
        f"{structural_edges['overall']:.6f}"
    )

    if (
        "test_accuracy"
        in native_metrics
    ):

        print(
            "\nHRNet Test Accuracy"
        )

        print(
            f"  Final running accuracy : "
            f"{native_metrics['test_accuracy']:.6f}"
        )

    print(
        "\nSaved to:"
    )

    print(
        save_dir
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HRNet-W32 on the "
            "InnoCount frozen test split."
        )
    )

    parser.add_argument(
        "--hrnet-repo",
        default=str(
            Path.home()
            / "HRNet-Human-Pose-Estimation"
        ),
        help=(
            "Path to "
            "HRNet-Human-Pose-Estimation"
        ),
    )

    parser.add_argument(
        "--weights",
        required=True,
        help=(
            "Path to HRNet checkpoint."
        ),
    )

    parser.add_argument(
        "--cfg",
        default=str(
            Path.home()
            / "HRNet-Human-Pose-Estimation"
            / "experiments"
            / "innocount"
            / "hrnet_w32_640.yaml"
        ),
        help=(
            "HRNet config."
        ),
    )

    parser.add_argument(
        "--annotation-file",
        default=str(
            project_root
            / "datasets"
            / "InnoCount_KeyPoint_ModelReady_v2"
            / "splits"
            / "test"
            / "annotations.json"
        ),
        help=(
            "Frozen InnoCount test annotation file."
        ),
    )

    parser.add_argument(
        "--project",
        default=str(
            project_root
            / "experiments"
            / "HRNet_Runs"
        ),
        help=(
            "Evaluation output root."
        ),
    )

    parser.add_argument(
        "--name",
        default=(
            "E07_HRNet_W32_ProfileEval"
        ),
        help=(
            "Evaluation run name."
        ),
    )

    parser.add_argument(
        "--device",
        default="0",
        help=(
            "GPU device index."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help=(
            "PCK normalized distance "
            "threshold."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    evaluate(
        args
    )