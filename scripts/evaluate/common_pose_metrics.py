#!/usr/bin/env python3
"""
InnoCount KeyPoint - Common Profile-Aware Pose Metrics

Shared evaluation metrics for:
    - HRNet-W32
    - YOLO26-Pose
    - Other compatible pose models

All metrics operate on the same representation:

    predictions : [N, 6, 2] pixel or normalized coordinates
    gt          : [N, 6, 2] same coordinate system
    gt_vis      : [N, 6]
    bboxes      : [N, 4] in [x, y, width, height]
    category_ids: [N]

The evaluator normalizes keypoint error using:

    max(bbox_width, bbox_height)

PCK threshold default:

    0.05

IMPORTANT
---------
This file intentionally contains the corrected Angle Bar equivalence:

    Angle Bar: G2 <-> G3

because Angle Bar uses:
    G1 = corner / heel
    G2 = tip
    G3 = tip

Only explicitly equivalent keypoints are allowed to exchange
prediction assignments.

HI-Beam upper/lower groups are NOT treated as interchangeable.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

GLOBAL_K = 6

CLASS_ID_TO_NAME = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}

CLASS_NAMES = list(CLASS_ID_TO_NAME.values())


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

        # Corrected:
        # G2 and G3 are the two equivalent free-leg tips.
        "equivalent_groups": [
            (1, 2),
        ],

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

        "equivalent_groups": [],

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

        "equivalent_groups": [
            (1, 2),
        ],

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

        # Do NOT assume upper/lower interchangeability.
        "equivalent_groups": [
            (1, 2),
            (4, 5),
        ],

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

        "equivalent_groups": [
            (0, 3),
            (1, 2),
        ],

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


def normalize_profile_name(name):
    if name is None:
        return None

    key = str(name).strip().lower()

    return PROFILE_ALIASES.get(
        key,
        str(name).strip(),
    )


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def _as_arrays(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    prediction_valid=None,
):
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
        category_ids,
    )

    if gt_vis.ndim == 3 and gt_vis.shape[-1] == 1:
        gt_vis = gt_vis[:, :, 0]

    if predictions.ndim != 3:
        raise ValueError(
            f"predictions must have shape [N,K,2], got "
            f"{predictions.shape}"
        )

    if gt.ndim != 3:
        raise ValueError(
            f"gt must have shape [N,K,2], got "
            f"{gt.shape}"
        )

    if predictions.shape[1] != GLOBAL_K:
        raise ValueError(
            f"predictions must have K={GLOBAL_K}, got "
            f"{predictions.shape}"
        )

    if gt.shape[1] != GLOBAL_K:
        raise ValueError(
            f"gt must have K={GLOBAL_K}, got "
            f"{gt.shape}"
        )

    if gt_vis.shape[1] != GLOBAL_K:
        raise ValueError(
            f"gt_vis must have K={GLOBAL_K}, got "
            f"{gt_vis.shape}"
        )

    if bboxes.shape[1] != 4:
        raise ValueError(
            f"bboxes must have shape [N,4], got "
            f"{bboxes.shape}"
        )

    n = min(
        len(predictions),
        len(gt),
        len(gt_vis),
        len(bboxes),
        len(category_ids),
    )

    if n == 0:
        raise RuntimeError(
            "No samples available for evaluation."
        )

    predictions = predictions[:n]
    gt = gt[:n]
    gt_vis = gt_vis[:n]
    bboxes = bboxes[:n]
    category_ids = category_ids[:n]

    if prediction_valid is None:
        prediction_valid = np.ones(
            (n, GLOBAL_K),
            dtype=bool,
        )
    else:
        prediction_valid = np.asarray(
            prediction_valid,
            dtype=bool,
        )

        if prediction_valid.ndim == 1:
            prediction_valid = np.repeat(
                prediction_valid[:, None],
                GLOBAL_K,
                axis=1,
            )

        prediction_valid = prediction_valid[:n]

        if prediction_valid.shape != (n, GLOBAL_K):
            raise ValueError(
                "prediction_valid must have shape "
                f"[{n},{GLOBAL_K}], got "
                f"{prediction_valid.shape}"
            )

    return (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    )


def bbox_scales(bboxes):
    """
    Return max(width, height) for every bbox.
    """

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    scales = np.maximum(
        bboxes[:, 2],
        bboxes[:, 3],
    )

    scales = scales.astype(
        np.float32,
    )

    scales[scales <= 0] = 1.0

    return scales


def normalized_errors(
    predictions,
    gt,
    bboxes,
):
    """
    Per-keypoint normalized Euclidean error.

    Returns:
        [N, K]
    """

    pred_xy = np.asarray(
        predictions,
        dtype=np.float32,
    )[:, :, :2]

    gt_xy = np.asarray(
        gt,
        dtype=np.float32,
    )[:, :, :2]

    scales = bbox_scales(bboxes)

    distances = np.linalg.norm(
        pred_xy - gt_xy,
        axis=2,
    )

    return distances / scales[:, None]


# ============================================================
# STANDARD FIXED-INDEX PCK
# ============================================================

def calculate_joint_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids=None,
    category_map=None,
    threshold=0.05,
    prediction_valid=None,
):
    """
    Standard fixed-index PCK.

    Every visible GT keypoint is evaluated against the prediction
    at the same global keypoint index.

    No equivalence/permutation is applied.
    """

    (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    ) = _as_arrays(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids
        if category_ids is not None
        else np.zeros(len(predictions), dtype=int),
        prediction_valid,
    )

    errors = normalized_errors(
        predictions,
        gt,
        bboxes,
    )

    gt_valid = gt_vis > 0

    correct = (
        errors <= threshold
    )

    # Missing YOLO detections are failures.
    correct &= prediction_valid

    total_correct = int(
        np.sum(
            correct & gt_valid
        )
    )

    total_valid = int(
        np.sum(gt_valid)
    )

    overall = (
        float(total_correct / total_valid)
        if total_valid > 0
        else 0.0
    )

    joint_pck = {}

    for j in range(GLOBAL_K):

        joint_valid = gt_valid[:, j]

        joint_total = int(
            np.sum(joint_valid)
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
                joint_correct / joint_total
            )
            if joint_total > 0
            else 0.0
        )

    return {
        "overall": overall,
        "joint_pck": joint_pck,
        "total_correct": total_correct,
        "total_valid": total_valid,
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
    prediction_valid=None,
):
    """
    Fixed-index PCK calculated independently for each profile.

    Only profile-defined keypoints are evaluated.
    """

    (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    ) = _as_arrays(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
    )

    errors = normalized_errors(
        predictions,
        gt,
        bboxes,
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

        schema = PROFILE_SCHEMAS[profile]

        stats[profile]["instances"] += 1

        for joint_id, role in schema["roles"].items():

            if role == "unused":
                continue

            if joint_id >= GLOBAL_K:
                continue

            if not gt_vis[i, joint_id] > 0:
                continue

            stats[profile]["valid"] += 1

            if (
                prediction_valid[i, joint_id]
                and errors[i, joint_id] <= threshold
            ):
                stats[profile]["correct"] += 1

    for profile, item in stats.items():

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
    prediction_valid=None,
):
    """
    PCK grouped by semantic role.

    Example:
        corner
        tip
        flange_center
        flange_tip
        web_tip
        ...

    Roles remain tied to their semantic global keypoint index.
    """

    (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    ) = _as_arrays(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
    )

    errors = normalized_errors(
        predictions,
        gt,
        bboxes,
    )

    role_correct = {}
    role_valid = {}

    profile_role_stats = {}

    for profile, schema in PROFILE_SCHEMAS.items():

        profile_role_stats[profile] = {}

        for joint_id, role in schema["roles"].items():

            if role == "unused":
                continue

            profile_role_stats[profile][role] = {
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

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[profile]

        for joint_id, role in schema["roles"].items():

            if role == "unused":
                continue

            if not gt_vis[i, joint_id] > 0:
                continue

            role_valid[role] += 1

            profile_role_stats[
                profile
            ][role]["valid"] += 1

            if (
                prediction_valid[i, joint_id]
                and errors[i, joint_id] <= threshold
            ):
                role_correct[role] += 1

                profile_role_stats[
                    profile
                ][role]["correct"] += 1

    overall_role_pck = {}

    for role in role_valid:

        if role_valid[role] > 0:
            overall_role_pck[role] = float(
                role_correct[role]
                / role_valid[role]
            )
        else:
            overall_role_pck[role] = 0.0

    for profile in profile_role_stats:

        for role in profile_role_stats[profile]:

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
        "overall_role_pck": overall_role_pck,
        "profile_role_pck": profile_role_stats,
    }


# ============================================================
# ORDER-INVARIANT PCK
# ============================================================

def calculate_order_invariant_pck(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
    prediction_valid=None,
):
    """
    Equivalence-aware PCK.

    Equivalent keypoints may exchange prediction assignments.

    Corrected equivalence definitions:

        Angle Bar:
            G2 <-> G3

        T-beam:
            G2 <-> G3

        HI-Beam:
            G2 <-> G3
            G5 <-> G6

        Sheet Pile:
            G1 <-> G4
            G2 <-> G3

        Channel:
            none

    Each GT keypoint can receive at most one prediction.
    """

    (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    ) = _as_arrays(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
    )

    equivalent_groups = {
        profile: schema["equivalent_groups"]
        for profile, schema in PROFILE_SCHEMAS.items()
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

    errors = normalized_errors(
        predictions,
        gt,
        bboxes,
    )

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[profile]

        schema_joints = []

        for joint_id, role in schema["roles"].items():

            if role == "unused":
                continue

            if joint_id >= GLOBAL_K:
                continue

            if gt_vis[i, joint_id] > 0:
                schema_joints.append(joint_id)

        valid_count = len(schema_joints)

        if valid_count == 0:
            continue

        matched = np.zeros(
            GLOBAL_K,
            dtype=bool,
        )

        grouped_joints = set()

        for group in equivalent_groups.get(
            profile,
            [],
        ):
            grouped_joints.update(group)

        # ----------------------------------------------------
        # Fixed-index joints
        # ----------------------------------------------------

        for joint_id in schema_joints:

            if joint_id in grouped_joints:
                continue

            if (
                prediction_valid[i, joint_id]
                and errors[i, joint_id] <= threshold
            ):
                matched[joint_id] = True

        # ----------------------------------------------------
        # Equivalent groups
        # ----------------------------------------------------

        for group in equivalent_groups.get(
            profile,
            [],
        ):

            valid_gt = [
                joint_id
                for joint_id in group
                if (
                    joint_id in schema_joints
                    and gt_vis[i, joint_id] > 0
                )
            ]

            if not valid_gt:
                continue

            valid_pred = [
                joint_id
                for joint_id in group
                if prediction_valid[i, joint_id]
            ]

            # If a prediction is missing, the corresponding GT
            # remains incorrect rather than disappearing.
            #
            # We solve the assignment only over available
            # predictions, then count unmatched GT as failures.

            if not valid_pred:
                continue

            best_assignment = None
            best_total = float("inf")

            for permutation_length in range(
                min(
                    len(valid_gt),
                    len(valid_pred),
                ),
                min(
                    len(valid_gt),
                    len(valid_pred),
                ) + 1,
            ):
                pass

            # Because groups are tiny (maximum 2), enumerate
            # all assignments directly.

            if len(valid_gt) == 1:

                gt_idx = valid_gt[0]

                best_pred = min(
                    valid_pred,
                    key=lambda p: errors[i, p]
                )

                if errors[i, best_pred] <= threshold:
                    matched[gt_idx] = True

            else:

                for permutation in itertools.permutations(
                    valid_pred,
                    len(valid_gt),
                ):

                    total_distance = 0.0

                    for gt_idx, pred_idx in zip(
                        valid_gt,
                        permutation,
                    ):
                        total_distance += (
                            errors[i, pred_idx]
                        )

                    if total_distance < best_total:

                        best_total = total_distance
                        best_assignment = permutation

                if best_assignment is not None:

                    for gt_idx, pred_idx in zip(
                        valid_gt,
                        best_assignment,
                    ):

                        if (
                            errors[i, pred_idx]
                            <= threshold
                        ):
                            matched[gt_idx] = True

        correct_count = int(
            np.sum(
                matched[schema_joints]
            )
        )

        total_correct += correct_count
        total_valid += valid_count

        profile_stats[
            profile
        ]["instances"] += 1

        profile_stats[
            profile
        ]["correct"] += correct_count

        profile_stats[
            profile
        ]["valid"] += valid_count

    overall = (
        float(
            total_correct
            / total_valid
        )
        if total_valid > 0
        else 0.0
    )

    if not (
        0.0 <= overall <= 1.0
    ):
        raise RuntimeError(
            "Order-invariant PCK outside [0,1]: "
            f"{overall}"
        )

    for profile, item in profile_stats.items():

        if item["valid"] > 0:
            item["pck"] = float(
                item["correct"]
                / item["valid"]
            )
        else:
            item["pck"] = 0.0

    return {
        "overall": overall,
        "correct": int(total_correct),
        "total": int(total_valid),
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
    prediction_valid=None,
):
    """
    An edge is correct when the mean endpoint localization
    error is <= threshold * bbox scale.

    Both GT endpoints must be visible.

    Missing predictions cause the edge to fail.
    """

    (
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
        n,
    ) = _as_arrays(
        predictions,
        gt,
        gt_vis,
        bboxes,
        category_ids,
        prediction_valid,
    )

    errors = normalized_errors(
        predictions,
        gt,
        bboxes,
    )

    total_edges = 0
    correct_edges = 0

    profile_stats = {}

    for profile in PROFILE_SCHEMAS:

        profile_stats[profile] = {
            "edges": 0,
            "correct": 0,
        }

    for i in range(n):

        profile = category_map.get(
            int(category_ids[i])
        )

        if profile not in PROFILE_SCHEMAS:
            continue

        schema = PROFILE_SCHEMAS[profile]

        for a, b in schema["edges"]:

            if not gt_vis[i, a] > 0:
                continue

            if not gt_vis[i, b] > 0:
                continue

            total_edges += 1

            profile_stats[
                profile
            ]["edges"] += 1

            if not prediction_valid[i, a]:
                continue

            if not prediction_valid[i, b]:
                continue

            endpoint_error = (
                errors[i, a]
                + errors[i, b]
            ) / 2.0

            if endpoint_error <= threshold:

                correct_edges += 1

                profile_stats[
                    profile
                ]["correct"] += 1

    overall = (
        float(
            correct_edges
            / total_edges
        )
        if total_edges > 0
        else 0.0
    )

    for profile, item in profile_stats.items():

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
        "correct": int(correct_edges),
        "total": int(total_edges),
        "profile": profile_stats,
    }


# ============================================================
# COMPLETE METRIC BUNDLE
# ============================================================

def calculate_all_metrics(
    predictions,
    gt,
    gt_vis,
    bboxes,
    category_ids,
    category_map,
    threshold=0.05,
    prediction_valid=None,
):
    """
    Calculate the complete common evaluation suite.
    """

    standard = calculate_joint_pck(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=threshold,
        prediction_valid=prediction_valid,
    )

    profile_pck = calculate_profile_pck(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=threshold,
        prediction_valid=prediction_valid,
    )

    semantic_roles = calculate_role_pck(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=threshold,
        prediction_valid=prediction_valid,
    )

    order_invariant = calculate_order_invariant_pck(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=threshold,
        prediction_valid=prediction_valid,
    )

    structural_edges = calculate_edge_accuracy(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=threshold,
        prediction_valid=prediction_valid,
    )

    return {
        "standard_pck": standard,
        "profile_pck": profile_pck,
        "semantic_roles": semantic_roles,
        "order_invariant_pck": order_invariant,
        "structural_edges": structural_edges,
        "profile_schemas": PROFILE_SCHEMAS,
    }


# ============================================================
# CONVENIENCE
# ============================================================

def make_category_map_from_coco(annotation_data):
    """
    Create:

        category_id -> canonical profile name

    from a COCO annotations dictionary.
    """

    category_map = {}

    for category in annotation_data.get(
        "categories",
        [],
    ):

        category_map[
            int(category["id"])
        ] = normalize_profile_name(
            category["name"]
        )

    return category_map