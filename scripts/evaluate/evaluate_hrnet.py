#!/usr/bin/env python3
"""
InnoCount KeyPoint - HRNet Final Test Evaluation

HRNet-specific evaluation wrapper.

This script:
    1. Optionally runs the HRNet test inference.
    2. Loads the exact prediction/GT arrays exported by
       InnoCountDataset.evaluate().
    3. Passes those arrays to common_pose_metrics.py.
    4. Saves the common metrics and audit files.

Shared metrics are implemented ONLY in:

    scripts/evaluate/common_pose_metrics.py

This ensures HRNet and YOLO26-Pose use the same:
    - Standard PCK@0.05
    - Profile PCK
    - Semantic-role PCK
    - Order-invariant PCK
    - Structural Edge Accuracy

IMPORTANT
---------
HRNet already evaluates fixed GT instances and exports the
prediction/GT arrays in exactly the same order. Therefore no
instance matching is performed here.

The prediction array and metadata arrays are loaded together
from HRNet's exported evaluation files.

The corrected Angle Bar equivalence is therefore inherited
from common_pose_metrics.py:

    G2 <-> G3

No upper/lower HI-Beam interchangeability is assumed.

The frozen test annotation file is:

    InnoCount_KeyPoint_ModelReady_v2/test/annotations.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# IMPORT COMMON METRICS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )

from common_pose_metrics import (  # noqa: E402
    PROFILE_SCHEMAS,
    calculate_all_metrics,
    make_category_map_from_coco,
)


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(path_arg: str | Path) -> Path:
    return (
        Path(path_arg)
        .expanduser()
        .resolve()
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths(
    hrnet_repo: Path,
    weights: Path,
    cfg: Path,
    annotation_file: Path,
):
    """
    Validate all files required for an HRNet test run.
    """

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
            f"HRNet checkpoint not found:\n"
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
# SYNCHRONIZE INNOCOUNT DATASET LOADER
# ============================================================

def synchronize_innocount_loader(
    hrnet_repo: Path,
):
    """
    Copy the project's current InnoCount loader into the
    HRNet repository.

    This is important because the loader contains the corrected
    Angle Bar mapping:

        [1, 2]

    rather than the old incorrect [1, 3].
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    source = (
        project_root
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

    shutil.copy2(
        source,
        destination,
    )

    print(
        "\n✓ Synced InnoCount dataset loader:"
    )

    print(
        f"  SOURCE:      {source}"
    )

    print(
        f"  DESTINATION: {destination}"
    )


# ============================================================
# COCO METADATA
# ============================================================

def load_coco(
    annotation_file: Path,
):
    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# HRNET OUTPUT FILE DISCOVERY
# ============================================================

def find_required_file(
    output_dir: Path,
    filename: str,
) -> Path:
    """
    Search recursively for one required HRNet evaluation file.
    """

    candidates = sorted(
        output_dir.rglob(
            filename
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "\nRequired HRNet evaluation file "
            "was not found.\n\n"
            f"Filename: {filename}\n"
            f"Searched under: {output_dir}"
        )

    if len(candidates) > 1:

        print(
            f"\nWARNING: multiple copies of "
            f"{filename} found:"
        )

        for candidate in candidates:
            print(
                f"  {candidate}"
            )

        print(
            "\nUsing the first sorted result."
        )

    return candidates[0]


# ============================================================
# LOAD HRNET EXPORTED ARRAYS
# ============================================================

def load_hrnet_arrays(
    output_dir: Path,
):
    """
    Load arrays exported by InnoCountDataset.evaluate().

    Expected files:

        hrnet_predictions.npy
        hrnet_gt.npy
        hrnet_gt_visibility.npy
        hrnet_bboxes.npy
        hrnet_image_ids.npy
        hrnet_category_ids.npy
        hrnet_annotation_ids.npy
    """

    output_dir = Path(
        output_dir
    )

    files = {
        "predictions":
            find_required_file(
                output_dir,
                "hrnet_predictions.npy",
            ),

        "gt":
            find_required_file(
                output_dir,
                "hrnet_gt.npy",
            ),

        "gt_vis":
            find_required_file(
                output_dir,
                "hrnet_gt_visibility.npy",
            ),

        "bboxes":
            find_required_file(
                output_dir,
                "hrnet_bboxes.npy",
            ),

        "image_ids":
            find_required_file(
                output_dir,
                "hrnet_image_ids.npy",
            ),

        "category_ids":
            find_required_file(
                output_dir,
                "hrnet_category_ids.npy",
            ),

        "annotation_ids":
            find_required_file(
                output_dir,
                "hrnet_annotation_ids.npy",
            ),
    }

    arrays = {
        key: np.load(
            path
        )
        for key, path in files.items()
    }

    print(
        "\nHRNet exported arrays"
    )

    print(
        "-" * 50
    )

    for key, path in files.items():

        print(
            f"{key:<18}: "
            f"{path}"
        )

        print(
            f"{'':<18}  "
            f"shape={arrays[key].shape}"
        )

    # --------------------------------------------------------
    # Alignment validation
    # --------------------------------------------------------

    n = len(
        arrays["predictions"]
    )

    for key, array in arrays.items():

        if len(array) != n:

            raise ValueError(
                "\nHRNet evaluation array "
                "length mismatch.\n"
                f"predictions: {n}\n"
                f"{key}: {len(array)}"
            )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    predictions = arrays[
        "predictions"
    ]

    gt = arrays[
        "gt"
    ]

    gt_vis = arrays[
        "gt_vis"
    ]

    bboxes = arrays[
        "bboxes"
    ]

    if predictions.ndim != 3:
        raise ValueError(
            f"Unexpected HRNet prediction shape: "
            f"{predictions.shape}"
        )

    if gt.ndim != 3:
        raise ValueError(
            f"Unexpected HRNet GT shape: "
            f"{gt.shape}"
        )

    if predictions.shape[1] != 6:
        raise ValueError(
            f"HRNet predictions have "
            f"{predictions.shape[1]} keypoints; "
            f"expected 6."
        )

    if gt.shape[1] != 6:
        raise ValueError(
            f"HRNet GT has "
            f"{gt.shape[1]} keypoints; "
            f"expected 6."
        )

    if gt_vis.ndim == 3:
        if gt_vis.shape[-1] == 1:
            arrays["gt_vis"] = (
                gt_vis[:, :, 0]
            )

    if arrays["gt_vis"].ndim != 2:
        raise ValueError(
            f"Unexpected HRNet visibility shape: "
            f"{arrays['gt_vis'].shape}"
        )

    if bboxes.ndim != 2:
        raise ValueError(
            f"Unexpected HRNet bbox shape: "
            f"{bboxes.shape}"
        )

    if bboxes.shape[1] != 4:
        raise ValueError(
            f"HRNet bboxes must have 4 values; "
            f"got {bboxes.shape}"
        )

    print(
        f"\n✓ All HRNet arrays aligned: "
        f"{n} instances"
    )

    return arrays


# ============================================================
# COORDINATE NORMALIZATION
# ============================================================

def normalize_hrnet_coordinates(
    predictions,
    gt,
    bboxes,
):
    """
    HRNet's exported coordinates are expected to already be
    in the same coordinate system as its GT/bboxes.

    This function handles the common cases explicitly:

    Case A:
        predictions and GT are pixel coordinates
        bbox = [x,y,w,h] pixels

    Case B:
        predictions and GT are normalized [0,1]
        bbox = normalized [x,y,w,h]

    The common metric implementation only requires that all
    three use the same coordinate system.

    Therefore, if coordinates already appear normalized,
    they are left untouched.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    gt = np.asarray(
        gt,
        dtype=np.float32,
    )

    bboxes = np.asarray(
        bboxes,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Determine whether data is already normalized.
    #
    # HRNet's InnoCount evaluator normally exports pixel-space
    # coordinates. For pixel-space coordinates, leave them as
    # they are.
    #
    # The common metrics only use relative bbox dimensions, so
    # no image-size normalization is actually necessary.
    # --------------------------------------------------------

    return (
        predictions,
        gt,
        bboxes,
    )


# ============================================================
# SAVE PROFILE CSV
# ============================================================

def save_profile_csv(
    path: Path,
    metrics,
):
    """
    Save profile-level summary.
    """

    profile_pck = metrics[
        "profile_pck"
    ]

    semantic_roles = metrics[
        "semantic_roles"
    ]

    structural_edges = metrics[
        "structural_edges"
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        import csv

        writer = csv.writer(f)

        writer.writerow(
            [
                "Profile",
                "Instances",
                "PCK",
                "Correct",
                "Valid",
                "Order_Invariant_PCK",
                "Semantic_Role_Mean_PCK",
                "Structural_Edge_Accuracy",
            ]
        )

        for profile in PROFILE_SCHEMAS:

            pck_item = (
                profile_pck[
                    profile
                ]
            )

            invariant_item = (
                metrics[
                    "order_invariant_pck"
                ][
                    "profile"
                ][
                    profile
                ]
            )

            roles = (
                semantic_roles[
                    "profile_role_pck"
                ][
                    profile
                ]
            )

            role_values = [
                item["pck"]
                for item in roles.values()
                if item["valid"] > 0
            ]

            role_mean = (
                float(
                    np.mean(
                        role_values
                    )
                )
                if role_values
                else 0.0
            )

            edge_accuracy = (
                structural_edges[
                    "profile"
                ][
                    profile
                ][
                    "accuracy"
                ]
            )

            writer.writerow(
                [
                    profile,
                    pck_item[
                        "instances"
                    ],
                    f"{pck_item['pck']:.6f}",
                    pck_item[
                        "correct"
                    ],
                    pck_item[
                        "valid"
                    ],
                    f"{invariant_item['pck']:.6f}",
                    f"{role_mean:.6f}",
                    f"{edge_accuracy:.6f}",
                ]
            )


# ============================================================
# SAVE ROLE CSV
# ============================================================

def save_role_csv(
    path: Path,
    metrics,
):
    role_data = (
        metrics[
            "semantic_roles"
        ][
            "overall_role_pck"
        ]
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        import csv

        writer = csv.writer(f)

        writer.writerow(
            [
                "Semantic_Role",
                "PCK",
            ]
        )

        for role, pck in role_data.items():

            writer.writerow(
                [
                    role,
                    f"{pck:.6f}",
                ]
            )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

def save_text_report(
    path: Path,
    report,
):
    metrics = report[
        "metrics"
    ]

    standard = metrics[
        "standard_pck"
    ]

    invariant = metrics[
        "order_invariant_pck"
    ]

    profile = metrics[
        "profile_pck"
    ]

    roles = metrics[
        "semantic_roles"
    ]

    edges = metrics[
        "structural_edges"
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "INNOCOUNT KEYPOINT\n"
        )

        f.write(
            "HRNET FINAL TEST EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Model       : HRNet-W32\n"
        )

        f.write(
            f"Weights     : "
            f"{report['weights']}\n"
        )

        f.write(
            f"Config      : "
            f"{report['cfg']}\n"
        )

        f.write(
            f"Annotations : "
            f"{report['annotation_file']}\n"
        )

        f.write(
            f"Threshold   : "
            f"{report['threshold']}\n"
        )

        f.write(
            f"Instances   : "
            f"{report['num_instances']}\n"
        )

        f.write(
            f"Generated   : "
            f"{report['timestamp']}\n\n"
        )

        # ----------------------------------------------------
        # Standard PCK
        # ----------------------------------------------------

        f.write(
            "STANDARD PCK@0.05\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        f.write(
            f"Overall : "
            f"{standard['overall']:.6f}\n"
        )

        f.write(
            f"Correct : "
            f"{standard['total_correct']}\n"
        )

        f.write(
            f"Valid   : "
            f"{standard['total_valid']}\n"
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

        f.write(
            f"Overall : "
            f"{invariant['overall']:.6f}\n"
        )

        gain = (
            invariant["overall"]
            - standard["overall"]
        )

        f.write(
            f"Gain    : "
            f"{gain:.6f} "
            f"({gain * 100:.2f} pp)\n"
        )

        for profile_name, item in (
            invariant[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile_name:<15}: "
                f"{item['pck']:.6f}\n"
            )

        # ----------------------------------------------------
        # Profile PCK
        # ----------------------------------------------------

        f.write(
            "\nPROFILE PCK\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        for profile_name, item in (
            profile.items()
        ):

            f.write(
                f"{profile_name:<15}: "
                f"{item['pck']:.6f} "
                f"(n={item['instances']})\n"
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
            roles[
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

        f.write(
            f"Overall : "
            f"{edges['overall']:.6f}\n"
        )

        for profile_name, item in (
            edges[
                "profile"
            ].items()
        ):

            f.write(
                f"{profile_name:<15}: "
                f"{item['accuracy']:.6f}\n"
            )


# ============================================================
# RUN HRNET INFERENCE
# ============================================================

def run_hrnet_test(
    hrnet_repo: Path,
    cfg: Path,
    weights: Path,
    output_dir: Path,
    annotation_file: Path,
    device: str,
):
    """
    Run the existing HRNet tools/test.py.

    NOTE:
    The exact command is kept compatible with the existing
    project structure.
    """

    test_script = (
        hrnet_repo
        / "tools"
        / "test.py"
    )

    command = [
        sys.executable,
        str(test_script),
        str(cfg),
        str(weights),
        "--out",
        str(
            output_dir
            / "hrnet_predictions.pkl"
        ),
    ]

    if device:

        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = str(device)

    print(
        "\nRunning HRNet test inference:"
    )

    print(
        " ".join(
            f'"{x}"'
            if " " in x
            else x
            for x in command
        )
    )

    result = subprocess.run(
        command,
        cwd=str(
            hrnet_repo
        ),
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "HRNet test.py failed with "
            f"exit code {result.returncode}."
        )


# ============================================================
# MAIN
# ============================================================

def evaluate(args):

    timestamp = (
        datetime.now().isoformat()
    )

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

    annotation_file = resolve_path(
        args.annotation_file
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_paths(
        hrnet_repo=hrnet_repo,
        weights=weights,
        cfg=cfg,
        annotation_file=annotation_file,
    )

    print(
        "=" * 70
    )

    print(
        "HRNET-W32 COMMON TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Weights       : {weights}"
    )

    print(
        f"Config        : {cfg}"
    )

    print(
        f"Annotations   : {annotation_file}"
    )

    print(
        f"Output        : {output_dir}"
    )

    print(
        f"Threshold     : {args.threshold}"
    )

    # --------------------------------------------------------
    # Synchronize loader
    # --------------------------------------------------------

    if not args.skip_loader_sync:

        synchronize_innocount_loader(
            hrnet_repo
        )

    # --------------------------------------------------------
    # Optional fresh inference
    # --------------------------------------------------------

    if args.run_inference:

        run_hrnet_test(
            hrnet_repo=hrnet_repo,
            cfg=cfg,
            weights=weights,
            output_dir=output_dir,
            annotation_file=annotation_file,
            device=args.device,
        )

    # --------------------------------------------------------
    # Locate/load exported arrays
    # --------------------------------------------------------

    arrays = load_hrnet_arrays(
        output_dir
    )

    predictions = arrays[
        "predictions"
    ]

    gt = arrays[
        "gt"
    ]

    gt_vis = arrays[
        "gt_vis"
    ]

    bboxes = arrays[
        "bboxes"
    ]

    image_ids = arrays[
        "image_ids"
    ]

    category_ids = arrays[
        "category_ids"
    ]

    annotation_ids = arrays[
        "annotation_ids"
    ]

    # --------------------------------------------------------
    # Load category mapping from exact frozen annotation file
    # --------------------------------------------------------

    coco = load_coco(
        annotation_file
    )

    category_map = (
        make_category_map_from_coco(
            coco
        )
    )

    print(
        "\nCategory mapping:"
    )

    for category_id in sorted(
        category_map
    ):

        print(
            f"  {category_id}: "
            f"{category_map[category_id]}"
        )

    # --------------------------------------------------------
    # Normalize coordinate representation
    # --------------------------------------------------------

    (
        predictions,
        gt,
        bboxes,
    ) = normalize_hrnet_coordinates(
        predictions=predictions,
        gt=gt,
        bboxes=bboxes,
    )

    # --------------------------------------------------------
    # Verify metadata consistency
    # --------------------------------------------------------

    if len(
        image_ids
    ) != len(predictions):

        raise ValueError(
            "image_ids are not aligned "
            "with predictions."
        )

    if len(
        category_ids
    ) != len(predictions):

        raise ValueError(
            "category_ids are not aligned "
            "with predictions."
        )

    if len(
        annotation_ids
    ) != len(predictions):

        raise ValueError(
            "annotation_ids are not aligned "
            "with predictions."
        )

    # --------------------------------------------------------
    # Common metrics
    # --------------------------------------------------------

    print(
        "\nCalculating common pose metrics..."
    )

    metrics = calculate_all_metrics(
        predictions=predictions,
        gt=gt,
        gt_vis=gt_vis,
        bboxes=bboxes,
        category_ids=category_ids,
        category_map=category_map,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # Report object
    # --------------------------------------------------------

    report = {
        "model": "HRNet-W32",

        "weights": str(
            weights
        ),

        "cfg": str(
            cfg
        ),

        "hrnet_repo": str(
            hrnet_repo
        ),

        "annotation_file": str(
            annotation_file
        ),

        "test_split": "test",

        "timestamp": timestamp,

        "threshold": float(
            args.threshold
        ),

        "num_instances": int(
            len(predictions)
        ),

        "metrics": metrics,

        "array_sources": {
            "predictions": str(
                find_required_file(
                    output_dir,
                    "hrnet_predictions.npy",
                )
            ),

            "gt": str(
                find_required_file(
                    output_dir,
                    "hrnet_gt.npy",
                )
            ),

            "gt_visibility": str(
                find_required_file(
                    output_dir,
                    "hrnet_gt_visibility.npy",
                )
            ),

            "bboxes": str(
                find_required_file(
                    output_dir,
                    "hrnet_bboxes.npy",
                )
            ),

            "image_ids": str(
                find_required_file(
                    output_dir,
                    "hrnet_image_ids.npy",
                )
            ),

            "category_ids": str(
                find_required_file(
                    output_dir,
                    "hrnet_category_ids.npy",
                )
            ),

            "annotation_ids": str(
                find_required_file(
                    output_dir,
                    "hrnet_annotation_ids.npy",
                )
            ),
        },

        "evaluation_method": {
            "instance_matching":
                "none; HRNet evaluator exports "
                "prediction/GT arrays in aligned order",

            "pck_normalization":
                "Euclidean keypoint error divided by "
                "max(bbox_width, bbox_height)",

            "pck_threshold":
                float(args.threshold),

            "order_invariant":
                "equivalent keypoint groups defined "
                "in common_pose_metrics.py",

            "structural_edges":
                "profile-specific edge definitions "
                "from common_pose_metrics.py",
        },

        "profile_schemas":
            PROFILE_SCHEMAS,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path = (
        output_dir
        / "common_evaluation_metrics.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save profile CSV
    # --------------------------------------------------------

    save_profile_csv(
        output_dir
        / "profile_metrics.csv",
        metrics,
    )

    # --------------------------------------------------------
    # Save role CSV
    # --------------------------------------------------------

    save_role_csv(
        output_dir
        / "semantic_role_metrics.csv",
        metrics,
    )

    # --------------------------------------------------------
    # Save aligned arrays again under common names
    # --------------------------------------------------------

    np.save(
        output_dir
        / "common_predictions.npy",
        predictions,
    )

    np.save(
        output_dir
        / "common_gt.npy",
        gt,
    )

    np.save(
        output_dir
        / "common_gt_visibility.npy",
        gt_vis,
    )

    np.save(
        output_dir
        / "common_bboxes.npy",
        bboxes,
    )

    np.save(
        output_dir
        / "common_image_ids.npy",
        image_ids,
    )

    np.save(
        output_dir
        / "common_category_ids.npy",
        category_ids,
    )

    np.save(
        output_dir
        / "common_annotation_ids.npy",
        annotation_ids,
    )

    # --------------------------------------------------------
    # Save text report
    # --------------------------------------------------------

    save_text_report(
        output_dir
        / "common_evaluation_report.txt",
        report,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    standard = metrics[
        "standard_pck"
    ]

    invariant = metrics[
        "order_invariant_pck"
    ]

    profile = metrics[
        "profile_pck"
    ]

    roles = metrics[
        "semantic_roles"
    ]

    edges = metrics[
        "structural_edges"
    ]

    gain = (
        invariant["overall"]
        - standard["overall"]
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL COMMON HRNET RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nStandard PCK@0.05 : "
        f"{standard['overall']:.6f}"
    )

    print(
        f"Order-invariant   : "
        f"{invariant['overall']:.6f}"
    )

    print(
        f"Gain              : "
        f"{gain:.6f} "
        f"({gain * 100:.2f} pp)"
    )

    print(
        f"\nStructural Edge Accuracy : "
        f"{edges['overall']:.6f}"
    )

    print(
        "\nProfile PCK:"
    )

    for profile_name, item in (
        profile.items()
    ):

        print(
            f"  {profile_name:<15} "
            f"{item['pck']:.6f} "
            f"(n={item['instances']})"
        )

    print(
        "\nSemantic Role PCK:"
    )

    for role, value in (
        roles[
            "overall_role_pck"
        ].items()
    ):

        print(
            f"  {role:<35} "
            f"{value:.6f}"
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {output_dir / 'common_evaluation_report.txt'}"
    )

    print(
        f"  {output_dir / 'profile_metrics.csv'}"
    )

    print(
        f"  {output_dir / 'semantic_role_metrics.csv'}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    repo = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HRNet-W32 on the frozen "
            "InnoCount test split using "
            "common_pose_metrics.py."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        help=(
            "HRNet checkpoint, e.g. "
            "model_best.pth"
        ),
    )

    parser.add_argument(
        "--cfg",
        required=True,
        help=(
            "HRNet experiment config, "
            "e.g. hrnet_w32_640.yaml"
        ),
    )

    parser.add_argument(
        "--hrnet-repo",
        default=(
            "/home/owner/HRNet-Human-Pose-Estimation"
        ),
        help=(
            "Path to HRNet repository."
        ),
    )

    parser.add_argument(
        "--annotation-file",
        default=str(
            repo
            / "datasets"
            / "InnoCount_KeyPoint_ModelReady_v2"
            / "test"
            / "annotations.json"
        ),
        help=(
            "Exact frozen ModelReady v2 "
            "test annotations."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory containing or receiving "
            "HRNet exported evaluation arrays."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help=(
            "PCK threshold. Default: 0.05"
        ),
    )

    parser.add_argument(
        "--device",
        default="0",
        help=(
            "CUDA device used when running "
            "fresh HRNet inference."
        ),
    )

    parser.add_argument(
        "--run-inference",
        action="store_true",
        help=(
            "Run HRNet tools/test.py before "
            "calculating common metrics."
        ),
    )

    parser.add_argument(
        "--skip-loader-sync",
        action="store_true",
        help=(
            "Do not copy the local InnoCount "
            "loader into the HRNet repository."
        ),
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_args()

    evaluate(
        args
    )


if __name__ == "__main__":
    main()

'''
cd /home/owner/skeletal_keypoint_detection

python scripts/evaluate/evaluate_hrnet.py \
    --weights "/home/owner/skeletal_keypoint_detection/experiments/HRNet_Runs/E08_HRNet_W32/innocount/pose_hrnet/hrnet_w32_640/model_best.pth" \
    --cfg "/home/owner/HRNet-Human-Pose-Estimation/experiments/innocount/hrnet_w32_640.yaml" \
    --hrnet-repo "/home/owner/HRNet-Human-Pose-Estimation" \
    --annotation-file "/home/owner/skeletal_keypoint_detection/datasets/InnoCount_KeyPoint_ModelReady_v2/test/annotations.json" \
    --output-dir "/home/owner/skeletal_keypoint_detection/experiments/HRNet_Runs/E08_HRNet_W32_FinalTest" \
    --threshold 0.05
'''