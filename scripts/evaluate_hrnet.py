#!/usr/bin/env python3
"""
InnoCount KeyPoint - HRNet Test Evaluation

Evaluate a trained HRNet pose model on the frozen InnoCount test split.

The wrapper uses HRNet's native tools/test.py evaluation pipeline.

Outputs:

experiments/HRNet_Runs/<name>/
    <timestamp>/
        evaluation_metrics.json
        evaluation_report.txt
        hrnet_test_output.txt
        hrnet_eval_config.yaml
"""

from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

import yaml


# ============================================================
# HELPERS
# ============================================================

def resolve_path(path_arg: str) -> Path:
    """Resolve a path and return an absolute Path."""
    return Path(path_arg).expanduser().resolve()


def validate_paths(
    hrnet_repo: Path,
    weights: Path,
    cfg: Path,
):
    """Validate required files/directories."""

    if not hrnet_repo.exists():
        raise FileNotFoundError(
            f"HRNet repository not found:\n{hrnet_repo}"
        )

    test_script = hrnet_repo / "tools" / "test.py"

    if not test_script.exists():
        raise FileNotFoundError(
            f"HRNet test script not found:\n{test_script}"
        )

    if not weights.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{weights}"
        )

    if not cfg.exists():
        raise FileNotFoundError(
            f"HRNet config not found:\n{cfg}"
        )


# ============================================================
# CONFIG
# ============================================================

def create_eval_config(
    cfg_path: Path,
    output_dir: Path,
) -> Path:
    """
    Create a temporary evaluation config.

    The original training config is NOT modified.

    Changes:
        DATASET.TEST_SET = test
        OUTPUT_DIR       = evaluation output directory
    """

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Force held-out test set
    cfg["DATASET"]["TEST_SET"] = "test"

    # Keep evaluation outputs isolated
    cfg["OUTPUT_DIR"] = str(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eval_cfg = output_dir / "hrnet_eval_config.yaml"

    with open(eval_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    return eval_cfg


# ============================================================
# PARSE HRNET OUTPUT
# ============================================================

def parse_hrnet_metrics(output: str):
    """
    Extract all available metrics from HRNet evaluation output.

    Expected format from _print_name_value():

        | Arch | PCK | PCK_joint_1 | ... | OrderInvariant_PCK | ... |
        |---|---|---|---|
        | pose_hrnet | 0.xxx | 0.xxx | ... | 0.xxx | ... |
    """

    metrics = {}

    # --------------------------------------------------------
    # Locate the HRNet markdown table
    # --------------------------------------------------------

    lines = output.splitlines()

    header = None
    values = None

    for i, line in enumerate(lines):

        if "| Arch" in line:
            header = line

            # The values row should follow the separator row.
            for j in range(i + 1, min(i + 4, len(lines))):

                candidate = lines[j].strip()

                if (
                    candidate.startswith("|")
                    and "pose_hrnet" in candidate
                ):
                    values = candidate
                    break

            if values is not None:
                break

    if header is not None and values is not None:

        header_fields = [
            x.strip()
            for x in header.strip("|").split("|")
        ]

        value_fields = [
            x.strip()
            for x in values.strip("|").split("|")
        ]

        # Usually both have identical lengths.
        for name, value in zip(
            header_fields,
            value_fields,
        ):

            if name == "Arch":
                continue

            try:
                metrics[name] = float(value)
            except ValueError:
                pass

    # --------------------------------------------------------
    # Fallback: explicitly search common metric names
    # --------------------------------------------------------

    # This protects against minor formatting changes.
    metric_patterns = {
        "PCK": r"PCK\s*[:|]\s*([0-9]*\.?[0-9]+)",
        "OrderInvariant_PCK":
            r"OrderInvariant_PCK\s*[:|]\s*([0-9]*\.?[0-9]+)",
    }

    for name, pattern in metric_patterns.items():

        if name not in metrics:

            match = re.search(
                pattern,
                output,
                re.IGNORECASE,
            )

            if match:
                metrics[name] = float(
                    match.group(1)
                )

    # --------------------------------------------------------
    # Running test accuracy
    # --------------------------------------------------------

    matches = re.findall(
        r"Test:\s*\[\d+/\d+\].*?"
        r"Accuracy\s+[0-9.]+\s+\(([0-9.]+)\)",
        output,
    )

    if matches:
        metrics["test_accuracy"] = float(
            matches[-1]
        )

    return metrics


# ============================================================
# ORGANIZE METRICS
# ============================================================

def organize_metrics(raw_metrics):
    """
    Organize metrics into a thesis-friendly structure.
    """

    standard = {}
    order_invariant = {}

    for key, value in raw_metrics.items():

        if key.startswith("PCK_joint_"):
            joint_id = key.replace(
                "PCK_joint_",
                "",
            )
            standard[f"joint_{joint_id}"] = value

        elif key.startswith(
            "OrderInvariant_PCK_joint_"
        ):
            joint_id = key.replace(
                "OrderInvariant_PCK_joint_",
                "",
            )
            order_invariant[
                f"joint_{joint_id}"
            ] = value

        elif key == "PCK":
            standard["overall"] = value

        elif key == "OrderInvariant_PCK":
            order_invariant["overall"] = value

    organized = {
        "standard_pck": standard,
        "order_invariant_pck": order_invariant,
    }

    if "test_accuracy" in raw_metrics:
        organized["test_accuracy"] = (
            raw_metrics["test_accuracy"]
        )

    # --------------------------------------------------------
    # Difference between evaluation methods
    # --------------------------------------------------------

    if (
        "overall" in standard
        and "overall" in order_invariant
    ):

        difference = (
            order_invariant["overall"]
            - standard["overall"]
        )

        organized["order_invariance_gain"] = {
            "absolute": difference,
            "percentage_points": difference * 100,
        }

    return organized


# ============================================================
# REPORT
# ============================================================

def make_report(
    save_dir: Path,
    weights: Path,
    cfg: Path,
    metrics,
    return_code,
    start_time,
    end_time,
):

    timestamp = datetime.now().isoformat()

    report = {
        "timestamp": timestamp,
        "model": "HRNet-W32",
        "weights": str(weights),
        "config": str(cfg),
        "test_split": "test",
        "start_time": start_time,
        "end_time": end_time,
        "metrics": metrics,
        "process": {
            "return_code": return_code,
            "status": (
                "completed"
                if return_code == 0
                else "failed"
            ),
        },
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_path = (
        save_dir / "evaluation_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # TXT REPORT
    # --------------------------------------------------------

    txt_path = (
        save_dir / "evaluation_report.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "HRNET-W32 TEST EVALUATION\n"
        )
        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Timestamp    : {timestamp}\n"
        )
        f.write(
            f"Start time   : {start_time}\n"
        )
        f.write(
            f"End time     : {end_time}\n"
        )
        f.write(
            f"Checkpoint   : {weights}\n"
        )
        f.write(
            f"Config       : {cfg}\n"
        )
        f.write(
            "Test split   : test\n\n"
        )

        # ----------------------------------------------------
        # Standard PCK
        # ----------------------------------------------------

        f.write(
            "STANDARD PCK\n"
        )
        f.write(
            "-" * 40 + "\n"
        )

        standard = metrics.get(
            "standard_pck",
            {},
        )

        if standard:

            if "overall" in standard:
                f.write(
                    f"Overall      : "
                    f"{standard['overall']:.6f}\n"
                )

            for joint in sorted(
                [
                    x
                    for x in standard
                    if x.startswith("joint_")
                ],
                key=lambda x: int(
                    x.replace("joint_", "")
                ),
            ):

                f.write(
                    f"{joint:<12}: "
                    f"{standard[joint]:.6f}\n"
                )

        else:
            f.write("Not detected\n")

        f.write("\n")

        # ----------------------------------------------------
        # Order-Invariant PCK
        # ----------------------------------------------------

        f.write(
            "ORDER-INVARIANT PCK\n"
        )
        f.write(
            "-" * 40 + "\n"
        )

        invariant = metrics.get(
            "order_invariant_pck",
            {},
        )

        if invariant:

            if "overall" in invariant:
                f.write(
                    f"Overall      : "
                    f"{invariant['overall']:.6f}\n"
                )

            for joint in sorted(
                [
                    x
                    for x in invariant
                    if x.startswith("joint_")
                ],
                key=lambda x: int(
                    x.replace("joint_", "")
                ),
            ):

                f.write(
                    f"{joint:<12}: "
                    f"{invariant[joint]:.6f}\n"
                )

        else:
            f.write("Not detected\n")

        f.write("\n")

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        f.write(
            "ORDER-INVARIANCE ANALYSIS\n"
        )
        f.write(
            "-" * 40 + "\n"
        )

        gain = metrics.get(
            "order_invariance_gain"
        )

        if gain:

            f.write(
                f"Absolute gain : "
                f"{gain['absolute']:.6f}\n"
            )

            f.write(
                f"Percentage pts: "
                f"{gain['percentage_points']:.2f}\n"
            )

        else:
            f.write(
                "Comparison unavailable\n"
            )

        # ----------------------------------------------------
        # Running accuracy
        # ----------------------------------------------------

        if "test_accuracy" in metrics:

            f.write("\n")
            f.write(
                "HRNET TEST ACCURACY\n"
            )
            f.write(
                "-" * 40 + "\n"
            )

            f.write(
                f"Final running accuracy: "
                f"{metrics['test_accuracy']:.6f}\n"
            )

        f.write("\n")
        f.write(
            f"HRNet test.py return code: "
            f"{return_code}\n"
        )

    return report


# ============================================================
# EVALUATION
# ============================================================

def evaluate(args):

    repo_root = Path(
        __file__
    ).resolve().parents[1]

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

    # --------------------------------------------------------
    # Timestamped evaluation directory
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
    )

    # --------------------------------------------------------
    # Create evaluation config
    # --------------------------------------------------------

    eval_cfg = create_eval_config(
        cfg,
        save_dir,
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("=" * 70)
    print("HRNET-W32 TEST EVALUATION")
    print("=" * 70)

    print(f"HRNet repo : {hrnet_repo}")
    print(f"Weights    : {weights}")
    print(f"Config     : {cfg}")
    print("Test split : test")
    print(f"Device     : {args.device}")
    print(f"Output     : {save_dir}")
    print()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = str(
        args.device
    )

    env["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"
    )

    # Make HRNet lib importable
    lib_path = hrnet_repo / "lib"

    existing_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    if existing_pythonpath:

        env["PYTHONPATH"] = (
            f"{lib_path}:{existing_pythonpath}"
        )

    else:

        env["PYTHONPATH"] = str(
            lib_path
        )

    # --------------------------------------------------------
    # Run native HRNet evaluation
    # --------------------------------------------------------

    cmd = [
        sys.executable,
        "tools/test.py",
        "--cfg",
        str(eval_cfg),
        "TEST.MODEL_FILE",
        str(weights),
    ]

    print("=" * 70)
    print("RUNNING HRNET TEST")
    print("=" * 70)
    print()

    start_time = datetime.now().isoformat()

    result = subprocess.run(
        cmd,
        cwd=hrnet_repo,
        env=env,
        capture_output=True,
        text=True,
    )

    end_time = datetime.now().isoformat()

    # --------------------------------------------------------
    # Save complete output
    # --------------------------------------------------------

    output = result.stdout

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

        f.write(output)

    # Also show output in terminal
    print(output)

    # --------------------------------------------------------
    # Parse metrics
    # --------------------------------------------------------

    combined_output = (
        result.stdout
        + "\n"
        + result.stderr
    )

    raw_metrics = parse_hrnet_metrics(
        combined_output
    )

    metrics = organize_metrics(
        raw_metrics
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = make_report(
        save_dir=save_dir,
        weights=weights,
        cfg=cfg,
        metrics=metrics,
        return_code=result.returncode,
        start_time=start_time,
        end_time=end_time,
    )

    # --------------------------------------------------------
    # Fail if HRNet test failed
    # --------------------------------------------------------

    if result.returncode != 0:

        print()
        print("=" * 70)
        print("HRNET EVALUATION FAILED")
        print("=" * 70)

        print(
            f"Return code: "
            f"{result.returncode}"
        )

        print(
            f"\nFull output saved to:\n"
            f"{raw_output_path}"
        )

        raise RuntimeError(
            "HRNet tools/test.py failed."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    standard = metrics.get(
        "standard_pck",
        {},
    )

    invariant = metrics.get(
        "order_invariant_pck",
        {},
    )

    print("\nStandard PCK")

    if "overall" in standard:
        print(
            f"  Overall : "
            f"{standard['overall']:.6f}"
        )

    print("\nOrder-Invariant PCK")

    if "overall" in invariant:
        print(
            f"  Overall : "
            f"{invariant['overall']:.6f}"
        )

    gain = metrics.get(
        "order_invariance_gain"
    )

    if gain:

        print("\nOrder-Invariance Gain")

        print(
            f"  Absolute         : "
            f"{gain['absolute']:.6f}"
        )

        print(
            f"  Percentage points: "
            f"{gain['percentage_points']:.2f}"
        )

    if "test_accuracy" in metrics:

        print(
            "\nFinal test accuracy : "
            f"{metrics['test_accuracy']:.6f}"
        )

    print("\nSaved to:")
    print(save_dir)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HRNet-W32 on the "
            "InnoCount test split."
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
            "Path to HRNet model_best.pth"
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
        help="HRNet training config",
    )

    parser.add_argument(
        "--project",
        default=str(
            repo_root
            / "experiments"
            / "HRNet_Runs"
        ),
        help="Evaluation output root",
    )

    parser.add_argument(
        "--name",
        default=(
            "E07_HRNet_W32_TestEval"
        ),
        help="Evaluation run name",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPU device index",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()