#!/usr/bin/env python3
"""
Experiment Summary Generator

Scans all YOLO26-Pose evaluation folders and generates a
publication-ready comparison table and plots.

Expected structure
------------------
experiments/
└── YOLO26_Pose_Runs/
    ├── E01_.../
    ├── E01_..._TestEval/
    │   └── evaluation_summary.txt
    ├── E02_.../
    └── E02_..._TestEval/

Outputs
-------
experiments/YOLO26_Pose_Runs/Summary/
    experiment_summary.csv
    experiment_ranking.csv
    box_map50_95.png
    pose_map50_95.png
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments" / "YOLO26_Pose_Runs"
OUT_DIR = EXP_ROOT / "Summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_summary(txt_file: Path):
    text = txt_file.read_text(encoding="utf-8")

    def grab(section, label):
        pattern = rf"{label}\s*:\s*([0-9.]+)"
        m = re.search(pattern, section)
        return float(m.group(1)) if m else None

    # Experiment name
    exp_name = txt_file.parent.name.replace("_TestEval", "")

    # Checkpoint
    ckpt = re.search(r"Checkpoint\s*:\s*(.+)", text)
    checkpoint = ckpt.group(1).strip() if ckpt else ""

    # Split into BOX and POSE sections
    box_match = re.search(
        r"BOX DETECTION(.*?)POSE ESTIMATION",
        text,
        flags=re.S
    )

    pose_match = re.search(
        r"POSE ESTIMATION(.*)",
        text,
        flags=re.S
    )

    box = box_match.group(1) if box_match else ""
    pose = pose_match.group(1) if pose_match else ""

    return {
        "Experiment": exp_name,
        "Checkpoint": checkpoint,

        "Box Precision": grab(box, "Precision"),
        "Box Recall": grab(box, "Recall"),
        "Box mAP50": grab(box, "mAP@50"),
        "Box mAP50-95": grab(box, "mAP50-95"),

        "Pose Precision": grab(pose, "Precision"),
        "Pose Recall": grab(pose, "Recall"),
        "Pose mAP50": grab(pose, "mAP@50"),
        "Pose mAP50-95": grab(pose, "mAP50-95"),
    }


records = []

for folder in sorted(EXP_ROOT.glob("*_TestEval")):
    summary = folder / "evaluation_report.txt"
    if summary.exists():
        records.append(parse_summary(summary))

if not records:
    raise FileNotFoundError("No evaluation_summary.txt found.")

df = pd.DataFrame(records)

df = df.sort_values("Experiment").reset_index(drop=True)

csv_path = OUT_DIR / "experiment_summary.csv"
df.to_csv(csv_path, index=False)

ranking = df.sort_values("Pose mAP50-95", ascending=False)
ranking_path = OUT_DIR / "experiment_ranking.csv"
ranking.to_csv(ranking_path, index=False)


def plot(metric, filename, title):
    plot_df = df.dropna(subset=[metric]).copy()

    plt.figure(figsize=(8, 4))
    plt.bar(plot_df["Experiment"], plot_df[metric])
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()


plot("Box mAP50-95", "box_map50_95.png", "Box Detection Performance")
plot("Pose mAP50-95", "pose_map50_95.png", "Pose Estimation Performance")

best = ranking.iloc[0]

print("=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)
print(df[[
    "Experiment",
    "Box mAP50",
    "Box mAP50-95",
    "Pose mAP50",
    "Pose mAP50-95"
]])

print("\nBEST MODEL")
print("-" * 70)
print(f"Experiment     : {best['Experiment']}")
print(f"Pose mAP50-95 : {best['Pose mAP50-95']:.4f}")

print("\nSaved to:")
print(OUT_DIR)