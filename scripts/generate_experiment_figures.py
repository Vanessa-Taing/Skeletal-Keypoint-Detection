from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

EXP_DIR = ROOT / "experiments" / "YOLO26_Pose_Runs"
SUMMARY_DIR = EXP_DIR / "Summary"

CSV = SUMMARY_DIR / "experiment_summary.csv"

OUT = ROOT / "docs" / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Matplotlib style (thesis friendly)
# ==========================================================

plt.rcParams.update({
    "figure.dpi":300,
    "font.family":"DejaVu Sans",
    "font.size":11,
    "axes.titlesize":13,
    "axes.labelsize":11,
    "legend.fontsize":10,
    "xtick.labelsize":10,
    "ytick.labelsize":10,
})

df = pd.read_csv(CSV)

# ==========================================================
# Helper functions
# ==========================================================

def overall_chart():
    order = ["E01","E02","E03","E04","E05"]
    values = df["Pose mAP50-95"].tolist()

    fig, ax = plt.subplots(figsize=(6.5,4))

    bars = ax.bar(
        order,
        values,
        color=["#D9D9D9","#BDBDBD","#969696","#525252","#252525"],
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_ylabel("Pose mAP50-95")
    ax.set_ylim(0,0.36)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for b,v in zip(bars,values):
        ax.text(
            b.get_x()+b.get_width()/2,
            v+0.008,
            f"{v:.3f}",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(OUT/"Figure_4_0_OverallComparison.png",
                bbox_inches="tight")
    plt.close()


def comparison(exp1, exp2, filename, title):

    metrics = [
        "Box mAP50",
        "Box mAP50-95",
        "Pose mAP50",
        "Pose mAP50-95",
    ]

    a = df[df["Experiment"]==exp1].iloc[0]
    b = df[df["Experiment"]==exp2].iloc[0]

    y1 = [a[m] for m in metrics]
    y2 = [b[m] for m in metrics]

    labels = ["B50","B95","P50","P95"]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7,4))

    bars1 = ax.bar(
        x-width/2,
        y1,
        width,
        label=exp1[:3],
        color="white",
        edgecolor="black",
        hatch="///",
        linewidth=1
    )

    bars2 = ax.bar(
        x+width/2,
        y2,
        width,
        label=exp2[:3],
        color="#666666",
        edgecolor="black",
        linewidth=1
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    for bars in [bars1,bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x()+bar.get_width()/2,
                h+0.02,
                f"{h:.2f}",
                ha="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(OUT/filename, bbox_inches="tight")
    plt.close()


# ==========================================================
# Crop augmentation visualization
# ==========================================================

def crop_training_batch():

    img_path = EXP_DIR/"E03_YOLO26s_Augmentation"/"train_batch0.jpg"

    if not img_path.exists():
        return

    img = Image.open(img_path)

    w,h = img.size

    # Remove right statistics panel (~28%)
    crop = img.crop((0,0,int(w*0.72),h))

    crop.save(
        OUT/"Figure_4_2_AugmentedTrainingSamples.jpg",
        quality=95
    )


# ==========================================================
# Prediction montage
# ==========================================================

def montage(image_list, filename, title_prefix):

    thumb=(420,420)
    cols=2
    rows=2

    canvas=Image.new(
        "RGB",
        (thumb[0]*cols, thumb[1]*rows),
        "white"
    )

    draw=ImageDraw.Draw(canvas)

    for i,path in enumerate(image_list):

        img=Image.open(path).convert("RGB")
        img=ImageOps.fit(img,thumb)

        x=(i%2)*thumb[0]
        y=(i//2)*thumb[1]

        canvas.paste(img,(x,y))

        draw.rectangle(
            [x,y,x+thumb[0]-1,y+thumb[1]-1],
            outline="black",
            width=2
        )

        draw.text((x+10,y+10),f"{title_prefix}{i+1}",fill="white")

    canvas.save(OUT/filename,quality=95)


# ==========================================================
# Generate all figures
# ==========================================================

overall_chart()

comparison(
    "E01_YOLO26n_Baseline",
    "E02_YOLO26s_Baseline",
    "Figure_4_1_E01_vs_E02.png",
    "Effect of Model Scaling"
)

crop_training_batch()

comparison(
    "E03_YOLO26s_Augmentation",
    "E04_YOLO26s_Augmentation",
    "Figure_4_3_E03_vs_E04.png",
    "Effect of Refined Augmentation"
)

comparison(
    "E04_YOLO26s_Augmentation",
    "E05_YOLO26m_Augmentation",
    "Figure_4_4_E04_vs_E05.png",
    "Effect of Larger Backbone"
)

pred_dir = EXP_DIR/"E04_YOLO26s_Augmentation_TestEval"/"predictions"/"images"

imgs = sorted(pred_dir.glob("*.jpg"))

if len(imgs)>=8:

    montage(
        imgs[:4],
        "Figure_4_5_SuccessCases.jpg",
        "S"
    )

    montage(
        imgs[-4:],
        "Figure_4_6_FailureCases.jpg",
        "F"
    )

print("Thesis figures saved to:", OUT)