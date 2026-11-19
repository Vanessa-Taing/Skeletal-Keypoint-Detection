#!/usr/bin/env python3
"""RO3 H1-C: indexed Channel GT/prediction visual audit."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets/InnoCount_COCO_Patched_UniqueIDs_v1"

BASE = ROOT / "experiments/RO3"

INPUT = (
    BASE / "H1B_Channel_Signature_Crosstab_v1"
    / "visual_review_candidates.csv"
)

VIT = ROOT / (
    "experiments/RO2_Final_Patched/"
    "ViTPoseB_E03_SharedProposals"
)

OUT = BASE / "H1C_Channel_Indexed_Overlays_v1"

assert not OUT.exists(), f"Refusing overwrite: {OUT}"

with INPUT.open(newline="") as f:
    cases = list(csv.DictReader(f))

assert cases

coco = json.loads(
    (DATA / "annotations/test.json").read_text()
)

images = {
    int(im["id"]): im for im in coco["images"]
}

annotations = {
    (int(a["image_id"]), int(a["id"])): a
    for a in coco["annotations"]
}

predictions = np.load(VIT / "vitpose_predictions.npy")
image_ids = np.load(VIT / "vitpose_image_ids.npy")
annotation_ids = np.load(VIT / "vitpose_annotation_ids.npy")

assert predictions.shape == (1424, 6, 2)

pred_map = {
    (int(image_ids[i]), int(annotation_ids[i])):
        predictions[i]
    for i in range(len(predictions))
}

assert len(pred_map) == 1424

manifest_path = ROOT / (
    "experiments/RO2_Final_Patched/"
    "YOLO26n_E09_FrozenSharedProposals/"
    "shared_proposals.jsonl"
)

proposals = {}

with manifest_path.open() as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            key = (
                int(row["image_id"]),
                int(row["annotation_id"])
            )
            proposals[key] = row

GREEN = (0, 220, 0)
PINK = (220, 30, 255)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)

OUT.mkdir(parents=True)

def panel(base, mode, gt, predicted, gt_box, pred_box,
          x0, y0, scale, annotation_id, metadata):

    resized = cv2.resize(
        base, None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_LINEAR
    )

    canvas = cv2.copyMakeBorder(
        resized, 68, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(35, 35, 35)
    )

    def pt(xy):
        return (
            int(round((xy[0]-x0)*scale)),
            int(round((xy[1]-y0)*scale + 68))
        )

    def draw_box(box, color):
        cv2.rectangle(
            canvas,
            pt(box[:2]),
            pt(box[2:]),
            color,
            2
        )

    def draw_points(points, prefix, color, dy):
        for j in range(4):
            loc = pt(points[j])

            cv2.circle(canvas, loc, 5, color, -1)

            cv2.putText(
                canvas,
                f"{prefix}{j+1}",
                (loc[0]+7, loc[1]+dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA
            )

    if mode in ("GT", "BOTH"):
        draw_box(gt_box, GREEN)
        draw_points(gt, "G", GREEN, -7)

    if mode in ("PRED", "BOTH"):
        draw_box(pred_box, YELLOW)
        draw_points(predicted, "P", PINK, 18)

    cv2.putText(
        canvas,
        f"{mode} | annotation {annotation_id}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, WHITE, 2, cv2.LINE_AA
    )

    cv2.putText(
        canvas, metadata,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44, WHITE, 1, cv2.LINE_AA
    )

    return canvas


cache = {}

for row in cases:
    key = (
        int(row["Image"].replace(".jpg", "")),
        int(row["Annotation_ID"])
    )

    assert key in annotations
    assert key in pred_map
    assert key in proposals
    assert proposals[key]["matched"]

    image_id, ann_id = key

    if image_id not in cache:
        path = (
            DATA / "images/test"
            / Path(images[image_id]["file_name"]).name
        )
        cache[image_id] = cv2.imread(str(path))
        assert cache[image_id] is not None, path

    image = cache[image_id]
    H, W = image.shape[:2]

    ann = annotations[key]

    kp = np.asarray(
        ann["keypoints"], dtype=np.float32
    ).reshape(6, 3)

    assert np.all(kp[:4, 2] > 0)

    gt = kp[:4, :2]
    predicted = pred_map[key][:4, :2]

    x, y, w, h = map(float, ann["bbox"])
    gt_box = np.array(
        [x, y, x+w, y+h], dtype=float
    )

    pred_box = np.asarray(
        proposals[key]["proposal_bbox_xyxy"],
        dtype=float
    )

    all_points = np.vstack([
        gt,
        predicted,
        gt_box.reshape(2, 2),
        pred_box.reshape(2, 2),
    ])

    low = all_points.min(axis=0)
    high = all_points.max(axis=0)

    margin = max(
        22, 0.15 * float((high-low).max())
    )

    x0 = max(0, int(np.floor(low[0]-margin)))
    y0 = max(0, int(np.floor(low[1]-margin)))
    x1 = min(W, int(np.ceil(high[0]+margin)))
    y1 = min(H, int(np.ceil(high[1]+margin)))

    assert x1 > x0 and y1 > y0

    base = image[y0:y1, x0:x1]

    scale = min(
        4.0,
        580 / base.shape[1],
        650 / base.shape[0],
    )

    normalizer = max(w, h)
    distances = np.linalg.norm(
        gt-predicted, axis=1
    ) / normalizer

    tolerance = 0.05 * normalizer

    meta = (
        f"Fixed={row['Fixed_correct_005']}/4 "
        f"Spatial={row['Spatial_correct_005']}/4 "
        f"Tol={tolerance:.1f}px"
    )

    views = [
        panel(
            base, mode,
            gt, predicted,
            gt_box, pred_box,
            x0, y0, scale,
            ann_id, meta
        )
        for mode in ("GT", "PRED", "BOTH")
    ]

    combined = cv2.hconcat(views)

    filename = (
        f"indexed_{image_id:06d}"
        f"_ann_{ann_id}.png"
    )

    assert cv2.imwrite(
        str(OUT / filename), combined
    )

    print(
        filename,
        "| mapping:",
        row["Nearest_mapping"],
        "| same-index errors:",
        np.round(distances, 3).tolist()
    )

print("\nPASS: Indexed visual overlays generated")
print("Cases:", len(cases))
print("Output:", OUT)
