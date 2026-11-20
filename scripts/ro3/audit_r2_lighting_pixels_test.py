#!/usr/bin/env python3
"""RO3 R2: Read-only photometric clipping audit."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from prepare_r2_lighting import (
    LEVELS, NEUTRAL, condition_id,
    to_linear, to_srgb, render_lighting,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/RO3/R2_Lighting_v1"
OUT = ROOT / "experiments/RO3/R2_LightingDiagnostics_v1/test"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_sources(mode):
    folder = SOURCE / f"test_{mode}"
    meta = json.loads(
        (folder / "protocol_metadata.json").read_text()
    )
    manifest = folder / "lighting_manifest.jsonl"

    require(meta["split"] == "test", "Wrong split")
    require(meta["mode"] == mode, "Wrong mode")
    require(meta["levels"] == LEVELS[mode], "Wrong levels")

    neutral_id = condition_id(mode, NEUTRAL[mode])

    rows = {}
    for line in manifest.read_text().splitlines():
        rec = json.loads(line)
        if rec["condition_id"] == neutral_id:
            iid = int(rec["image_id"])
            require(iid not in rows, "Duplicate source image")
            rows[iid] = rec

    require(len(rows) == 74, "Expected 74 source images")

    coco_path = ROOT / meta["annotation_path"]
    require(
        sha256(coco_path) == meta["annotation_sha256"],
        "COCO annotation hash differs",
    )

    coco = json.loads(coco_path.read_text())
    require(len(coco["annotations"]) == 1626, "Wrong GT count")

    return rows, meta, coco


def main():
    require(not OUT.exists(), f"Output already exists: {OUT}")

    exposure, exp_meta, coco = get_sources("exposure")
    contrast, con_meta, _ = get_sources("contrast")

    require(
        set(exposure) == set(contrast),
        "Lighting axes have different source images",
    )
    require(
        exp_meta["annotation_sha256"]
        == con_meta["annotation_sha256"],
        "Annotation source differs between lighting axes",
    )

    boxes = defaultdict(list)
    for ann in coco["annotations"]:
        boxes[int(ann["image_id"])].append(ann["bbox"])

    results = []
    first_verified = set()

    for image_id in sorted(exposure):
        rec = exposure[image_id]
        other = contrast[image_id]
        require(
            rec["source_sha256"] == other["source_sha256"],
            "Different source images between axes",
        )

        path = ROOT / rec["source_image"]
        require(path.is_file(), f"Missing image: {path}")
        require(
            sha256(path) == rec["source_sha256"],
            f"Source image changed: {path}",
        )

        src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(src is not None, f"Cannot read: {path}")

        height, width = src.shape[:2]
        require(
            (height, width) == (rec["height"], rec["width"]),
            "Source dimensions differ",
        )

        linear = to_linear(src.astype(np.float32) / 255.0)

        luminance = (
            0.0722 * linear[:, :, 0]
            + 0.7152 * linear[:, :, 1]
            + 0.2126 * linear[:, :, 2]
        )
        pivot = float(luminance.mean())

        for mode in ("exposure", "contrast"):
            for value in LEVELS[mode]:
                if value == NEUTRAL[mode]:
                    changed = src.copy()
                    raw = linear
                else:
                    if mode == "exposure":
                        raw = linear * (2.0 ** value)
                    else:
                        raw = pivot + value * (linear - pivot)

                    changed = np.rint(
                        to_srgb(raw) * 255.0
                    ).astype(np.uint8)

                # Check that diagnostics use the exact
                # photometric function used for inference.
                check_id = (mode, value)

                if check_id not in first_verified:
                    reference = render_lighting(
                        {"mode": mode, "value": value},
                        src,
                    )
                    require(
                        np.array_equal(changed, reference),
                        f"Render mismatch: {mode} {value}",
                    )
                    first_verified.add(check_id)

                # Newly clipped channels exclude original
                # channels already at their limiting value.
                low = (raw < 0) & (linear > 0)
                high = (raw > 1) & (linear < 1)
                newly_clipped = low | high
                clipped_pixel = np.any(newly_clipped, axis=2)

                changed_pixel = np.any(
                    changed != src, axis=2
                )

                black_before = np.all(src == 0, axis=2)
                black_after = np.all(changed == 0, axis=2)
                white_before = np.all(src == 255, axis=2)
                white_after = np.all(changed == 255, axis=2)

                new_black = black_after & ~black_before
                new_white = white_after & ~white_before

                box_pixels = 0
                box_clipped_pixels = 0

                # Box-weighted counts deliberately double-
                # count pixels covered by overlapping GT
                # boxes. They are NOT steel-foreground masks.
                for x, y, w, h in boxes[image_id]:
                    x0 = max(0, min(width, int(np.floor(x))))
                    y0 = max(0, min(height, int(np.floor(y))))
                    x1 = max(
                        0, min(width, int(np.ceil(x + w)))
                    )
                    y1 = max(
                        0, min(height, int(np.ceil(y + h)))
                    )

                    require(
                        x1 > x0 and y1 > y0,
                        "Invalid clipped GT box",
                    )

                    box_pixels += (x1 - x0) * (y1 - y0)
                    box_clipped_pixels += int(
                        clipped_pixel[y0:y1, x0:x1].sum()
                    )

                results.append({
                    "mode": mode,
                    "value": value,
                    "condition_id": condition_id(mode, value),
                    "image_id": image_id,
                    "gt_objects": len(boxes[image_id]),
                    "pixels": height * width,
                    "channels": height * width * 3,
                    "changed_pixels": int(changed_pixel.sum()),
                    "new_clipped_channels": int(
                        newly_clipped.sum()
                    ),
                    "new_clipped_pixels": int(
                        clipped_pixel.sum()
                    ),
                    "new_black_pixels": int(new_black.sum()),
                    "new_white_pixels": int(new_white.sum()),
                    "gt_box_weighted_pixels": box_pixels,
                    "gt_box_weighted_new_clipped_pixels":
                        box_clipped_pixels,
                })

        print(f"Audited image {image_id}")

    summary = []

    for mode in ("exposure", "contrast"):
        for value in LEVELS[mode]:
            selected = [
                r for r in results
                if r["mode"] == mode and r["value"] == value
            ]

            require(len(selected) == 74, "Incomplete condition")

            def total(name):
                return sum(r[name] for r in selected)

            total_pixels = total("pixels")
            total_box_pixels = total("gt_box_weighted_pixels")

            summary.append({
                "mode": mode,
                "value": value,
                "condition_id": condition_id(mode, value),
                "source_images": len(selected),
                "changed_pixel_pct": (
                    100 * total("changed_pixels") / total_pixels
                ),
                "newly_clipped_pixel_pct": (
                    100 * total("new_clipped_pixels") / total_pixels
                ),
                "newly_clipped_channel_pct": (
                    100 * total("new_clipped_channels")
                    / total("channels")
                ),
                "new_black_pixel_pct": (
                    100 * total("new_black_pixels") / total_pixels
                ),
                "new_white_pixel_pct": (
                    100 * total("new_white_pixels") / total_pixels
                ),
                "gt_box_weighted_newly_clipped_pct": (
                    100 * total(
                        "gt_box_weighted_new_clipped_pixels"
                    ) / total_box_pixels
                ),
            })

    OUT.mkdir(parents=True)

    for filename, rows in (
        ("image_pixel_diagnostics.csv", results),
        ("lighting_clipping_summary.csv", summary),
    ):
        with (OUT / filename).open("x", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(rows[0])
            )
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "protocol": "RO3_R2_LightingPixelAudit_v1",
        "split": "test",
        "source_images": 74,
        "generator_sha256": sha256(
            ROOT / "scripts/ro3/prepare_r2_lighting.py"
        ),
        "audit_script_sha256": sha256(Path(__file__)),
        "interpretation": (
            "Newly clipped pixels have at least one color "
            "channel pushed outside the valid linear range "
            "that was not already at its input extreme. "
            "GT-box coverage is box-pixel-weighted and "
            "does not represent segmented steel foreground."
        ),
    }

    (OUT / "audit_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nR2 TEST PIXEL CLIPPING AUDIT")
    print("=" * 75)
    print(
        f"{'Condition':<15}"
        f"{'New clip %':>13}"
        f"{'New black %':>13}"
        f"{'New white %':>13}"
        f"{'GT-box clip %':>16}"
    )

    for row in summary:
        print(
            f"{row['condition_id']:<15}"
            f"{row['newly_clipped_pixel_pct']:>13.2f}"
            f"{row['new_black_pixel_pct']:>13.2f}"
            f"{row['new_white_pixel_pct']:>13.2f}"
            f"{row['gt_box_weighted_newly_clipped_pct']:>16.2f}"
        )

    print("\nPASS: Read-only pixel audit completed")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
