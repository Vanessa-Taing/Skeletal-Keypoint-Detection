from pathlib import Path
from PIL import Image, ImageDraw
import argparse


CLASS_NAMES = {
    0: "Channel",
    1: "T-beam",
    2: "HI-Beam",
    3: "Angle Bar",
    4: "Sheet Pile",
}

# Active keypoints for each class.
# Global representation has K=6.
ACTIVE_K = {
    0: 4,  # Channel
    1: 4,  # T-beam
    2: 6,  # HI-Beam
    3: 3,  # Angle Bar
    4: 4,  # Sheet Pile
}

# Same skeleton logic as your class-specific schemas,
# expressed using global K indices.
SKELETON = {
    0: [(1, 2), (2, 3), (3, 4)],          # Channel
    1: [(1, 2), (1, 3), (1, 4)],          # T-beam
    2: [(1, 2), (1, 3), (1, 4), (4, 5), (4, 6)],  # HI-Beam
    3: [(1, 3), (1, 2)],                  # Angle Bar
    4: [(1, 2), (2, 3), (3, 4)],          # Sheet Pile
}


def parse_label(label_path):
    records = []

    with open(label_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            values = line.strip().split()

            if not values:
                continue

            if len(values) != 23:
                raise ValueError(
                    f"{label_path}:{line_no}: "
                    f"expected 23 values, got {len(values)}"
                )

            cls = int(values[0])

            bbox = list(map(float, values[1:5]))

            keypoints = []

            for i in range(6):
                base = 5 + i * 3

                x = float(values[base])
                y = float(values[base + 1])
                v = int(float(values[base + 2]))

                keypoints.append((x, y, v))

            records.append({
                "class": cls,
                "bbox": bbox,
                "keypoints": keypoints,
            })

    return records


def draw_pose(image, records, output_path):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    W, H = image.size

    for obj_idx, obj in enumerate(records):

        cls = obj["class"]
        name = CLASS_NAMES.get(cls, f"class-{cls}")

        # -------------------------------------------------
        # Bounding box
        # -------------------------------------------------

        cx, cy, bw, bh = obj["bbox"]

        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            f"{obj_idx}: {name}",
            fill="red",
        )

        # -------------------------------------------------
        # Keypoints
        # -------------------------------------------------

        kp_pixels = {}

        for kp_idx, (x, y, v) in enumerate(obj["keypoints"], 1):

            if v <= 0:
                continue

            px = x * W
            py = y * H

            kp_pixels[kp_idx] = (px, py)

            r = 6

            draw.ellipse(
                [
                    px - r,
                    py - r,
                    px + r,
                    py + r,
                ],
                fill="yellow",
                outline="black",
            )

            draw.text(
                (px + 7, py - 7),
                f"K{kp_idx}",
                fill="yellow",
            )

        # -------------------------------------------------
        # Skeleton
        # -------------------------------------------------

        for a, b in SKELETON.get(cls, []):

            if a not in kp_pixels or b not in kp_pixels:
                continue

            draw.line(
                [
                    kp_pixels[a],
                    kp_pixels[b],
                ],
                fill="cyan",
                width=3,
            )

    image.save(output_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Image filename, e.g. 00002.jpg",
    )

    parser.add_argument(
        "--output",
        default="yolo_pose_inspection.jpg",
    )

    args = parser.parse_args()

    dataset = Path(args.dataset)

    image_path = (
        dataset
        / "images"
        / args.split
        / args.image
    )

    label_path = (
        dataset
        / "labels"
        / args.split
        / Path(args.image).with_suffix(".txt")
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if not label_path.exists():
        raise FileNotFoundError(label_path)

    image = Image.open(image_path)

    records = parse_label(label_path)

    print(f"Image : {image_path}")
    print(f"Label : {label_path}")
    print(f"Size  : {image.size}")
    print(f"Objects: {len(records)}")
    print()

    for i, obj in enumerate(records):

        cls = obj["class"]

        print(
            f"Object {i}: "
            f"{CLASS_NAMES.get(cls, cls)}"
        )

        print(
            f"  bbox = {obj['bbox']}"
        )

        for k, (x, y, v) in enumerate(
            obj["keypoints"], 1
        ):
            print(
                f"  K{k}: "
                f"({x:.6f}, {y:.6f}) "
                f"v={v}"
            )

        print()

    draw_pose(
        image,
        records,
        args.output,
    )

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()