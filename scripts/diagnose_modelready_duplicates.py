import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/owner/skeletal_keypoint_detection")
DATASET = ROOT / "datasets/InnoCount_KeyPoint_ModelReady_v2"

SPLITS = ["train", "valid", "test"]


def annotation_signature(annotation):
    return (
        annotation.get("category_id"),
        tuple(annotation.get("bbox", [])),
        tuple(annotation.get("keypoints", [])),
    )


def main():
    print("=" * 70)
    print("MODELREADY DUPLICATE ANNOTATION DIAGNOSTIC")
    print("=" * 70)

    total_duplicate_rows = 0

    for split in SPLITS:
        path = DATASET / "splits" / split / "annotations.json"

        with open(path) as f:
            data = json.load(f)

        images = {
            img["id"]: img["file_name"]
            for img in data["images"]
        }

        grouped = defaultdict(list)

        for ann in data["annotations"]:
            signature = annotation_signature(ann)

            grouped[
                (ann["image_id"], signature)
            ].append(ann)

        print(f"\n[{split.upper()}]")
        print("-" * 70)

        found = False

        for (image_id, signature), anns in grouped.items():
            if len(anns) > 1:
                found = True

                print(
                    f"\nImage: {images.get(image_id, image_id)}"
                )
                print(
                    f"  category: {signature[0]}"
                )
                print(
                    f"  bbox: {signature[1]}"
                )
                print(
                    f"  annotation IDs: {[a['id'] for a in anns]}"
                )

                total_duplicate_rows += len(anns) - 1

        if not found:
            print("No exact duplicate annotations found.")

    print("\n" + "=" * 70)
    print(f"Total duplicate annotation rows: {total_duplicate_rows}")
    print("=" * 70)


if __name__ == "__main__":
    main()