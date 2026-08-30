from pathlib import Path
import shutil

# scripts/converter/convert_to_hrnet.py
ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "datasets" / "InnoCount_KeyPoint_Frozen_v2"
DST = ROOT / "datasets" / "HRNet_COCO"

SPLITS = ["train", "valid", "test"]


def convert(split):
    src_img = SRC / "splits" / split / "images"
    src_ann = SRC / "splits" / split / "annotations.json"

    dst_img = DST / "images" / split
    dst_ann = DST / "annotations"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)

    for img in src_img.glob("*"):
        shutil.copy2(img, dst_img / img.name)

    shutil.copy2(src_ann, dst_ann / f"{split}.json")
    print(f"✓ {split} complete")


def main():
    if DST.exists():
        shutil.rmtree(DST)

    for split in SPLITS:
        convert(split)

    print("\nHRNet dataset created successfully.")
    print(DST)


if __name__ == "__main__":
    main()