from pathlib import Path
from collections import Counter

ROOT = Path("/home/owner/skeletal_keypoint_detection")
DATASET = ROOT / "datasets/InnoCount_KeyPoint_YOLO26_Pose_v2"

SPLITS = ["train", "valid", "test"]


def main():
    print("=" * 70)
    print("YOLO LABEL DUPLICATE DIAGNOSTIC")
    print("=" * 70)

    total_files = 0
    total_rows = 0
    total_duplicate_rows = 0

    for split in SPLITS:
        label_dir = DATASET / "labels" / split

        if not label_dir.exists():
            print(f"\nWARNING: missing {label_dir}")
            continue

        print(f"\n[{split.upper()}]")
        print("-" * 70)

        split_files = 0
        split_rows = 0
        split_duplicate_rows = 0

        for label_file in sorted(label_dir.glob("*.txt")):
            lines = [
                line.strip()
                for line in label_file.read_text().splitlines()
                if line.strip()
            ]

            if not lines:
                continue

            counts = Counter(lines)
            duplicates = {
                row: count
                for row, count in counts.items()
                if count > 1
            }

            if duplicates:
                duplicate_rows = sum(count - 1 for count in duplicates.values())

                print(f"\n{label_file.name}")
                print(f"  total rows       : {len(lines)}")
                print(f"  unique rows      : {len(counts)}")
                print(f"  duplicate rows   : {duplicate_rows}")

                for row, count in duplicates.items():
                    print(f"  repeated {count}x:")
                    print(f"    {row}")

                split_duplicate_rows += duplicate_rows

            split_files += 1
            split_rows += len(lines)

        total_files += split_files
        total_rows += split_rows
        total_duplicate_rows += split_duplicate_rows

        print(f"\n{split} summary:")
        print(f"  label files      : {split_files}")
        print(f"  label rows       : {split_rows}")
        print(f"  duplicate rows   : {split_duplicate_rows}")

    print("\n" + "=" * 70)
    print("TOTAL")
    print("=" * 70)
    print(f"Label files        : {total_files}")
    print(f"Label rows         : {total_rows}")
    print(f"Exact duplicate rows: {total_duplicate_rows}")


if __name__ == "__main__":
    main()