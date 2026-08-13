#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

EXPECTED = {
    "Channel": {
        "keypoints": ["new-point-0","new-point-1","new-point-2","new-point-3"],
        "skeleton": [[1,2],[2,3],[3,4]],
    },
    "T-beam": {
        "keypoints": ["new-point-0","new-point-1","new-point-2","new-point-3"],
        "skeleton": [[1,2],[1,3],[1,4]],
    },
    "HI-Beam": {
        "keypoints": ["new-point-0","new-point-1","new-point-2","new-point-3","new-point-4","new-point-5"],
        "skeleton": [[1,2],[1,3],[1,4],[4,5],[4,6]],
    },
    "Angle Bar": {
        "keypoints": ["new-point-0","new-point-1","new-point-3"],
        "skeleton": [[1,3],[1,2]],
    },
    "Sheet Pile": {
        "keypoints": ["new-point-0","new-point-1","new-point-2","new-point-3"],
        "skeleton": [[1,2],[2,3],[3,4]],
    },
}

SPLITS = ("train", "valid", "test")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    args = p.parse_args()

    schemas = {}

    for split in SPLITS:
        path = args.dataset / "splits" / split / "annotations.json"
        data = json.loads(path.read_text(encoding="utf-8"))

        current = {}

        for cat in data["categories"]:
            name = cat["name"]
            current[name] = {
                "keypoints": cat.get("keypoints", []),
                "skeleton": cat.get("skeleton", []),
            }

        schemas[split] = current

    print("CLASS-SPECIFIC SCHEMA VALIDATION")
    print("=" * 70)

    failed = False

    for split in SPLITS:
        print(f"\n{split}:")

        for name, expected in EXPECTED.items():
            actual = schemas[split].get(name)

            if actual != expected:
                failed = True
                print(f"  FAIL: {name}")
                print(f"    expected keypoints: {expected['keypoints']}")
                print(f"    actual keypoints  : {actual.get('keypoints') if actual else None}")
                print(f"    expected skeleton : {expected['skeleton']}")
                print(f"    actual skeleton   : {actual.get('skeleton') if actual else None}")
            else:
                print(f"  PASS: {name}")

    if failed:
        raise SystemExit("\nSCHEMA VALIDATION FAILED")

    print("\nPASS: all splits match the class-specific schema.")

if __name__ == "__main__":
    main()
