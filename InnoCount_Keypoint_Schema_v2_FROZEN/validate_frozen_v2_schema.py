#!/usr/bin/env python3
import argparse, json
from pathlib import Path

EXPECTED = {
    "Channel": (
        ["new-point-0","new-point-1","new-point-2","new-point-3"],
        [[1,2],[2,3],[3,4]]
    ),
    "T-beam": (
        ["new-point-0","new-point-1","new-point-2","new-point-3"],
        [[1,2],[1,3],[1,4]]
    ),
    "HI-Beam": (
        ["new-point-0","new-point-1","new-point-2","new-point-3","new-point-4","new-point-5"],
        [[1,2],[1,3],[1,4],[4,5],[4,6]]
    ),
    "Angle Bar": (
        ["new-point-0","new-point-1","new-point-3"],
        [[1,3],[1,2]]
    ),
    "Sheet Pile": (
        ["new-point-0","new-point-1","new-point-2","new-point-3"],
        [[1,2],[2,3],[3,4]]
    )
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    args = p.parse_args()

    failed = False

    for split in ("train","valid","test"):
        path = args.dataset/"splits"/split/"annotations.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        cats = {c["name"]: c for c in data["categories"]}

        print(f"\n{split}:")

        for name, (kps, skeleton) in EXPECTED.items():
            c = cats.get(name)
            ok = (
                c is not None
                and c.get("keypoints") == kps
                and c.get("skeleton") == skeleton
            )

            print(f"  {'PASS' if ok else 'FAIL'}: {name}")

            if not ok:
                failed = True
                if c:
                    print("    actual keypoints:", c.get("keypoints"))
                    print("    actual skeleton:", c.get("skeleton"))

    if failed:
        raise SystemExit("\nFROZEN SEMANTIC SCHEMA VALIDATION FAILED")

    print("\nPASS: Frozen v2 structural schema matches the final semantic contract.")
    print("Semantic meanings are stored in keypoint_schema_v2.json and keypoint_schema_v2.txt.")

if __name__ == "__main__":
    main()
