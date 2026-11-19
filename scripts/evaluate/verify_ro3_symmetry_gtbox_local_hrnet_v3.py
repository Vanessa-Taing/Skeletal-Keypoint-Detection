#!/usr/bin/env python3
"""
RO3 H1-E: Independently compare original and revised
order-invariant PCK on frozen GT-box predictions.

Reads the existing source ZIP without extracting or
modifying its contents.

Checks:
  - Original and revised evaluator versions
  - All GT object identities
  - GT coordinates, visibility, boxes and categories
  - Original RO2 metric reproduction
  - No changes to fixed-index/edge metrics
  - Per-profile Channel and Sheet Pile OI differences
"""

import hashlib
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluate"))

import common_pose_metrics as original
import common_pose_metrics_ro3_constrained_symmetry_v1 as revised

ARCHIVE = ROOT / "RO3_Symmetry_Verification_Sources.zip"

OUT = ROOT / (
    "experiments/RO3/"
    "H1E_GTBox_Symmetry_Verification_v3"
)

ANNOTATIONS = (
    "datasets/InnoCount_COCO_Patched_UniqueIDs_v1/"
    "annotations/test.json"
)

MODELS = {
    "HRNet-W32": {
        "directory": (
            "experiments/HRNet_Runs/"
            "E09_HRNet_W32_Patched_v1/"
            "innocount/pose_hrnet/hrnet_w32_640"
        ),
        "prefix": "hrnet",
        "expected_correct": 2376,
        "expected_original_oi": 0.3740956163994893,
        "expected_edge": 0.24377650746819104,
    },
    "ViTPose-B": {
        "directory": (
            "experiments/RO2_Final_Patched/"
            "ViTPoseB_E03_CommonTest"
        ),
        "prefix": "vitpose",
        "expected_correct": 4325,
        "expected_original_oi": 0.7419492126542772,
        "expected_edge": 0.539553752535497,
    },
}

PROFILES = [
    "Angle Bar",
    "Channel",
    "T-beam",
    "HI-Beam",
    "Sheet Pile",
]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_array(archive, directory, prefix, field):
    # The HRNet E09 training output directory contains
    # 1551 validation rows, not the final test export.
    #
    # Load the verified 1626-row RO2 final-test export
    # directly from its original local directory.
    if prefix == "hrnet":
        source = ROOT / (
            "experiments/RO2_Final_Patched/"
            "HRNet_Output/innocount/pose_hrnet/"
            "hrnet_w32_640"
        ) / f"hrnet_{field}.npy"

        if not source.is_file():
            raise FileNotFoundError(source)

        return np.load(
            source,
            allow_pickle=False,
        )

    # ViTPose E03 remains sourced from the original ZIP.
    filename = f"{directory}/{prefix}_{field}.npy"

    return np.load(
        io.BytesIO(archive.read(filename)),
        allow_pickle=False,
    )


def validate_schema():
    for profile in ("Channel", "Sheet Pile"):
        schema = revised.PROFILE_SCHEMAS[profile]

        require(
            schema["equivalent_groups"] == [],
            f"{profile}: unexpected independent groups",
        )

        allowed = schema["allowed_full_permutations"]

        require(
            allowed == [
                (0, 1, 2, 3, 4, 5),
                (3, 2, 1, 0, 4, 5),
            ],
            f"{profile}: incorrect allowed permutations",
        )

        edge_set = {
            frozenset(edge) for edge in schema["edges"]
        }

        for permutation in allowed:
            transformed = {
                frozenset(
                    (permutation[a], permutation[b])
                )
                for a, b in schema["edges"]
            }

            require(
                transformed == edge_set,
                f"{profile}: permutation breaks skeleton",
            )

            for index in range(4):
                require(
                    schema["roles"][index]
                    == schema["roles"][permutation[index]],
                    f"{profile}: permutation changes role",
                )

    for profile in ("Angle Bar", "T-beam", "HI-Beam"):
        require(
            original.PROFILE_SCHEMAS[profile]
            == revised.PROFILE_SCHEMAS[profile],
            f"{profile}: schema changed unexpectedly",
        )


def main():
    require(
        ARCHIVE.is_file(),
        f"Missing ZIP: {ARCHIVE}",
    )
    require(
        not OUT.exists(),
        f"Refusing to overwrite existing output: {OUT}",
    )

    validate_schema()

    results = {}

    with ZipFile(ARCHIVE) as archive:
        # Ensure the local evaluator files are identical
        # to the versions included in the submitted ZIP.
        for filename in (
            "scripts/evaluate/common_pose_metrics.py",
            "scripts/evaluate/"
            "common_pose_metrics_ro3_constrained_symmetry_v1.py",
        ):
            stored = archive.read(filename)
            local = (ROOT / filename).read_bytes()

            require(
                hash_bytes(stored) == hash_bytes(local),
                f"Evaluator mismatch: {filename}",
            )

        coco = json.loads(
            archive.read(ANNOTATIONS).decode("utf-8")
        )

        category_map = (
            original.make_category_map_from_coco(coco)
        )

        gt_lookup = {}

        for ann in coco["annotations"]:
            key = (
                int(ann["image_id"]),
                int(ann["id"]),
            )

            require(
                key not in gt_lookup,
                f"Duplicate GT identity: {key}",
            )
            gt_lookup[key] = ann

        require(
            len(gt_lookup) == 1626,
            f"Expected 1626 GT objects, got {len(gt_lookup)}",
        )

        print("PASS: COCO GT contains 1626 unique objects")
        print("PASS: Constrained symmetry rules validated")

        for model_name, config in MODELS.items():
            print("\n" + "=" * 72)
            print(model_name)
            print("=" * 72)

            directory = config["directory"]
            prefix = config["prefix"]

            fields = (
                "predictions",
                "gt",
                "gt_visibility",
                "bboxes",
                "image_ids",
                "annotation_ids",
                "category_ids",
            )

            a = {
                field: read_array(
                    archive, directory, prefix, field
                )
                for field in fields
            }

            n = len(a["predictions"])

            require(
                n == 1626,
                f"{model_name}: expected 1626 predictions, got {n}",
            )

            for field in fields:
                require(
                    len(a[field]) == n,
                    f"{model_name}: {field} length mismatch",
                )

            require(
                a["predictions"].shape == (n, 6, 2),
                f"{model_name}: unexpected prediction shape",
            )

            require(
                a["gt"].shape == (n, 6, 2),
                f"{model_name}: unexpected GT shape",
            )

            require(
                a["bboxes"].shape == (n, 4),
                f"{model_name}: unexpected bbox shape",
            )

            raw_visibility = a["gt_visibility"]

            print(
                "Exported visibility shape:",
                raw_visibility.shape,
            )

            # Some HRNet exports contain a singleton
            # dimension: (N, 6, 1).
            # Remove only that dimension.
            if raw_visibility.shape == (n, 6, 1):
                a["gt_visibility"] = raw_visibility[:, :, 0]

            elif raw_visibility.shape == (n, 6):
                pass

            else:
                raise ValueError(
                    f"{model_name}: unsupported visibility shape "
                    f"{raw_visibility.shape}. "
                    "Expected (N, 6) or (N, 6, 1)."
                )

            visible = a["gt_visibility"] > 0

            require(
                visible.shape == (n, 6),
                f"{model_name}: unexpected visibility shape",
            )

            require(
                int(visible.sum()) == 7049,
                f"{model_name}: GT denominator is not 7049",
            )

            seen = set()

            for i in range(n):
                key = (
                    int(a["image_ids"][i]),
                    int(a["annotation_ids"][i]),
                )

                require(
                    key in gt_lookup,
                    f"{model_name}: unknown annotation {key}",
                )

                require(
                    key not in seen,
                    f"{model_name}: duplicate annotation {key}",
                )

                seen.add(key)
                ann = gt_lookup[key]

                require(
                    int(a["category_ids"][i])
                    == int(ann["category_id"]),
                    f"{model_name}: category mismatch {key}",
                )

                ann_kp = np.asarray(
                    ann["keypoints"], dtype=np.float32
                ).reshape(6, 3)

                require(
                    np.array_equal(
                        visible[i],
                        ann_kp[:, 2] > 0,
                    ),
                    f"{model_name}: visibility mismatch {key}",
                )

                require(
                    np.allclose(
                        a["gt"][i][visible[i]],
                        ann_kp[:, :2][visible[i]],
                        atol=0.01,
                        rtol=0,
                    ),
                    f"{model_name}: GT coordinates mismatch {key}",
                )

                require(
                    np.allclose(
                        a["bboxes"][i],
                        np.asarray(
                            ann["bbox"], dtype=np.float32
                        ),
                        atol=0.01,
                        rtol=0,
                    ),
                    f"{model_name}: GT bbox mismatch {key}",
                )

            require(
                seen == set(gt_lookup),
                f"{model_name}: missing GT identities",
            )

            args = dict(
                predictions=a["predictions"],
                gt=a["gt"],
                gt_vis=a["gt_visibility"],
                bboxes=a["bboxes"],
                category_ids=a["category_ids"],
                category_map=category_map,
                threshold=0.05,
            )

            old = original.calculate_all_metrics(**args)
            new = revised.calculate_all_metrics(**args)

            # Primary metrics must remain completely unchanged.
            for metric in (
                "standard_pck",
                "profile_pck",
                "semantic_roles",
                "structural_edges",
            ):
                require(
                    old[metric] == new[metric],
                    f"{model_name}: unexpected change in {metric}",
                )

            standard = old["standard_pck"]

            require(
                standard["total_correct"]
                == config["expected_correct"],
                f"{model_name}: original fixed PCK not reproduced",
            )

            require(
                standard["total_valid"] == 7049,
                f"{model_name}: original denominator mismatch",
            )

            old_oi = old["order_invariant_pck"]
            new_oi = new["order_invariant_pck"]

            require(
                abs(
                    old_oi["overall"]
                    - config["expected_original_oi"]
                ) < 1e-7,
                f"{model_name}: original OI baseline mismatch",
            )

            require(
                abs(
                    old["structural_edges"]["overall"]
                    - config["expected_edge"]
                ) < 1e-7,
                f"{model_name}: edge baseline mismatch",
            )

            # The other three profiles must have identical OI.
            for profile in ("Angle Bar", "T-beam", "HI-Beam"):
                require(
                    old_oi["profile"][profile]
                    == new_oi["profile"][profile],
                    f"{model_name}: {profile} OI changed",
                )

            # Complete reversal adds an option to Channel.
            require(
                new_oi["profile"]["Channel"]["correct"]
                >= old_oi["profile"]["Channel"]["correct"],
                f"{model_name}: Channel OI unexpectedly decreased",
            )

            # Complete reversal restricts the old Sheet Pile
            # independent-swap possibilities.
            require(
                new_oi["profile"]["Sheet Pile"]["correct"]
                <= old_oi["profile"]["Sheet Pile"]["correct"],
                f"{model_name}: Sheet Pile OI unexpectedly increased",
            )

            print("PASS: All 1626 GT identities aligned")
            print("PASS: All 7049 valid GT keypoints aligned")
            print("PASS: Original RO2 baselines reproduced")
            print("PASS: All non-OI metrics unchanged")
            print()

            print(
                f"{'Profile':<14}"
                f"{'Original OI':>14}"
                f"{'Revised OI':>14}"
                f"{'Change':>12}"
            )

            profile_results = {}

            for profile in PROFILES:
                before = old_oi["profile"][profile]
                after = new_oi["profile"][profile]

                delta = (
                    after["pck"] - before["pck"]
                ) * 100

                print(
                    f"{profile:<14}"
                    f"{before['pck']:>14.4%}"
                    f"{after['pck']:>14.4%}"
                    f"{delta:>+11.2f} pp"
                )

                profile_results[profile] = {
                    "original": before,
                    "revised": after,
                    "change_percentage_points": delta,
                }

            overall_delta = (
                new_oi["overall"] - old_oi["overall"]
            ) * 100

            print()
            print(
                "Fixed PCK:",
                f"{standard['overall']:.6%}",
            )
            print(
                "Original overall OI:",
                f"{old_oi['overall']:.6%}",
            )
            print(
                "Revised overall OI:",
                f"{new_oi['overall']:.6%}",
            )
            print(
                "OI change:",
                f"{overall_delta:+.4f} pp",
            )

            results[model_name] = {
                "objects": n,
                "valid_gt_keypoints": int(visible.sum()),
                "fixed_pck": standard["overall"],
                "structural_edge_accuracy":
                    old["structural_edges"]["overall"],
                "original_order_invariant": old_oi["overall"],
                "revised_order_invariant": new_oi["overall"],
                "change_percentage_points": overall_delta,
                "profiles": profile_results,
            }

    OUT.mkdir(parents=True)

    output_file = OUT / "gtbox_symmetry_verification.json"

    output_file.write_text(
        json.dumps(results, indent=2)
    )

    print("\n" + "=" * 72)
    print("PASS: GT-box constrained symmetry verification completed")
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
