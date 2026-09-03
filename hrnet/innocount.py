import os
import itertools
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO

from dataset.JointsDataset import JointsDataset


class InnoCountDataset(JointsDataset):

    NUM_JOINTS = 6

    # ------------------------------------------------------------------
    # Profile-aware equivalent keypoint groups
    # ------------------------------------------------------------------
    #
    # Keypoint indices are zero-based.
    #
    # These groups define keypoints that are geometrically equivalent
    # within a particular structural profile.
    #
    PROFILE_GROUPS = {

        "Angle Bar": [
            [1, 2],     # two equivalent tips
        ],

        "Channel": [
            [0, 3],     # two equivalent tips
            [1, 2],     # two equivalent corners
        ],

        "T-beam": [
            [1, 2],     # two flange tips
        ],

        "HI-Beam": [
            [1, 2],     # upper flange tips
            [4, 5],     # lower flange tips
        ],

        "Sheet Pile": [
            [0, 3],     # two equivalent tips
            [1, 2],     # two equivalent corners
        ],
    }

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        cfg,
        root,
        image_set,
        is_train,
        transform=None,
    ):

        super().__init__(
            cfg,
            root,
            image_set,
            is_train,
            transform,
        )

        self.num_joints = self.NUM_JOINTS
        self.flip_pairs = []
        self.parent_ids = []

        ann_file = os.path.join(
            root,
            "annotations",
            f"{image_set}.json",
        )

        self.coco = COCO(
            ann_file
        )

        self.db = self._get_db()

    # ------------------------------------------------------------------
    # Keypoint padding
    # ------------------------------------------------------------------

    def _pad_keypoints(
        self,
        keypoints,
    ):

        kp = np.array(
            keypoints,
            dtype=np.float32,
        ).reshape(-1, 3)

        if len(kp) > self.NUM_JOINTS:

            raise ValueError(
                f"Annotation contains {len(kp)} keypoints, "
                f"but NUM_JOINTS={self.NUM_JOINTS}"
            )

        padded = np.zeros(
            (self.NUM_JOINTS, 3),
            dtype=np.float32,
        )

        padded[
            :len(kp)
        ] = kp

        return padded

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    def _get_db(self):

        gt_db = []

        for img_id in self.coco.getImgIds():

            img = self.coco.loadImgs(
                img_id
            )[0]

            ann_ids = self.coco.getAnnIds(
                imgIds=img_id
            )

            anns = self.coco.loadAnns(
                ann_ids
            )

            for ann in anns:

                kp = self._pad_keypoints(
                    ann["keypoints"]
                )

                x, y, w, h = ann["bbox"]

                category_id = ann[
                    "category_id"
                ]

                category = self.coco.loadCats(
                    category_id
                )[0]

                category_name = category[
                    "name"
                ]

                gt_db.append(
                    {
                        "image": os.path.join(
                            self.root,
                            "images",
                            self.image_set,
                            img["file_name"],
                        ),

                        "center": np.array(
                            [
                                x + w / 2,
                                y + h / 2,
                            ],
                            dtype=np.float32,
                        ),

                        "scale": np.array(
                            [
                                w / 200,
                                h / 200,
                            ],
                            dtype=np.float32,
                        ),

                        "joints_3d": kp,

                        "joints_3d_vis": (
                            kp[:, 2:] > 0
                        ).astype(
                            np.float32
                        ),

                        "bbox": ann["bbox"],

                        "score": 1.0,

                        "filename": "",

                        "imgnum": 0,

                        "category_id": (
                            category_id
                        ),

                        "category_name": (
                            category_name
                        ),

                        # Keep the COCO annotation ID
                        # for reference only.
                        "annotation_id": int(
                            ann["id"]
                        ),

                        "image_id": int(
                            ann["image_id"]
                        ),
                    }
                )

        return gt_db

    # ------------------------------------------------------------------
    # Profile-aware matching
    # ------------------------------------------------------------------

    def _role_aware_distances(
        self,
        pred,
        gt,
        valid,
        scale,
        profile_name,
    ):

        if profile_name not in self.PROFILE_GROUPS:

            raise ValueError(
                f"Unknown profile "
                f"'{profile_name}'. "
                f"Known profiles: "
                f"{list(self.PROFILE_GROUPS)}"
            )

        matched = np.zeros(
            self.NUM_JOINTS,
            dtype=bool,
        )

        # --------------------------------------------------------------
        # First match all fixed / non-equivalent joints.
        # --------------------------------------------------------------

        grouped_indices = set()

        for group in self.PROFILE_GROUPS[
            profile_name
        ]:
            grouped_indices.update(
                group
            )

        for joint_idx in range(
            self.NUM_JOINTS
        ):

            if joint_idx in grouped_indices:
                continue

            if not valid[joint_idx]:
                continue

            error = np.linalg.norm(
                pred[joint_idx]
                - gt[joint_idx]
            )

            matched[joint_idx] = (
                error <= 0.05 * scale
            )

        # --------------------------------------------------------------
        # Match equivalent groups using minimum total distance.
        # --------------------------------------------------------------

        for group in self.PROFILE_GROUPS[
            profile_name
        ]:

            valid_gt = [
                idx
                for idx in group
                if valid[idx]
            ]

            if not valid_gt:
                continue

            valid_pred = [
                idx
                for idx in group
            ]

            if len(valid_gt) != len(
                valid_pred
            ):
                continue

            best_assignment = None
            best_total = float("inf")

            for permutation in itertools.permutations(
                valid_pred
            ):

                total = 0.0

                for gt_idx, pred_idx in zip(
                    valid_gt,
                    permutation,
                ):

                    total += np.linalg.norm(
                        pred[pred_idx]
                        - gt[gt_idx]
                    )

                if total < best_total:

                    best_total = total
                    best_assignment = (
                        permutation
                    )

            if best_assignment is None:
                continue

            for gt_idx, pred_idx in zip(
                valid_gt,
                best_assignment,
            ):

                error = np.linalg.norm(
                    pred[pred_idx]
                    - gt[gt_idx]
                )

                matched[gt_idx] = (
                    error <= 0.05 * scale
                )

        return matched

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cfg,
        preds,
        output_dir,
        all_boxes,
        img_path,
        *args,
        **kwargs,
    ):

        preds = np.asarray(
            preds,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # IMPORTANT
        #
        # HRNet's `preds[i]` corresponds directly to `self.db[i]`.
        #
        # We therefore save the exact GT, visibility, bbox and metadata
        # in this same order.
        #
        # This avoids trying to reconstruct HRNet's internal ordering
        # from COCO annotation IDs.
        # --------------------------------------------------------------

        if output_dir is not None:

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_predictions.npy",
                ),
                preds,
            )

            hrnet_gt = np.asarray(
                [
                    record["joints_3d"]
                    for record in self.db
                ],
                dtype=np.float32,
            )

            hrnet_gt_vis = np.asarray(
                [
                    record["joints_3d_vis"]
                    for record in self.db
                ],
                dtype=np.float32,
            )

            hrnet_bboxes = np.asarray(
                [
                    record["bbox"]
                    for record in self.db
                ],
                dtype=np.float32,
            )

            hrnet_image_ids = np.asarray(
                [
                    record["image_id"]
                    for record in self.db
                ],
                dtype=np.int64,
            )

            hrnet_category_ids = np.asarray(
                [
                    record["category_id"]
                    for record in self.db
                ],
                dtype=np.int64,
            )

            hrnet_annotation_ids = np.asarray(
                [
                    record["annotation_id"]
                    for record in self.db
                ],
                dtype=np.int64,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_gt.npy",
                ),
                hrnet_gt,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_gt_visibility.npy",
                ),
                hrnet_gt_vis,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_bboxes.npy",
                ),
                hrnet_bboxes,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_image_ids.npy",
                ),
                hrnet_image_ids,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_category_ids.npy",
                ),
                hrnet_category_ids,
            )

            np.save(
                os.path.join(
                    output_dir,
                    "hrnet_annotation_ids.npy",
                ),
                hrnet_annotation_ids,
            )

        # --------------------------------------------------------------
        # Validate prediction shape
        # --------------------------------------------------------------

        if (
            preds.ndim != 3
            or preds.shape[1]
            != self.NUM_JOINTS
        ):

            raise ValueError(
                f"Unexpected prediction shape: "
                f"{preds.shape}. "
                f"Expected "
                f"[N, {self.NUM_JOINTS}, 2+]."
            )

        pred_xy = preds[
            :,
            :,
            :2,
        ]

        gt_xy = np.array(
            [
                record["joints_3d"][
                    :,
                    :2
                ]
                for record in self.db
            ],
            dtype=np.float32,
        )

        gt_vis = np.array(
            [
                record["joints_3d_vis"][
                    :,
                    0
                ]
                for record in self.db
            ],
            dtype=np.float32,
        )

        n = min(
            len(pred_xy),
            len(gt_xy),
            len(self.db),
        )

        if n == 0:

            raise RuntimeError(
                "No predictions available "
                "for evaluation."
            )

        pred_xy = pred_xy[:n]
        gt_xy = gt_xy[:n]
        gt_vis = gt_vis[:n]

        # --------------------------------------------------------------
        # Normalization scale
        # --------------------------------------------------------------

        scales = np.array(
            [
                max(
                    float(
                        record["bbox"][2]
                    ),
                    float(
                        record["bbox"][3]
                    ),
                )
                for record in self.db[:n]
            ],
            dtype=np.float32,
        )

        scales[
            scales <= 0
        ] = 1.0

        # --------------------------------------------------------------
        # Standard fixed-index PCK
        # --------------------------------------------------------------

        distances = np.linalg.norm(
            pred_xy - gt_xy,
            axis=2,
        )

        normalized_distances = (
            distances
            / scales[:, None]
        )

        threshold = 0.05

        correct = (
            normalized_distances
            <= threshold
        )

        valid = (
            gt_vis > 0
        )

        num_correct = np.sum(
            correct & valid
        )

        num_valid = np.sum(
            valid
        )

        pck = (
            float(
                num_correct
                / num_valid
            )
            if num_valid > 0
            else 0.0
        )

        # --------------------------------------------------------------
        # Standard per-joint PCK
        # --------------------------------------------------------------

        joint_pck = {}

        for j in range(
            self.NUM_JOINTS
        ):

            joint_valid = (
                valid[:, j]
            )

            if np.sum(
                joint_valid
            ) == 0:

                joint_pck[j] = 0.0

            else:

                joint_pck[j] = float(
                    np.sum(
                        correct[:, j]
                        & joint_valid
                    )
                    / np.sum(
                        joint_valid
                    )
                )

        # --------------------------------------------------------------
        # Profile-specific analysis
        # --------------------------------------------------------------

        profiles = [
            "Angle Bar",
            "Channel",
            "T-beam",
            "HI-Beam",
            "Sheet Pile",
        ]

        profile_stats = {}

        for profile in profiles:

            indices = [
                i
                for i, record
                in enumerate(
                    self.db[:n]
                )
                if record[
                    "category_name"
                ] == profile
            ]

            if not indices:
                continue

            profile_correct = 0
            profile_valid = 0
            role_correct = 0

            for i in indices:

                valid_i = valid[i]

                profile_valid += int(
                    np.sum(valid_i)
                )

                profile_correct += int(
                    np.sum(
                        correct[i]
                        & valid_i
                    )
                )

                matched = (
                    self._role_aware_distances(
                        pred_xy[i],
                        gt_xy[i],
                        valid_i,
                        scales[i],
                        profile,
                    )
                )

                role_correct += int(
                    np.sum(
                        matched
                        & valid_i
                    )
                )

            standard = (
                profile_correct
                / profile_valid
                if profile_valid > 0
                else 0.0
            )

            role_aware = (
                role_correct
                / profile_valid
                if profile_valid > 0
                else 0.0
            )

            profile_stats[
                profile
            ] = {
                "instances": len(
                    indices
                ),
                "valid_joints": int(
                    profile_valid
                ),
                "standard_pck": float(
                    standard
                ),
                "role_aware_pck": float(
                    role_aware
                ),
                "gain": float(
                    role_aware
                    - standard
                ),
            }

        # --------------------------------------------------------------
        # Results
        # --------------------------------------------------------------

        name_values = (
            OrderedDict()
        )

        name_values[
            "PCK"
        ] = pck

        for j in range(
            self.NUM_JOINTS
        ):

            name_values[
                f"PCK_joint_{j + 1}"
            ] = joint_pck[j]

        # --------------------------------------------------------------
        # Save profile analysis
        # --------------------------------------------------------------

        if output_dir is not None:

            os.makedirs(
                output_dir,
                exist_ok=True,
            )

            result_file = os.path.join(
                output_dir,
                "pck_results.txt",
            )

            with open(
                result_file,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    "HRNet InnoCount Evaluation\n"
                )

                f.write(
                    "=" * 70
                    + "\n\n"
                )

                f.write(
                    f"Threshold: "
                    f"{threshold}\n"
                )

                f.write(
                    f"Valid joints: "
                    f"{num_valid}\n"
                )

                f.write(
                    f"Correct joints: "
                    f"{num_correct}\n\n"
                )

                f.write(
                    "Standard PCK\n"
                )

                f.write(
                    f"Overall: "
                    f"{pck:.6f}\n\n"
                )

                for j in range(
                    self.NUM_JOINTS
                ):

                    f.write(
                        f"Joint {j + 1}: "
                        f"{joint_pck[j]:.6f}\n"
                    )

                f.write(
                    "\nProfile Analysis\n"
                )

                f.write(
                    "-" * 70
                    + "\n"
                )

                for profile, stats in (
                    profile_stats.items()
                ):

                    f.write(
                        f"\n{profile}\n"
                    )

                    f.write(
                        f"  Instances: "
                        f"{stats['instances']}\n"
                    )

                    f.write(
                        f"  Valid joints: "
                        f"{stats['valid_joints']}\n"
                    )

                    f.write(
                        f"  Standard PCK: "
                        f"{stats['standard_pck']:.6f}\n"
                    )

                    f.write(
                        f"  Role-aware PCK: "
                        f"{stats['role_aware_pck']:.6f}\n"
                    )

                    f.write(
                        f"  Gain: "
                        f"{stats['gain']:.6f}\n"
                    )

        return (
            name_values,
            pck,
        )