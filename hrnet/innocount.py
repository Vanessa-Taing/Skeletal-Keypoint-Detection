import os
from collections import OrderedDict

import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO

from dataset.JointsDataset import JointsDataset


class InnoCountDataset(JointsDataset):
    """
    HRNet dataset loader for InnoCount.

    Original annotations contain variable-length keypoints:
        Angle Bar : 3
        Channel   : 4
        T-beam    : 4
        Sheet Pile : 4
        HI-Beam   : 6

    HRNet requires a fixed K=6, therefore missing joints are
    padded with invisible (0, 0, 0).

    Evaluation provides two metrics:

    1. Standard PCK
       Assumes prediction joint i corresponds to GT joint i.

    2. Order-Invariant PCK
       Does not assume that the arbitrary annotation numbering
       corresponds to the prediction numbering.

       Visible GT joints are matched to predicted joints using
       minimum-cost Hungarian matching.
    """

    NUM_JOINTS = 6

    def __init__(
        self,
        cfg,
        root,
        image_set,
        is_train,
        transform=None
    ):
        super().__init__(
            cfg,
            root,
            image_set,
            is_train,
            transform
        )

        self.num_joints = self.NUM_JOINTS

        # No left/right semantic symmetry is assumed.
        self.flip_pairs = []
        self.parent_ids = []

        ann_file = os.path.join(
            root,
            "annotations",
            f"{image_set}.json"
        )

        self.coco = COCO(ann_file)

        self.db = self._get_db()

    # ------------------------------------------------------------------
    # Annotation loading
    # ------------------------------------------------------------------

    def _pad_keypoints(self, keypoints):
        """
        Convert variable-length COCO keypoints into fixed K=6 format.

        Example:

            3 joints:
                [x1,y1,v1, x2,y2,v2, x3,y3,v3]
                ->
                [joint0, joint1, joint2, 0, 0, 0]

            4 joints:
                ->
                [joint0, joint1, joint2, joint3, 0, 0]

            6 joints:
                ->
                [joint0 ... joint5]
        """

        kp = np.array(
            keypoints,
            dtype=np.float32
        ).reshape(-1, 3)

        if len(kp) > self.NUM_JOINTS:
            raise ValueError(
                f"Annotation contains {len(kp)} joints, "
                f"but NUM_JOINTS={self.NUM_JOINTS}."
            )

        padded = np.zeros(
            (self.NUM_JOINTS, 3),
            dtype=np.float32
        )

        padded[:len(kp)] = kp

        return padded

    def _get_db(self):

        gt_db = []

        for img_id in self.coco.getImgIds():

            img = self.coco.loadImgs(img_id)[0]

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
                                y + h / 2
                            ],
                            dtype=np.float32,
                        ),

                        "scale": np.array(
                            [
                                w / 200,
                                h / 200
                            ],
                            dtype=np.float32,
                        ),

                        "joints_3d": kp,

                        "joints_3d_vis": (
                            kp[:, 2:] > 0
                        ).astype(np.float32),

                        "bbox": ann["bbox"],

                        "score": 1.0,

                        "filename": "",

                        "imgnum": 0,
                    }
                )

        return gt_db

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_scale(bbox):
        """
        Return the normalization scale used by PCK.

        scale = max(width, height)
        """

        x, y, w, h = bbox

        scale = max(
            float(w),
            float(h)
        )

        if scale <= 0:
            scale = 1.0

        return scale

    @staticmethod
    def _normalized_distance_matrix(
        pred_xy,
        gt_xy,
        scale
    ):
        """
        Calculate pairwise normalized Euclidean distances.

        pred_xy:
            [N_pred, 2]

        gt_xy:
            [N_gt, 2]

        Returns:
            [N_pred, N_gt]
        """

        diff = (
            pred_xy[:, None, :]
            - gt_xy[None, :, :]
        )

        distances = np.linalg.norm(
            diff,
            axis=2
        )

        return distances / scale

    # ------------------------------------------------------------------
    # Standard PCK
    # ------------------------------------------------------------------

    def _standard_pck(
        self,
        pred_xy,
        gt_xy,
        gt_vis,
        scales,
        threshold
    ):
        """
        Existing index-based PCK.

        Prediction joint i is compared directly against
        GT joint i.
        """

        distances = np.linalg.norm(
            pred_xy - gt_xy,
            axis=2
        )

        normalized_distances = (
            distances / scales[:, None]
        )

        correct = (
            normalized_distances <= threshold
        )

        valid = gt_vis > 0

        num_correct = np.sum(
            correct & valid
        )

        num_valid = np.sum(valid)

        if num_valid == 0:
            pck = 0.0
        else:
            pck = float(
                num_correct / num_valid
            )

        joint_pck = {}

        for j in range(self.NUM_JOINTS):

            joint_valid = valid[:, j]

            if np.sum(joint_valid) == 0:

                joint_pck[j] = 0.0

            else:

                joint_pck[j] = float(
                    np.sum(
                        correct[:, j]
                        & joint_valid
                    )
                    / np.sum(joint_valid)
                )

        return (
            pck,
            joint_pck,
            num_correct,
            num_valid
        )

    # ------------------------------------------------------------------
    # Order-invariant PCK
    # ------------------------------------------------------------------

    def _order_invariant_pck(
        self,
        pred_xy,
        gt_xy,
        gt_vis,
        scales,
        threshold
    ):
        """
        Order-invariant PCK.

        For each object:

            1. Extract visible GT joints.
            2. Consider all six predicted joint locations.
            3. Construct a pairwise normalized-distance matrix.
            4. Use Hungarian matching to find the minimum-cost
               assignment.
            5. Count matched pairs whose normalized distance
               <= threshold.

        Importantly, GT annotation indices are NOT assumed
        to carry semantic meaning.

        Returns:
            overall PCK
            per-GT-joint PCK
            number of correct matches
            number of valid GT joints
        """

        total_correct = 0
        total_valid = 0

        # Per original GT index.
        # This is mainly useful diagnostically.
        joint_correct = np.zeros(
            self.NUM_JOINTS,
            dtype=np.int64
        )

        joint_valid = np.zeros(
            self.NUM_JOINTS,
            dtype=np.int64
        )

        num_samples = len(pred_xy)

        for i in range(num_samples):

            # ----------------------------------------------------------
            # Visible GT joints
            # ----------------------------------------------------------

            valid_gt_indices = np.where(
                gt_vis[i] > 0
            )[0]

            if len(valid_gt_indices) == 0:
                continue

            current_gt = gt_xy[
                i,
                valid_gt_indices
            ]

            # ----------------------------------------------------------
            # Predictions
            # ----------------------------------------------------------

            current_pred = pred_xy[i]

            # ----------------------------------------------------------
            # Pairwise normalized distances
            # ----------------------------------------------------------

            distance_matrix = (
                self._normalized_distance_matrix(
                    current_pred,
                    current_gt,
                    scales[i]
                )
            )

            # ----------------------------------------------------------
            # Hungarian matching
            #
            # linear_sum_assignment expects:
            #
            #     rows = predictions
            #     cols = GT joints
            #
            # Since there are always 6 predictions and at most
            # 6 visible GT joints, every GT joint receives a
            # unique prediction.
            # ----------------------------------------------------------

            pred_indices, gt_indices = (
                linear_sum_assignment(
                    distance_matrix
                )
            )

            # ----------------------------------------------------------
            # Evaluate matched pairs
            # ----------------------------------------------------------

            for pred_idx, gt_local_idx in zip(
                pred_indices,
                gt_indices
            ):

                gt_original_idx = (
                    valid_gt_indices[
                        gt_local_idx
                    ]
                )

                distance = (
                    distance_matrix[
                        pred_idx,
                        gt_local_idx
                    ]
                )

                joint_valid[
                    gt_original_idx
                ] += 1

                total_valid += 1

                if distance <= threshold:

                    joint_correct[
                        gt_original_idx
                    ] += 1

                    total_correct += 1

        # --------------------------------------------------------------
        # Overall PCK
        # --------------------------------------------------------------

        if total_valid == 0:
            pck = 0.0
        else:
            pck = float(
                total_correct / total_valid
            )

        # --------------------------------------------------------------
        # Per-joint PCK
        # --------------------------------------------------------------

        joint_pck = {}

        for j in range(self.NUM_JOINTS):

            if joint_valid[j] == 0:

                joint_pck[j] = 0.0

            else:

                joint_pck[j] = float(
                    joint_correct[j]
                    / joint_valid[j]
                )

        return (
            pck,
            joint_pck,
            total_correct,
            total_valid
        )

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
        """
        Evaluate HRNet predictions using two PCK metrics.

        Standard PCK:
            prediction[i] <-> GT[i]

        Order-Invariant PCK:
            predicted points are optimally matched to visible
            GT points using Hungarian matching.

        Normalization:
            Euclidean distance / max(bbox_width, bbox_height)

        Correct threshold:
            normalized distance <= 0.05
        """

        preds = np.asarray(
            preds,
            dtype=np.float32
        )

        # --------------------------------------------------------------
        # Validate prediction shape
        # --------------------------------------------------------------

        if (
            preds.ndim != 3
            or preds.shape[1] != self.NUM_JOINTS
            or preds.shape[2] < 2
        ):
            raise ValueError(
                f"Unexpected prediction shape: {preds.shape}. "
                f"Expected [N, {self.NUM_JOINTS}, 2+]."
            )

        # HRNet gives:
        #
        # [x, y, confidence]
        #
        pred_xy = preds[:, :, :2]

        # --------------------------------------------------------------
        # Ground truth
        # --------------------------------------------------------------

        gt_xy = np.array(
            [
                record["joints_3d"][:, :2]
                for record in self.db
            ],
            dtype=np.float32
        )

        gt_vis = np.array(
            [
                record["joints_3d_vis"][:, 0]
                for record in self.db
            ],
            dtype=np.float32
        )

        # --------------------------------------------------------------
        # Prediction / GT count
        # --------------------------------------------------------------

        n = min(
            len(pred_xy),
            len(gt_xy)
        )

        if n == 0:
            raise RuntimeError(
                "No predictions available for evaluation."
            )

        pred_xy = pred_xy[:n]
        gt_xy = gt_xy[:n]
        gt_vis = gt_vis[:n]

        # --------------------------------------------------------------
        # Bounding-box normalization
        # --------------------------------------------------------------

        scales = np.asarray(
            [
                self._bbox_scale(
                    record["bbox"]
                )
                for record in self.db[:n]
            ],
            dtype=np.float32
        )

        # --------------------------------------------------------------
        # PCK threshold
        # --------------------------------------------------------------

        threshold = 0.05

        # ==============================================================
        # 1. STANDARD PCK
        # ==============================================================

        (
            standard_pck,
            standard_joint_pck,
            standard_correct,
            standard_valid
        ) = self._standard_pck(
            pred_xy,
            gt_xy,
            gt_vis,
            scales,
            threshold
        )

        # ==============================================================
        # 2. ORDER-INVARIANT PCK
        # ==============================================================

        (
            invariant_pck,
            invariant_joint_pck,
            invariant_correct,
            invariant_valid
        ) = self._order_invariant_pck(
            pred_xy,
            gt_xy,
            gt_vis,
            scales,
            threshold
        )

        # --------------------------------------------------------------
        # Results
        # --------------------------------------------------------------

        name_values = OrderedDict()

        # Existing metric
        name_values["PCK"] = standard_pck

        for j in range(self.NUM_JOINTS):

            name_values[
                f"PCK_joint_{j + 1}"
            ] = standard_joint_pck[j]

        # New diagnostic metric
        name_values[
            "OrderInvariant_PCK"
        ] = invariant_pck

        for j in range(self.NUM_JOINTS):

            name_values[
                f"OrderInvariant_PCK_joint_{j + 1}"
            ] = invariant_joint_pck[j]

        # For now, keep the original PCK as the official
        # performance indicator used by HRNet.
        perf_indicator = standard_pck

        # --------------------------------------------------------------
        # Save detailed results
        # --------------------------------------------------------------

        if output_dir is not None:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            result_file = os.path.join(
                output_dir,
                "pck_results.txt"
            )

            with open(
                result_file,
                "w"
            ) as f:

                f.write(
                    "HRNet InnoCount PCK Evaluation\n"
                )

                f.write(
                    "=" * 60 + "\n\n"
                )

                f.write(
                    f"Threshold : {threshold}\n\n"
                )

                # ------------------------------------------------------
                # Standard
                # ------------------------------------------------------

                f.write(
                    "STANDARD PCK\n"
                )

                f.write(
                    "-" * 30 + "\n"
                )

                f.write(
                    "Prediction joint i is compared "
                    "with GT joint i.\n\n"
                )

                f.write(
                    f"Valid joints   : "
                    f"{standard_valid}\n"
                )

                f.write(
                    f"Correct joints : "
                    f"{standard_correct}\n"
                )

                f.write(
                    f"Overall PCK    : "
                    f"{standard_pck:.6f}\n\n"
                )

                f.write(
                    "Per-joint PCK\n"
                )

                for j in range(
                    self.NUM_JOINTS
                ):

                    f.write(
                        f"Joint {j + 1}: "
                        f"{standard_joint_pck[j]:.6f}\n"
                    )

                # ------------------------------------------------------
                # Order invariant
                # ------------------------------------------------------

                f.write(
                    "\n"
                )

                f.write(
                    "ORDER-INVARIANT PCK\n"
                )

                f.write(
                    "-" * 30 + "\n"
                )

                f.write(
                    "Predicted joints are matched "
                    "to visible GT joints using "
                    "Hungarian minimum-cost matching.\n\n"
                )

                f.write(
                    f"Valid joints   : "
                    f"{invariant_valid}\n"
                )

                f.write(
                    f"Correct joints : "
                    f"{invariant_correct}\n"
                )

                f.write(
                    f"Overall PCK    : "
                    f"{invariant_pck:.6f}\n\n"
                )

                f.write(
                    "Per-joint PCK\n"
                )

                for j in range(
                    self.NUM_JOINTS
                ):

                    f.write(
                        f"Joint {j + 1}: "
                        f"{invariant_joint_pck[j]:.6f}\n"
                    )

                # ------------------------------------------------------
                # Diagnostic interpretation
                # ------------------------------------------------------

                gap = (
                    invariant_pck
                    - standard_pck
                )

                f.write(
                    "\n"
                )

                f.write(
                    "DIAGNOSTIC\n"
                )

                f.write(
                    "-" * 30 + "\n"
                )

                f.write(
                    f"Order-invariance gain: "
                    f"{gap:+.6f}\n"
                )

                if gap > 0.05:

                    f.write(
                        "The model performs substantially "
                        "better when arbitrary joint ordering "
                        "is ignored.\n"
                    )

                    f.write(
                        "This suggests that annotation ordering "
                        "may be affecting the standard PCK.\n"
                    )

                else:

                    f.write(
                        "There is no large improvement from "
                        "ignoring joint ordering.\n"
                    )

                    f.write(
                        "Poor localization may therefore be "
                        "a more important source of error.\n"
                    )

        return (
            name_values,
            perf_indicator
        )