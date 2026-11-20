### R3-A: In-Plane Orientation Robustness

The R3-A experiment evaluated the robustness of the frozen ViTPose-B E03 model to seven predefined in-plane image rotations ranging from −45° to +45°. Complete source photographs were rotated together with their ground-truth keypoints and bounding boxes. The primary evaluation used fixed-index PCK@0.05 normalized by the original ground-truth bounding-box maximum dimension, thereby avoiding an artificially relaxed correctness threshold caused by the enlargement of axis-aligned boxes during rotation.

On validation, the clean PCK of 77.32% decreased by 5.80–9.17 percentage points across the six nonzero rotation conditions. All six individual source-image-cluster bootstrap 95% confidence intervals for the paired changes lay below zero.

On the frozen test partition, the clean PCK of 61.36% decreased by 1.69–4.36 percentage points. The reductions were more pronounced at ±30° and ±45°, with individual cluster-bootstrap confidence intervals below zero, whereas the intervals for ±15° included zero.

Profile-level analysis revealed heterogeneous sensitivity. Angle Bar exhibited the largest observed decline, reaching −18.94 percentage points at −45°. T-beam also showed substantial reductions, including −7.07 percentage points at +45°, across 37 independent test source photographs. HI-Beam exhibited smaller changes under the reported extreme rotations. Channel and Sheet Pile results require particular caution because they contain only two and one independent test photographs, respectively.

A secondary evaluation using transformed axis-aligned bounding-box normalization produced systematically higher PCK values, demonstrating the influence of normalization choice on measured rotation robustness.

Overall, R3-A provides evidence that the frozen skeletal keypoint localization model is sensitive to synthetic in-plane orientation changes, with performance degradation depending on rotation magnitude, profile category and scene composition. Because the procedure also introduces interpolation, expanded-canvas borders and transformed top-down box proposals, the measured decline cannot be attributed exclusively to the model's lack of rotational equivariance.

These findings characterize in-plane image orientation robustness, not true three-dimensional viewpoint robustness. Viewpoint-like projective distortions are evaluated separately in R3-B.