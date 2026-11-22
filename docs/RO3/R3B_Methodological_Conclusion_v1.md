# RO3 R3-B — Simulated Projective Viewpoint Robustness

Date: 22 November 2026
Model: Frozen ViTPose-B E03, epoch 90
Status: Validation and frozen test evaluation completed

## Experimental design

Nine predefined conditions were evaluated: one neutral condition
and horizontal-like (yaw) and vertical-like (pitch) projective
deformations at 6% and 12% edge-inset severity.

Each homography was applied to the complete source photograph,
ground-truth keypoints, and ground-truth bounding boxes.

The severity values specify fractional image-edge insets, not
physical camera rotation angles. These transformations are
synthetic 2D projective warps, not real 3D viewpoint acquisitions.

The primary metric was fixed-index PCK@0.05 normalized by the
original ground-truth bounding-box maximum dimension. The
transformed-box-normalized PCK was retained as a secondary metric.

All comparisons used paired ground-truth identities. Confidence
intervals were computed by bootstrapping original source-image
clusters rather than treating keypoints as independent samples.

## Dataset and parity

Validation: 74 photographs, 72 annotated source photographs,
1,551 GT objects, 6,720 valid GT keypoints.

Test: 74 photographs, 71 annotated source photographs,
1,626 GT objects, 7,049 valid GT keypoints.

Frozen clean validation PCK: 5,196/6,720 = 77.3214%.
Frozen clean test PCK: 4,325/7,049 = 61.3562%.

Neutral evaluations reproduced the frozen clean baselines and
passed the coordinate-compatibility checks.

## Overall findings

Validation changes under nonneutral transformations ranged from
-2.72 to +0.24 percentage points.

Test changes ranged from -0.84 to +0.71 percentage points.

Six of eight nonneutral test conditions showed a small positive
observed change, while two showed a negative change. All eight
individual test source-image-cluster bootstrap 95% confidence
intervals included zero. The eight validation intervals also
included zero.

The test findings therefore do not establish a reliable average
PCK improvement or degradation for any tested projective condition.

The stronger horizontal-compression losses observed on validation
did not consistently reproduce on the frozen test partition.

## Profile-level interpretation

The 12% test conditions revealed heterogeneous category responses.

HI-Beam changes under yaw-left and yaw-right compression were
+2.09 and +1.02 percentage points, respectively.

T-beam changes under the same conditions were -0.44 and -2.08
percentage points. Other profiles showed smaller positive and
negative changes depending on deformation direction.

The opposing category-level changes partly offset in aggregate.
These results are descriptive. No profile-specific statistical
significance is established from the reported subgroup table.

Channel comprises only two independent test source photographs,
and Sheet Pile only one. Findings for these categories are
especially sensitive to source-scene composition.

## Comparison with R3-A

R3-A synthetic in-plane rotations yielded test PCK changes of
-1.69 to -4.36 percentage points, with four of six individual
nonneutral bootstrap intervals excluding zero.

R3-B synthetic projective warps yielded smaller and inconsistent
test PCK changes, with none of the eight individual intervals
excluding zero.

These transformation families differ in their geometry, severity
definitions, interpolation effects, artificial borders, and
top-down GT-box proposals. Their numerical effects cannot be
used to rank true physical camera-viewpoint robustness.

## Methodological limitations

1. Projective image warps do not reveal newly visible 3D surfaces.
2. Bilinear interpolation and artificial background borders
   can influence model inference.
3. GT-box size changes spatially and nonuniformly.
4. Original-box normalization is not projectively invariant.
5. Evaluation uses GT-conditioned top-down pose inference, not
   independently detected bounding boxes.
6. Some profile categories have very few independent scenes.
7. Bootstrap intervals are individual, unadjusted comparisons.

## Reproducibility and analysis correction

The frozen protocol definitions and prediction arrays were
preserved. A post-inference path correction was required in
summarize_r3b_test.py: its evaluation-folder selector was changed
from valid_{condition_id} to test_{condition_id}.

This correction changed the input directory used by the test
summary, not the model predictions, deformation definitions,
scoring formula, or bootstrap specification.

The final corrected summarizer and its output must be preserved
alongside the original pre-test protocol lock.

## Conclusion

Under the evaluated 6%-12% synthetic projective deformation
conditions, ViTPose-B E03 exhibited small, direction-dependent,
and statistically unresolved average changes in test fixed-index
PCK. Profile-level responses were heterogeneous and could offset
in aggregate.

The findings characterize robustness to the specified synthetic
projective warps only. They do not demonstrate robustness to
actual three-dimensional camera viewpoint changes.
