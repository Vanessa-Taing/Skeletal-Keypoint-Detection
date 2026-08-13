# InnoCount KeyPoint Frozen v2

This dataset is derived from `InnoCount_KeyPoint_Frozen_v1`.

The purpose of v2 is to improve split representativeness while preserving
the source-aware split constraint.

## What changed

Only the train/valid/test assignment of complete source groups changed.

The following were NOT changed:

- images
- image pixels
- annotations
- class IDs
- class names
- bounding boxes
- keypoint coordinates
- keypoint visibility
- skeletons
- exact duplicate canonicalization
- negative/background images

## Split policy

Target:

- train ≈ 80%
- valid ≈ 10%
- test ≈ 10%

The optimizer also targets approximately 80/10/10 object distributions
for each of the five classes.

Complete source groups are always kept in a single split.

## Reproducibility

The split optimizer is deterministic for a fixed seed.

The command and seed should be recorded in the research experiment log.
