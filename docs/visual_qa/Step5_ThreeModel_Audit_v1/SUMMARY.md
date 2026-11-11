# Three-model visual audit

- Reviewed dataset: `datasets/InnoCount_COCO_UniqueIDs_v1`
- Total sampled objects: 48
- Scoring: fixed-index PCK@0.05 in original-image pixel space.
- Model comparisons remain protocol-mixed: YOLO predicted boxes; HRNet/ViTPose GT boxes.
- Red crosses = ground truth; blue dots = predicted points; yellow rectangle = ground-truth bbox.
- Sampling is deliberately *targeted*, not representative of dataset-level prevalence.

## Strata (eligible cases / selected)

- `01_channel_near_G3G4`: 276 eligible, 8 selected
- `02_channel_separated_G3G4`: 144 eligible, 8 selected
- `03_channel_model_disagreement`: 14 eligible, 8 selected
- `04_other_shared_failures`: 64 eligible, 8 selected
- `05_vitpose_strong_yolo_weak`: 556 eligible, 8 selected
- `06_channel_near_G1G2`: 122 eligible, 8 selected

## Test-set image concentration

- **Angle Bar**: 173 objects in 7 images; largest image 000616.jpg contains 39
- **Channel**: 598 objects in 2 images; largest image 000477.jpg contains 498
- **HI-Beam**: 375 objects in 24 images; largest image 000075.jpg contains 120
- **Sheet Pile**: 83 objects in 1 images; largest image 000530.jpg contains 83
- **T-beam**: 397 objects in 37 images; largest image 000556.jpg contains 148

**Do not treat 598 Channel objects from two images as 598 independent visual scenes.**
These targeted test samples are for diagnosing existing results, not tuning hyperparameters on the held-out test set.

## Reviewer instructions

Open PNGs in `overlays/` and fill `review_worksheet.csv` without editing the dataset.
For a suspected wrong label, record the **image_id + annotation_id + affected G number(s)** and why.
Distinguish physical keypoint ambiguity from equivalence under mirror/rotation.
Do not infer occlusion, lighting, or viewpoint categories solely from PCK values.
If a full-image context is required, open the matching source in the dataset `images/test/` folder.

## Model-ground-truth consistency

All three prediction bundles were checked for identical composite identities and COCO GT visible coordinates/bboxes prior to sampling.
