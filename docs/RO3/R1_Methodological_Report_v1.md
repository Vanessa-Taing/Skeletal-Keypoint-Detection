# RO3 R1 Methodological and Results Report
## Robustness of Skeletal Keypoint Localization Under Natural Stacking Conditions and Controlled Synthetic Occlusion

**Research project:** Skeletal Keypoint Detection for Structural Steel Profile Classification in Logistics Environments  
**Research component:** Research Objective 3, Experiment R1  
**Model:** ViTPose-B, experiment E03, epoch 90  
**Dataset:** InnoCount COCO Patched UniqueIDs v1  
**Version:** 1.0 — Final R1 report  
**Report date:** 19 November 2026  
**Status:** Experiments completed; evidence preservation to be finalized and verified separately

---

## 1. Purpose and Research Scope

RO3 R1 investigates the robustness of skeletal keypoint localization when structural steel profiles are observed under stacking, crowding, and occlusion conditions representative of logistics environments.

The investigation focuses on five structural steel categories: Angle Bar, Channel, T-beam, HI-Beam, and Sheet Pile. Each profile uses a predefined semantic skeletal representation with up to six global keypoint slots, G1–G6. Unused slots are masked according to the profile's annotation schema.

R1 evaluates the frozen ViTPose-B E03 model selected during RO2. The objective is not to improve or retrain the model, but to quantify how localization accuracy varies when visual evidence is obstructed.

Two complementary analyses are performed:

**R1-A — Natural stacking and occlusion audit:** Observational evaluation of existing logistics photographs, using original keypoint visibility annotations, source-image object density, and ground-truth bounding-box overlap.

**R1-B — Controlled synthetic occlusion robustness:** Paired evaluation of the same ground-truth objects under deterministic opaque-strip perturbations of increasing bounding-box-area coverage.

The two studies are interpreted separately because naturally occurring scene complexity cannot be reduced to a known physical occlusion percentage, while synthetic obstruction provides controlled severity but has limited physical realism.

Counting, object detection recall, and instance enumeration are outside the primary R1 assessment and are reserved for a subsequent application extension.

## 2. Dataset and Frozen Model

The patched COCO dataset consists of 594 training images, 74 validation images, and 74 test images, with 12,200, 1,551, and 1,626 annotated instances, respectively.

The frozen test partition contains 7,049 labeled, valid ground-truth keypoints across 1,626 objects. Of 74 test photographs, 71 contain annotated objects and three are negative images without annotated instances.

The test-set distribution is:

| Profile | Objects | Valid GT keypoints | Source images |
|---|---:|---:|---:|
| Angle Bar | 173 | 491 | 7 |
| Channel | 598 | 2,392 | 2 |
| T-beam | 397 | 1,584 | 37 |
| HI-Beam | 375 | 2,250 | 24 |
| Sheet Pile | 83 | 332 | 1 |
| **Total** | **1,626** | **7,049** | **71 annotated images** |

The model checkpoint is `experiments/ViTPose_Runs/E03_ViTPoseB_Patched_v1/epoch_90.pth`. The experiment uses the corresponding patched E03 configuration.

Inference is performed using ground-truth bounding boxes. The provided box identifies the target instance; ViTPose predicts its skeleton without independently detecting all objects in the photograph.

For synthetic experiments, a separate virtual image is generated for each target object, severity level, and replicate. Neighboring profiles remain visible as scene context unless intersected by the synthetic blocker. Each virtual image produces one evaluated target pose.

## 3. Evaluation Metrics and Reproducibility

The **primary metric** is fixed-index Percentage of Correct Keypoints at a threshold of 0.05, abbreviated PCK@0.05.

For a labeled keypoint, localization is counted as correct when the Euclidean distance between predicted and ground-truth coordinates, divided by the maximum width or height of the original ground-truth bounding box, does not exceed 0.05. Results are micro-averaged over all valid ground-truth keypoints.

COCO visibility values are interpreted as `v=0` for unlabeled, `v=1` for labeled but not visible, and `v=2` for labeled and visible. The primary PCK denominator includes points with `v>0`.

Secondary metrics include constrained order-invariant PCK and predefined structural edge accuracy. The revised symmetry calculation is a post-hoc sensitivity analysis from the RO3 H1 investigation. Physical-semantic equivalence for the affected profile endpoints remains subject to independent schema confirmation; consequently, the revised metric does not replace the frozen RO2 fixed-index baseline.

Structural edge accuracy evaluates predefined pairs of skeletal endpoints using the project's localization-error criterion. It should not be interpreted as an independent measure of true geometric angle, topological validity, or physical structural integrity.

The clean test evaluation reproduced the original ViTPose-B E03 baseline exactly:

- Fixed-index PCK@0.05: **4,325/7,049 = 61.3562%**.
- Constrained order-invariant PCK: **79.5716%**.
- Structural edge accuracy: **53.9554%**.

The maximum coordinate difference from the archived E03 prediction reference was approximately **0.00873 pixels**. The clean single-object versus batched-inference verification found a maximum difference of approximately **0.000061 pixels**, with no PCK decision changes among 47 checked valid keypoints.

These checks support compatibility of the new R1 inference pipeline with the frozen E03 evaluation.

## 4. R1-A: Natural Stacking and Occlusion Audit

### 4.1 Method

R1-A performs a read-only analysis of the original test photographs and the frozen clean predictions.

Three observational variables were extracted.

**Image object density:** The number of annotated objects in a source photograph, categorized as low (1–9), medium (10–49), or high (50 or more).

**Maximum GT bounding-box overlap:** For each target object, the largest pairwise intersection area between its clipped GT box and another annotated GT box, divided by the target's clipped GT-box area. Categories were no overlap, greater than 0–10%, greater than 10–30%, and greater than 30%.

**Original keypoint visibility:** Existing COCO visibility values distinguishing labeled-visible and labeled-occluded landmarks.

Bounding-box overlap is asymmetric because the intersection area is normalized by the target object's area. It measures projected spatial overlap, not visible steel-surface coverage or depth ordering.

### 4.2 Results by Image Density

| Scene density | GT objects | Source images | Fixed-index PCK |
|---|---:|---:|---:|
| Low: 1–9 objects | 199 | 45 | 74.87% |
| Medium: 10–49 objects | 424 | 20 | 69.13% |
| High: 50 or more | 1,003 | 6 | 55.34% |

The difference between low- and high-density groups is approximately **19.53 percentage points**.

This association is consistent with greater localization difficulty in densely populated scenes. It does not establish that density is itself the cause, because different density groups contain different profiles, object scales, viewpoints, and source images.

### 4.3 Results by Bounding-Box Overlap

| Maximum target-box overlap | GT objects | Source images | Fixed-index PCK |
|---|---:|---:|---:|
| None | 27 | 13 | 59.17% |
| Greater than 0–10% | 21 | 10 | 71.43% |
| Greater than 10–30% | 30 | 13 | 64.18% |
| Greater than 30% | 1,548 | 60 | 61.20% |

Approximately **95.2%** of annotated objects are classified in the greater-than-30% overlap group. This highly imbalanced distribution limits comparative analysis.

Within-profile results reinforce the limitation. All 598 Channel instances and all 83 Sheet Pile instances belong to the high-overlap group, while 170 of 173 Angle Bars also fall into that group. T-beams in the high-overlap group have 78.76% PCK, compared with 51.39% among the small set without overlapping boxes. Thus, bounding-box overlap does not show a consistently negative association with accuracy.

The overlap variable is retained as an indicator of local projected crowding, rather than being treated as a physical occlusion percentage.

### 4.4 Results by Original Keypoint Visibility

| Original visibility | Correct / valid | Fixed-index PCK |
|---|---:|---:|
| Labeled and visible (`v=2`) | 4,299 / 6,992 | 61.48% |
| Labeled but occluded (`v=1`) | 26 / 57 | 45.61% |

Originally occluded keypoints show an approximately **15.87-percentage-point** lower fixed-index PCK.

However, only 57 labeled keypoints across 20 photographs are marked originally occluded. The groups are not matched by profile or scene characteristics, so the observed difference is descriptive.

### 4.5 R1-A Interpretation

The natural-scene audit provides evidence that keypoint accuracy varies with scene crowding and original visibility annotations. It also demonstrates that simple bounding-box overlap is not a sufficiently discriminative or physically meaningful substitute for observed occlusion severity.

No causal relationship between stacking density, overlap area, and localization error is claimed.

## 5. R1-B: Controlled Synthetic Occlusion

### 5.1 Experimental Design

The frozen model is evaluated on unchanged source photographs and on virtual versions containing one opaque, steel-gray rectangular strip.

The strip is placed within the clipped ground-truth bounding box of one selected target object. Its orientation is horizontal or vertical and its placement is generated deterministically from a fixed seed, dataset split, composite object identity, placement mode, and replicate number.

The primary placement mode is **uniform**: strip orientation and anchor are sampled without using the target keypoint coordinates. The strip grows in thickness as severity increases while retaining its orientation and anchor within the same object–replicate series.

Severity is defined as:

**Synthetic occlusion severity (%) = 100 × opaque-strip area inside the clipped GT box / clipped GT-box area.**

The predefined severity levels are **0%, 10%, 20%, 30%, 40%, and 50%**.

This is an area fraction of the bounding rectangle, not the proportion of actual steel foreground surface hidden. Empty spaces, background, truck components, and other contents of the target rectangle may be covered. Image pixels outside the target rectangle do not contribute to its severity.

An exploratory keypoint-targeted placement mode was also generated on validation data. Its anchor is selected using originally visible target-keypoint coordinates, with a uniform fallback when no eligible landmark exists. This is a separate diagnostic stress test and is not included in the primary frozen test results.

### 5.2 Ground-Truth Policy

The synthetic blocker is rendered into the model input image without drawing GT points or annotations.

The original GT box, keypoint coordinates, semantic indices, category, and visibility values remain unchanged.

A separate metadata field identifies previously visible (`v=2`) keypoints whose coordinates are covered by the synthetic strip. These points remain in the evaluation denominator.

This is essential: removing hidden keypoints from the scoring denominator would create a severity-dependent evaluation population and could overstate performance.

The experiments preserve object identity using the composite pair `(image_id, annotation_id)`.

### 5.3 Evaluation Dataset and Repetitions

The validation split contains 1,551 annotated instances and 6,720 valid GT keypoints. The primary uniform-placement development benchmark uses one perturbation replicate per nonzero severity, plus one clean evaluation.

The frozen test split contains 1,626 instances and 7,049 valid GT keypoints. Three deterministic perturbation replicates are evaluated at each nonzero severity, while the clean baseline is evaluated once.

Each nonzero test severity therefore contains **21,147 repeated valid-keypoint evaluations** (7,049 original keypoints × three perturbations). These are repeated observations of the same original objects, not 21,147 independent keypoints or new photographs.

The 0% input reproduces the original GT-box E03 evaluation. The model checkpoint and preprocessing are held fixed across all conditions.

### 5.4 Validation Results

| Synthetic coverage | Fixed PCK@0.05 | Change from clean |
|---|---:|---:|
| 0% | 77.32% | Baseline |
| 10% | 57.49% | −19.84 pp |
| 20% | 51.25% | −26.07 pp |
| 30% | 45.83% | −31.49 pp |
| 40% | 41.16% | −36.16 pp |
| 50% | 36.58% | −40.74 pp |

The validation experiment shows a progressive decline across increasing perturbation levels. The largest step occurs between the clean condition and the first evaluated nonzero severity.

### 5.5 Frozen Test Results

| Synthetic coverage | Fixed PCK | Constrained OI PCK | Structural edge accuracy | Fixed PCK change |
|---|---:|---:|---:|---:|
| 0% | 61.36% | 79.57% | 53.96% | Baseline |
| 10% | 50.09% | 63.36% | 42.84% | −11.27 pp |
| 20% | 45.67% | 57.89% | 37.27% | −15.69 pp |
| 30% | 42.91% | 54.42% | 33.57% | −18.45 pp |
| 40% | 40.55% | 51.18% | 30.02% | −20.81 pp |
| 50% | **37.88%** | **47.90%** | **26.46%** | **−23.48 pp** |

The actual mean rasterized coverage closely matched the requested levels, reaching approximately 49.88% for the 50% condition.

At 50% synthetic coverage, fixed-index PCK decreased by **23.48 percentage points** relative to the clean test baseline.

The source-image-cluster bootstrap 95% interval for this paired difference was approximately **−36.55 to −16.51 percentage points**.

All nonzero-severity overall fixed-index differences had bootstrap intervals entirely below zero. These intervals account for clustering by original image, but cannot eliminate limitations from restricted scene diversity or profile composition.

### 5.6 Profile-Specific Robustness

| Profile | Clean fixed PCK | PCK at 50% | Paired change | Test source images |
|---|---:|---:|---:|---:|
| Angle Bar | 71.49% | 20.57% | −50.92 pp | 7 |
| Channel | 40.09% | 32.40% | −7.69 pp | 2 |
| T-beam | 77.02% | 29.57% | −47.45 pp | 37 |
| HI-Beam | 68.13% | 51.73% | −16.40 pp | 24 |
| Sheet Pile | 78.92% | 48.69% | −30.22 pp | 1 |

Angle Bar and T-beam exhibit the largest paired declines at 50% severity, while HI-Beam exhibits a smaller decline.

Channel's fixed-index accuracy decreases initially but rises modestly between 30% and 50% obstruction. This nonmonotonic behavior should not be interpreted as evidence that severe occlusion improves Channel localization: its clean fixed-index baseline is already relatively low, and all Channel test instances come from two photographs.

Sheet Pile measurements originate from a single test photograph. Profile-specific findings for Channel and Sheet Pile are therefore descriptive of the tested scenes rather than broadly generalizable scene-level estimates.

Differences among profiles may reflect geometry, scale, visual context, semantic-index ambiguity, and scene composition. This experiment does not identify which of those factors causes the observed differences.

### 5.7 Paired Keypoint Coverage Analysis

The paired analysis divides previously visible keypoints into two groups for each severity: those whose coordinates fall under the synthetic strip and those remaining outside it.

Each group's occluded predictions are compared against **the clean predictions of the exact same GT keypoints**.

| Coverage level | Change for newly hidden keypoints | Change for visible, uncovered keypoints |
|---|---:|---:|
| 10% | −14.59 pp | −10.95 pp |
| 20% | −36.16 pp | −11.84 pp |
| 30% | −42.70 pp | −11.78 pp |
| 40% | **−44.25 pp** | **−11.93 pp** |
| 50% | −42.78 pp | −12.89 pp |

At 40% severity, newly hidden points decline from 62.87% clean PCK to 18.62% after obstruction. Their paired change is −44.25 percentage points, with a reported image-cluster bootstrap interval of approximately −58.53 to −34.74 pp.

At the same severity, originally visible but uncovered points decline from 60.96% to 49.03%. Their paired change is −11.93 percentage points, with a corresponding interval of approximately −19.32 to −7.51 pp.

The nonzero decrease among uncovered landmarks indicates that perturbing part of the target box can affect predictions at other locations. This is consistent with dependence on wider skeletal or contextual visual information, although it does not reveal the model's internal mechanism.

The composition of newly hidden points changes across severity levels. Consequently, absolute PCK values of the hidden group should not be compared across severity as though they represent one unchanged keypoint population. The within-severity paired clean-to-occluded difference is the appropriate subgroup result.

## 6. Relationship Between R1-A and R1-B

R1-A and R1-B provide complementary evidence.

R1-A establishes descriptive relationships between existing logistics-scene properties and localization accuracy. It shows lower PCK in highly crowded source images and lower accuracy for the limited set of originally occluded keypoints.

R1-B measures the response of the frozen model to controlled visual obstruction while keeping ground-truth object identities, bounding boxes, semantic keypoint labels, and evaluation definitions unchanged.

The two approaches are consistent in showing that incomplete visual information is associated with lower localization accuracy. They are not numerically interchangeable because one measures observational scene properties while the other modifies image content using synthetic blockers.

Neither analysis provides a direct measurement of true three-dimensional stacking depth or the physical percentage of steel foreground obscured.

## 7. Limitations and Threats to Validity

**Synthetic occluder realism:** Opaque rectangular strips do not reproduce the precise shapes, textures, shadows, specular reflections, interlocking surfaces, and depth relationships found in physical steel stacks.

**Bounding-box denominator:** Coverage percentages measure clipped rectangular GT-box area rather than segmented steel foreground area. This can include substantial background or non-profile content.

**Top-down GT-box conditioning:** The primary experiment isolates pose localization with a supplied GT box. It does not test whether an object detector could discover the same target under occlusion. Consequently, the results cannot be interpreted as full-pipeline detection or counting performance.

**Scene concentration:** Channel instances occupy only two test source images, and Sheet Pile instances occupy one. The 1,626 GT objects do not represent 1,626 independent scene conditions.

**Original visibility scarcity:** Only 57 valid test keypoints have the original COCO occluded label, limiting the strength of natural-visibility comparisons.

**Semantic symmetry uncertainty:** Revised constrained order-invariant scores are retained as secondary sensitivity results pending physical-semantic validation of permitted profile reversals. The primary fixed-index metric is unaffected.

**Photometric side effects:** A solid synthetic occluder changes image appearance, not merely the availability of object evidence. The experiment cannot separate lost visual information from the model's response to an unfamiliar artificial patch.

**Generalizability:** Findings characterize the specified dataset, frozen checkpoint, preprocessing configuration, GT-box protocol and perturbation distribution. They are not directly transferable to other imaging systems, unseen logistics sites or different object detectors without additional evaluation.

## 8. Reproducibility and Evidence Preservation

The R1 implementation comprises the occlusion manifest generator, frozen ViTPose inference runner, validation and test execution scripts, paired statistical summarizers, and natural-stacking audit.

Primary source scripts:

- `scripts/ro3/prepare_r1_occlusion.py`
- `scripts/ro3/evaluate_r1_vitpose.py`
- `scripts/ro3/summarize_r1_validation.py`
- `scripts/ro3/summarize_r1_test.py`
- `scripts/ro3/audit_r1_natural_stacking.py`
- `scripts/ro3/run_r1_validation_remaining.sh`
- `scripts/ro3/run_r1_test_uniform.sh`

Evaluation dependencies include `scripts/evaluate/common_pose_metrics.py` and its versioned RO3 constrained-symmetry counterpart.

Principal result directories:

- `experiments/RO3/R1_Occlusion_v1/`
- `experiments/RO3/R1_Eval_v1/`
- `experiments/RO3/R1_Validation_PairedSummary_v1/`
- `experiments/RO3/R1_Test_PairedSummary_v1/`
- `experiments/RO3/R1_NaturalStackingAudit_v1/`

Source annotation and model fingerprints are to be included in a versioned SHA-256 evidence ledger. A separate local archive will preserve the complete derived R1 evidence, while GitHub will retain the source scripts, methodological report, evidence ledger, and archive checksum. The archive is considered frozen only after the preservation script completes and checksum verification succeeds.

The original datasets, model checkpoints and frozen RO2 reproducibility archive are not to be overwritten or regenerated.

## 9. Conclusion

The RO3 R1 investigation demonstrates that the frozen ViTPose-B E03 model is sensitive to incomplete visual evidence under both naturally complex logistics scenes and controlled synthetic occlusion.

The natural-scene audit identifies lower fixed-index keypoint accuracy in more crowded images and among the limited set of originally occluded landmarks, while showing that GT bounding-box overlap cannot be treated as a validated measure of physical occlusion.

The controlled experiment provides stronger paired evidence. On the frozen test set, fixed-index PCK decreases from **61.36% without synthetic obstruction to 37.88% under 50% target GT-box coverage**. Covered keypoints exhibit pronounced localization losses, and points outside the blocker also show measurable declines relative to their own clean predictions.

These results establish a reproducible benchmark of skeletal keypoint robustness to controlled visual obstruction. They also identify model limitations under incomplete structural evidence, without conflating pose localization with object detection or counting.

RO3 R1 is considered experimentally complete. Subsequent RO3 experiments will investigate lighting variation (R2) and viewpoint/orientation variation (R3), applying comparable paired evaluation principles while respecting the different types of transformations involved. Counting will be evaluated separately as an application extension.

---

**Report status:** Final experimental interpretation, version 1.0.  
**Evidence status:** Frozen archive and SHA-256 verification to be completed using the accompanying preservation procedure.  
**Scientific boundary:** Primary findings use fixed-index PCK; post-hoc constrained symmetry remains a secondary sensitivity measure.