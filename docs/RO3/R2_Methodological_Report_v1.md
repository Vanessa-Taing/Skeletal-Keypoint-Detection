# RO3 R2 — Lighting Robustness of Skeletal Keypoint Localization

**Project:** Skeletal Keypoint Detection for Structural Steel Profile Classification in Logistics Environments  
**Research component:** RO3 R2-A and R2-B  
**Model:** Frozen ViTPose-B E03, epoch 90  
**Dataset:** InnoCount COCO Patched UniqueIDs v1  
**Report version:** 1.0  
**Date:** 20 November 2026  
**Status:** Exposure and contrast experiments complete; evidence freeze to be verified separately

## 1. Objective

The R2 investigation evaluates how controlled global photometric transformations affect the localization of semantic skeletal keypoints on five structural steel profile categories: Angle Bar, Channel, T-beam, HI-Beam and Sheet Pile.

The study extends the RO3 robustness investigation without retraining or modifying the selected ViTPose-B E03 checkpoint. It isolates top-down skeletal keypoint localization using original ground-truth bounding boxes, rather than evaluating independent object detection, object counting or full-pipeline performance.

Two separate experimental axes are considered: global exposure adjustment (R2-A) and global contrast adjustment (R2-B). Localized shadows and nonuniform illumination are outside the scope of these completed experiments.

## 2. Experimental Protocol

### 2.1 Dataset and frozen inference

The validation partition contains 74 photographs, 72 annotated source photographs, 1,551 GT objects and 6,720 valid GT keypoints.

The test partition contains 74 photographs, 71 annotated source photographs, 1,626 GT objects and 7,049 valid GT keypoints.

The same frozen ViTPose-B E03 configuration and epoch-90 checkpoint are used for every condition. Each complete source photograph is photometrically transformed once, and all of its annotated GT boxes are supplied to the top-down pose model.

Original ground-truth image coordinates, bounding boxes, semantic keypoint indices, categories and visibility annotations are unchanged. The transformations do not alter image geometry.

### 2.2 Global exposure experiment

Exposure is simulated by converting 8-bit sRGB image values to linear light, scaling the linear-light channels by a factor of \(2^{EV}\), clipping as necessary, and converting back to 8-bit sRGB.

Five predefined exposure settings are evaluated:

- −2 EV
- −1 EV
- 0 EV
- +1 EV
- +2 EV

The 0 EV transformation returns a pixel-identical copy of the original photograph.

### 2.3 Global contrast experiment

The original image is converted to linear-light RGB. Each channel is scaled about the source photograph's mean linear luminance using:

\[
I_{\mathrm{out}} = \mu + c(I_{\mathrm{in}}-\mu)
\]

where \(\mu\) is the source image's mean linear luminance and \(c\) is the predefined contrast multiplier.

The multipliers are 0.65, 0.80, 1.00, 1.20 and 1.40.

Values outside the valid linear-light range are clipped before conversion back to sRGB. The 1.00× neutral condition is pixel-identical to the original image.

The exposure and contrast transformations constitute separate experimental axes. Their settings are not treated as levels of one common lighting-severity scale.

### 2.4 Metrics and statistical analysis

The primary metric is fixed-index PCK@0.05, normalized by the maximum dimension of each original GT bounding box. Only labeled GT keypoints with COCO visibility \(v>0\) contribute to the denominator.

Every photometric condition is paired with the same GT objects and clean GT keypoints. Condition-specific predictions are compared against frozen clean E03 predictions using composite `(image_id, annotation_id)` identity.

Reported uncertainty uses 2,000 bootstrap resamples of the original source photographs. Individual objects and keypoints in the same photograph are not treated as independent sampling units.

Constrained order-invariant PCK and predefined structural edge accuracy are secondary measures. The revised constrained symmetry definition remains a post-hoc sensitivity metric pending independent physical-semantic equivalence validation.

The transformations are deterministic and therefore each condition is evaluated once; repeated evaluation of identical transformed pixels would not provide new independent perturbation evidence.

## 3. Reproducibility Checks

The neutral validation evaluations reproduced **5,196/6,720 = 77.3214%** fixed-index PCK.

The neutral test evaluations reproduced **4,325/7,049 = 61.3562%** fixed-index PCK.

Coordinate-level comparisons against existing clean E03 predictions passed the configured 0.25-pixel compatibility tolerance.

All ten validation conditions and all ten test conditions completed, including duplicate neutral anchors across the two photometric axes.

The test protocol was locked using SHA-256 fingerprints of the generator, evaluator, summarizer, validation results, clipping audit, and validation/test transformation manifests before frozen test inference.

## 4. Primary Results

### 4.1 Exposure robustness

| Exposure | Validation PCK | Validation change | Test PCK | Test change |
|---|---:|---:|---:|---:|
| −2 EV | 67.87% | −9.45 pp | 57.61% | −3.75 pp |
| −1 EV | 74.51% | −2.81 pp | 59.90% | −1.46 pp |
| 0 EV | 77.32% | Baseline | 61.36% | Baseline |
| +1 EV | 75.58% | −1.74 pp | 62.53% | +1.18 pp |
| +2 EV | 68.71% | −8.62 pp | 60.85% | −0.51 pp |

Validation exhibits greater degradation under ±2 EV than under moderate adjustments.

On test, the largest observed exposure reduction is −3.75 pp at −2 EV. All nonneutral overall test exposure changes have source-image-cluster bootstrap 95% confidence intervals that include zero.

Consequently, the frozen test does not establish a consistently directed average effect across resampled source images, despite condition-specific descriptive changes.

### 4.2 Contrast robustness

| Contrast | Validation PCK | Validation change | Test PCK | Test change |
|---|---:|---:|---:|---:|
| 0.65× | 74.73% | −2.59 pp | 63.51% | +2.16 pp |
| 0.80× | 76.98% | −0.34 pp | 61.99% | +0.64 pp |
| 1.00× | 77.32% | Baseline | 61.36% | Baseline |
| 1.20× | 60.58% | −16.74 pp | 58.87% | −2.48 pp |
| 1.40× | 50.13% | **−27.19 pp** | 56.38% | **−4.98 pp** |

Contrast enhancement is associated with pronounced validation degradation. The effect is substantially smaller on the frozen test partition.

At 1.40× contrast, the test paired change has a source-image-cluster bootstrap 95% interval of approximately **−12.66 to +1.13 pp**, which includes zero.

The discrepancy between validation and test precludes a claim that increased contrast produces a similarly large accuracy reduction across both partitions.

## 5. Photometric Clipping Audit

A read-only audit quantified channel clipping, newly black or white pixels, and the proportion of transformed GT-box pixel evaluations affected by clipping.

| Condition | Validation GT-box clipping | Test GT-box clipping |
|---|---:|---:|
| −2 EV | 0.00% | 0.00% |
| −1 EV | 0.00% | 0.00% |
| +1 EV | 11.21% | 14.86% |
| +2 EV | 22.19% | 31.12% |
| 0.65× contrast | 0.00% | 0.00% |
| 0.80× contrast | 0.00% | 0.00% |
| 1.20× contrast | 40.05% | 40.31% |
| 1.40× contrast | 51.33% | 49.01% |

The GT-box clipping fraction is box-pixel-weighted. Pixels in overlapping GT boxes can be counted multiple times; it does not represent a segmented steel-foreground area fraction.

At 1.40× contrast, validation and test have similar overall GT-box clipping fractions despite markedly different PCK changes. Thus, aggregate clipping fraction alone does not explain the sensitivity difference.

Contrast enhancement creates substantial black-region saturation, particularly at 1.20× and 1.40×. These settings should be characterized as **contrast-and-saturation stress tests**, not isolated contrast changes in the absence of information loss.

Negative exposure produces no out-of-range linear-light channels, although 8-bit quantization can introduce newly black pixels.

## 6. Source-Image Attribution

To determine whether high-contrast degradation was distributed across scenes or concentrated in individual photographs, the paired PCK difference was decomposed by original source image.

For a source photograph, its signed contribution to overall PCK change is calculated as:

\[
100\frac{C_{\mathrm{transformed}}-C_{\mathrm{clean}}}
{N_{\mathrm{valid,\ split}}}
\]

where \(C\) denotes the number of correctly localized keypoints in that photograph and the denominator is the total number of valid GT keypoints for the entire split.

Summing contributions over all annotated source images exactly reproduces the corresponding global PCK change.

### 6.1 Validation at 1.40× contrast

Five source images dominate the net change:

| Image | Dominant profile | Contribution |
|---|---|---:|
| `000297.jpg` | HI-Beam | −7.69 pp |
| `000300.jpg` | HI-Beam | −6.19 pp |
| `000642.jpg` | Channel | −5.13 pp |
| `000044.jpg` | Channel | −3.88 pp |
| `000292.jpg` | Channel | −3.85 pp |

Together they contribute approximately **−26.76 pp**, or **98.4% of the −27.19 pp net validation change**.

These five photographs contain 895 GT objects and 4,264 valid keypoints. Their GT-box clipping fractions range from approximately 73.4% to 90.4%.

The concentration is partly a property of micro-averaging: photographs containing many annotated keypoints have correspondingly greater influence on overall PCK.

### 6.2 Test at 1.40× contrast

Two source images dominate the negative net change:

| Image | Dominant profile | Contribution |
|---|---|---:|
| `000075.jpg` | HI-Beam | −3.63 pp |
| `000477.jpg` | Channel | −1.08 pp |

Their combined contribution is approximately **−4.71 pp**, or **94.6% of the −4.98 pp net test change**.

Other test scenes exhibit smaller negative changes, no changes, or improvements. Their contributions partially cancel.

### 6.3 Interpretation

The attribution audit demonstrates that measured photometric robustness is heterogeneous across source photographs. A small number of heavily annotated scenes account for most of the net high-contrast decline in both splits.

This is an exact accounting decomposition, not a causal attribution model. The most influential photographs were identified after observing the results; selecting or excluding them post hoc must not replace the prespecified whole-split evaluation.

Aggregate clipping is not sufficient to determine the direction or magnitude of localization changes. The affected structural cues, original image conditions, model prediction behavior, object scale and scene composition remain plausible but unisolated contributors.

## 7. Profile-Specific Observations

At +2 EV on test, Angle Bar decreases by approximately **12.42 pp**, with a reported image-cluster interval entirely below zero.

At 1.40× contrast on test, T-beam decreases by approximately **3.28 pp**, with a corresponding interval from roughly **−5.57 to −1.02 pp**.

Conversely, other categories and photometric settings show small changes or improvements.

The test Channel distribution is restricted to two photographs and the Sheet Pile distribution to one. Their category-level changes cannot be generalized as independently replicated scene-level findings.

Differences in profile accuracy may be confounded by scene characteristics, the underlying clean accuracy, and the concentration of annotated objects in particular images.

## 8. Limitations

1. **Synthetic rather than measured lighting:** The transformations model exposure and contrast changes but do not directly represent physically measured illumination conditions.
2. **Image clipping:** Extreme exposure and contrast can irreversibly remove tonal detail, so the observed responses reflect both photometric transformation and possible information saturation.
3. **GT-box conditioning:** The experiment evaluates skeletal localization given target bounding boxes, not detection, missed instances, counting or tracking.
4. **Uneven source-image support:** Channel and Sheet Pile contain very few independently sampled test scenes.
5. **Concentrated micro-average influence:** Dense photographs contribute disproportionately to the number of scored keypoints.
6. **Bootstrap uncertainty:** Source-image resampling accounts for within-image dependence but cannot establish generalization to unseen logistics facilities.
7. **Interpretation of secondary metrics:** Revised constrained order-invariant PCK is retained as a sensitivity measure, not the official RO2 fixed-index metric.
8. **Scope of lighting conditions:** Nonuniform illumination, directional glare and localized shadows were not evaluated in R2-A or R2-B.

## 9. Conclusion

The frozen ViTPose-B E03 model exhibits measurable but strongly scene-dependent sensitivity to global photometric transformations.

On validation, the strongest responses occur under enhanced contrast, particularly at 1.40×, where fixed-index PCK decreases by 27.19 pp. However, the analogous frozen test change is only −4.98 pp, with a source-image-cluster interval spanning zero.

Photometric audits establish that the high-contrast settings introduce extensive pixel saturation on both splits. Source-image attribution further shows that most of the net degradation arises from a small number of heavily annotated HI-Beam and Channel photographs.

Accordingly, R2 does not support a universal claim that all lighting changes substantially reduce skeletal keypoint localization accuracy. Instead, it demonstrates that robustness depends on the transformation, scene composition, and the visual information retained after photometric processing.

**R2-A (global exposure) and R2-B (global contrast) are complete.** Their scripts, protocol lock, predictions, paired summaries, clipping audits and attribution evidence should be preserved as a distinct versioned research release. The next planned RO3 investigation concerns viewpoint and orientation robustness.

---

**Primary metric:** Fixed-index PCK@0.05  
**Experimental status:** Completed  
**Evidence status:** Awaiting final archive and checksum verification  
**No model retraining or checkpoint selection was performed during R2.**