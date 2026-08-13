# InnoCount KeyPoint Frozen-v1

This directory is the frozen research dataset produced from the original
Roboflow COCO ZIP.

## Final classes

| ID | Class |
|---:|---|
| 0 | Channel |
| 1 | T-beam |
| 2 | HI-Beam |
| 3 | Angle Bar |
| 4 | Sheet Pile |

## Rules

- Roboflow train/valid/test are treated as one master pool.
- Exact duplicate JPEGs are detected by MD5.
- Exact duplicates are canonicalized before splitting.
- Source-linked images are grouped using `image.extra.name`.
- 26 genuine negative/background images are retained.
- No new augmentation is added.
- The final target is approximately 80/10/10.
- The split is source-aware.
- Validation/test are not augmented.
- The master manifest is the lockfile for the frozen split.

## Reproduction

Run:

```bash
python create_frozen_dataset.py     --input "InnoCount KeyPoint.v3i.coco.zip"     --output "InnoCount_KeyPoint_Frozen_v1"
```

For a previously frozen dataset, `reports/master_manifest.csv` should be
kept permanently as the assignment lockfile.
