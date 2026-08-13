# InnoCount Keypoint Schema v2 — FROZEN

The semantic definitions have been confirmed.

Run:

```bash
conda activate innocount-freeze

python validate_frozen_v2_schema.py     --dataset "InnoCount_KeyPoint_Frozen_v2"
```

Expected:

```text
PASS: Frozen v2 structural schema matches the final semantic contract.
```

Files:
- `keypoint_schema_v2.json` — machine-readable final schema
- `keypoint_schema_v2.txt` — human-readable final specification
- `validate_frozen_v2_schema.py` — structural validator

The semantic reference frame is class-local steel-profile geometry, not image
orientation.

Angle Bar K3 remains `new-point-3` by provenance.
