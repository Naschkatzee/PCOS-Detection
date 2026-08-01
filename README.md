# PCOS Foundation Model Project — Step 1: Dataset Audit

## Expected dataset layout

```text
data/raw/ovarian_ultrasound/
├── Dominant_Follicle/
├── Normal/
└── PCO/
```

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the audit

```bash
python audit_dataset.py \
  --data data/raw/ovarian_ultrasound \
  --output reports/dataset_audit
```

Generated files:

- `summary.json`
- `image_inventory.csv`
- `resolution_counts.csv`
- `exact_duplicates.json`
- `corrupted_files.txt`

Do not create train/validation/test splits until duplicates have been reviewed.
