# Riya — Feature Engineering

## My contribution to this project

This folder covers the feature-engineering side of the fraud detection
dataset, starting from Sheetal's validated/cleaned output
(`sheetal_03_validated.csv`) and turning it into a final, model-ready dataset.

### Files

| File | Task | Description |
|---|---|---|
| `01_data_processing.py` | Data processing | Drops unreliable columns (flagged during validation) and pure-identifier/free-text/high-cardinality columns not usable for modeling |
| `02_feature_engineering.py` | Feature engineering | Master script — combines temporal, geospatial, and customer-behavior features, then encodes categoricals into the final dataset |
| `03_temporal_features.py` | Temporal features | Real `customer_age` (from dob), `trans_hour`, `trans_day_of_week`, `trans_month`, `is_weekend` |
| `04_geospatial_features.py` | Geospatial features | Haversine distance (km) between customer's home location and merchant's location |
| `05_customer_behavior_features.py` | Customer behavior features | Per-customer transaction count, average spend, and deviation of the current transaction from that customer's average spend |
| `06_documentation.md` | Documentation | This file |

### How to run (in order)

```bash
cd riya
python 01_data_processing.py
python 03_temporal_features.py
python 04_geospatial_features.py
python 05_customer_behavior_features.py
python 02_feature_engineering.py   # master script, produces the final dataset
```

`02_feature_engineering.py` internally re-runs the same logic as steps 3-5 in
sequence and additionally handles categorical encoding, so it can also be run
directly right after `01_data_processing.py`.

### Features created

- **Temporal:** `customer_age`, `trans_hour`, `trans_day_of_week`, `trans_month`, `is_weekend`
- **Geospatial:** `customer_merchant_distance_km`
- **Customer behavior:** `customer_txn_count`, `customer_avg_amt`, `amt_vs_customer_avg`
- **Encoded categoricals:** `gender` (binary), one-hot encoded `category`, `state`, `Transaction_Type`, `Payment_Method`

### Output

`final_preprocessed_dataset.csv` — 100,000 rows x 84 columns, fully numeric,
0 missing values, `is_fraud` as the last column.

### Note on scaling / class imbalance

Feature scaling (StandardScaler) and class-imbalance handling (SMOTE /
class weights) are **not** applied in this dataset on purpose — both should
be fit only on the training split after `train_test_split(..., stratify=is_fraud)`,
to avoid data leakage into the test set.
