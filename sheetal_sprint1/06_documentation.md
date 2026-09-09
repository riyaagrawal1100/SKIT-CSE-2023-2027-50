# Sheetal — Data Cleaning & Validation

## My contribution to this project

This folder covers the cleaning, validation, and exploratory-analysis side of
the fraud detection dataset (`fraud_detection_credit_card_small.csv`,
100,000 rows).

### Files

| File | Task | Description |
|---|---|---|
| `01_data_cleaning.py` | Data cleaning | Drops the leftover index column, removes duplicate rows/IDs, fixes data types (dates, cc_num, zip), zero-pads zip codes, trims whitespace |
| `02_missing_value_handling.py` | Missing value handling | Finds all missing values (`merch_zipcode`, ~15% of rows) and imputes them using nearest-neighbor matching on merchant lat/long |
| `03_data_validation.py` | Data validation | Sanity-checks value ranges (amount, age, score, lat/long, fraud flag) and flags inconsistent "enrichment" columns |
| `04_behavioral_analysis.py` | Behavioral analysis | Explores fraud-rate patterns by category, gender, payment method, transaction type, and hour of day |
| `05_visualizations.py` | Visualizations | Generates 4 charts: class distribution, amount distribution, fraud rate by category, fraud rate by hour |
| `06_documentation.md` | Documentation | This file |

### How to run (in order)

```bash
cd sheetal
python 01_data_cleaning.py
python 02_missing_value_handling.py
python 03_data_validation.py
python 04_behavioral_analysis.py
python 05_visualizations.py
```

### Key findings

- Original dataset had 0 duplicate rows, but ~15% missing `merch_zipcode`
  values and integer-stored zip codes that had lost their leading zeros.
- After cleaning + imputation, the dataset has **0 missing values** across
  all 100,000 rows.
- `Transaction_Time`, `Customer_Age`, and `Merchant_Category` do not reliably
  match the real transaction timestamp / dob-derived age / true category —
  these should be treated with caution in downstream analysis.
- Fraud rate overall is ~0.6%, and varies meaningfully by category, payment
  method, and hour of day (see `behavioral_analysis_report.md`).

### Output

The final validated file (`sheetal_03_validated.csv`) is handed off to
Riya's feature-engineering pipeline as the starting point.
