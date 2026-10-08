"""
09_feature_validation.py
Author: Riya
Task: Sprint 1 — Final feature validation

What this does:
Validates `data/final_preprocessed_dataset.csv` — the model-ready dataset
produced by 02_feature_engineering.py — before it's handed off for Sprint 2
modeling work. This is a READ-ONLY validation utility: it never modifies
`final_preprocessed_dataset.csv`, never regenerates features, and never uses
`is_fraud` to validate or derive any feature (it only confirms the column
exists and is well-formed, exactly like every other column).

Checks performed:
  1. Required columns exist (all engineered feature columns Riya's and
     Sheetal's Sprint 2 scripts are expected to have produced)
  2. `is_fraud` target column exists
  3. Row count matches the known/expected row count of the dataset
  4. Duplicate rows
  5. Missing values (per column)
  6. Data types (numeric feature columns are actually numeric)
  7. Numeric columns contain valid (finite, non-NaN) values
  8. Location-change features (07_location_change_features.py) have valid
     values (distances >= 0, speeds >= 0, flags in {0,1}, etc.)
  9. Transaction-amount anomaly features (08_transaction_amount_features.py)
     have valid values (ratios >= 0, flags in {0,1}, etc.)
 10. No unexpected negative/invalid values in columns that should never be
     negative (amounts, distances, counts, ages, etc.)
 11. Overall pass/fail verdict on whether the dataset is suitable to hand
     off to Sprint 2

Usage:
    python 09_feature_validation.py
    python 09_feature_validation.py --input /path/to/final_preprocessed_dataset.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "final_preprocessed_dataset.csv")

# Row count the dataset is expected to have (raw_fraud_data.csv has 100,000
# rows and no row-dropping step exists anywhere in the pipeline after
# sheetal/01_data_cleaning.py's de-dup, which is a no-op on this data).
EXPECTED_ROW_COUNT = 100_000

# --------------------------------------------------------------------------
# Column groups, using the ACTUAL column names produced by the pipeline
# --------------------------------------------------------------------------

TARGET_COL = "is_fraud"

# Columns carried through from riya/01_data_processing.py / raw data,
# still present (not one-hot-encoded, not dropped) in the final dataset.
BASE_REQUIRED_COLUMNS = [
    "amt",
    "gender",
    "lat",
    "long",
    "city_pop",
    "merch_lat",
    "merch_long",
    "Customer_Satisfaction_Score",
    "Loyalty_Points_Earned",
]

# riya/03_temporal_features.py
TEMPORAL_COLUMNS = [
    "customer_age",
    "trans_hour",
    "trans_day_of_week",
    "trans_month",
    "is_weekend",
]

# riya/04_geospatial_features.py
GEO_COLUMNS = ["customer_merchant_distance_km"]

# riya/05_customer_behavior_features.py
CUSTOMER_BEHAVIOR_COLUMNS = [
    "customer_txn_count",
    "customer_avg_amt",
    "amt_vs_customer_avg",
]

# riya/07_location_change_features.py ("impossible travel" features)
LOCATION_CHANGE_COLUMNS = [
    "prev_trans_distance_km",
    "hours_since_prev_trans",
    "location_changed",
    "implied_travel_speed_kmh",
    "rapid_location_change",
]

# riya/08_transaction_amount_features.py ("suddenly large amount" features)
AMOUNT_ANOMALY_COLUMNS = [
    "customer_hist_avg_amt",
    "customer_hist_std_amt",
    "amount_ratio_to_normal",
    "customer_relative_amount_deviation",
    "large_transaction_flag",
]

# sheetal/07_odd_timing_features.py + sheetal/08_transaction_frequency_features.py
# (produced by Sheetal's scripts but folded into the same final dataset by
# 02_feature_engineering.py, so their presence is validated here too since
# this file validates the FINAL dataset as a whole)
BEHAVIORAL_COLUMNS = [
    "night_transaction",
    "customer_hist_avg_hour",
    "hour_deviation_from_normal",
    "unusual_transaction_hour",
    "txn_count_last_5min",
    "txn_count_last_30min",
    "txn_count_last_60min",
    "transaction_velocity_per_min",
    "rapid_transaction_flag",
]

REQUIRED_COLUMNS = (
    BASE_REQUIRED_COLUMNS
    + TEMPORAL_COLUMNS
    + GEO_COLUMNS
    + CUSTOMER_BEHAVIOR_COLUMNS
    + LOCATION_CHANGE_COLUMNS
    + AMOUNT_ANOMALY_COLUMNS
    + BEHAVIORAL_COLUMNS
    + [TARGET_COL]
)

# Binary flag columns: must contain only {0, 1}
BINARY_FLAG_COLUMNS = [
    "gender",
    "is_weekend",
    "location_changed",
    "rapid_location_change",
    "large_transaction_flag",
    "night_transaction",
    "unusual_transaction_hour",
    "rapid_transaction_flag",
    TARGET_COL,
]

# Columns that must never be negative
NON_NEGATIVE_COLUMNS = [
    "amt",
    "city_pop",
    "Loyalty_Points_Earned",
    "customer_age",
    "customer_merchant_distance_km",
    "customer_txn_count",
    "customer_avg_amt",
    "prev_trans_distance_km",
    "hours_since_prev_trans",
    "implied_travel_speed_kmh",
    "customer_hist_avg_amt",
    "customer_hist_std_amt",
    "amount_ratio_to_normal",
    "customer_hist_avg_hour",
    "hour_deviation_from_normal",
    "txn_count_last_5min",
    "txn_count_last_30min",
    "txn_count_last_60min",
    "transaction_velocity_per_min",
]

# All-numeric columns expected in the final dataset (everything except the
# one-hot-encoded boolean dummy columns from category/state/Transaction_Type/
# Payment_Method, which are checked separately as boolean/0-1 columns).
NUMERIC_COLUMNS = (
    BASE_REQUIRED_COLUMNS
    + TEMPORAL_COLUMNS
    + GEO_COLUMNS
    + CUSTOMER_BEHAVIOR_COLUMNS
    + LOCATION_CHANGE_COLUMNS
    + AMOUNT_ANOMALY_COLUMNS
    + BEHAVIORAL_COLUMNS
    + [TARGET_COL]
)

# Reasonable bounds for sanity-checking specific columns
RANGE_CHECKS = {
    "lat": (-90, 90),
    "merch_lat": (-90, 90),
    "long": (-180, 180),
    "merch_long": (-180, 180),
    "Customer_Satisfaction_Score": (1, 10),
    "trans_hour": (0, 23),
    "trans_day_of_week": (0, 6),
    "trans_month": (1, 12),
    "customer_age": (0, 120),
}


class ValidationResult:
    """Collects pass/fail check results and a human-readable log."""

    def __init__(self):
        self.checks = []  # list of (name, passed: bool, detail: str)

    def add(self, name, passed, detail=""):
        self.checks.append((name, passed, detail))

    @property
    def all_passed(self):
        return all(passed for _, passed, _ in self.checks)

    def summary_lines(self):
        lines = []
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            line = f"  [{status}] {name}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        return lines


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}\n"
            "Run riya/02_feature_engineering.py first to generate "
            "final_preprocessed_dataset.csv (this script only validates it, "
            "it never generates it)."
        )
    return pd.read_csv(path)


def check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    result.add(
        "Required columns present",
        len(missing) == 0,
        "all expected columns found" if not missing else f"missing: {missing}",
    )


def check_target_column(df: pd.DataFrame, result: ValidationResult) -> None:
    present = TARGET_COL in df.columns
    detail = ""
    if present:
        bad_values = df[~df[TARGET_COL].isin([0, 1])][TARGET_COL].unique().tolist()
        detail = (
            "contains only 0/1"
            if not bad_values
            else f"unexpected values found: {bad_values}"
        )
        present = present and not bad_values
    result.add("'is_fraud' target column exists and is binary", present, detail)


def check_row_count(df: pd.DataFrame, result: ValidationResult) -> None:
    actual = len(df)
    result.add(
        "Row count matches expected dataset size",
        actual == EXPECTED_ROW_COUNT,
        f"expected {EXPECTED_ROW_COUNT:,}, found {actual:,}",
    )


def check_duplicate_rows(df: pd.DataFrame, result: ValidationResult) -> None:
    dup_count = int(df.duplicated().sum())
    result.add(
        "No duplicate rows",
        dup_count == 0,
        f"{dup_count} duplicate row(s) found" if dup_count else "0 duplicates",
    )


def check_missing_values(df: pd.DataFrame, result: ValidationResult) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0]
    result.add(
        "No missing values",
        len(missing) == 0,
        "0 missing values" if len(missing) == 0 else f"columns with NaNs: {missing.to_dict()}",
    )


def check_data_types(df: pd.DataFrame, result: ValidationResult) -> None:
    bad_types = []
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            bad_types.append((col, str(df[col].dtype)))
    result.add(
        "Numeric feature columns have numeric dtypes",
        len(bad_types) == 0,
        "all numeric columns correctly typed" if not bad_types else f"non-numeric: {bad_types}",
    )


def check_numeric_columns_valid(df: pd.DataFrame, result: ValidationResult) -> None:
    """Numeric columns must be finite (no inf/-inf) and non-NaN."""
    bad_cols = {}
    for col in NUMERIC_COLUMNS:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_inf = int(np.isinf(df[col]).sum())
        n_nan = int(df[col].isna().sum())
        if n_inf or n_nan:
            bad_cols[col] = {"inf": n_inf, "nan": n_nan}
    result.add(
        "Numeric columns contain only finite, valid values",
        len(bad_cols) == 0,
        "no inf/NaN found" if not bad_cols else f"invalid values: {bad_cols}",
    )


def check_range_bounds(df: pd.DataFrame, result: ValidationResult) -> None:
    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        n_bad = int((~df[col].between(lo, hi)).sum())
        if n_bad:
            violations[col] = f"{n_bad} value(s) outside [{lo}, {hi}]"
    result.add(
        "Values within expected ranges",
        len(violations) == 0,
        "all within range" if not violations else str(violations),
    )


def check_binary_flags(df: pd.DataFrame, result: ValidationResult) -> None:
    bad = {}
    for col in BINARY_FLAG_COLUMNS:
        if col not in df.columns:
            continue
        bad_values = df[~df[col].isin([0, 1])][col].unique().tolist()
        if bad_values:
            bad[col] = bad_values
    result.add(
        "Binary flag columns contain only 0/1",
        len(bad) == 0,
        "all binary flags valid" if not bad else f"invalid: {bad}",
    )


def check_non_negative_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    violations = {}
    for col in NON_NEGATIVE_COLUMNS:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            violations[col] = n_neg
    result.add(
        "No unexpected negative values",
        len(violations) == 0,
        "no negative values" if not violations else f"negative counts: {violations}",
    )


def check_location_change_features(df: pd.DataFrame, result: ValidationResult) -> None:
    """Sanity checks specific to riya/07_location_change_features.py output."""
    if not all(c in df.columns for c in LOCATION_CHANGE_COLUMNS):
        result.add("Location-change features valid", False, "one or more columns missing")
        return

    problems = []

    if (df["prev_trans_distance_km"] < 0).any():
        problems.append("prev_trans_distance_km has negative values")
    if (df["hours_since_prev_trans"] < 0).any():
        problems.append("hours_since_prev_trans has negative values")
    if (df["implied_travel_speed_kmh"] < 0).any():
        problems.append("implied_travel_speed_kmh has negative values")
    if not df["location_changed"].isin([0, 1]).all():
        problems.append("location_changed has non-binary values")
    if not df["rapid_location_change"].isin([0, 1]).all():
        problems.append("rapid_location_change has non-binary values")

    # Note: rapid_location_change (implied-speed threshold) and
    # location_changed (distance threshold) use independent thresholds by
    # design (see riya/07_location_change_features.py) — a short distance
    # covered in a very short time can still exceed the speed threshold, so
    # rapid_location_change=1 with location_changed=0 is expected and valid,
    # not a consistency violation.

    result.add(
        "Location-change features internally valid",
        len(problems) == 0,
        "all checks passed" if not problems else "; ".join(problems),
    )


def check_amount_anomaly_features(df: pd.DataFrame, result: ValidationResult) -> None:
    """Sanity checks specific to riya/08_transaction_amount_features.py output."""
    if not all(c in df.columns for c in AMOUNT_ANOMALY_COLUMNS):
        result.add("Transaction-amount anomaly features valid", False, "one or more columns missing")
        return

    problems = []

    if (df["customer_hist_avg_amt"] < 0).any():
        problems.append("customer_hist_avg_amt has negative values")
    if (df["customer_hist_std_amt"] < 0).any():
        problems.append("customer_hist_std_amt has negative values (std cannot be negative)")
    if (df["amount_ratio_to_normal"] < 0).any():
        problems.append("amount_ratio_to_normal has negative values")
    if not df["large_transaction_flag"].isin([0, 1]).all():
        problems.append("large_transaction_flag has non-binary values")

    result.add(
        "Transaction-amount anomaly features internally valid",
        len(problems) == 0,
        "all checks passed" if not problems else "; ".join(problems),
    )


def run_all_checks(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    check_required_columns(df, result)
    check_target_column(df, result)
    check_row_count(df, result)
    check_duplicate_rows(df, result)
    check_missing_values(df, result)
    check_data_types(df, result)
    check_numeric_columns_valid(df, result)
    check_range_bounds(df, result)
    check_binary_flags(df, result)
    check_non_negative_columns(df, result)
    check_location_change_features(df, result)
    check_amount_anomaly_features(df, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate the final engineered fraud-detection dataset.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to final_preprocessed_dataset.csv (default: ../data/final_preprocessed_dataset.csv)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RIYA — Sprint 1 Feature Validation")
    print(f"Validating: {os.path.abspath(args.input)}")
    print("=" * 70)

    try:
        df = load_dataset(args.input)
    except FileNotFoundError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)

    print(f"\nLoaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    result = run_all_checks(df)

    print("Validation checks:")
    for line in result.summary_lines():
        print(line)

    print("\n" + "=" * 70)
    if result.all_passed:
        print("OVERALL RESULT: PASS — dataset is suitable to hand off to Sprint 2.")
    else:
        n_failed = sum(1 for _, passed, _ in result.checks if not passed)
        print(f"OVERALL RESULT: FAIL — {n_failed} check(s) failed. Review above before Sprint 2.")
    print("=" * 70)

    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
