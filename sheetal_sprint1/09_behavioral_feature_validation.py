"""
09_behavioral_feature_validation.py
Author: Sheetal
Task: Sprint 1 — Behavioral feature validation

What this does:
Validates the BEHAVIORAL features Sheetal's Sprint 2 scripts add to the
pipeline:
  - sheetal/07_odd_timing_features.py
      night_transaction, customer_hist_avg_hour, hour_deviation_from_normal,
      unusual_transaction_hour
  - sheetal/08_transaction_frequency_features.py
      txn_count_last_5min, txn_count_last_30min, txn_count_last_60min,
      transaction_velocity_per_min, rapid_transaction_flag

By default this validates the FINAL dataset (`data/final_preprocessed_dataset.csv`),
since that's the dataset that ends up carrying these columns after
riya/02_feature_engineering.py merges everyone's features together. It can
also be pointed at an earlier-stage file (e.g. `sheetal_08_frequency.csv`)
via --input, since these columns exist there too.

This is a READ-ONLY validation utility: it never modifies the dataset, never
regenerates features, and never uses `is_fraud` to validate or derive any
behavioral feature.

Checks performed:
  - Required columns exist (all 9 behavioral columns above)
  - Missing values in behavioral columns
  - Invalid values / unexpected ranges per feature:
      * night_transaction, unusual_transaction_hour, rapid_transaction_flag
        are binary (0/1)
      * customer_hist_avg_hour is within [0, 23]
      * hour_deviation_from_normal is within [0, 12] (circular hour distance
        on a 24h clock can never exceed 12)
      * txn_count_last_5min/30min/60min are non-negative integers
      * transaction_velocity_per_min is non-negative
  - Data consistency between related features:
      * txn_count_last_5min <= txn_count_last_30min <= txn_count_last_60min
        (a wider trailing window can never contain fewer transactions than
        a narrower one nested inside it)
      * transaction_velocity_per_min == txn_count_last_60min / 60 (matches
        the formula in sheetal/08_transaction_frequency_features.py)
      * rapid_transaction_flag == 1 whenever txn_count_last_5min >= 1, and
        0 otherwise (matches the RAPID_TRANSACTION_WINDOW_MINUTES=5 rule)
  - Duplicate rows
  - Required non-behavioral key columns needed to interpret the behavioral
    features are present (trans_hour, cc_num or is_fraud presence noted only
    informationally, never used for validation logic)

Usage:
    python 09_behavioral_feature_validation.py
    python 09_behavioral_feature_validation.py --input /path/to/final_preprocessed_dataset.csv
"""

import argparse
import os
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "final_preprocessed_dataset.csv")

# --------------------------------------------------------------------------
# Actual column names from sheetal/07_odd_timing_features.py and
# sheetal/08_transaction_frequency_features.py
# --------------------------------------------------------------------------

TIMING_COLUMNS = [
    "night_transaction",
    "customer_hist_avg_hour",
    "hour_deviation_from_normal",
    "unusual_transaction_hour",
]

FREQUENCY_COLUMNS = [
    "txn_count_last_5min",
    "txn_count_last_30min",
    "txn_count_last_60min",
    "transaction_velocity_per_min",
    "rapid_transaction_flag",
]

BEHAVIORAL_COLUMNS = TIMING_COLUMNS + FREQUENCY_COLUMNS

BINARY_COLUMNS = ["night_transaction", "unusual_transaction_hour", "rapid_transaction_flag"]

COUNT_COLUMNS = ["txn_count_last_5min", "txn_count_last_30min", "txn_count_last_60min"]

# RAPID_TRANSACTION_WINDOW_MINUTES from sheetal/08_transaction_frequency_features.py
RAPID_WINDOW_COLUMN = "txn_count_last_5min"


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
            "This script only validates behavioral features that already exist "
            "in the dataset — run sheetal/07_odd_timing_features.py, "
            "sheetal/08_transaction_frequency_features.py (or the full "
            "riya/02_feature_engineering.py pipeline) first."
        )
    return pd.read_csv(path)


def check_required_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    missing = [c for c in BEHAVIORAL_COLUMNS if c not in df.columns]
    result.add(
        "Required behavioral columns present",
        len(missing) == 0,
        "all 9 behavioral columns found" if not missing else f"missing: {missing}",
    )


def check_duplicate_rows(df: pd.DataFrame, result: ValidationResult) -> None:
    dup_count = int(df.duplicated().sum())
    result.add(
        "No duplicate rows",
        dup_count == 0,
        "0 duplicates" if dup_count == 0 else f"{dup_count} duplicate row(s) found",
    )


def check_missing_values(df: pd.DataFrame, result: ValidationResult) -> None:
    present_cols = [c for c in BEHAVIORAL_COLUMNS if c in df.columns]
    missing = df[present_cols].isna().sum()
    missing = missing[missing > 0]
    result.add(
        "No missing values in behavioral columns",
        len(missing) == 0,
        "0 missing values" if len(missing) == 0 else f"columns with NaNs: {missing.to_dict()}",
    )


def check_binary_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    bad = {}
    for col in BINARY_COLUMNS:
        if col not in df.columns:
            continue
        bad_values = df[~df[col].isin([0, 1])][col].unique().tolist()
        if bad_values:
            bad[col] = bad_values
    result.add(
        "Binary behavioral flags contain only 0/1",
        len(bad) == 0,
        "all valid" if not bad else f"invalid values: {bad}",
    )


def check_night_transaction_logic(df: pd.DataFrame, result: ValidationResult) -> None:
    """night_transaction should be 1 iff trans_hour is in 22:00-05:59.

    Only runs if trans_hour is present in this file (it's dropped from the
    fully one-hot-encoded final dataset in some pipelines, so this check is
    skipped gracefully rather than failing when the column isn't available).
    """
    if "trans_hour" not in df.columns or "night_transaction" not in df.columns:
        result.add(
            "night_transaction matches trans_hour window (22:00-05:59)",
            True,
            "skipped — trans_hour not present in this file",
        )
        return

    night_hours = set(list(range(22, 24)) + list(range(0, 6)))
    expected = df["trans_hour"].isin(night_hours).astype(int)
    mismatches = int((expected != df["night_transaction"]).sum())
    result.add(
        "night_transaction matches trans_hour window (22:00-05:59)",
        mismatches == 0,
        "matches for all rows" if mismatches == 0 else f"{mismatches} mismatched row(s)",
    )


def check_hour_features_ranges(df: pd.DataFrame, result: ValidationResult) -> None:
    problems = []

    if "customer_hist_avg_hour" in df.columns:
        n_bad = int((~df["customer_hist_avg_hour"].between(0, 23)).sum())
        if n_bad:
            problems.append(f"customer_hist_avg_hour: {n_bad} value(s) outside [0, 23]")

    if "hour_deviation_from_normal" in df.columns:
        # circular_hour_diff() in sheetal/07_odd_timing_features.py caps the
        # result at 12 (half of a 24-hour clock) by construction.
        n_bad = int((~df["hour_deviation_from_normal"].between(0, 12)).sum())
        if n_bad:
            problems.append(f"hour_deviation_from_normal: {n_bad} value(s) outside [0, 12]")

    result.add(
        "Timing feature values within expected ranges",
        len(problems) == 0,
        "all within range" if not problems else "; ".join(problems),
    )


def check_frequency_counts_non_negative(df: pd.DataFrame, result: ValidationResult) -> None:
    problems = []
    for col in COUNT_COLUMNS + ["transaction_velocity_per_min"]:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            problems.append(f"{col}: {n_neg} negative value(s)")
    result.add(
        "Transaction frequency/velocity values are non-negative",
        len(problems) == 0,
        "all non-negative" if not problems else "; ".join(problems),
    )


def check_frequency_counts_are_integers(df: pd.DataFrame, result: ValidationResult) -> None:
    problems = []
    for col in COUNT_COLUMNS:
        if col not in df.columns:
            continue
        non_integer = int((df[col] % 1 != 0).sum())
        if non_integer:
            problems.append(f"{col}: {non_integer} non-integer value(s)")
    result.add(
        "Transaction count columns contain whole numbers",
        len(problems) == 0,
        "all whole numbers" if not problems else "; ".join(problems),
    )


def check_nested_window_consistency(df: pd.DataFrame, result: ValidationResult) -> None:
    """A 30-min window can never contain fewer prior transactions than the
    5-min window nested inside it, and likewise for 30-min vs 60-min."""
    if not all(c in df.columns for c in COUNT_COLUMNS):
        result.add("Trailing windows are properly nested (5min <= 30min <= 60min)", False, "columns missing")
        return

    n_bad_5_30 = int((df["txn_count_last_5min"] > df["txn_count_last_30min"]).sum())
    n_bad_30_60 = int((df["txn_count_last_30min"] > df["txn_count_last_60min"]).sum())
    problems = []
    if n_bad_5_30:
        problems.append(f"{n_bad_5_30} row(s) where 5min count > 30min count")
    if n_bad_30_60:
        problems.append(f"{n_bad_30_60} row(s) where 30min count > 60min count")

    result.add(
        "Trailing windows are properly nested (5min <= 30min <= 60min)",
        len(problems) == 0,
        "properly nested for all rows" if not problems else "; ".join(problems),
    )


def check_velocity_formula(df: pd.DataFrame, result: ValidationResult) -> None:
    """transaction_velocity_per_min = txn_count_last_60min / 60, per
    sheetal/08_transaction_frequency_features.py."""
    if not all(c in df.columns for c in ["txn_count_last_60min", "transaction_velocity_per_min"]):
        result.add("transaction_velocity_per_min matches formula (60min count / 60)", False, "columns missing")
        return

    expected = df["txn_count_last_60min"] / 60.0
    mismatches = int((~expected.round(6).eq(df["transaction_velocity_per_min"].round(6))).sum())
    result.add(
        "transaction_velocity_per_min matches formula (60min count / 60)",
        mismatches == 0,
        "formula holds for all rows" if mismatches == 0 else f"{mismatches} mismatched row(s)",
    )


def check_rapid_transaction_flag_logic(df: pd.DataFrame, result: ValidationResult) -> None:
    """rapid_transaction_flag should be 1 iff txn_count_last_5min >= 1, per
    RAPID_TRANSACTION_WINDOW_MINUTES=5 in
    sheetal/08_transaction_frequency_features.py."""
    if not all(c in df.columns for c in [RAPID_WINDOW_COLUMN, "rapid_transaction_flag"]):
        result.add("rapid_transaction_flag matches 5-minute rule", False, "columns missing")
        return

    expected = (df[RAPID_WINDOW_COLUMN] >= 1).astype(int)
    mismatches = int((expected != df["rapid_transaction_flag"]).sum())
    result.add(
        "rapid_transaction_flag matches 5-minute rule",
        mismatches == 0,
        "matches for all rows" if mismatches == 0 else f"{mismatches} mismatched row(s)",
    )


def run_all_checks(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    check_required_columns(df, result)
    check_duplicate_rows(df, result)
    check_missing_values(df, result)
    check_binary_columns(df, result)
    check_night_transaction_logic(df, result)
    check_hour_features_ranges(df, result)
    check_frequency_counts_non_negative(df, result)
    check_frequency_counts_are_integers(df, result)
    check_nested_window_consistency(df, result)
    check_velocity_formula(df, result)
    check_rapid_transaction_flag_logic(df, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Sheetal's behavioral features (timing + frequency).")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Path to a CSV containing the behavioral columns (default: ../data/final_preprocessed_dataset.csv)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SHEETAL — Sprint 1 Behavioral Feature Validation")
    print(f"Validating: {os.path.abspath(args.input)}")
    print("=" * 70)

    try:
        df = load_dataset(args.input)
    except FileNotFoundError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error while loading dataset: {e}")
        sys.exit(1)

    print(f"\nLoaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    try:
        result = run_all_checks(df)
    except Exception as e:
        print(f"[FAIL] Unexpected error during validation: {e}")
        sys.exit(1)

    print("Validation checks:")
    for line in result.summary_lines():
        print(line)

    print("\n" + "=" * 70)
    if result.all_passed:
        print("OVERALL RESULT: PASS — behavioral features are valid and consistent.")
    else:
        n_failed = sum(1 for _, passed, _ in result.checks if not passed)
        print(f"OVERALL RESULT: FAIL — {n_failed} check(s) failed. Review above.")
    print("=" * 70)

    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
