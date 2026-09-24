"""
Sprint 1 - Processed Dataset Summary Utility

Owner: Riya
Purpose:
    Generate a reproducible summary of the final Sprint 1 processed dataset
    before entering Sprint 2.

Expected input:
    data/final_preprocessed_dataset.csv

The script does not modify the dataset.
"""

from pathlib import Path
import sys
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "final_preprocessed_dataset.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the final processed dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{path}\n"
            "Run the Sprint 1 preprocessing pipeline first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    return df


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    print("\n" + "=" * 60)
    print("SPRINT 1 - PROCESSED DATASET SUMMARY")
    print("=" * 60)

    print(f"Rows             : {len(df):,}")
    print(f"Columns          : {len(df.columns):,}")
    print(f"Duplicate rows   : {df.duplicated().sum():,}")
    print(f"Missing values   : {int(df.isna().sum().sum()):,}")


def print_target_summary(df: pd.DataFrame) -> None:
    """Print fraud/legitimate transaction distribution."""
    print("\n" + "-" * 60)
    print("TARGET DISTRIBUTION")
    print("-" * 60)

    if "is_fraud" not in df.columns:
        print("WARNING: 'is_fraud' column was not found.")
        return

    counts = df["is_fraud"].value_counts(dropna=False).sort_index()

    for value, count in counts.items():
        percentage = (count / len(df)) * 100
        label = "Fraud" if value == 1 else "Legitimate" if value == 0 else str(value)
        print(f"{label:<15}: {count:>10,} ({percentage:6.2f}%)")


def print_missing_values(df: pd.DataFrame) -> None:
    """Print columns containing missing values."""
    print("\n" + "-" * 60)
    print("MISSING VALUE CHECK")
    print("-" * 60)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("PASS: No missing values found.")
    else:
        print(f"WARNING: {len(missing)} columns contain missing values.")
        print(missing.to_string())


def print_data_types(df: pd.DataFrame) -> None:
    """Print data-type distribution."""
    print("\n" + "-" * 60)
    print("DATA TYPES")
    print("-" * 60)

    print(df.dtypes.value_counts().to_string())


def print_feature_groups(df: pd.DataFrame) -> None:
    """Identify the new Sprint 1 behavioral feature groups by column names."""
    print("\n" + "-" * 60)
    print("BEHAVIORAL FEATURE CHECK")
    print("-" * 60)

    groups = {
        "Location Change": [
            "location_changed",
            "distance_from_previous_location_km",
            "rapid_location_change",
        ],
        "Large Amount": [
            "customer_median_amt",
            "amount_deviation",
            "amount_ratio_to_customer_median",
            "large_transaction_flag",
        ],
        "Odd Timing": [
            "night_transaction",
            "unusual_transaction_hour",
            "customer_usual_hour",
            "hour_deviation_from_customer_pattern",
        ],
        "Transaction Frequency": [
            "transactions_last_5min",
            "transactions_last_30min",
            "transactions_last_1hour",
            "rapid_transaction_flag",
        ],
    }

    for group, expected_columns in groups.items():
        found = [column for column in expected_columns if column in df.columns]
        missing = [column for column in expected_columns if column not in df.columns]

        print(f"\n{group}:")
        print(f"  Found : {len(found)}/{len(expected_columns)}")

        if found:
            print("  Columns:")
            for column in found:
                print(f"    ✓ {column}")

        if missing:
            print("  Missing:")
            for column in missing:
                print(f"    ✗ {column}")


def print_numeric_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for numeric columns."""
    print("\n" + "-" * 60)
    print("NUMERIC FEATURE SUMMARY")
    print("-" * 60)

    numeric_df = df.select_dtypes(include="number")

    print(f"Numeric columns: {len(numeric_df.columns)}")

    if numeric_df.empty:
        print("No numeric columns found.")
        return

    summary = numeric_df.describe().T[
        ["min", "max", "mean", "std"]
    ].round(3)

    # Avoid dumping hundreds of rows to the terminal.
    print(summary.to_string(max_rows=30))


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run basic Sprint 1 readiness checks."""
    print("\n" + "-" * 60)
    print("SPRINT 1 READINESS CHECK")
    print("-" * 60)

    checks = {
        "Dataset is not empty": len(df) > 0,
        "Target column exists": "is_fraud" in df.columns,
        "No duplicate rows": not df.duplicated().any(),
        "No missing values": not df.isna().any().any(),
    }

    if "is_fraud" in df.columns:
        checks["Target contains only 0/1"] = df["is_fraud"].dropna().isin([0, 1]).all()

    all_passed = True

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check}")
        if not passed:
            all_passed = False

    return all_passed


def main() -> None:
    """Run the complete dataset summary."""
    try:
        df = load_dataset(DATA_PATH)

        print_dataset_overview(df)
        print_target_summary(df)
        print_missing_values(df)
        print_data_types(df)
        print_feature_groups(df)
        print_numeric_summary(df)

        ready = validate_dataset(df)

        print("\n" + "=" * 60)
        if ready:
            print("RESULT: Sprint 1 dataset passed the basic readiness checks.")
        else:
            print("RESULT: Review the failed checks before Sprint 2.")
        print("=" * 60)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
