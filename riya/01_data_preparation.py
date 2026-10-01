"""
01_data_preparation.py
Author: Riya
Sprint 2 — Data preparation for XGBoost modeling

What this does:
1. Loads `final_preprocessed_dataset.csv` (Sprint 1 output) and validates it
   (shape, dtypes, missing values, duplicate rows, target column sanity).
2. Separates features (X) from the target (y = `is_fraud`). Defensively
   double-checks that no known identifier/leakage columns (e.g. `cc_num`,
   `trans_num`, `dob`, raw timestamp) made it into X — Sprint 1 already drops
   these, but this is a cheap safety net for Sprint 2.
3. Casts the one-hot-encoded boolean columns from Sprint 1 (`pd.get_dummies`
   output) to int — this is a pure dtype cast, NOT a value change, so it
   carries no information from the dataset and is safe to do before the
   split.
4. Splits into train/test (stratified on `is_fraud`) BEFORE anything that
   could learn from the data (no scaling, no imputation, no oversampling is
   applied at this stage — see the module docstring note below).
5. Reports the class imbalance (fraud vs legitimate counts/percentages) on
   the full dataset and confirms the split preserved that ratio.
6. Saves X_train/X_test/y_train/y_test to Sprint_2/data/ so every downstream
   script (training, tuning, evaluation, comparison, and Sheetal's own
   pipeline) reads the exact same split — this keeps the model comparison
   at the end fair (same train/test rows for both models).

IMPORTANT — leakage discipline:
The train/test split happens here, first, before any class-imbalance
handling or feature preparation. Class imbalance is NOT corrected at this
stage: it's handled per-model, on the training data only, inside each
model's own training script (see 02_xgboost_training.py), so nothing learned
from resampling ever touches the test set.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SPRINT2_DIR = os.path.join(PROJECT_ROOT, "Sprint_2")

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "final_preprocessed_dataset.csv")
DATA_OUT_DIR = os.path.join(SPRINT2_DIR, "data")

TARGET_COL = "is_fraud"

# Columns that must NEVER end up in X — if any of these are present, Sprint 1
# leaked an identifier / raw-timestamp / target-derived column into the
# "final" dataset, and we should fail loudly rather than train on it.
FORBIDDEN_COLS = ["cc_num", "trans_num", "dob", "trans_date_trans_time", "unix_time"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_validate(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run Sprint 1's riya/02_feature_engineering.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("final_preprocessed_dataset.csv is empty.")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    missing = df.isna().sum().sum()
    if missing > 0:
        raise ValueError(f"Dataset has {missing} missing values — expected 0 after Sprint 1.")

    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        print(f"WARNING: {dup_rows} exact duplicate rows found (not removed automatically).")

    bad_target_values = set(df[TARGET_COL].unique()) - {0, 1}
    if bad_target_values:
        raise ValueError(f"Target column has unexpected values: {bad_target_values}")

    present_forbidden = [c for c in FORBIDDEN_COLS if c in df.columns]
    if present_forbidden:
        raise ValueError(
            f"Leakage risk: found identifier/raw-timestamp columns still in the "
            f"dataset: {present_forbidden}. These should have been dropped in "
            f"Sprint 1's riya/02_feature_engineering.py."
        )

    non_numeric = df.drop(columns=[TARGET_COL]).select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Found non-numeric feature columns that still need encoding: {non_numeric}"
        )

    return df


def prepare_features_and_target(df: pd.DataFrame):
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    return X, y


def report_class_balance(y: pd.Series, label: str) -> None:
    counts = y.value_counts().sort_index()
    pct = y.value_counts(normalize=True).sort_index() * 100

    fraud_n = int(counts.get(1, 0))
    legit_n = int(counts.get(0, 0))
    fraud_pct = float(pct.get(1, 0.0))
    legit_pct = float(pct.get(0, 0.0))

    print(f"\n[{label}] Class distribution ({len(y):,} rows)")
    print(f"  Fraud transactions:       {fraud_n:,}")
    print(f"  Legitimate transactions:  {legit_n:,}")
    print(f"  Fraud percentage:         {fraud_pct:.3f}%")
    print(f"  Legitimate percentage:    {legit_pct:.3f}%")


def main():
    os.makedirs(DATA_OUT_DIR, exist_ok=True)

    print(f"Loading {INPUT_PATH} ...")
    df = load_and_validate(INPUT_PATH)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    X, y = prepare_features_and_target(df)
    print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
    print(f"Feature columns ({X.shape[1]}): {list(X.columns)[:8]} ... (+{X.shape[1] - 8} more)")

    report_class_balance(y, "FULL DATASET")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    report_class_balance(y_train, "TRAIN SPLIT")
    report_class_balance(y_test, "TEST SPLIT")

    X_train.to_csv(os.path.join(DATA_OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_OUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_OUT_DIR, "y_test.csv"), index=False)

    with open(os.path.join(DATA_OUT_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(X.columns))

    print(f"\nSaved train/test split to {DATA_OUT_DIR}/")
    print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv, feature_columns.txt")
    print(
        "\nNote: class imbalance is intentionally NOT corrected here. It is "
        "handled per-model on the training data only (see 02_xgboost_training.py)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
