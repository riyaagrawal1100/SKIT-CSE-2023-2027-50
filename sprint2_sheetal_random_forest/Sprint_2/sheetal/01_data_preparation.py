"""
Sprint 2 - Sheetal (Random Forest)
Step 1: Data preparation

Pipeline implemented here:
    final_preprocessed_dataset.csv
        -> data validation
        -> separate X and y (is_fraud)
        -> stratified train/test split          (BEFORE anything is learned from data)
        -> class imbalance analysis + strategy
        -> feature preparation (fit on TRAIN only, applied to TEST)
        -> save prepared data for 02_random_forest_training.py

Data-leakage rules followed:
    * The split happens before imputation / encoding are fitted.
    * The imputer and encoder are fitted on the training set only.
    * No SMOTE / resampling is used (see choose_imbalance_strategy for why).
    * is_fraud and any column whose name contains "fraud" are never used as features.
    * Scaling is intentionally NOT applied: Random Forest splits are scale-invariant.

Run:
    python Sprint_2/sheetal/01_data_preparation.py
    python Sprint_2/sheetal/01_data_preparation.py --data path/to/final_preprocessed_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TARGET_COL = "is_fraud"
DATASET_NAME = "final_preprocessed_dataset.csv"
TEST_SIZE = 0.20
LEAKAGE_CORR_THRESHOLD = 0.95   # |corr(feature, target)| above this is flagged
MAX_OHE_CARDINALITY = 50        # categorical columns above this look like identifiers
MIN_CLASS_COUNT = 10            # minimum samples per class needed for a sane split
TIME_HINTS = ("time", "date", "timestamp", "unix")

THIS_DIR = Path(__file__).resolve().parent            # Sprint_2/sheetal
SPRINT2_DIR = THIS_DIR.parent                         # Sprint_2
PROJECT_ROOT = SPRINT2_DIR.parent                     # project root
RESULTS_DIR = SPRINT2_DIR / "results"
PREPARED_DIR = THIS_DIR / "prepared_data"
PREPARED_FILE = PREPARED_DIR / "random_forest_prepared_data.joblib"
REPORT_FILE = RESULTS_DIR / "sheetal_data_preparation_report.json"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_dataset(cli_path: str | None) -> Path:
    """Return the dataset path: the CLI path if given, otherwise search the project."""
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found at: {path}")
        return path

    candidates = [
        p for p in PROJECT_ROOT.rglob(DATASET_NAME)
        if SPRINT2_DIR not in p.parents  # never pick up copies inside Sprint_2
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find '{DATASET_NAME}' under {PROJECT_ROOT}. "
            "Pass the location with --data <path>."
        )
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    if len(candidates) > 1:
        print(f"[WARN] {len(candidates)} copies of {DATASET_NAME} found; using the shallowest:")
        for c in candidates:
            print(f"       - {c}")
    return candidates[0]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the CSV with clear error messages."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - surface any parsing problem clearly
        raise ValueError(f"Could not read dataset {path}: {exc}") from exc
    print(f"[OK] Loaded {path}")
    print(f"     Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    return df


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def coerce_target(series: pd.Series) -> pd.Series:
    """Validate that the target is binary (0/1) and return it as int."""
    if series.isna().any():
        raise ValueError(f"Target '{TARGET_COL}' has {int(series.isna().sum())} missing values.")
    if series.dtype == bool:
        return series.astype(int)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"Target '{TARGET_COL}' contains non-numeric values.")
    values = set(numeric.unique())
    if not values.issubset({0, 1}):
        raise ValueError(f"Target '{TARGET_COL}' must be 0/1 but has values: {sorted(values)}")
    if len(values) < 2:
        raise ValueError(f"Target '{TARGET_COL}' contains only one class: {sorted(values)}")
    return numeric.astype(int)


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run data-quality checks. Raises on fatal problems, reports the rest."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Available columns: {list(df.columns)}"
        )

    features = df.drop(columns=[TARGET_COL])
    numeric_cols = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    non_numeric_cols = [c for c in features.columns if c not in numeric_cols]

    missing = df.isna().sum()
    missing = missing[missing > 0]
    inf_counts = {
        c: int(np.isinf(features[c]).sum())
        for c in features.select_dtypes(include=[np.number]).columns
        if np.isinf(features[c]).any()
    }
    constant_cols = [c for c in features.columns if features[c].nunique(dropna=False) <= 1]
    time_like = [c for c in features.columns if any(h in c.lower() for h in TIME_HINTS)]

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET_COL,
        "numeric_feature_columns": len(numeric_cols),
        "non_numeric_feature_columns": non_numeric_cols,
        "columns_with_missing_values": {k: int(v) for k, v in missing.items()},
        "columns_with_infinite_values": inf_counts,
        "constant_columns": constant_cols,
        "duplicate_rows": int(df.duplicated().sum()),
        "time_like_columns": time_like,
    }

    print("\n=== DATA VALIDATION ===")
    print(f"Rows x columns          : {report['rows']:,} x {report['columns']:,}")
    print(f"Target column           : {TARGET_COL}")
    print(f"Numeric feature columns : {len(numeric_cols)}")
    print(f"Non-numeric columns     : {non_numeric_cols if non_numeric_cols else 'none'}")
    print(f"Columns with NaN        : {list(report['columns_with_missing_values']) or 'none'}")
    print(f"Columns with +/-inf     : {list(inf_counts) or 'none'}")
    print(f"Constant columns        : {constant_cols or 'none'}")
    print(f"Duplicate rows          : {report['duplicate_rows']:,}")
    if time_like:
        print(f"[WARN] Time-like columns found: {time_like}. If transactions are time-ordered, "
              "a random split can mix past and future; mention this in the report.")
    return report


# --------------------------------------------------------------------------- #
# Features / target
# --------------------------------------------------------------------------- #
def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split into X and y. Drops any other column whose name contains 'fraud'."""
    y = coerce_target(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL]).copy()

    # Name-based target-derived columns (e.g. a 'fraud_flag' copy of the label).
    dropped = [c for c in X.columns if "fraud" in c.lower()]
    if dropped:
        print(f"[WARN] Dropping target-derived columns (name contains 'fraud'): {dropped}")
        X = X.drop(columns=dropped)

    # Booleans -> ints; +/-inf -> NaN (value-independent, no statistics learned).
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(int)
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    if TARGET_COL in X.columns:
        raise RuntimeError("Target column leaked into the feature matrix.")
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after removing the target.")
    print(f"\n[OK] Features: {X.shape[1]} columns | Target: {TARGET_COL}")
    return X, y, dropped


# --------------------------------------------------------------------------- #
# Class distribution / imbalance
# --------------------------------------------------------------------------- #
def summarize_class_distribution(y: pd.Series, title: str) -> dict:
    """Print and return fraud/legitimate counts and percentages."""
    total = int(len(y))
    fraud = int((y == 1).sum())
    legit = int((y == 0).sum())
    summary = {
        "total": total,
        "fraud_transactions": fraud,
        "legitimate_transactions": legit,
        "fraud_percentage": round(100 * fraud / total, 4),
        "legitimate_percentage": round(100 * legit / total, 4),
    }
    print(f"\n--- Class distribution: {title} ---")
    print(f"Fraud transactions      : {fraud:,}")
    print(f"Legitimate transactions : {legit:,}")
    print(f"Fraud percentage        : {summary['fraud_percentage']:.4f}%")
    print(f"Legitimate percentage   : {summary['legitimate_percentage']:.4f}%")
    return summary


def choose_imbalance_strategy(y_train: pd.Series) -> dict:
    """
    Pick the imbalance strategy from the TRAINING distribution.

    Random Forest supports cost-sensitive learning through class_weight.
    'balanced_subsample' recomputes the weights inside every bootstrap sample,
    which suits a forest because each tree sees a different resample.
    Reasons this is preferred over SMOTE here:
      * No synthetic transactions: SMOTE interpolates between rows, which is
        unreliable with one-hot or behavioural columns.
      * Predicted probabilities stay closer to reality, which Sprint 3 needs
        for risk scoring.
      * No resampling inside cross-validation, so synthetic neighbours of
        validation rows cannot leak into training folds.
      * The test set keeps its true fraud rate.
    The reference weights below are informational; the model uses the string
    option so weights are recomputed per bootstrap sample.
    """
    fraud = int((y_train == 1).sum())
    legit = int((y_train == 0).sum())
    total = fraud + legit
    fraud_pct = 100 * fraud / total

    if fraud < MIN_CLASS_COUNT:
        raise ValueError(f"Only {fraud} fraud rows in the training set; too few to train.")

    reference_weights = {
        "legitimate_weight": round(total / (2 * legit), 4),
        "fraud_weight": round(total / (2 * fraud), 4),
    }
    if fraud_pct >= 30:
        strategy = {
            "strategy": "none",
            "class_weight": None,
            "reason": "Classes are roughly balanced (fraud >= 30%), so no correction is applied.",
        }
    else:
        strategy = {
            "strategy": "class_weight",
            "class_weight": "balanced_subsample",
            "reason": (
                "Fraud is the minority class. Weighting fraud errors more heavily inside "
                "each bootstrap sample teaches every tree to care about fraud without "
                "creating synthetic data. SMOTE/oversampling is not used."
            ),
        }
    strategy["reference_weights_full_train"] = reference_weights

    if fraud < 100:
        strategy["warning"] = (
            f"Only {fraud} fraud rows in training; bootstrap samples may hold very few "
            "frauds, so metrics and tuning will be noisy."
        )
        print(f"[WARN] {strategy['warning']}")

    print("\n=== IMBALANCE STRATEGY ===")
    print(f"Training fraud rate     : {fraud_pct:.4f}%")
    print(f"Chosen strategy         : {strategy['strategy']}")
    print(f"class_weight            : {strategy['class_weight']}")
    print(f"Reference weights       : {reference_weights}")
    print(f"Why                     : {strategy['reason']}")
    return strategy


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. Nothing has been fitted on the data before this."""
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    if y.value_counts().min() < MIN_CLASS_COUNT:
        raise ValueError(f"Each class needs at least {MIN_CLASS_COUNT} samples to split.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"\n[OK] Stratified split (test_size={test_size}, random_state={seed})")
    print(f"     Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Leakage audit (train only, diagnostic)
# --------------------------------------------------------------------------- #
def leakage_audit(X_train: pd.DataFrame, y_train: pd.Series) -> list[dict]:
    """Flag numeric features almost perfectly correlated with the target."""
    findings = []
    numeric = X_train.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if numeric[col].nunique(dropna=True) < 2:
            continue
        corr = numeric[col].corr(y_train)
        if pd.notna(corr) and abs(corr) >= LEAKAGE_CORR_THRESHOLD:
            findings.append({"feature": col, "correlation_with_target": round(float(corr), 4)})

    print("\n=== LEAKAGE AUDIT (training set) ===")
    if findings:
        for f in findings:
            print(f"[WARN] {f['feature']} has correlation {f['correlation_with_target']} "
                  "with the target. Verify it is not derived from is_fraud.")
    else:
        print(f"No numeric feature has |corr| >= {LEAKAGE_CORR_THRESHOLD} with the target.")
    return findings


# --------------------------------------------------------------------------- #
# Feature preparation (fit on train only)
# --------------------------------------------------------------------------- #
def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    """
    Build (not yet fit) the preprocessing transformer from column TYPES only.
    Numeric: median imputation. Categorical: mode imputation + one-hot encoding.
    High-cardinality text columns look like identifiers and are excluded.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    excluded = [c for c in categorical_cols if X_train[c].nunique(dropna=True) > MAX_OHE_CARDINALITY]
    if excluded:
        print(f"[WARN] Excluding high-cardinality non-numeric columns (likely identifiers): {excluded}")
    categorical_cols = [c for c in categorical_cols if c not in excluded]

    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))
    if not transformers:
        raise ValueError("No usable feature columns after preprocessing rules.")

    print(f"\n[OK] Preprocessing plan: {len(numeric_cols)} numeric, "
          f"{len(categorical_cols)} categorical (one-hot). No scaling (tree model).")
    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre, excluded


def prepare_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, list[str]]:
    """Fit on train only, then transform both train and test."""
    pre, excluded = build_preprocessor(X_train)
    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    if list(X_train_p.columns) != list(X_test_p.columns):
        raise RuntimeError("Train and test feature columns do not match after preprocessing.")
    if X_train_p.isna().any().any() or X_test_p.isna().any().any():
        raise RuntimeError("NaN values remain after preprocessing.")
    if TARGET_COL in X_train_p.columns:
        raise RuntimeError("Target column present among prepared features.")
    print(f"[OK] Prepared matrices | Train: {X_train_p.shape} | Test: {X_test_p.shape}")
    return X_train_p, X_test_p, pre, excluded


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_outputs(bundle: dict, report: dict) -> None:
    """Persist prepared data (for step 2) and the JSON report (for the write-up)."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, PREPARED_FILE)
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[OK] Prepared data saved : {PREPARED_FILE}")
    print(f"[OK] Report saved        : {REPORT_FILE}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 2 (Sheetal): Random Forest data preparation")
    parser.add_argument("--data", type=str, default=None,
                        help=f"Path to {DATASET_NAME} (auto-detected if omitted)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data_path = find_dataset(args.data)
        df = load_dataset(data_path)

        validation = validate_dataset(df)
        distribution_full = summarize_class_distribution(
            coerce_target(df[TARGET_COL]), "full dataset"
        )

        X, y, dropped_cols = separate_features_target(df)
        X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

        distribution_train = summarize_class_distribution(y_train, "training set")
        distribution_test = summarize_class_distribution(y_test, "test set")
        imbalance = choose_imbalance_strategy(y_train)

        leakage = leakage_audit(X_train, y_train)
        X_train_p, X_test_p, preprocessor, excluded = prepare_features(X_train, X_test)

        bundle = {
            "X_train": X_train_p,
            "X_test": X_test_p,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "preprocessor": preprocessor,
            "feature_names": X_train_p.columns.tolist(),
            "imbalance": imbalance,
            "random_state": args.seed,
        }
        report = {
            "dataset_path": str(data_path),
            "random_state": args.seed,
            "test_size": args.test_size,
            "validation": validation,
            "dropped_target_derived_columns": dropped_cols,
            "excluded_high_cardinality_columns": excluded,
            "class_distribution": {
                "full": distribution_full,
                "train": distribution_train,
                "test": distribution_test,
            },
            "imbalance_strategy": imbalance,
            "leakage_findings": leakage,
            "n_prepared_features": len(bundle["feature_names"]),
        }
        save_outputs(bundle, report)
        print("\nData preparation complete. Next: 02_random_forest_training.py")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
