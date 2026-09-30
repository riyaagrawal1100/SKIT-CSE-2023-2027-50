"""
Sprint 2 - Sheetal (Random Forest)
Step 2: Baseline Random Forest training

Loads the prepared data from 01_data_preparation.py, then:
    * validates the prepared data
    * cross-validates a baseline Random Forest on the TRAINING set only
      (stratified folds; class weighting is applied inside the model, so no
      resampling ever touches a validation fold)
    * fits the baseline on the full training set
    * reports test-set metrics (Precision, Recall, F1, ROC-AUC, PR-AUC,
      confusion matrix). Accuracy is NOT used as the main metric because
      fraud data is highly imbalanced.
    * saves the baseline model, metrics and feature importances

The test set is only used to REPORT the baseline. Tuning (step 3) uses
cross-validation on the training set only.

Run:
    python Sprint_2/sheetal/02_random_forest_training.py
    python Sprint_2/sheetal/02_random_forest_training.py --cv-folds 3 --n-estimators 100
    python Sprint_2/sheetal/02_random_forest_training.py --skip-cv      # for very large data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TARGET_COL = "is_fraud"
CV_FOLDS = 5
N_ESTIMATORS = 200
DECISION_THRESHOLD = 0.5

THIS_DIR = Path(__file__).resolve().parent            # Sprint_2/sheetal
SPRINT2_DIR = THIS_DIR.parent                         # Sprint_2
MODELS_DIR = SPRINT2_DIR / "models"
RESULTS_DIR = SPRINT2_DIR / "results"
PREPARED_FILE = THIS_DIR / "prepared_data" / "random_forest_prepared_data.joblib"
BASELINE_MODEL_FILE = MODELS_DIR / "random_forest_baseline_model.pkl"
BASELINE_METRICS_FILE = RESULTS_DIR / "random_forest_baseline_metrics.json"
BASELINE_CM_FILE = RESULTS_DIR / "random_forest_baseline_confusion_matrix.csv"
FEATURE_IMPORTANCE_FILE = RESULTS_DIR / "random_forest_feature_importance.csv"

REQUIRED_KEYS = ("X_train", "X_test", "y_train", "y_test", "feature_names", "imbalance")


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_prepared_data(path: Path) -> dict:
    """Load the bundle written by 01_data_preparation.py."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {path}\n"
            "Run Sprint_2/sheetal/01_data_preparation.py first."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read prepared data {path}: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Prepared data is missing keys: {missing}. Re-run step 1.")
    print(f"[OK] Loaded prepared data: {path}")
    return bundle


def validate_bundle(bundle: dict) -> None:
    """Sanity checks before training."""
    X_train, X_test = bundle["X_train"], bundle["X_test"]
    y_train, y_test = bundle["y_train"], bundle["y_test"]

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target lengths do not match.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test columns differ.")
    if TARGET_COL in X_train.columns:
        raise ValueError(f"Target '{TARGET_COL}' found among the features (leakage).")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Prepared features contain NaN values.")
    if not all(np.issubdtype(t, np.number) for t in X_train.dtypes):
        raise ValueError("Prepared features must all be numeric.")
    for name, y in (("train", y_train), ("test", y_test)):
        if y.nunique() < 2:
            raise ValueError(f"The {name} target has only one class.")

    print(f"[OK] Validation passed | features: {X_train.shape[1]} | "
          f"train rows: {len(X_train):,} | test rows: {len(X_test):,}")
    print(f"     Train fraud rate: {100 * y_train.mean():.4f}% | "
          f"Test fraud rate: {100 * y_test.mean():.4f}%")


# --------------------------------------------------------------------------- #
# Model and metrics
# --------------------------------------------------------------------------- #
def build_model(class_weight, n_estimators: int, seed: int) -> RandomForestClassifier:
    """Baseline Random Forest. class_weight comes from the imbalance strategy in step 1."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,          # smooths leaves; avoids memorising single rows
        class_weight=class_weight,   # 'balanced_subsample' when fraud is the minority
        n_jobs=-1,
        random_state=seed,
    )


def compute_metrics(y_true, y_proba, threshold: float = DECISION_THRESHOLD) -> dict:
    """Threshold metrics plus threshold-free ROC-AUC and PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }


def print_metrics(title: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print(f"\n--- {title} ---")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print(f"ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"PR-AUC    : {m['pr_auc']:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted)")
    print(f"              Pred 0     Pred 1")
    print(f"  Actual 0  {cm['true_negative']:>8,}   {cm['false_positive']:>8,}")
    print(f"  Actual 1  {cm['false_negative']:>8,}   {cm['true_positive']:>8,}")


# --------------------------------------------------------------------------- #
# Cross-validation (training set only)
# --------------------------------------------------------------------------- #
def run_cross_validation(model, X_train, y_train, folds: int, seed: int) -> dict:
    """Stratified CV on the training set. Returns mean and std per metric."""
    n_fraud = int(y_train.sum())
    folds = min(folds, n_fraud)
    if folds < 2:
        raise ValueError("Not enough fraud rows in the training set for cross-validation.")

    scoring = {
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    print(f"\n[..] Running {folds}-fold stratified cross-validation on the training set")
    start = time.time()
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                            n_jobs=1, error_score="raise")
    print(f"[OK] Cross-validation finished in {time.time() - start:.1f}s")

    summary = {"folds": folds}
    print("\n--- Cross-validation (mean +/- std) ---")
    for name in scoring:
        vals = scores[f"test_{name}"]
        summary[name] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"{name:<10}: {vals.mean():.4f} +/- {vals.std():.4f}")
    return summary


# --------------------------------------------------------------------------- #
# Feature importance and saving
# --------------------------------------------------------------------------- #
def feature_importance_table(model: RandomForestClassifier, names: list[str]) -> pd.DataFrame:
    """Impurity-based importances sorted from most to least important."""
    table = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def save_outputs(model, metrics: dict, importance: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, BASELINE_MODEL_FILE)
    with open(BASELINE_METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    cm = metrics["test"]["confusion_matrix"]
    pd.DataFrame(
        [[cm["true_negative"], cm["false_positive"]],
         [cm["false_negative"], cm["true_positive"]]],
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    ).to_csv(BASELINE_CM_FILE)
    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print(f"\n[OK] Baseline model saved     : {BASELINE_MODEL_FILE}")
    print(f"[OK] Baseline metrics saved   : {BASELINE_METRICS_FILE}")
    print(f"[OK] Confusion matrix saved   : {BASELINE_CM_FILE}")
    print(f"[OK] Feature importance saved : {FEATURE_IMPORTANCE_FILE}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 2 (Sheetal): baseline Random Forest")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation (useful for very large datasets)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        bundle = load_prepared_data(PREPARED_FILE)
        validate_bundle(bundle)

        X_train, X_test = bundle["X_train"], bundle["X_test"]
        y_train, y_test = bundle["y_train"], bundle["y_test"]
        class_weight = bundle["imbalance"].get("class_weight")
        print(f"\nImbalance strategy: {bundle['imbalance']['strategy']} "
              f"(class_weight={class_weight})")

        model = build_model(class_weight, args.n_estimators, args.seed)

        cv_summary = None
        if not args.skip_cv:
            cv_summary = run_cross_validation(model, X_train, y_train, args.cv_folds, args.seed)

        print("\n[..] Fitting baseline Random Forest on the full training set")
        start = time.time()
        model.fit(X_train, y_train)
        print(f"[OK] Training finished in {time.time() - start:.1f}s")

        train_metrics = compute_metrics(y_train, model.predict_proba(X_train)[:, 1])
        test_metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1])
        print_metrics("Baseline - TRAIN (expect optimistic values)", train_metrics)
        print_metrics("Baseline - TEST", test_metrics)

        gap = train_metrics["pr_auc"] - test_metrics["pr_auc"]
        if gap > 0.15:
            print(f"\n[WARN] PR-AUC gap train-test = {gap:.3f}: possible overfitting. "
                  "Step 3 tunes depth and leaf size to address this.")

        importance = feature_importance_table(model, bundle["feature_names"])
        print("\nTop 10 features by importance:")
        print(importance.head(10).to_string(index=False))

        metrics = {
            "model": "RandomForestClassifier (baseline)",
            "random_state": args.seed,
            "n_estimators": args.n_estimators,
            "class_weight": class_weight,
            "cross_validation": cv_summary,
            "train": train_metrics,
            "test": test_metrics,
        }
        save_outputs(model, metrics, importance)
        print("\nBaseline training complete. Next: 03_random_forest_tuning.py")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
