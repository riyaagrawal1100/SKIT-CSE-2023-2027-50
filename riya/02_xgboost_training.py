"""
02_xgboost_training.py
Author: Riya
Sprint 2 — Baseline XGBoost training

What this does:
1. Loads the train split saved by 01_data_preparation.py (X_train, y_train).
   The test set is NOT loaded here — it stays untouched until final
   evaluation in 04_xgboost_evaluation.py.
2. Carves a validation set out of the TRAINING data only (a second,
   stratified split of X_train/y_train) so XGBoost can use early stopping
   without ever looking at the real test set.
3. Handles class imbalance via `scale_pos_weight`, computed only from the
   training sub-split — NOT SMOTE. See "Why class weighting" below.
4. Trains a baseline XGBoost classifier (sensible defaults, not yet tuned —
   tuning happens in 03_xgboost_tuning.py) and reports validation metrics
   appropriate for a heavily imbalanced problem: confusion matrix,
   precision, recall, F1, ROC-AUC, PR-AUC.
5. Saves the baseline model and its validation metrics to Sprint_2/results/
   so it can be compared against the tuned model later.

Why class weighting instead of SMOTE:
Fraud is only ~0.6% of this dataset. SMOTE would synthesize thousands of
artificial fraud rows by interpolating between existing fraud examples in
feature space — with so few real fraud examples (and many of them behavior-
derived features like `amount_ratio_to_normal`), those synthetic points can
end up unrealistic and can also leak the *shape* of the minority class into
the model in a way that doesn't reflect real transactions. XGBoost has a
built-in, standard way to handle imbalance without touching the data at all:
`scale_pos_weight`, which multiplies the gradient/loss contribution of
positive (fraud) examples so the model is pushed to pay attention to them,
while still training on 100% real transactions. That keeps the training
distribution honest and avoids the extra leakage-prone step of fitting a
resampler. `scale_pos_weight` is computed here strictly from the training
sub-split, never from validation or test data.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SPRINT2_DIR = os.path.join(PROJECT_ROOT, "Sprint_2")

DATA_DIR = os.path.join(SPRINT2_DIR, "data")
MODELS_DIR = os.path.join(SPRINT2_DIR, "models")
RESULTS_DIR = os.path.join(SPRINT2_DIR, "results")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20  # carved out of the training set only


def load_training_data():
    x_path = os.path.join(DATA_DIR, "X_train.csv")
    y_path = os.path.join(DATA_DIR, "y_train.csv")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(
            "Training data not found. Run 01_data_preparation.py first."
        )

    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path).iloc[:, 0]

    return X_train, y_train


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive (fraud) examples in the training sub-split.")
    return n_neg / n_pos


def evaluate(model, X, y, label: str) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    metrics = {
        "confusion_matrix": cm.tolist(),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
        "pr_auc": float(average_precision_score(y, y_proba)),
    }

    print(f"\n[{label}] Confusion matrix (rows=actual, cols=predicted):")
    print(cm)
    print(f"[{label}] Precision: {metrics['precision']:.4f}")
    print(f"[{label}] Recall:    {metrics['recall']:.4f}")
    print(f"[{label}] F1-score:  {metrics['f1_score']:.4f}")
    print(f"[{label}] ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"[{label}] PR-AUC:    {metrics['pr_auc']:.4f}")

    return metrics


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train_full, y_train_full = load_training_data()
    print(f"Loaded training data: {X_train_full.shape[0]:,} rows x {X_train_full.shape[1]} columns")

    # Validation split carved out of TRAINING data only (test set untouched).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    print(f"Training sub-split: {X_tr.shape[0]:,} rows | Validation split: {X_val.shape[0]:,} rows")

    scale_pos_weight = compute_scale_pos_weight(y_tr)
    print(f"\nComputed scale_pos_weight from training sub-split only: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining baseline XGBoost model (with early stopping on validation set)...")
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration: {model.best_iteration}")

    val_metrics = evaluate(model, X_val, y_val, "VALIDATION (baseline)")

    model_path = os.path.join(MODELS_DIR, "xgboost_baseline_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nSaved baseline model to {model_path}")

    results_path = os.path.join(RESULTS_DIR, "xgboost_baseline_metrics.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": "XGBoost (baseline, untuned)",
                "scale_pos_weight_used": scale_pos_weight,
                "best_iteration": int(model.best_iteration),
                "validation_metrics": val_metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved baseline metrics to {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
