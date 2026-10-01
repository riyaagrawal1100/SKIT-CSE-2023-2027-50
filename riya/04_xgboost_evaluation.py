"""
04_xgboost_evaluation.py
Author: Riya
Sprint 2 — Final XGBoost training + one-time test-set evaluation

What this does:
1. Loads the best hyperparameters found by 03_xgboost_tuning.py (cross-
   validated on the training set only) and the full X_train/y_train.
2. Trains the FINAL XGBoost model on the full training set using those
   hyperparameters, with a small internal early-stopping validation split
   carved out of the training data (never from the test set).
3. Loads X_test/y_test — untouched until this point — and evaluates the
   final model on it EXACTLY ONCE. This is the only place in the whole
   Sprint 2 XGBoost pipeline the test set is used for anything.
4. Reports confusion matrix, precision, recall, F1-score, ROC-AUC, and
   PR-AUC, and saves:
     - the trained model -> Sprint_2/models/xgboost_fraud_model.pkl
     - metrics (json) -> Sprint_2/results/xgboost_final_metrics.json
     - confusion matrix, ROC curve, and PR curve plots -> Sprint_2/results/
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
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
VALIDATION_SIZE = 0.20  # for early stopping only, carved from TRAIN


def load_split(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 01_data_preparation.py first.")
    return pd.read_csv(path)


def load_best_params() -> dict:
    path = os.path.join(RESULTS_DIR, "xgboost_best_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 03_xgboost_tuning.py first.")
    with open(path) as f:
        payload = json.load(f)
    return payload["best_params"]


def evaluate(model, X, y, label: str) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)
    metrics = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["legit (0)", "fraud (1)"],
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

    return metrics, y_pred, y_proba


def save_plots(model, X_test, y_test, y_pred, y_proba):
    cm_fig, cm_ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"], cmap="Blues", ax=cm_ax
    )
    cm_ax.set_title("XGBoost — Confusion Matrix (Test Set)")
    cm_fig.savefig(os.path.join(RESULTS_DIR, "xgboost_confusion_matrix.png"), bbox_inches="tight")
    plt.close(cm_fig)

    roc_fig, roc_ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=roc_ax, name="XGBoost")
    roc_ax.set_title("XGBoost — ROC Curve (Test Set)")
    roc_fig.savefig(os.path.join(RESULTS_DIR, "xgboost_roc_curve.png"), bbox_inches="tight")
    plt.close(roc_fig)

    pr_fig, pr_ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=pr_ax, name="XGBoost")
    pr_ax.set_title("XGBoost — Precision-Recall Curve (Test Set)")
    pr_fig.savefig(os.path.join(RESULTS_DIR, "xgboost_pr_curve.png"), bbox_inches="tight")
    plt.close(pr_fig)

    top_n = 20
    importances = pd.Series(model.feature_importances_, index=model.get_booster().feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)
    fi_fig, fi_ax = plt.subplots(figsize=(8, 7))
    fi_ax.barh(top_features.index[::-1], top_features.values[::-1], color="#4C72B0")
    fi_ax.set_title(f"XGBoost — Top {top_n} Feature Importances")
    fi_fig.savefig(os.path.join(RESULTS_DIR, "xgboost_feature_importance.png"), bbox_inches="tight")
    plt.close(fi_fig)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train = load_split("X_train")
    y_train = load_split("y_train").iloc[:, 0]
    X_test = load_split("X_test")
    y_test = load_split("y_test").iloc[:, 0]
    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows (test untouched until now)")

    best_params = load_best_params()
    print("\nUsing tuned hyperparameters from 03_xgboost_tuning.py:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Small validation split (from TRAIN only) purely to enable early stopping
    # on the final fit — the test set plays no role in this.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )

    print("\nTraining final XGBoost model on the training set (tuned hyperparameters)...")
    final_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Best iteration (early stopping): {final_model.best_iteration}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET (used only once)")
    print("=" * 60)
    test_metrics, y_pred, y_proba = evaluate(final_model, X_test, y_test, "TEST")

    save_plots(final_model, X_test, y_test, y_pred, y_proba)

    model_path = os.path.join(MODELS_DIR, "xgboost_fraud_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"\nSaved final XGBoost model to {model_path}")

    results_path = os.path.join(RESULTS_DIR, "xgboost_final_metrics.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": "XGBoost (tuned, final)",
                "hyperparameters": best_params,
                "best_iteration": int(final_model.best_iteration),
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved final metrics to {results_path}")
    print(f"Saved plots to {RESULTS_DIR}/ (confusion matrix, ROC curve, PR curve, feature importance)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
