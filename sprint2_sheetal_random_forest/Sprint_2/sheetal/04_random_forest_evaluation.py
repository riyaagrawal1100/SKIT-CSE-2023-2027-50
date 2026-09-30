"""
Sprint 2 - Sheetal (Random Forest)
Step 4: Final evaluation and model saving

What this script does:
    * loads the tuned model from 03_random_forest_tuning.py and the untouched test set
    * checks that the model's features match the test features and contain no target
    * evaluates on the test set at the tuned decision threshold (chosen from
      out-of-fold training predictions in step 3) and, for reference, at 0.5
      - Precision, Recall, F1-score, ROC-AUC, PR-AUC, confusion matrix
      - Accuracy is NOT reported as a headline metric (imbalanced data)
    * compares against the baseline from step 2
    * measures prediction speed and model size (useful for the model comparison)
    * saves the FINAL model, its metadata, the preprocessor, test predictions,
      metrics, confusion matrix and plots

Important: the final model is the tuned model chosen by cross-validation. It is
NOT picked by looking at test results, so the test set stays an honest estimate.

Run:
    python Sprint_2/sheetal/04_random_forest_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TARGET_COL = "is_fraud"
DEFAULT_THRESHOLD = 0.5
OVERFIT_WARNING_GAP = 0.10   # CV vs test PR-AUC difference that triggers a warning

THIS_DIR = Path(__file__).resolve().parent            # Sprint_2/sheetal
SPRINT2_DIR = THIS_DIR.parent                         # Sprint_2
MODELS_DIR = SPRINT2_DIR / "models"
RESULTS_DIR = SPRINT2_DIR / "results"

PREPARED_FILE = THIS_DIR / "prepared_data" / "random_forest_prepared_data.joblib"
TUNED_MODEL_FILE = MODELS_DIR / "random_forest_tuned_model.pkl"
BEST_PARAMS_FILE = RESULTS_DIR / "random_forest_best_params.json"
BASELINE_METRICS_FILE = RESULTS_DIR / "random_forest_baseline_metrics.json"

FINAL_MODEL_FILE = MODELS_DIR / "random_forest_fraud_model.pkl"
PREPROCESSOR_FILE = MODELS_DIR / "random_forest_preprocessor.pkl"
METADATA_FILE = MODELS_DIR / "random_forest_model_metadata.json"
FINAL_METRICS_FILE = RESULTS_DIR / "random_forest_final_metrics.json"
CONFUSION_CSV_FILE = RESULTS_DIR / "random_forest_confusion_matrix.csv"
CONFUSION_PNG_FILE = RESULTS_DIR / "random_forest_confusion_matrix.png"
CURVES_PNG_FILE = RESULTS_DIR / "random_forest_roc_pr_curves.png"
PREDICTIONS_FILE = RESULTS_DIR / "random_forest_test_predictions.csv"
IMPORTANCE_FILE = RESULTS_DIR / "random_forest_tuned_feature_importance.csv"

REQUIRED_KEYS = ("X_train", "X_test", "y_train", "y_test", "feature_names", "preprocessor")


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_artifacts() -> tuple[dict, object, dict]:
    """Load prepared data, the tuned model and the step-3 summary."""
    for path, hint in (
        (PREPARED_FILE, "01_data_preparation.py"),
        (TUNED_MODEL_FILE, "03_random_forest_tuning.py"),
        (BEST_PARAMS_FILE, "03_random_forest_tuning.py"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}\nRun Sprint_2/sheetal/{hint} first.")

    try:
        bundle = joblib.load(PREPARED_FILE)
        model = joblib.load(TUNED_MODEL_FILE)
        with open(BEST_PARAMS_FILE, encoding="utf-8") as fh:
            tuning = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not load a required artifact: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f"Prepared data is missing keys: {missing}. Re-run step 1.")
    print("[OK] Loaded prepared data, tuned model and tuning summary")
    return bundle, model, tuning


def validate_inputs(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> None:
    """Make sure the model and test data are consistent and leak-free."""
    if len(X_test) != len(y_test):
        raise ValueError("Test features and target lengths do not match.")
    if TARGET_COL in X_test.columns:
        raise ValueError(f"Target '{TARGET_COL}' found among the test features (leakage).")
    if X_test.isna().any().any():
        raise ValueError("Test features contain NaN values.")
    if y_test.nunique() < 2:
        raise ValueError("The test target has only one class; metrics are undefined.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"Decision threshold must be in (0, 1], got {threshold}.")

    trained_on = getattr(model, "feature_names_in_", None)
    if trained_on is not None and list(trained_on) != list(X_test.columns):
        raise ValueError("Model feature names differ from the test feature names.")
    if getattr(model, "n_features_in_", X_test.shape[1]) != X_test.shape[1]:
        raise ValueError("Model feature count differs from the test data.")
    print(f"[OK] Inputs valid | test rows: {len(X_test):,} | fraud rate: {100 * y_test.mean():.4f}%")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    """Threshold metrics plus threshold-free ROC-AUC and PR-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
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
    print(f"\n--- {title} (threshold = {m['threshold']:.4f}) ---")
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print(f"ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"PR-AUC    : {m['pr_auc']:.4f}")
    print("Confusion matrix (rows = actual, cols = predicted)")
    print(f"              Pred 0     Pred 1")
    print(f"  Actual 0  {cm['true_negative']:>8,}   {cm['false_positive']:>8,}")
    print(f"  Actual 1  {cm['false_negative']:>8,}   {cm['true_positive']:>8,}")


def measure_efficiency(model, X_test: pd.DataFrame) -> dict:
    """Prediction speed and model size, for the efficiency side of the model comparison."""
    start = time.perf_counter()
    model.predict_proba(X_test)
    elapsed = time.perf_counter() - start
    size_mb = FINAL_MODEL_FILE.stat().st_size / (1024 ** 2) if FINAL_MODEL_FILE.is_file() else None
    return {
        "predict_seconds_total": float(elapsed),
        "predict_ms_per_1000_rows": float(1000 * elapsed / len(X_test) * 1000),
        "model_size_mb": None if size_mb is None else float(size_mb),
    }


# --------------------------------------------------------------------------- #
# Plots (optional: skipped if matplotlib is unavailable)
# --------------------------------------------------------------------------- #
def save_plots(y_true, y_proba, metrics: dict) -> bool:
    """Confusion matrix plus ROC and Precision-Recall curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping plots (pip install matplotlib).")
        return False

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["true_negative"], cm["false_positive"]],
                       [cm["false_negative"], cm["true_positive"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Random Forest - confusion matrix\n(threshold {metrics['threshold']:.3f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG_FILE, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision, label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    axes[1].axhline(np.mean(y_true), ls="--", color="grey", label="Fraud rate (no skill)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(CURVES_PNG_FILE, dpi=150)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_outputs(model, bundle: dict, tuning: dict, final: dict, at_default: dict,
                 baseline_test: dict | None, efficiency: dict,
                 y_test: pd.Series, y_proba: np.ndarray) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, FINAL_MODEL_FILE)
    joblib.dump(bundle["preprocessor"], PREPROCESSOR_FILE)
    efficiency["model_size_mb"] = FINAL_MODEL_FILE.stat().st_size / (1024 ** 2)

    metadata = {
        "model": "RandomForestClassifier (final)",
        "owner": "Sheetal",
        "target_column": TARGET_COL,
        "decision_threshold": final["threshold"],
        "random_state": tuning.get("random_state", RANDOM_STATE),
        "best_params": tuning.get("best_params"),
        "feature_names": bundle["feature_names"],
        "preprocessor_file": PREPROCESSOR_FILE.name,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    results = {
        "model": "Random Forest",
        "test_at_tuned_threshold": final,
        "test_at_default_threshold": at_default,
        "baseline_test_at_default_threshold": baseline_test,
        "cv_best_scores": tuning.get("best_cv_scores"),
        "efficiency": efficiency,
    }
    with open(FINAL_METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    cm = final["confusion_matrix"]
    pd.DataFrame(
        [[cm["true_negative"], cm["false_positive"]],
         [cm["false_negative"], cm["true_positive"]]],
        index=["actual_legitimate", "actual_fraud"],
        columns=["predicted_legitimate", "predicted_fraud"],
    ).to_csv(CONFUSION_CSV_FILE)

    pd.DataFrame({"y_true": y_test.to_numpy(), "y_proba": y_proba}).to_csv(
        PREDICTIONS_FILE, index=False)

    if hasattr(model, "feature_importances_"):
        pd.DataFrame({"feature": bundle["feature_names"],
                      "importance": model.feature_importances_}
                     ).sort_values("importance", ascending=False).to_csv(IMPORTANCE_FILE, index=False)

    print(f"\n[OK] FINAL model saved   : {FINAL_MODEL_FILE}")
    print(f"[OK] Preprocessor saved  : {PREPROCESSOR_FILE}")
    print(f"[OK] Metadata saved      : {METADATA_FILE}")
    print(f"[OK] Metrics saved       : {FINAL_METRICS_FILE}")
    print(f"[OK] Confusion matrix    : {CONFUSION_CSV_FILE}")
    print(f"[OK] Test predictions    : {PREDICTIONS_FILE}")


def load_baseline_test() -> dict | None:
    """Baseline test metrics from step 2 (threshold 0.5), if available."""
    if not BASELINE_METRICS_FILE.is_file():
        return None
    try:
        with open(BASELINE_METRICS_FILE, encoding="utf-8") as fh:
            return json.load(fh).get("test")
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        bundle, model, tuning = load_artifacts()
        X_test, y_test = bundle["X_test"], bundle["y_test"]
        threshold = float(tuning.get("decision_threshold", {}).get("threshold", DEFAULT_THRESHOLD))
        validate_inputs(model, X_test, y_test, threshold)

        y_proba = model.predict_proba(X_test)[:, 1]
        final = compute_metrics(y_test, y_proba, threshold)
        at_default = compute_metrics(y_test, y_proba, DEFAULT_THRESHOLD)
        print_metrics("FINAL Random Forest - TEST, tuned threshold", final)
        print_metrics("Reference - TEST, default threshold", at_default)

        baseline_test = load_baseline_test()
        if baseline_test:
            print("\n--- Baseline (step 2, threshold 0.5) vs tuned model on TEST ---")
            print(f"{'metric':<10}{'baseline':>10}{'tuned@0.5':>11}{'tuned@thr':>11}")
            for name in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
                print(f"{name:<10}{baseline_test[name]:>10.4f}{at_default[name]:>11.4f}{final[name]:>11.4f}")

        cv_pr = (tuning.get("best_cv_scores") or {}).get("pr_auc", {}).get("mean")
        if cv_pr is not None and abs(cv_pr - final["pr_auc"]) > OVERFIT_WARNING_GAP:
            print(f"\n[WARN] CV PR-AUC ({cv_pr:.3f}) and test PR-AUC ({final['pr_auc']:.3f}) differ "
                  "noticeably. Check for overfitting or a small number of test frauds.")
        if int(y_test.sum()) < 30:
            print(f"[WARN] Only {int(y_test.sum())} fraud rows in the test set; "
                  "metrics have wide uncertainty.")

        efficiency = measure_efficiency(model, X_test)
        save_outputs(model, bundle, tuning, final, at_default, baseline_test,
                     efficiency, y_test, y_proba)
        print(f"\nPrediction time : {efficiency['predict_ms_per_1000_rows']:.2f} ms per 1000 rows")
        print(f"Model size      : {efficiency['model_size_mb']:.2f} MB")

        if save_plots(y_test, y_proba, final):
            print(f"[OK] Plots saved         : {CONFUSION_PNG_FILE.name}, {CURVES_PNG_FILE.name}")
        print("\nRandom Forest evaluation complete. Final model is ready for the comparison step.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
