"""
Sprint 2 - Sheetal (Random Forest)
Step 3: Hyperparameter tuning

What this script does (all on the TRAINING set only; the test set is untouched):
    * loads the prepared data from 01_data_preparation.py
    * runs a randomized search with stratified cross-validation
      - the model is refit on PR-AUC (average precision), the most informative
        metric for a rare-fraud problem; Precision, Recall, F1 and ROC-AUC are
        recorded for every candidate too
      - class imbalance is handled by class_weight inside the model, so no
        resampling ever touches a validation fold
    * chooses a decision threshold from out-of-fold predictions (maximises F1),
      because the default 0.5 is rarely optimal for rare events
    * refits the best configuration on the full training set
    * saves the tuning table, best parameters, and the tuned model

Step 4 evaluates the tuned model on the untouched test set.

Run:
    python Sprint_2/sheetal/03_random_forest_tuning.py
    python Sprint_2/sheetal/03_random_forest_tuning.py --n-iter 10 --cv-folds 3
    python Sprint_2/sheetal/03_random_forest_tuning.py --tune-sample-size 200000   # big data
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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TARGET_COL = "is_fraud"
CV_FOLDS = 3
N_ITER = 20
MIN_TUNE_FRAUD = 30      # below this the search is too noisy to trust (warning only)

THIS_DIR = Path(__file__).resolve().parent            # Sprint_2/sheetal
SPRINT2_DIR = THIS_DIR.parent                         # Sprint_2
MODELS_DIR = SPRINT2_DIR / "models"
RESULTS_DIR = SPRINT2_DIR / "results"
PREPARED_FILE = THIS_DIR / "prepared_data" / "random_forest_prepared_data.joblib"
BASELINE_METRICS_FILE = RESULTS_DIR / "random_forest_baseline_metrics.json"
TUNING_TABLE_FILE = RESULTS_DIR / "random_forest_tuning_results.csv"
BEST_PARAMS_FILE = RESULTS_DIR / "random_forest_best_params.json"
TUNED_MODEL_FILE = MODELS_DIR / "random_forest_tuned_model.pkl"

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


def validate_training_data(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Sanity checks on the training data used for tuning."""
    if len(X_train) != len(y_train):
        raise ValueError("Feature and target lengths do not match.")
    if TARGET_COL in X_train.columns:
        raise ValueError(f"Target '{TARGET_COL}' found among the features (leakage).")
    if X_train.isna().any().any():
        raise ValueError("Training features contain NaN values.")
    if y_train.nunique() < 2:
        raise ValueError("The training target has only one class.")
    print(f"[OK] Training data valid | rows: {len(X_train):,} | features: {X_train.shape[1]} | "
          f"fraud rate: {100 * y_train.mean():.4f}%")


def stratified_subsample(X: pd.DataFrame, y: pd.Series, size: int | None, seed: int):
    """Optionally shrink the tuning set (stratified) to keep the search fast on big data."""
    if size is None or size >= len(X):
        return X, y
    if size < 1000:
        raise ValueError("--tune-sample-size must be at least 1000.")
    X_s, _, y_s, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
    print(f"[OK] Tuning on a stratified sample of {len(X_s):,} rows "
          f"(fraud: {int(y_s.sum()):,}); final model is still refit on all training rows.")
    return X_s, y_s


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def build_param_distributions(class_weight) -> dict:
    """
    Hyperparameters that matter most for a forest on imbalanced data:
      * max_depth / min_samples_leaf / min_samples_split control overfitting
      * max_features controls how different the trees are from each other
      * max_samples subsamples each bootstrap, which also speeds up training
      * class_weight: keep the step-1 choice, and try the alternative
        'balanced' variant when a weighting strategy was selected
    """
    if class_weight is None:
        weights = [None]
    else:
        weights = sorted({class_weight, "balanced", "balanced_subsample"}, key=str)
    return {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "max_samples": [None, 0.5, 0.7, 0.9],
        "class_weight": weights,
    }


def build_scoring() -> dict:
    """Metrics recorded for every candidate. Accuracy is deliberately absent."""
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
    }


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def run_search(X, y, class_weight, n_iter: int, folds: int, seed: int) -> RandomizedSearchCV:
    """Randomized search with stratified CV, refit on PR-AUC."""
    n_fraud = int(y.sum())
    if n_fraud < folds:
        raise ValueError(f"Only {n_fraud} fraud rows; need at least {folds} for {folds}-fold CV.")
    if n_fraud < MIN_TUNE_FRAUD:
        print(f"[WARN] Only {n_fraud} fraud rows in the tuning set; results will be noisy.")

    base = RandomForestClassifier(n_jobs=-1, random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=build_param_distributions(class_weight),
        n_iter=n_iter,
        scoring=build_scoring(),
        refit="pr_auc",
        cv=cv,
        random_state=seed,
        n_jobs=1,                # the forest itself already uses all cores
        verbose=1,
        error_score="raise",
        return_train_score=False,
    )
    print(f"\n[..] Randomized search: {n_iter} candidates x {folds} folds "
          f"= {n_iter * folds} model fits (refit metric: PR-AUC)")
    start = time.time()
    search.fit(X, y)
    print(f"[OK] Search finished in {(time.time() - start) / 60:.1f} min")
    return search


def tuning_table(search: RandomizedSearchCV) -> pd.DataFrame:
    """Readable table of every candidate, best PR-AUC first."""
    res = pd.DataFrame(search.cv_results_)
    keep = ["rank_test_pr_auc"]
    for m in ("pr_auc", "roc_auc", "f1", "precision", "recall"):
        keep += [f"mean_test_{m}", f"std_test_{m}"]
    keep += ["mean_fit_time", "params"]
    table = res[keep].sort_values("rank_test_pr_auc").reset_index(drop=True)
    table["params"] = table["params"].astype(str)
    return table


def select_threshold(estimator, X, y, folds: int, seed: int) -> dict:
    """
    Choose the probability threshold that maximises F1 on OUT-OF-FOLD predictions.
    Every prediction comes from a model that did not see that row, so the choice
    is not biased by training fit, and the test set is never involved.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    print("\n[..] Out-of-fold predictions for threshold selection")
    proba = cross_val_predict(clone(estimator), X, y, cv=cv,
                              method="predict_proba", n_jobs=1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    result = {
        "threshold": float(thresholds[best]),
        "oof_precision": float(precision[best]),
        "oof_recall": float(recall[best]),
        "oof_f1": float(f1[best]),
    }
    print(f"[OK] Selected threshold {result['threshold']:.4f} "
          f"(OOF precision {result['oof_precision']:.4f}, recall {result['oof_recall']:.4f}, "
          f"F1 {result['oof_f1']:.4f})")
    return result


# --------------------------------------------------------------------------- #
# Reporting and saving
# --------------------------------------------------------------------------- #
def baseline_cv_pr_auc() -> float | None:
    """Baseline CV PR-AUC from step 2, if available, for a before/after comparison."""
    if not BASELINE_METRICS_FILE.is_file():
        return None
    try:
        with open(BASELINE_METRICS_FILE, encoding="utf-8") as fh:
            cv = json.load(fh).get("cross_validation")
        return float(cv["pr_auc"]["mean"]) if cv else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def summarize_best(search: RandomizedSearchCV) -> dict:
    """Mean CV scores of the best candidate."""
    idx = search.best_index_
    res = search.cv_results_
    return {m: {"mean": float(res[f"mean_test_{m}"][idx]), "std": float(res[f"std_test_{m}"][idx])}
            for m in ("pr_auc", "roc_auc", "f1", "precision", "recall")}


def save_outputs(table: pd.DataFrame, summary: dict, model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TUNING_TABLE_FILE, index=False)
    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    joblib.dump(model, TUNED_MODEL_FILE)
    print(f"\n[OK] Tuning table saved : {TUNING_TABLE_FILE}")
    print(f"[OK] Best params saved  : {BEST_PARAMS_FILE}")
    print(f"[OK] Tuned model saved  : {TUNED_MODEL_FILE}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 2 (Sheetal): Random Forest tuning")
    parser.add_argument("--n-iter", type=int, default=N_ITER, help="Random candidates to try")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--tune-sample-size", type=int, default=None,
                        help="Tune on a stratified sample of this many training rows")
    parser.add_argument("--skip-threshold", action="store_true",
                        help="Skip out-of-fold threshold selection (keeps 0.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.n_iter < 1 or args.cv_folds < 2:
            raise ValueError("--n-iter must be >= 1 and --cv-folds must be >= 2.")

        bundle = load_prepared_data(PREPARED_FILE)
        X_train, y_train = bundle["X_train"], bundle["y_train"]
        validate_training_data(X_train, y_train)
        class_weight = bundle["imbalance"].get("class_weight")

        X_tune, y_tune = stratified_subsample(X_train, y_train, args.tune_sample_size, args.seed)
        folds = min(args.cv_folds, int(y_tune.sum()))

        search = run_search(X_tune, y_tune, class_weight, args.n_iter, folds, args.seed)
        table = tuning_table(search)
        best_scores = summarize_best(search)

        print("\n=== BEST CONFIGURATION ===")
        print(json.dumps(search.best_params_, indent=2, default=str))
        print("\nBest candidate, cross-validated on training data (mean +/- std):")
        for name, s in best_scores.items():
            print(f"  {name:<10}: {s['mean']:.4f} +/- {s['std']:.4f}")

        base_pr = baseline_cv_pr_auc()
        if base_pr is not None:
            delta = best_scores["pr_auc"]["mean"] - base_pr
            print(f"\nBaseline CV PR-AUC (step 2): {base_pr:.4f} -> tuned: "
                  f"{best_scores['pr_auc']['mean']:.4f} ({delta:+.4f})")
            if args.tune_sample_size:
                print("(note: tuned score comes from a sample, so the comparison is approximate)")

        threshold_info = {"threshold": 0.5, "note": "default; threshold selection skipped"}
        if not args.skip_threshold:
            threshold_info = select_threshold(search.best_estimator_, X_tune, y_tune,
                                              folds, args.seed)

        print("\n[..] Refitting the best configuration on the full training set")
        final_model = clone(search.best_estimator_)
        final_model.fit(X_train, y_train)
        print("[OK] Refit complete")

        summary = {
            "model": "RandomForestClassifier (tuned)",
            "random_state": args.seed,
            "search": {"n_iter": args.n_iter, "cv_folds": folds,
                       "refit_metric": "pr_auc", "tuning_rows": int(len(X_tune))},
            "best_params": search.best_params_,
            "best_cv_scores": best_scores,
            "baseline_cv_pr_auc": base_pr,
            "decision_threshold": threshold_info,
        }
        save_outputs(table, summary, final_model)
        print("\nTuning complete. Next: 04_random_forest_evaluation.py")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
