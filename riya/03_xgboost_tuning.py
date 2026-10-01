"""
03_xgboost_tuning.py
Author: Riya
Sprint 2 — XGBoost hyperparameter tuning (cross-validation on TRAIN only)

What this does:
1. Loads the full training split (X_train, y_train) from 01_data_preparation.py.
   The test set is never loaded in this file.
2. Runs `RandomizedSearchCV` with `StratifiedKFold` cross-validation, scored
   on PR-AUC (average precision) — a much more informative metric than
   accuracy or even ROC-AUC for a ~0.6%-fraud dataset, since PR-AUC focuses
   on how well the model ranks/precision-targets the rare positive class.
3. `scale_pos_weight` is fixed for the search using the full training set's
   class ratio (train-only, computed once here — not recomputed per fold,
   which keeps the search fast; this is a standard simplification since the
   class ratio is very stable across folds of the same stratified split).
4. Cross-validation itself guards against overfitting the hyperparameters
   to one particular train/validation split — every candidate parameter set
   is scored across several stratified folds of the TRAINING data only.
5. Saves the best hyperparameters and the full CV results table to
   Sprint_2/results/, for 04_xgboost_evaluation.py to pick up and use to
   train the final model.

Note on search size: N_ITER / N_FOLDS below are set to a modest size so the
full cross-validated search finishes in a reasonable time on modest hardware
(single-core CI/grading environments included). Widen the parameter ranges
and/or raise these constants if more compute is available.
"""

import json
import os
import sys

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SPRINT2_DIR = os.path.join(PROJECT_ROOT, "Sprint_2")

DATA_DIR = os.path.join(SPRINT2_DIR, "data")
RESULTS_DIR = os.path.join(SPRINT2_DIR, "results")

RANDOM_STATE = 42
N_ITER = 12
N_FOLDS = 3


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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train = load_training_data()
    print(f"Loaded training data: {X_train.shape[0]:,} rows x {X_train.shape[1]} columns")

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(f"Training class balance -> fraud: {n_pos:,} | legit: {n_neg:,}")
    print(f"scale_pos_weight fixed for the search: {scale_pos_weight:.2f}")

    # n_jobs=1 on the estimator itself (parallelism is handled at the
    # RandomizedSearchCV level below) to avoid CPU oversubscription between
    # the outer search loop and each individual XGBoost fit.
    base_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    param_distributions = {
        "n_estimators": randint(100, 350),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.03, 0.27),        # 0.03 - 0.30
        "subsample": uniform(0.6, 0.4),               # 0.6 - 1.0
        "colsample_bytree": uniform(0.6, 0.4),         # 0.6 - 1.0
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0.0, 0.5),
        "reg_alpha": uniform(0.0, 1.0),
        "reg_lambda": uniform(0.5, 1.5),
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="average_precision",  # PR-AUC — appropriate for severe imbalance
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=False,  # 04_xgboost_evaluation.py retrains the final model explicitly
    )

    print(
        f"\nRunning RandomizedSearchCV: {N_ITER} candidates x {N_FOLDS} folds "
        f"= {N_ITER * N_FOLDS} fits, scored on PR-AUC (average_precision)..."
    )
    search.fit(X_train, y_train)

    print(f"\nBest CV PR-AUC: {search.best_score_:.4f}")
    print("Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_params = dict(search.best_params_)
    best_params["scale_pos_weight"] = scale_pos_weight

    with open(os.path.join(RESULTS_DIR, "xgboost_best_params.json"), "w") as f:
        json.dump(
            {
                "best_cv_pr_auc": float(search.best_score_),
                "best_params": best_params,
                "cv_folds": N_FOLDS,
                "search_iterations": N_ITER,
                "scoring": "average_precision (PR-AUC)",
            },
            f,
            indent=2,
        )
    print(f"\nSaved best hyperparameters to {RESULTS_DIR}/xgboost_best_params.json")

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results = cv_results.sort_values("rank_test_score")
    cv_results_path = os.path.join(RESULTS_DIR, "xgboost_cv_results.csv")
    cv_results.to_csv(cv_results_path, index=False)
    print(f"Saved full CV results table to {cv_results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
