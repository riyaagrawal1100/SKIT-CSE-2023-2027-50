"""
02_feature_engineering.py
Author: Riya
Task: Feature engineering

What this does:
This is the master script that ties the individual feature modules together
into one final, model-ready dataset:
    1. Start from the processed data (01_data_processing.py output)
    2. Add temporal features (03_temporal_features.py)
    3. Add geospatial features (04_geospatial_features.py)
    4. Add customer behavior features (05_customer_behavior_features.py)
    5. Drop `cc_num` and the raw `trans_date_trans_time`/`dob` columns, now
       that everything useful has been extracted from them
    6. Encode remaining categorical columns (gender, category, state,
       Transaction_Type, Payment_Method) into numeric/one-hot form
    7. Move the target column (is_fraud) to the end

Class imbalance and feature scaling are intentionally NOT applied here —
both should be fit only on the training split (after train_test_split) to
avoid data leakage. See 06_documentation.md for the recommended next steps.
"""

import os
import pandas as pd

# Local module filenames start with a number (01_, 03_, ...) so they can't be
# imported with a plain `import` statement — load them dynamically instead.
import importlib.util


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_01_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "final_preprocessed_dataset.csv")

temporal_mod = _load_module(os.path.join(BASE_DIR, "03_temporal_features.py"), "temporal_features")
geo_mod = _load_module(os.path.join(BASE_DIR, "04_geospatial_features.py"), "geospatial_features")
behavior_mod = _load_module(os.path.join(BASE_DIR, "05_customer_behavior_features.py"), "customer_behavior_features")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = temporal_mod.add_temporal_features(df)
    df = geo_mod.add_geospatial_features(df)
    df = behavior_mod.add_customer_behavior_features(df)

    # Drop columns no longer needed now that their information has been extracted
    df = df.drop(columns=["cc_num", "trans_date_trans_time", "dob", "zip", "merch_zipcode",
                           "merch_zipcode_missing"])

    # Encode categoricals
    df["gender"] = df["gender"].map({"M": 0, "F": 1})
    df = pd.get_dummies(df, columns=["category", "state", "Transaction_Type", "Payment_Method"],
                         drop_first=True)

    # Move target column to the end
    target = "is_fraud"
    df = df[[c for c in df.columns if c != target] + [target]]

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    print(f"Loaded processed data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_final = engineer_features(df)
    print(f"After feature engineering: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")
    print(f"Missing values: {df_final.isna().sum().sum()}")

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final model-ready dataset to {OUTPUT_PATH}")
