"""
01_data_processing.py
Author: Riya
Task: Data processing

What this does:
- Takes the cleaned, validated dataset (handed off from Sheetal's pipeline)
- Drops columns confirmed unreliable during validation (Transaction_Time,
  Customer_Age, Merchant_Category)
- Drops pure-identifier / free-text / very-high-cardinality columns that
  carry no reusable modeling signal (trans_num, unix_time, first, last,
  street, job, city, merchant)
- NOTE: `cc_num` is deliberately KEPT at this stage (even though it's an ID)
  because 05_customer_behavior_features.py needs it to group transactions by
  customer. It gets dropped later, in 02_feature_engineering.py, after those
  customer-level features are computed.
- Prepares the base dataframe that all later feature-engineering steps build on
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_03_validated.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_01_processed.csv")

UNRELIABLE_COLS = ["Transaction_Time", "Customer_Age", "Merchant_Category"]
ID_COLS = ["trans_num", "unix_time", "first", "last", "street", "job", "city", "merchant"]


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in UNRELIABLE_COLS if c in df.columns])
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str})
    print(f"Loaded validated data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_processed = process_data(df)
    print(f"After processing: {df_processed.shape[0]:,} rows x {df_processed.shape[1]} columns")
    print(f"Dropped columns: {UNRELIABLE_COLS + ID_COLS}")

    df_processed.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed data to {OUTPUT_PATH}")
