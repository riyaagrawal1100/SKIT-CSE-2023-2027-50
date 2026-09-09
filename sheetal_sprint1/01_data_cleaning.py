"""
01_data_cleaning.py
Author: Sheetal
Task: Data cleaning

What this does:
- Drops the leftover/meaningless index column
- Removes exact duplicate rows and duplicate transaction IDs (if any)
- Fixes data types (dates -> datetime, cc_num/zip -> string so leading zeros aren't lost)
- Zero-pads zip codes back to 5 digits
- Trims stray whitespace / standardizes casing on text columns
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "raw_fraud_data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_01_cleaned.csv")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the leftover pandas index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove exact duplicate rows and duplicate transaction IDs
    df = df.drop_duplicates()
    if "trans_num" in df.columns:
        df = df.drop_duplicates(subset="trans_num")

    # Fix data types
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["cc_num"] = df["cc_num"].astype(str)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["merch_zipcode"] = df["merch_zipcode"].apply(
        lambda x: str(int(x)).zfill(5) if pd.notna(x) else np.nan
    )

    # Trim whitespace / standardize casing on text columns
    text_cols = ["merchant", "first", "last", "gender", "street", "city", "state",
                 "job", "category", "Merchant_Category", "Transaction_Type", "Payment_Method"]
    for c in text_cols:
        df[c] = df[c].str.strip()
    df["state"] = df["state"].str.upper()

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded raw data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_clean = clean_data(df)
    print(f"After cleaning: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned data to {OUTPUT_PATH}")
