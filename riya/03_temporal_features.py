"""
03_temporal_features.py
Author: Riya
Task: Temporal features

What this does:
- Computes the customer's real age from dob + transaction date
  (replaces the unreliable Customer_Age column that was dropped earlier)
- Extracts hour, day-of-week, month, and a weekend flag from the
  transaction timestamp — fraud often follows a time-of-day / weekday pattern
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_01_processed.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_03_temporal.csv")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    trans_dt = pd.to_datetime(df["trans_date_trans_time"])
    dob_dt = pd.to_datetime(df["dob"])

    df["customer_age"] = (trans_dt - dob_dt).dt.days // 365
    df["trans_hour"] = trans_dt.dt.hour
    df["trans_day_of_week"] = trans_dt.dt.dayofweek  # 0 = Monday
    df["trans_month"] = trans_dt.dt.month
    df["is_weekend"] = df["trans_day_of_week"].isin([5, 6]).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str})
    df_out = add_temporal_features(df)

    new_cols = ["customer_age", "trans_hour", "trans_day_of_week", "trans_month", "is_weekend"]
    print(f"Added temporal features: {new_cols}")
    print(df_out[new_cols].head())

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with temporal features to {OUTPUT_PATH}")
