"""
07_odd_timing_features.py
Author: Sheetal
Task: "Odd Timing" features (fraud signal: a transaction happening at a time
that's unusual for that specific customer, or in general at night)

What this does:
- night_transaction: 1 if the transaction happened between 12 AM and 5 AM
  (a common high-risk window for fraud, when the real cardholder is usually
  asleep)
- customer_usual_transaction_hour: this customer's average transaction hour,
  calculated using ONLY that customer's transactions strictly BEFORE the
  current one (historical only, no leakage from the current/future transactions)
- hour_deviation_from_normal_behavior: how far the current transaction's hour
  is from the customer's usual hour, using a 24-hour circular distance
  (so e.g. 11 PM vs 1 AM is treated as close together (2 hours), not far apart)
- unusual_transaction_hour: 1 if that deviation is large (default > 6 hours
  circular distance from the customer's normal transaction time)

A customer's first-ever transaction has no history to compare against, so
`hour_deviation_from_normal_behavior` defaults to 0 and `unusual_transaction_hour`
defaults to 0 for those rows.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_03_validated.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_07_odd_timing.csv")

NIGHT_START_HOUR = 0   # 12 AM
NIGHT_END_HOUR = 5     # up to, but not including, 5 AM
UNUSUAL_HOUR_DEVIATION_THRESHOLD = 6  # hours (circular distance)


def circular_hour_distance(h1, h2):
    """Distance between two hours-of-day on a 24-hour clock (e.g. 23 vs 1 -> 2, not 22)."""
    diff = (h1 - h2).abs()
    return np.minimum(diff, 24 - diff)


def add_odd_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour

    df["night_transaction"] = df["trans_hour"].between(
        NIGHT_START_HOUR, NIGHT_END_HOUR - 1
    ).astype(int)

    df_sorted = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()

    grp = df_sorted.groupby("cc_num")["trans_hour"]
    cum_sum_incl_current = grp.cumsum()
    cum_count_incl_current = grp.cumcount() + 1

    sum_before_current = cum_sum_incl_current - df_sorted["trans_hour"]
    count_before_current = cum_count_incl_current - 1
    has_history = count_before_current > 0

    usual_hour = sum_before_current / count_before_current.replace(0, np.nan)
    df_sorted["customer_usual_transaction_hour"] = usual_hour  # NaN for first-ever transaction

    deviation = circular_hour_distance(df_sorted["trans_hour"], usual_hour)
    df_sorted["hour_deviation_from_normal_behavior"] = deviation.where(has_history, 0.0)

    df_sorted["unusual_transaction_hour"] = (
        (df_sorted["hour_deviation_from_normal_behavior"] > UNUSUAL_HOUR_DEVIATION_THRESHOLD)
        & has_history
    ).astype(int)

    df_out = df_sorted.sort_index()
    return df_out


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_out = add_odd_timing_features(df)

    new_cols = ["night_transaction", "customer_usual_transaction_hour",
                "hour_deviation_from_normal_behavior", "unusual_transaction_hour"]
    print(f"Added features: {new_cols}")
    print(df_out[new_cols].describe())
    print(f"\nnight_transaction = 1 for {df_out['night_transaction'].sum():,} rows")
    print(f"unusual_transaction_hour = 1 for {df_out['unusual_transaction_hour'].sum():,} rows")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with odd-timing features to {OUTPUT_PATH}")
