"""
08_transaction_amount_features.py
Author: Riya
Task: "Suddenly Large Amount" features (fraud signal: a transaction that's
unusually large compared to what this specific customer normally spends)

What this does, per customer (grouped by cc_num, sorted chronologically):
- customer_avg_amt_so_far: the customer's average spend calculated using
  ONLY transactions strictly BEFORE the current one (an expanding/historical
  average, not the customer's full-dataset average) -- this avoids leaking
  information from the current or future transactions into the baseline
- customer_relative_amount_deviation: amt - customer_avg_amt_so_far
  (how many dollars above/below their normal spend this transaction is)
- amount_ratio_to_normal_spending: amt / customer_avg_amt_so_far
  (e.g. 5.0 means "5x this customer's usual spend")
- large_transaction_flag: 1 if amount_ratio_to_normal_spending exceeds a
  threshold (default 3x) -- a simple, explainable "sudden large amount" rule

For a customer's first transaction (no prior history), there's no historical
average to compare against yet, so the ratio defaults to 1.0 (no deviation)
and the flag defaults to 0, rather than guessing.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_07_location_change.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_08_amount_anomaly.csv")

LARGE_TXN_RATIO_THRESHOLD = 3.0  # flag if amount is 3x+ this customer's historical average


def add_transaction_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df_sorted = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()

    grp = df_sorted.groupby("cc_num")["amt"]
    # Sum and count of the SAME customer's transactions strictly before the
    # current row (cumsum/cumcount include the current row, so we subtract it out).
    cum_sum_incl_current = grp.cumsum()
    cum_count_incl_current = grp.cumcount() + 1

    sum_before_current = cum_sum_incl_current - df_sorted["amt"]
    count_before_current = cum_count_incl_current - 1

    historical_avg = sum_before_current / count_before_current.replace(0, np.nan)

    has_history = count_before_current > 0
    df_sorted["customer_avg_amt_so_far"] = historical_avg  # NaN for first-ever transaction

    deviation = df_sorted["amt"] - historical_avg
    ratio = df_sorted["amt"] / historical_avg.replace(0, np.nan)

    # No history yet -> no deviation to measure (default to "normal": 0 deviation, ratio 1)
    df_sorted["customer_relative_amount_deviation"] = deviation.where(has_history, 0.0)
    df_sorted["amount_ratio_to_normal_spending"] = ratio.where(has_history, 1.0).fillna(1.0)

    df_sorted["large_transaction_flag"] = (
        (df_sorted["amount_ratio_to_normal_spending"] >= LARGE_TXN_RATIO_THRESHOLD)
        & has_history
    ).astype(int)

    df_out = df_sorted.sort_index()
    return df_out


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_out = add_transaction_amount_features(df)

    new_cols = ["customer_avg_amt_so_far", "customer_relative_amount_deviation",
                "amount_ratio_to_normal_spending", "large_transaction_flag"]
    print(f"Added features: {new_cols}")
    print(df_out[new_cols].describe())
    print(f"\nlarge_transaction_flag = 1 for {df_out['large_transaction_flag'].sum():,} rows")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with transaction-amount features to {OUTPUT_PATH}")
