"""
08_transaction_frequency_features.py
Author: Sheetal
Task: "Frequent Back-to-Back Transactions" features (fraud signal: many
transactions from the same card in a very short window -- a classic card
testing / rapid-fraud pattern)

What this does, per customer (grouped by cc_num):
- txn_count_last_5min / txn_count_last_30min / txn_count_last_60min: how
  many of that customer's OTHER transactions happened in the trailing
  5 / 30 / 60 minutes before the current one (the current transaction itself
  is excluded from its own count -- this only looks backward in time, so
  there is no leakage from future transactions)
- transaction_velocity: transactions per hour for this customer, based on
  the trailing 60-minute count (a simple, interpretable "how fast is this
  card being used right now" measure)
- rapid_transaction_flag: 1 if there were 2 or more other transactions from
  this same card in the last 5 minutes (i.e. this transaction is itself part
  of a back-to-back burst)
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_07_odd_timing.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_08_frequency.csv")

RAPID_TXN_COUNT_THRESHOLD = 2  # other transactions within 5 minutes to trigger the flag


def _trailing_count(df_sorted: pd.DataFrame, window: str) -> pd.Series:
    """Count of a customer's transactions in the trailing `window`,
    excluding the current transaction itself (closed='left')."""
    counts = (
        df_sorted
        .set_index("trans_date_trans_time")
        .groupby("cc_num")["amt"]
        .rolling(window, closed="left")
        .count()
    )
    # counts has a (cc_num, trans_date_trans_time) MultiIndex; align back to df_sorted's row order
    return counts.reset_index(drop=True).set_axis(df_sorted.index)


def add_transaction_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    df_sorted = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()

    df_sorted["txn_count_last_5min"] = _trailing_count(df_sorted, "5min").fillna(0).astype(int)
    df_sorted["txn_count_last_30min"] = _trailing_count(df_sorted, "30min").fillna(0).astype(int)
    df_sorted["txn_count_last_60min"] = _trailing_count(df_sorted, "60min").fillna(0).astype(int)

    df_sorted["transaction_velocity"] = df_sorted["txn_count_last_60min"]  # transactions/hour

    df_sorted["rapid_transaction_flag"] = (
        df_sorted["txn_count_last_5min"] >= RAPID_TXN_COUNT_THRESHOLD
    ).astype(int)

    df_out = df_sorted.sort_index()
    return df_out


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_out = add_transaction_frequency_features(df)

    new_cols = ["txn_count_last_5min", "txn_count_last_30min", "txn_count_last_60min",
                "transaction_velocity", "rapid_transaction_flag"]
    print(f"Added features: {new_cols}")
    print(df_out[new_cols].describe())
    print(f"\nrapid_transaction_flag = 1 for {df_out['rapid_transaction_flag'].sum():,} rows")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with transaction-frequency features to {OUTPUT_PATH}")
