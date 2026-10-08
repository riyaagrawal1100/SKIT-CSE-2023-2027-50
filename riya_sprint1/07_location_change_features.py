"""
07_location_change_features.py
Author: Riya
Task: Location Change features (fraud signal: unusual/impossible travel between
transactions)

What this does, per customer (grouped by cc_num, sorted chronologically):
- distance_from_previous_transaction_km: haversine distance between the
  CURRENT transaction's merchant location and the customer's PREVIOUS
  transaction's merchant location (compares only with the immediately
  preceding transaction, never with future ones -> no data leakage)
- time_since_previous_transaction_hours: time gap to that previous transaction
- location_changed: 1 if the distance from the previous transaction exceeds
  a "moved to a new area" threshold (default 50 km)
- rapid_location_change: 1 if the location changed AND it happened in a very
  short time window (default < 1 hour) -- i.e. physically implausible travel
  speed, a classic card-fraud signal (transaction shows up far away, too soon
  after the last one)

A customer's very first transaction has no previous transaction to compare
against, so these features are set to 0 / not-applicable for those rows
(there's nothing suspicious about a first-ever transaction by definition).
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_05_customer_behavior.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_07_location_change.csv")

LOCATION_CHANGE_KM_THRESHOLD = 50.0     # distance beyond which we call it "changed location"
RAPID_CHANGE_HOURS_THRESHOLD = 1.0      # time window within which that change is "rapid"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_location_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Sort chronologically within each customer so "previous transaction"
    # always means the transaction that truly came right before it in time.
    df_sorted = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()

    grp = df_sorted.groupby("cc_num")
    prev_lat = grp["merch_lat"].shift(1)
    prev_long = grp["merch_long"].shift(1)
    prev_time = grp["trans_date_trans_time"].shift(1)

    has_previous = prev_lat.notna()

    distance = pd.Series(np.nan, index=df_sorted.index)
    distance[has_previous] = haversine_km(
        df_sorted.loc[has_previous, "merch_lat"], df_sorted.loc[has_previous, "merch_long"],
        prev_lat[has_previous], prev_long[has_previous],
    )
    df_sorted["distance_from_previous_transaction_km"] = distance.fillna(0.0)

    time_gap_hours = (df_sorted["trans_date_trans_time"] - prev_time).dt.total_seconds() / 3600.0
    # Use NaN (not inf) for a customer's first transaction, which has no previous
    # transaction to compare against -- keeps downstream stats (describe, mean, etc.) clean.
    df_sorted["time_since_previous_transaction_hours"] = time_gap_hours

    df_sorted["location_changed"] = (
        (df_sorted["distance_from_previous_transaction_km"] > LOCATION_CHANGE_KM_THRESHOLD)
        & has_previous
    ).astype(int)

    df_sorted["rapid_location_change"] = (
        (df_sorted["location_changed"] == 1)
        & (df_sorted["time_since_previous_transaction_hours"] < RAPID_CHANGE_HOURS_THRESHOLD)
    ).astype(int)

    # Restore the original row order
    df_out = df_sorted.sort_index()
    return df_out


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    print(f"Loaded data: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_out = add_location_change_features(df)

    new_cols = ["distance_from_previous_transaction_km", "time_since_previous_transaction_hours",
                "location_changed", "rapid_location_change"]
    print(f"Added features: {new_cols}")
    print(df_out[new_cols].describe())
    print(f"\nlocation_changed = 1 for {df_out['location_changed'].sum():,} rows")
    print(f"rapid_location_change = 1 for {df_out['rapid_location_change'].sum():,} rows")

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with location-change features to {OUTPUT_PATH}")
