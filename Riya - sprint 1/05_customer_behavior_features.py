"""
05_customer_behavior_features.py
Author: Riya
Task: Customer behavior features

What this does:
- Builds per-customer aggregate features that describe a customer's normal
  spending pattern, grouped by `cc_num` (each card = one customer):
    - customer_txn_count: how many transactions this customer has in the data
    - customer_avg_amt: this customer's average transaction amount
    - amt_vs_customer_avg: how far the current transaction's amount is from
      that customer's own average (unusual-for-this-customer spending is a
      strong fraud signal)
- `cc_num` itself is kept in the output here; it gets dropped in
  02_feature_engineering.py once these features have been computed, since
  the raw card number has no modeling value on its own.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_04_geo.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_05_customer_behavior.csv")

CUSTOMER_KEY = "cc_num"


def add_customer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    customer_stats = df.groupby(CUSTOMER_KEY)["amt"].agg(
        customer_txn_count="count",
        customer_avg_amt="mean",
    )
    df = df.merge(customer_stats, on=CUSTOMER_KEY, how="left")
    df["amt_vs_customer_avg"] = df["amt"] - df["customer_avg_amt"]
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str})
    df_out = add_customer_behavior_features(df)

    new_cols = ["customer_txn_count", "customer_avg_amt", "amt_vs_customer_avg"]
    print(f"Added customer behavior features: {new_cols}")
    print(df_out[new_cols].describe())

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with customer behavior features to {OUTPUT_PATH}")
