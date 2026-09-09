"""
02_missing_value_handling.py
Author: Sheetal
Task: Missing value handling

What this does:
- Checks every column for missing values
- Imputes missing `merch_zipcode` values using nearest-neighbor matching on
  merchant lat/long (each missing zip is filled with the zip of the
  geographically closest merchant transaction in the dataset)
- Adds a `merch_zipcode_missing` flag column so imputed rows stay identifiable
"""

import os
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_01_cleaned.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_02_no_missing.csv")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    print("Missing values before handling:")
    print(missing_summary if len(missing_summary) else "  None")

    # Flag rows where merch_zipcode was originally missing
    df["merch_zipcode_missing"] = df["merch_zipcode"].isna()

    known_mask = df["merch_zipcode"].notna()
    missing_mask = df["merch_zipcode"].isna()

    if missing_mask.sum() > 0:
        known_coords = df.loc[known_mask, ["merch_lat", "merch_long"]].to_numpy()
        known_zips = df.loc[known_mask, "merch_zipcode"].to_numpy()
        missing_coords = df.loc[missing_mask, ["merch_lat", "merch_long"]].to_numpy()

        tree = cKDTree(known_coords)
        _, nearest_idx = tree.query(missing_coords, k=1)
        df.loc[missing_mask, "merch_zipcode"] = known_zips[nearest_idx]

    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    df_filled = handle_missing_values(df)

    print("\nMissing values after handling:")
    remaining = df_filled.isna().sum()
    remaining = remaining[remaining > 0]
    print(remaining if len(remaining) else "  None - fully clean")

    df_filled.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with missing values handled to {OUTPUT_PATH}")
