"""
04_geospatial_features.py
Author: Riya
Task: Geospatial features

What this does:
- Computes the haversine distance (in km) between the customer's home
  location (lat/long) and the merchant's location (merch_lat/merch_long)
- Large customer-merchant distances are a common fraud signal
  (transaction happening far from where the customer actually lives)
"""

import numpy as np
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_03_temporal.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "riya_04_geo.csv")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    df["customer_merchant_distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str})
    df_out = add_geospatial_features(df)

    print("Added feature: customer_merchant_distance_km")
    print(df_out["customer_merchant_distance_km"].describe())

    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved data with geospatial features to {OUTPUT_PATH}")
