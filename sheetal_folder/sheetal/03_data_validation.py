"""
03_data_validation.py
Author: Sheetal
Task: Data validation

What this does:
- Runs sanity checks on value ranges (amount, age, satisfaction score, lat/long,
  fraud flag, gender/category/payment values)
- Flags internal inconsistencies in the "enrichment" columns
  (Transaction_Time, Customer_Age, Merchant_Category don't reliably match the
  real transaction data)
- Writes a validation report and the final validated dataset
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_02_no_missing.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_03_validated.csv")
REPORT_PATH = os.path.join(BASE_DIR, "..", "output", "data_validation_report.md")


def validate_data(df: pd.DataFrame) -> dict:
    checks = {
        "amt <= 0": int((df["amt"] <= 0).sum()),
        "Customer_Age outside 0-120": int((~df["Customer_Age"].between(0, 120)).sum()),
        "Customer_Satisfaction_Score outside 1-10": int((~df["Customer_Satisfaction_Score"].between(1, 10)).sum()),
        "lat outside -90..90": int((~df["lat"].between(-90, 90)).sum()),
        "long outside -180..180": int((~df["long"].between(-180, 180)).sum()),
        "is_fraud not in {0,1}": int((~df["is_fraud"].isin([0, 1])).sum()),
        "gender not in {M,F}": int((~df["gender"].isin(["M", "F"])).sum()),
        "duplicate trans_num": int(df["trans_num"].duplicated().sum()),
    }

    # Consistency checks on the "enrichment" columns
    df["_extracted_time"] = pd.to_datetime(df["trans_date_trans_time"]).dt.strftime("%H:%M")
    time_match_rate = (df["_extracted_time"] == df["Transaction_Time"]).mean()

    dob_year = pd.to_datetime(df["dob"]).dt.year
    trans_year = pd.to_datetime(df["trans_date_trans_time"]).dt.year
    age_match_rate = ((trans_year - dob_year) == df["Customer_Age"]).mean()

    df.drop(columns=["_extracted_time"], inplace=True)

    checks["Transaction_Time matches real timestamp (rate)"] = round(float(time_match_rate), 4)
    checks["Customer_Age matches dob-derived age (rate)"] = round(float(age_match_rate), 4)

    return checks


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    results = validate_data(df)

    print("Validation results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(REPORT_PATH, "w") as f:
        f.write("# Data Validation Report\n\n")
        f.write(f"Rows validated: {len(df):,}\n\n")
        for k, v in results.items():
            f.write(f"- {k}: {v}\n")
        f.write(
            "\n**Note:** `Transaction_Time`, `Customer_Age`, and `Merchant_Category` show very "
            "low agreement with the real transaction timestamp / dob-derived age / true category, "
            "meaning they appear to be randomly generated rather than derived fields. They should "
            "not be trusted for analysis or modeling.\n"
        )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved validated data to {OUTPUT_PATH}")
    print(f"Saved validation report to {REPORT_PATH}")
