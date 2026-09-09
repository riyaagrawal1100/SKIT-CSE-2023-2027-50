"""
04_behavioral_analysis.py
Author: Sheetal
Task: Behavioral analysis

What this does:
- Explores fraud rate patterns across category, gender, payment method,
  transaction type, and hour of day
- Looks at spending amount differences between fraud and non-fraud transactions
- Writes a short analysis report summarizing the findings
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_03_validated.csv")
REPORT_PATH = os.path.join(BASE_DIR, "..", "output", "behavioral_analysis_report.md")


def analyze(df: pd.DataFrame) -> str:
    lines = []
    overall_rate = df["is_fraud"].mean()
    lines.append(f"Overall fraud rate: {overall_rate:.3%} ({df['is_fraud'].sum():,} of {len(df):,} transactions)\n")

    lines.append("\n## Fraud rate by category\n")
    by_cat = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
    for cat, rate in by_cat.items():
        lines.append(f"- {cat}: {rate:.3%}\n")

    lines.append("\n## Fraud rate by gender\n")
    by_gender = df.groupby("gender")["is_fraud"].mean()
    for g, rate in by_gender.items():
        lines.append(f"- {g}: {rate:.3%}\n")

    lines.append("\n## Fraud rate by payment method\n")
    by_pay = df.groupby("Payment_Method")["is_fraud"].mean().sort_values(ascending=False)
    for pm, rate in by_pay.items():
        lines.append(f"- {pm}: {rate:.3%}\n")

    lines.append("\n## Fraud rate by transaction type\n")
    by_type = df.groupby("Transaction_Type")["is_fraud"].mean().sort_values(ascending=False)
    for tt, rate in by_type.items():
        lines.append(f"- {tt}: {rate:.3%}\n")

    lines.append("\n## Fraud rate by hour of day\n")
    hour = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    by_hour = df.groupby(hour)["is_fraud"].mean().sort_values(ascending=False).head(5)
    lines.append("Top 5 riskiest hours:\n")
    for h, rate in by_hour.items():
        lines.append(f"- {h}:00 -> {rate:.3%}\n")

    lines.append("\n## Amount comparison\n")
    avg_fraud_amt = df.loc[df["is_fraud"] == 1, "amt"].mean()
    avg_normal_amt = df.loc[df["is_fraud"] == 0, "amt"].mean()
    lines.append(f"- Average amount on fraud transactions: {avg_fraud_amt:.2f}\n")
    lines.append(f"- Average amount on normal transactions: {avg_normal_amt:.2f}\n")

    return "".join(lines)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})
    summary = analyze(df)
    print(summary)

    with open(REPORT_PATH, "w") as f:
        f.write("# Customer Behavioral Analysis\n\n")
        f.write(summary)

    print(f"\nSaved behavioral analysis report to {REPORT_PATH}")
