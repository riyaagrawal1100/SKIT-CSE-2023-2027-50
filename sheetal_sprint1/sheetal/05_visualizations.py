"""
05_visualizations.py
Author: Sheetal
Task: Visualizations

What this does:
- Generates and saves charts that summarize the dataset and fraud patterns:
    1. Fraud vs non-fraud class distribution
    2. Transaction amount distribution
    3. Fraud rate by transaction category
    4. Fraud rate by hour of day
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "..", "data", "sheetal_03_validated.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "visualizations")


def plot_class_distribution(df):
    counts = df["is_fraud"].value_counts().sort_index()
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=["Not Fraud", "Fraud"], autopct="%1.2f%%", colors=["#4C72B0", "#DD8452"])
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", bbox_inches="tight")
    plt.close()


def plot_amount_distribution(df):
    plt.figure(figsize=(7, 5))
    plt.hist(df.loc[df["amt"] < 500, "amt"], bins=50, color="#4C72B0")
    plt.title("Transaction Amount Distribution (amt < 500)")
    plt.xlabel("Amount ($)")
    plt.ylabel("Count")
    plt.savefig(f"{OUTPUT_DIR}/02_amount_distribution.png", bbox_inches="tight")
    plt.close()


def plot_fraud_rate_by_category(df):
    rates = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    plt.barh(rates.index, rates.values, color="#DD8452")
    plt.title("Fraud Rate by Category")
    plt.xlabel("Fraud Rate")
    plt.gca().invert_yaxis()
    plt.savefig(f"{OUTPUT_DIR}/03_fraud_rate_by_category.png", bbox_inches="tight")
    plt.close()


def plot_fraud_rate_by_hour(df):
    hour = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    rates = df.groupby(hour)["is_fraud"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(rates.index, rates.values, color="#55A868")
    plt.title("Fraud Rate by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Rate")
    plt.savefig(f"{OUTPUT_DIR}/04_fraud_rate_by_hour.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, dtype={"zip": str, "merch_zipcode": str, "cc_num": str})

    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_fraud_rate_by_category(df)
    plot_fraud_rate_by_hour(df)

    print(f"Saved 4 visualizations to {OUTPUT_DIR}/")
