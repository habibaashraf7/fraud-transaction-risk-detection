"""Load the card-transaction data and build leakage-free features.

Dataset (Kaggle, free account): https://www.kaggle.com/datasets/kartik2112/fraud-detection
Put fraudTrain.csv and fraudTest.csv in data/raw/.

Every feature only uses information available *when the transaction happens*:
the transaction itself + that customer's PAST transactions (never future ones, never fraud labels).
Personal data (names, street, card number) and gender (a protected attribute) are never model features.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "features.pkl"

# split dates (time-based: train on the past, test on the future)
VAL_START = pd.Timestamp("2020-03-01")    # train: 2019-01-01 .. 2020-02-29
TEST_START = pd.Timestamp("2020-06-21 12:14")  # validation: 2020-03-01 .. 2020-06-21 | test: the Kaggle test file

CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos", "health_fitness",
              "home", "kids_pets", "misc_net", "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel"]

FEATURES = ["amt", "log_amt", "category", "hour", "is_night", "day_of_week", "age", "log_city_pop",
            "distance_km", "cust_prior_txns", "amt_vs_cust_avg", "amt_vs_cust_cat_avg", "hours_since_prev",
            "txns_last_24h", "amt_last_24h", "new_category_for_cust"]

FEATURE_LABELS = {
    "amt": "Amount", "log_amt": "Amount (log)", "category": "Merchant category", "hour": "Hour of day",
    "is_night": "Night-time (10pm–4am)", "day_of_week": "Day of week", "age": "Customer age",
    "gender_f": "Gender", "log_city_pop": "City population", "distance_km": "Distance to merchant",
    "cust_prior_txns": "Customer history length", "amt_vs_cust_avg": "Amount vs customer's usual",
    "amt_vs_cust_cat_avg": "Amount vs usual in this category", "hours_since_prev": "Time since previous purchase",
    "txns_last_24h": "Purchases in last 24h", "amt_last_24h": "Spend in last 24h",
    "new_category_for_cust": "First purchase in this category",
}


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def build() -> pd.DataFrame:
    files = [RAW / "fraudTrain.csv", RAW / "fraudTest.csv"]
    if not all(f.exists() for f in files):
        raise FileNotFoundError(f"Put fraudTrain.csv and fraudTest.csv from Kaggle in {RAW} (see README).")
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender", "city", "state", "lat",
            "long", "city_pop", "dob", "trans_num", "merch_lat", "merch_long", "is_fraud"]
    df = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["time", "trans_num"], kind="stable").reset_index(drop=True)

    # --- transaction-level features
    df["log_amt"] = np.log1p(df["amt"])
    df["hour"] = df["time"].dt.hour
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 4)).astype(int)
    df["day_of_week"] = df["time"].dt.dayofweek
    df["age"] = ((df["time"] - pd.to_datetime(df["dob"])).dt.days / 365.25).round(1)
    df["gender_f"] = (df["gender"] == "F").astype(int)
    df["log_city_pop"] = np.log10(df["city_pop"])
    df["distance_km"] = _haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)

    # --- customer behaviour, using ONLY that customer's previous transactions
    g = df.groupby("cc_num", sort=False)
    df["cust_prior_txns"] = g.cumcount()
    prior_sum = g["amt"].cumsum() - df["amt"]
    prior_avg = prior_sum / df["cust_prior_txns"].replace(0, np.nan)
    df["amt_vs_cust_avg"] = (df["amt"] / prior_avg).fillna(1.0)

    gc = df.groupby(["cc_num", "category"], sort=False, observed=True)
    cat_n = gc.cumcount()
    cat_avg = (gc["amt"].cumsum() - df["amt"]) / cat_n.replace(0, np.nan)
    df["amt_vs_cust_cat_avg"] = (df["amt"] / cat_avg).fillna(1.0)
    df["new_category_for_cust"] = ((cat_n == 0) & (df["cust_prior_txns"] > 0)).astype(int)

    df["hours_since_prev"] = (g["time"].diff().dt.total_seconds() / 3600).fillna(24 * 30)

    # rolling 24h window per customer, excluding the current transaction
    # sort by customer then time, so the grouped rolling output (sorted by customer, then
    # original order within each customer) lines up row-for-row with `by_cust`
    by_cust = df.sort_values(["cc_num", "time"], kind="stable")
    roll = by_cust.set_index("time").groupby("cc_num", sort=True)["amt"].rolling("24h", closed="left")
    df.loc[by_cust.index, "txns_last_24h"] = roll.count().to_numpy()
    df.loc[by_cust.index, "amt_last_24h"] = roll.sum().to_numpy()
    df[["txns_last_24h", "amt_last_24h"]] = df[["txns_last_24h", "amt_last_24h"]].fillna(0)

    df["split"] = np.where(df["time"] >= TEST_START, "test", np.where(df["time"] >= VAL_START, "val", "train"))
    keep = ["trans_num", "time", "cc_num", "merchant", "city", "state", "is_fraud", "split"] + FEATURES
    return df[keep]


def load(rebuild: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    df = build()
    with open(CACHE, "wb") as f:
        pickle.dump(df, f)
    return df


if __name__ == "__main__":
    d = load(rebuild=True)
    print(d.groupby("split").agg(rows=("is_fraud", "size"), frauds=("is_fraud", "sum"),
                                 fraud_rate=("is_fraud", "mean"), start=("time", "min"), end=("time", "max")))
    print(d.groupby("is_fraud")[["amt_vs_cust_avg", "hours_since_prev", "txns_last_24h", "is_night",
                                 "new_category_for_cust", "distance_km"]].median())
