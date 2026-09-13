"""
data_loader.py

Loads the UCI ML Drug Review dataset (Drugs.com) and filters it down to a
single medical condition, per the project's narrowed scope.

Expected source columns (matches the real Kaggle/UCI file):
    uniqueID, drugName, condition, review, rating, date, usefulCount

Real dataset (not included in this repo due to size/licensing):
    https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

Until you download it, this module will fall back to the small synthetic
sample in data/sample_depression_reviews.csv so the pipeline can be run
and tested end to end.
"""

import os
import pandas as pd

REAL_DATA_CANDIDATES = [
    "data/drugsComTrain_raw.tsv",
    "data/drugsComTest_raw.tsv",
    "data/drugsComTrain_raw.csv",
]
SAMPLE_DATA_PATH = "data/sample_depression_reviews.csv"


def _read_any(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def load_raw_dataset() -> pd.DataFrame:
    """
    Loads whichever dataset is available: the real UCI file if present,
    otherwise the bundled synthetic sample (with a warning).
    """
    for path in REAL_DATA_CANDIDATES:
        if os.path.exists(path):
            print(f"[data_loader] Loading real dataset from {path}")
            return _read_any(path)

    print(
        "[data_loader] WARNING: real dataset not found in data/. "
        f"Falling back to synthetic sample at {SAMPLE_DATA_PATH}. "
        "Download the real file from "
        "https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018 "
        "and place drugsComTrain_raw.tsv in data/ to use the full dataset."
    )
    return _read_any(SAMPLE_DATA_PATH)


def filter_by_condition(df: pd.DataFrame, condition: str = "Depression") -> pd.DataFrame:
    """
    Narrows the dataset to a single condition, as justified in the report:
    isolating one condition keeps vocabulary and sentiment patterns coherent
    and avoids diluting the model with unrelated drug/condition pairs.
    """
    condition_mask = df["condition"].astype(str).str.strip().str.lower() == condition.lower()
    filtered = df.loc[condition_mask].copy()
    filtered = filtered.dropna(subset=["review", "rating"])
    filtered = filtered.drop_duplicates(subset=["review"])
    print(f"[data_loader] Filtered to condition='{condition}': {len(filtered)} rows")
    return filtered


def rating_to_sentiment(rating: float) -> str:
    """
    Buckets the 1-10 rating scale into three sentiment classes.
    Thresholds (1-4 negative, 5-6 neutral, 7-10 positive) follow the
    convention used in prior published work on this dataset (Gräßer et al.,
    2018) and are discussed/justified further in the report.
    """
    if rating <= 4:
        return "negative"
    elif rating <= 6:
        return "neutral"
    else:
        return "positive"


def load_condition_dataset(condition: str = "Depression") -> pd.DataFrame:
    df = load_raw_dataset()
    df = filter_by_condition(df, condition)
    df["sentiment"] = df["rating"].apply(rating_to_sentiment)
    return df


if __name__ == "__main__":
    data = load_condition_dataset("Depression")
    print(data["sentiment"].value_counts())
    print(data.head())
