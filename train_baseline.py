"""
train_baseline.py

Baseline sentiment classifier for depression-related drug reviews.

Approach: TF-IDF vectorisation + two classical ML models (Logistic
Regression and Linear SVM), chosen as the baseline because they are fast
to train, interpretable (inspectable coefficients/top features), and a
strong reference point before investing in a heavier transformer model
(see train_transformer.py and the report's model-comparison section).

Usage:
    python src/train_baseline.py --condition Depression
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from data_loader import load_condition_dataset
from preprocess import clean_dataframe

OUTPUT_DIR = "outputs"


def run(condition: str = "Depression", test_size: float = 0.2, random_state: int = 42):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load + filter + label
    df = load_condition_dataset(condition)
    df = clean_dataframe(df)

    if df["sentiment"].nunique() < 2 or len(df) < 10:
        raise ValueError(
            "Not enough labeled data to train. If you're on the synthetic "
            "sample, add more rows or plug in the real dataset."
        )

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Vectorise
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 3. Train two candidate models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_state),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, output_dict=True)

        results[name] = {"accuracy": acc, "f1_macro": f1_macro, "report": report}

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.3f} | Macro F1: {f1_macro:.3f}")
        print(classification_report(y_test, preds))

        # Confusion matrix plot
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, preds, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close(fig)

        joblib.dump(model, os.path.join(OUTPUT_DIR, f"model_{name}.joblib"))

    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))

    with open(os.path.join(OUTPUT_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 4. Best model's top features (interpretability, useful for the report)
    best_name = max(results, key=lambda k: results[k]["f1_macro"])
    print(f"\nBest baseline model: {best_name}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Depression")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    run(condition=args.condition, test_size=args.test_size)
