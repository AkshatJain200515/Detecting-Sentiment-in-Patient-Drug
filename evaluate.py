"""
evaluate.py

Loads baseline_results.json (written by train_baseline.py) and prints a
side-by-side comparison table. If transformer metrics are supplied, they
are folded into the same comparison for the report's evaluation section.
"""

import argparse
import json
import os

OUTPUT_DIR = "outputs"


def load_baseline_results():
    path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Run train_baseline.py first.")
    with open(path) as f:
        return json.load(f)


def print_comparison_table(results: dict, transformer_metrics: dict = None):
    header = f"{'Model':<25}{'Accuracy':<12}{'Macro F1':<12}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<25}{r['accuracy']:<12.3f}{r['f1_macro']:<12.3f}")
    if transformer_metrics:
        acc = transformer_metrics.get("eval_accuracy", float("nan"))
        f1 = transformer_metrics.get("eval_f1_macro", float("nan"))
        print(f"{'distilbert':<25}{acc:<12.3f}{f1:<12.3f}")


def top_misclassified_examples(y_true, y_pred, texts, n=5):
    """Utility for the report's error-analysis / discussion section."""
    mismatches = [
        (t, yt, yp) for t, yt, yp in zip(texts, y_true, y_pred) if yt != yp
    ]
    return mismatches[:n]


if __name__ == "__main__":
    results = load_baseline_results()
    print_comparison_table(results)
