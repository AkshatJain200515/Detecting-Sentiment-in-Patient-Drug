"""
train_transformer.py

Stronger model: fine-tunes a lightweight pretrained transformer
(distilbert-base-uncased) on the same 3-class sentiment task, to compare
against the TF-IDF + classical ML baseline in train_baseline.py.

This is the more expensive step in the pipeline (needs `transformers`,
`torch`, ideally a GPU for the full ~20-30k-row Depression subset). On a
CPU-only laptop, reduce --epochs and --max-samples for a quick smoke test.

Usage:
    python src/train_transformer.py --condition Depression --epochs 2
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_condition_dataset
from preprocess import clean_dataframe

OUTPUT_DIR = "outputs"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def run(condition: str = "Depression", epochs: int = 2, max_samples: int = None,
        model_name: str = "distilbert-base-uncased"):
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing transformer dependencies. Install with:\n"
            "  pip install torch transformers datasets --break-system-packages\n"
            f"Original error: {e}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_condition_dataset(condition)
    df = clean_dataframe(df)
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    df["label"] = df["sentiment"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_ds = Dataset.from_pandas(train_df[["clean_review", "label"]])
    test_ds = Dataset.from_pandas(test_df[["clean_review", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["clean_review"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "transformer_checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        learning_rate=2e-5,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final transformer eval metrics:", metrics)

    model.save_pretrained(os.path.join(OUTPUT_DIR, "distilbert_sentiment_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "distilbert_sentiment_model"))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="Depression")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run(condition=args.condition, epochs=args.epochs, max_samples=args.max_samples)
