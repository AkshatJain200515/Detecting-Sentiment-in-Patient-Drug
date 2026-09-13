# Depression Medication Sentiment Classifier

Individual project for **B198c7 AI Applications for Digital Business**

Classifies patient drug reviews for **Depression** medications into
**negative / neutral / positive** sentiment, using the UCI ML Drug Review
dataset. Framed as a monitoring tool a telehealth or pharma company could
use to catch efficacy or side-effect complaints early, without reading
every review manually.

## Project structure

```
drug-review-sentiment/
├── README.md
├── requirements.txt
├── data/                    # csv file will go here
├── src/
│   ├── data_loader.py       # loads + filters dataset to one condition
│   ├── preprocess.py        # text cleaning
│   ├── train_baseline.py    # TF-IDF + Logistic Regression / Linear SVM
│   ├── train_transformer.py # fine-tuned DistilBERT (stronger model)
│   └── evaluate.py          # model comparison table
└── outputs/                 # generated: metrics, confusion matrices, models
```

## Getting the real dataset

This repo ships only a small **synthetic** sample (`data/sample_depression_reviews.csv`)
so the code runs out of the box. For the real project, download the actual
dataset and drop the training file into `data/`:

1. Go to <https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018>
2. Download `drugsComTrain_raw.tsv`
3. Place it in `data/drugsComTrain_raw.tsv`

`data_loader.py` automatically prefers the real file when present and only
falls back to the synthetic sample otherwise — no code changes needed.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
# 1. Baseline models (TF-IDF + Logistic Regression / Linear SVM)
python src/train_baseline.py --condition Depression

# 2. Stronger model (DistilBERT fine-tuning) - optional, needs torch/transformers
python src/train_transformer.py --condition Depression --epochs 2

# 3. Compare results
python src/evaluate.py
```

Outputs (trained models, confusion matrix plots, metrics JSON) are written
to `outputs/`.

## Method summary

1. **Scope**: dataset filtered to `condition == "Depression"` only, to keep
   vocabulary and sentiment patterns coherent (see report for justification).
2. **Labels**: 1-10 star rating bucketed into 3 classes — negative (1-4),
   neutral (5-6), positive (7-10).
3. **Baseline**: TF-IDF (unigrams + bigrams) + Logistic Regression / Linear SVM.
4. **Stronger model**: fine-tuned `distilbert-base-uncased`.
5. **Evaluation**: accuracy, macro F1, per-class precision/recall, confusion
   matrices, qualitative error analysis.

## Known limitations (see report for full discussion)

- Class imbalance (ratings skew positive) — addressed with `class_weight="balanced"`
  for the baseline models, but should be re-checked once the real dataset is loaded.
- Single-condition scope trades generalisability for coherence; a
  production system would likely need one model per condition or a
  condition-aware architecture.
- The synthetic sample bundled here is for pipeline testing only — all
  reported metrics in the report were computed on the real dataset.

## Author

Akshat Jain
