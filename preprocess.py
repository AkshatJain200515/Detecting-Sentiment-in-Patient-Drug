"""
preprocess.py

Basic text cleaning for the review text before vectorisation.
Kept deliberately simple (regex + lowercasing + stopword removal) since
the modelling step (TF-IDF, transformers) handles most of the heavy
lifting -- see the report's "Detailed Design" section for the
justification of this choice over heavier preprocessing (lemmatisation,
spelling correction, etc.).
"""

import re
import html

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except ImportError:  # pragma: no cover
    ENGLISH_STOP_WORDS = set()

TAG_RE = re.compile(r"&#\d+;|<[^>]+>")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    if remove_stopwords:
        tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)

    return text


def clean_dataframe(df, text_col: str = "review", out_col: str = "clean_review"):
    df = df.copy()
    df[out_col] = df[text_col].apply(clean_text)
    df = df[df[out_col].str.len() > 0]
    return df


if __name__ == "__main__":
    sample = "This medication took about &#039;three&#039; weeks <br> to start working!!"
    print(clean_text(sample))
