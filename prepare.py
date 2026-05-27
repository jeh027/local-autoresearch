"""
One-time data preparation for autoresearch SVM experiments.

Downloads tabular training data, writes train/val parquet splits, fits a
TF-IDF + numeric feature pipeline (word length, vowel/consonant ratios),
and exposes runtime loaders + a fixed validation metric for train.py.

Usage:
    python prepare.py
    python prepare.py --val-fraction 0.2
"""

# ---------------------------------------------------------------------------
# Import Dependencies
# ---------------------------------------------------------------------------

import os
import sys
import time
import argparse
import pickle
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # training time budget in seconds (5 minutes)
RANDOM_SEED = 42
VAL_FRACTION = 0.2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PREPROCESSOR_DIR = os.path.join(CACHE_DIR, "preprocessor")
BASE_URL = "https://raw.githubusercontent.com/jeh027/autoresearch-template/refs/heads/main/train.csv"

RAW_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_PARQUET = os.path.join(DATA_DIR, "train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "val.parquet")
VECTORIZER_PATH = os.path.join(PREPROCESSOR_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(PREPROCESSOR_DIR, "label_encoder.pkl")
EXTRA_SCALER_PATH = os.path.join(PREPROCESSOR_DIR, "extra_scaler.pkl")
TRAIN_FEATURES_PATH = os.path.join(PREPROCESSOR_DIR, "train_features.npz")
VAL_FEATURES_PATH = os.path.join(PREPROCESSOR_DIR, "val_features.npz")

# Appended after TF-IDF columns: length, vowel_ratio, consonant_ratio
EXTRA_FEATURE_NAMES = ("word_length", "vowel_ratio", "consonant_ratio")

# Character n-grams work well for short words (Spanish vs French).
TFIDF_KWARGS = dict(
    analyzer="char",
    ngram_range=(1, 3),
    min_df=1,
    lowercase=True,
)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(max_attempts=5):
    """Download the CSV from BASE_URL with retries."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_CSV):
        print(f"Data: CSV already exists at {RAW_CSV}")
        return

    print(f"Data: downloading from {BASE_URL} ...")
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(BASE_URL, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = RAW_CSV + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, RAW_CSV)
            print(f"Data: saved to {RAW_CSV}")
            return
        except (requests.RequestException, OSError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [RAW_CSV + ".tmp", RAW_CSV]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    print("Data: download failed.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Parquet processing + train/val split
# ---------------------------------------------------------------------------

def _read_csv():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Missing {RAW_CSV}. Run prepare.py first.")
    
    df = pd.read_csv(RAW_CSV)
    expected = {"word", "label"}
    
    if not expected.issubset(df.columns):
        raise ValueError(f"CSV must have columns {expected}, got {list(df.columns)}")
    
    df = df.dropna(subset=["word", "label"])
    
    df["word"] = df["word"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["word"] != "") & (df["label"] != "")]
    
    return df


def process_parquet(val_fraction = VAL_FRACTION, random_seed = RANDOM_SEED):
    """Load CSV, hold out a validation set, write train/val parquet files."""

    if (os.path.exists(TRAIN_PARQUET) and os.path.exists(VAL_PARQUET)):
        
        train_n = pq.read_table(TRAIN_PARQUET).num_rows
        val_n = pq.read_table(VAL_PARQUET).num_rows
        
        print(f"Data: parquet splits already exist ({train_n} train, {val_n} val)")
        
        return

    df = _read_csv()
    train_df, val_df = train_test_split(
        df,
        test_size = val_fraction,
        random_state = random_seed,
        stratify = df["label"],
    )

    train_table = pa.Table.from_pandas(train_df, preserve_index=False)
    val_table = pa.Table.from_pandas(val_df, preserve_index=False)
    pq.write_table(train_table, TRAIN_PARQUET)
    pq.write_table(val_table, VAL_PARQUET)
    print(f"Data: wrote {len(train_df)} train + {len(val_df)} val rows to parquet")


def _load_split_parquet(split):
    path = TRAIN_PARQUET if split == "train" else VAL_PARQUET

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run prepare.py first.")

    return pq.read_table(path).to_pandas()


# ---------------------------------------------------------------------------
# Feature preprocessing (TF-IDF + extra numeric features)
# ---------------------------------------------------------------------------

def _strip_accents(char):
    decomposed = unicodedata.normalize("NFD", char)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _is_vowel(char):
    if not char.isalpha():
        return False
    return _strip_accents(char.lower()) in "aeiou"


def _extra_features_raw(words):
    """Per-word numeric features: length, vowel ratio, consonant ratio (letters only)."""
    rows = np.zeros((len(words), len(EXTRA_FEATURE_NAMES)), dtype=np.float64)
    for i, word in enumerate(words):
        w = str(word).lower().strip()
        rows[i, 0] = len(w)
        letters = [c for c in w if c.isalpha()]
        n_letters = len(letters)
        if n_letters == 0:
            continue
        n_vowels = sum(1 for c in letters if _is_vowel(c))
        n_consonants = n_letters - n_vowels
        rows[i, 1] = n_vowels / n_letters
        rows[i, 2] = n_consonants / n_letters
    return rows


def _combine_features(tfidf_X, extra_X_scaled):
    extra_sparse = sparse.csr_matrix(extra_X_scaled)
    return sparse.hstack([tfidf_X, extra_sparse], format="csr")


def _save_sparse_npz(path, X, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        data=X.data,
        indices=X.indices,
        indptr=X.indptr,
        shape=X.shape,
        y=y,
    )


def _load_sparse_npz(path):
    bundle = np.load(path)
    X = sparse.csr_matrix(
        (bundle["data"], bundle["indices"], bundle["indptr"]),
        shape=tuple(bundle["shape"]),
    )
    return X, bundle["y"]


def fit_preprocessor(random_seed=RANDOM_SEED):
    """Fit label encoder + TF-IDF + extra features on training split; materialize matrices."""
    artifacts = [
        VECTORIZER_PATH,
        LABEL_ENCODER_PATH,
        EXTRA_SCALER_PATH,
        TRAIN_FEATURES_PATH,
        VAL_FEATURES_PATH,
    ]
    if all(os.path.exists(p) for p in artifacts):
        print(f"Preprocessor: already fitted at {PREPROCESSOR_DIR}")
        return

    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)
    train_df = _load_split_parquet("train")
    val_df = _load_split_parquet("val")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])

    print("Preprocessor: fitting TF-IDF on training split...")
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    X_train_tfidf = vectorizer.fit_transform(train_df["word"])
    X_val_tfidf = vectorizer.transform(val_df["word"])

    print("Preprocessor: building length / vowel / consonant features...")
    extra_scaler = StandardScaler()
    X_train_extra = extra_scaler.fit_transform(_extra_features_raw(train_df["word"]))
    X_val_extra = extra_scaler.transform(_extra_features_raw(val_df["word"]))

    X_train = _combine_features(X_train_tfidf, X_train_extra)
    X_val = _combine_features(X_val_tfidf, X_val_extra)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    with open(EXTRA_SCALER_PATH, "wb") as f:
        pickle.dump(extra_scaler, f)

    _save_sparse_npz(TRAIN_FEATURES_PATH, X_train, y_train)
    _save_sparse_npz(VAL_FEATURES_PATH, X_val, y_val)
    n_tfidf = X_train_tfidf.shape[1]
    print(
        f"Preprocessor: {n_tfidf} TF-IDF + {len(EXTRA_FEATURE_NAMES)} extra "
        f"= {X_train.shape[1]} features, saved to {PREPROCESSOR_DIR}"
    )


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_label_encoder():
    with open(LABEL_ENCODER_PATH, "rb") as f:
        return pickle.load(f)

def load_vectorizer():
    with open(VECTORIZER_PATH, "rb") as f:
        return pickle.load(f)


def load_extra_scaler():
    with open(EXTRA_SCALER_PATH, "rb") as f:
        return pickle.load(f)


def vectorize_words(words):
    """Transform raw words into the full sparse feature matrix (TF-IDF + extras)."""
    vectorizer = load_vectorizer()
    extra_scaler = load_extra_scaler()
    tfidf_X = vectorizer.transform(words)
    extra_X = extra_scaler.transform(_extra_features_raw(words))
    return _combine_features(tfidf_X, extra_X)


def load_features(split="train"):
    """
    Return (X, y) sparse feature matrix and encoded labels.
    split: "train" or "val"
    """
    assert split in ("train", "val")
    path = TRAIN_FEATURES_PATH if split == "train" else VAL_FEATURES_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python prepare.py` to build features."
        )
    return _load_sparse_npz(path)


def load_train_val():
    """Convenience loader for SVM training in train.py."""
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    return X_train, X_val, y_train, y_val


def get_num_classes():
    return len(load_label_encoder().classes_)


def get_class_names():
    return list(load_label_encoder().classes_)

def get_time_budget():
    return TIME_BUDGET


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_accuracy(model, X_val=None, y_val=None):
    """
    Validation accuracy on the hold-out set. Higher is better.
    Uses the fixed val split written by prepare.py.
    """
    if X_val is None or y_val is None:
        X_val, y_val = load_features("val")
    y_pred = model.predict(X_val)
    return float(accuracy_score(y_val, y_pred))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(val_fraction=VAL_FRACTION, random_seed=RANDOM_SEED):
    print(f"Cache directory: {CACHE_DIR}")
    print()

    download_data()
    print()

    process_parquet(val_fraction=val_fraction, random_seed=random_seed)
    print()

    fit_preprocessor(random_seed=random_seed)
    print()
    print("Done! Ready to train.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data and features for autoresearch SVM experiments"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=VAL_FRACTION,
        help="Fraction of rows held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for train/val split",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")

    main(val_fraction=args.val_fraction, random_seed=args.seed)