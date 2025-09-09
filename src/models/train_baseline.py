# src/models/train_baseline.py
# Baseline: TF-IDF (1–2 n-gram) + Logistic Regression
# Trains on train.csv, evaluates on val.csv, saves a single Pipeline as joblib.
# Comments in English as requested.

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
from joblib import dump

def fit_baseline(train_path: str, val_path: str, outdir: str) -> None:
    # 1) Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train, y_train = train_df["text"].astype(str).tolist(), train_df["label"].astype(str).tolist()
    X_val, y_val = val_df["text"].astype(str).tolist(), val_df["label"].astype(str).tolist()

    # 2) Define pipeline (TF-IDF + Logistic Regression)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("lr", LogisticRegression(max_iter=200, class_weight="balanced"))  # balanced helps on tiny/imbalanced sets
    ])

    # 3) Train
    pipe.fit(X_train, y_train)

    # 4) Validate
    preds = pipe.predict(X_val)
    macro_f1 = f1_score(y_val, preds, average="macro")
    print("Validation report:")
    print(classification_report(y_val, preds, digits=4))
    print(f"Macro F1: {macro_f1:.4f}")

    # 5) Save whole pipeline
    out_path = f"{outdir}/baseline.joblib"
    dump(pipe, out_path)
    print(f"Saved model -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model.")
    ap.add_argument("--train", required=True, help="Path to processed train.csv")
    ap.add_argument("--val", required=True, help="Path to processed val.csv")
    ap.add_argument("--outdir", required=True, help="Directory to save baseline.joblib")
    args = ap.parse_args()

    fit_baseline(args.train, args.val, args.outdir)
