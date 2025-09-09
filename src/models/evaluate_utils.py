# src/models/evaluate.py
# Loads a saved pipeline (baseline.joblib) and evaluates on a CSV.
# Prints classification report and saves a confusion matrix.

import argparse
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    df = pd.read_csv(args.data)
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    model = load(args.model)  # pipeline (tfidf + lr)

    preds = model.predict(X)
    print(classification_report(y, preds, digits=4))

    labels = sorted(list(set(y)))
    cm = confusion_matrix(y, preds, labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()

    out_png = f"{args.out}/confusion_matrix.png"
    plt.savefig(out_png, dpi=200)
    print(f"Saved -> {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args)
