# src/models/eval_bert.py
# Evaluate fine-tuned BERT model on validation set

import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix

LABEL2ID = {"human": 0, "ai": 1}
ID2LABEL = {0: "human", 1: "ai"}

def main(args):
    # Load data
    df = pd.read_csv(args.data)

    # Handle label column (numeric vs string)
    if pd.api.types.is_numeric_dtype(df["label"]):
        # Kaggle dataset: already 0/1
        y_true = df["label"].astype(int).values
    else:
        # Old dataset: human/ai strings
        y_true = df["label"].str.lower().map(LABEL2ID).values

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Tokenize
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encodings)
        y_pred = torch.argmax(outputs.logits, dim=1).numpy()

    # Report
    print("=== BERT Validation Report ===")
    print(classification_report(y_true, y_pred, target_names=["human", "ai"]))
    print("\nConfusion Matrix (rows=true, cols=pred): [human, ai]")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Validation CSV")
    ap.add_argument("--model_dir", required=True, help="Fine-tuned BERT model directory")
    args = ap.parse_args()
    main(args)
