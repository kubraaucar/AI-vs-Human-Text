# src/models/eval_paraphrase.py
# Evaluate model (joblib baseline OR HuggingFace BERT) on a paraphrased CSV

import argparse, os
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def eval_baseline(model_path, df):
    from joblib import load
    pipe = load(model_path)
    y_true = df["label"].astype(str).tolist()
    y_pred = pipe.predict(df["paraphrase"].tolist())
    return y_true, y_pred

def eval_bert(model_dir, df, max_len=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    LABEL2ID = {"human": 0, "ai": 1}
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = df["paraphrase"].astype(str).tolist()
    enc = tok(texts, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    preds = logits.argmax(-1).cpu().numpy()
    y_true = df["label"].str.lower().map(LABEL2ID).values
    y_pred = preds
    return y_true, y_pred

def main(args):
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["paraphrase", "label"])
    df = df[df["paraphrase"].astype(str).str.strip() != ""]

    if os.path.isdir(args.model):
        # Assume HuggingFace BERT
        y_true, y_pred = eval_bert(args.model, df, args.max_len)
    else:
        # Assume baseline joblib
        y_true, y_pred = eval_baseline(args.model, df)

    print("=== Paraphrase Set Report ===")
    print(classification_report(y_true, y_pred, target_names=["human","ai"], digits=4))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    labels = [0,1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix (rows=true, cols=pred): [human, ai]")
    print(cm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to paraphrase CSV")
    ap.add_argument("--model", required=True, help="Path to model (joblib file or HF model dir)")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    main(args)
