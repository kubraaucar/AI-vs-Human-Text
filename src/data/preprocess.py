# src/data/preprocess.py
# Preprocess big CSV into train/val splits (AI vs Human)

import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    print(f"Loading {args.input} in chunks...")
    chunks = []
    for chunk in pd.read_csv(args.input, chunksize=50000):  # 50k satır parça parça okunur
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df)} rows.")
    print("Columns:", df.columns.tolist())

    # Kaggle dataset: kolonlar 'text' ve 'generated'
    if "generated" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"generated": "label"})
        print("Renamed 'generated' -> 'label'")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV içinde 'text' ve 'label' kolonları bulunamadı!")

    # 🔹 Label normalization
    if df["label"].dtype != "object":
        # numeric (0/1 veya 0.0/1.0) ise stringe çevir
        df["label"] = df["label"].map({0: "human", 1: "ai", 0.0: "human", 1.0: "ai"})
        print("Converted numeric labels -> 'human' / 'ai'")
    else:
        # string ise normalize et (küçük harfe çevir)
        df["label"] = df["label"].str.lower().map({"human": "human", "ai": "ai", "generated": "ai"})

    # 🔹 Hızlı test için veriyi küçült (20k satır)
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
        print("Subsampled to 20,000 rows for quick training.")

    # Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=args.val_size, random_state=args.seed, stratify=df["label"]
    )

    os.makedirs(args.outdir, exist_ok=True)
    train_path = os.path.join(args.outdir, "train.csv")
    val_path = os.path.join(args.outdir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved -> {train_path} ({len(train_df)} rows)")
    print(f"Saved -> {val_path} ({len(val_df)} rows)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw CSV")
    ap.add_argument("--outdir", required=True, help="Output directory for processed data")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
