# src/models/make_paraphrase_sheet.py
# Creates a small to-do CSV for paraphrasing from val.csv (balanced sample)

import argparse, pandas as pd

def main(args):
    df = pd.read_csv(args.val)
    # Keep only needed cols
    df = df[["text", "label"]].copy()
    # Balanced sample: take up to n_per_class from each class
    out_rows = []
    for cls in ["ai", "human"]:
        part = df[df["label"] == cls].sample(n=min(args.n_per_class, (df["label"] == cls).sum()), random_state=args.seed)
        out_rows.append(part)
    out_df = pd.concat(out_rows).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    out_df.rename(columns={"text": "original"}, inplace=True)
    out_df["paraphrase"] = ""  # you will fill this column manually
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved -> {args.out} (rows={len(out_df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="Path to data/processed/val.csv")
    ap.add_argument("--out", default="data/processed/val_paraphrase_todo.csv")
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
