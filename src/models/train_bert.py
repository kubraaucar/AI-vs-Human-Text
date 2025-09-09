import os
import argparse
import numpy as np
import pandas as pd
import inspect

from datasets import Dataset, DatasetDict
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

# Label mapping
LABEL2ID = {"human": 0, "ai": 1}
ID2LABEL = {0: "human", 1: "ai"}


def load_csv(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # label normalization -> {human, ai} -> {0,1}
    train_df["label"] = (
        train_df["label"].astype(str).str.strip().str.lower().map(LABEL2ID)
    )
    val_df["label"] = (
        val_df["label"].astype(str).str.strip().str.lower().map(LABEL2ID)
    )

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
        }
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main(args):
    model_name = "prajjwal1/bert-tiny"  # hızlı ve küçük

    # 1) Dataset
    ds = load_csv(args.train, args.val)

    # 2) Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
        )

    ds = ds.map(tokenize, batched=True)

    # 3) Model
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # 4) TrainingArguments — sürüme göre uyumlu ayar
    sig_params = set(inspect.signature(TrainingArguments).parameters.keys())

    # Ortak (her sürümde olan) argümanlar
    ta_kwargs = {
        "output_dir": args.outdir,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "logging_dir": os.path.join(args.outdir, "logs"),
        "logging_steps": 100,
        "save_total_limit": 2,
    }

    # Kaydetme stratejisi
    if "save_strategy" in sig_params:
        ta_kwargs["save_strategy"] = "epoch"
    elif "save_steps" in sig_params:
        # Çok eski sürümler: adım bazlı kaydetmeyi etkinleştirebiliriz
        ta_kwargs["save_steps"] = 500

    # Değerlendirme stratejisi (sürüme göre evaluation_strategy / eval_strategy / yok)
    added_eval = False
    if "evaluation_strategy" in sig_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
        added_eval = True
    elif "eval_strategy" in sig_params:
        ta_kwargs["eval_strategy"] = "epoch"
        added_eval = True

    # En iyi modeli yükleme (yalnızca destekleniyorsa ve eval varsa)
    if "load_best_model_at_end" in sig_params:
        ta_kwargs["load_best_model_at_end"] = bool(added_eval)

    # Bilinmeyen anahtarları ayıkla (tam uyum için)
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in sig_params}

    training_args = TrainingArguments(**ta_kwargs)

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) Train
    trainer.train()

    # 7) Eval
    metrics = trainer.evaluate()
    print("=== Eval after training ===")
    print(metrics)

    # 8) Save
    save_path = os.path.join(args.outdir, "bert_tiny")
    trainer.save_model(save_path)
    print(f"Saved fine-tuned model to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    main(args)
