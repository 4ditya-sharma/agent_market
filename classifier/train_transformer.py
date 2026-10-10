"""Optional heavier model: fine-tune DeBERTa-v3-small (or any HF encoder) for 3-class injection
detection. Run on a laptop/Colab with internet; the sklearn model stays as fallback.

    python -m classifier.train_transformer --model microsoft/deberta-v3-small --epochs 4

Anti-overfit measures: group-aware held-out test (same split logic as train_model.py),
early stopping on validation macro-F1, weight decay, label smoothing, class-weighted loss,
small LR with warmup, and we freeze nothing but train only a few epochs.
Exports models/injection_transformer/ which ai_layer/firewall.py uses automatically if present
(set FIREWALL_USE_TRANSFORMER=1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "injection_v2.csv"
OUT = ROOT / "models" / "injection_transformer"
LABELS = ["benign", "malicious", "malignant"]


class _DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels): self.enc, self.labels = enc, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        d["labels"] = torch.tensor(self.labels[i]); return d


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kw):
        super().__init__(**kw); self.cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self.cw.to(out.logits.device), label_smoothing=0.05)(out.logits, labels)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-small")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    a = ap.parse_args()

    df = pd.read_csv(DATA).dropna(subset=["text"]); df["group"] = df["group"].astype(str)
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(df, df["label"], df["group"]))
    train, test = df.iloc[tr_i], df.iloc[te_i]
    tr_i, va_i = next(GroupShuffleSplit(1, test_size=0.15, random_state=7).split(train, train["label"], train["group"]))
    train, val = train.iloc[tr_i], train.iloc[va_i]
    print(f"train {len(train)} val {len(val)} test {len(test)}")

    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda d: tok(d["text"].tolist(), truncation=True, max_length=a.max_len)
    ds_tr, ds_va, ds_te = (_DS(enc(d), d["label"].tolist()) for d in (train, val, test))
    counts = np.bincount(train["label"], minlength=3)
    cw = torch.tensor(len(train) / (3 * counts), dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3, id2label=dict(enumerate(LABELS)),
                                                               label2id={l: i for i, l in enumerate(LABELS)})
    args = TrainingArguments(output_dir=str(OUT / "ckpt"), num_train_epochs=a.epochs, learning_rate=a.lr,
                             per_device_train_batch_size=a.batch, per_device_eval_batch_size=64, weight_decay=0.05,
                             warmup_ratio=0.1, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
                             metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
                             logging_steps=50, report_to=[], fp16=torch.cuda.is_available())
    metrics = lambda p: {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}
    trainer = WeightedTrainer(cw, model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
                              callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()
    pred = trainer.predict(ds_te).predictions.argmax(-1)
    print(classification_report(test["label"], pred, target_names=LABELS, digits=3))
    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT)); tok.save_pretrained(str(OUT))
    (OUT / "report.json").write_text(json.dumps(classification_report(test["label"], pred, target_names=LABELS, output_dict=True), indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
