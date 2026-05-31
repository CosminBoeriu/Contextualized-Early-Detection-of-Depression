#!/usr/bin/env python3
"""
04_train.py
-----------
Fine-tune a transformer encoder for binary depression classification.

Features:
  - Weighted cross-entropy loss to handle severe class imbalance
  - Early stopping on val F1
  - Optuna hyperparameter search (pass --search)
  - Saves best checkpoint to models/<model_name>/

Usage:
  python scripts/04_train.py                # single run with config.yaml params
  python scripts/04_train.py --search       # Optuna HP search
  python scripts/04_train.py --lr 2e-5 --batch_size 8 --epochs 3
"""

import argparse
import json
import logging
import os
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

TRAIN_FILE  = Path(CFG["paths"]["train_file"])
VAL_FILE    = Path(CFG["paths"]["val_file"])
MODEL_DIR   = Path(CFG["paths"]["model_dir"])
LOG_DIR     = Path(CFG["paths"]["log_dir"])
MODEL_NAME  = CFG["model"]["name"]
NUM_LABELS  = CFG["model"]["num_labels"]
SEED        = CFG["split"]["random_seed"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found. Run previous pipeline steps first.")
    data = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class DepressionDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex    = self.examples[idx]
        text  = ex["formatted_text"]
        label = int(ex["label"])

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(examples: list[dict]) -> torch.Tensor:
    labels = [int(e["label"]) for e in examples]
    pos = sum(labels)
    neg = len(labels) - pos
    total = len(labels)
    w0 = total / (2 * neg) if neg > 0 else 1.0
    w1 = total / (2 * pos) if pos > 0 else 1.0
    logger.info(f"Class weights: neg={w0:.3f}, pos={w1:.3f}")
    return torch.tensor([w0, w1], dtype=torch.float)


# ── Metrics ───────────────────────────────────────────────────────────────────

def make_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = (probs[:, 1] >= threshold).astype(int)
        f1  = f1_score(labels, preds, zero_division=0)
        p   = precision_score(labels, preds, zero_division=0)
        r   = recall_score(labels, preds, zero_division=0)
        return {"f1": f1, "precision": p, "recall": r}
    return compute_metrics


# ── Custom Trainer with weighted loss ─────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device if self.args.device else "cpu")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        device = logits.device
        weights = self.class_weights.to(device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss     = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ── Training Function ─────────────────────────────────────────────────────────

def train(
    model_name:    str,
    train_data:    list[dict],
    val_data:      list[dict],
    lr:            float,
    batch_size:    int,
    num_epochs:    int,
    weight_decay:  float,
    output_dir:    Path,
    use_class_weights: bool = True,
    fp16:          bool = True,
) -> dict:
    """
    Fine-tune model_name on train_data, evaluate on val_data.
    Returns best val metrics dict.
    """
    logger.info(f"Loading tokenizer & model: {model_name}")
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    max_length = CFG["preprocessing"]["max_tokens"]
    train_ds   = DepressionDataset(train_data, tokenizer, max_length)
    val_ds     = DepressionDataset(val_data,   tokenizer, max_length)

    class_weights = compute_class_weights(train_data) if use_class_weights \
                    else torch.ones(NUM_LABELS)

    warmup_ratio  = CFG["training"]["warmup_ratio"]
    patience      = CFG["training"]["early_stopping_patience"]
    metric_best   = CFG["training"]["metric_for_best"]
    grad_accum    = CFG["training"]["gradient_accumulation_steps"]

    safe_model_name = model_name.replace("/", "_")
    run_dir = output_dir / safe_model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determine if GPU is available
    use_fp16 = fp16 and torch.cuda.is_available()

    args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        gradient_accumulation_steps=grad_accum,
        fp16=use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model=metric_best,
        greater_is_better=(metric_best != "loss"),
        save_total_limit=2,
        logging_dir=str(LOG_DIR),
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=CFG["training"]["dataloader_num_workers"],
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    logger.info("Starting training ...")
    trainer.train()

    # Save best model
    best_dir = run_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    logger.info(f"Best model saved to {best_dir}")

    # Final evaluation
    metrics = trainer.evaluate()
    logger.info(f"Val metrics: {metrics}")

    # Full classification report
    preds_out = trainer.predict(val_ds)
    logits    = preds_out.predictions
    probs     = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds     = (probs[:, 1] >= 0.5).astype(int)
    true      = [int(e["label"]) for e in val_data]

    report = classification_report(true, preds,
                                   target_names=["Control", "Depression"])
    logger.info(f"\nClassification Report:\n{report}")

    # Save predictions on val set
    pred_file = run_dir / "val_predictions.json"
    val_preds = [
        {
            "subject_id": e["subject_id"],
            "true_label": int(e["label"]),
            "pred_label": int(preds[i]),
            "score":      float(probs[i, 1]),
        }
        for i, e in enumerate(val_data)
    ]
    with open(pred_file, "w", encoding='utf-8', errors='replace') as f:
        json.dump(val_preds, f, indent=2)
    logger.info(f"Val predictions saved -> {pred_file}")

    return {k: v for k, v in metrics.items() if isinstance(v, (int, float))}


# ── Optuna Search ─────────────────────────────────────────────────────────────

def optuna_search(train_data: list[dict], val_data: list[dict]):
    try:
        import optuna
    except ImportError:
        sys.exit("[ERROR] Install optuna: pip install optuna")

    search_cfg = CFG["optuna"]
    n_trials   = search_cfg["n_trials"]
    lr_range   = search_cfg["search_space"]["learning_rate"]
    bs_choices = search_cfg["search_space"]["batch_size"]
    wd_range   = search_cfg["search_space"]["weight_decay"]

    def objective(trial):
        lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
        bs = trial.suggest_categorical("batch_size", bs_choices)
        wd = trial.suggest_categorical("weight_decay", wd_range)
        ep = 3

        trial_dir = MODEL_DIR / f"trial_{trial.number}"
        metrics   = train(
            model_name=MODEL_NAME,
            train_data=train_data,
            val_data=val_data,
            lr=lr,
            batch_size=bs,
            num_epochs=ep,
            weight_decay=wd,
            output_dir=trial_dir,
            use_class_weights=CFG["training"]["use_class_weights"],
            fp16=CFG["training"]["fp16"],
        )
        return metrics.get("eval_f1", 0.0)

    study = optuna.create_study(direction="maximize",
                                study_name="erisk_task2_hpsearch")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_trial.params}")
    logger.info(f"Best F1: {study.best_value:.4f}")

    results_path = MODEL_DIR / "optuna_results.json"
    trials_data  = [
        {"trial": t.number, "params": t.params, "f1": t.value}
        for t in study.trials
    ]
    with open(results_path, "w", encoding='utf-8', errors='replace') as f:
        json.dump(trials_data, f, indent=2)
    logger.info(f"Optuna results -> {results_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train depression classifier")
    p.add_argument("--search",      action="store_true",
                   help="Run Optuna hyperparameter search instead of single run")
    p.add_argument("--model",       default=MODEL_NAME,
                   help="HuggingFace model name or path")
    p.add_argument("--lr",          type=float,
                   default=CFG["training"]["learning_rate"])
    p.add_argument("--batch_size",  type=int,
                   default=CFG["training"]["batch_size"])
    p.add_argument("--epochs",      type=int,
                   default=CFG["training"]["num_epochs"])
    p.add_argument("--weight_decay",type=float,
                   default=CFG["training"]["weight_decay"])
    return p.parse_args()


def main():
    args = parse_args()

    logger.info(f"[TRAIN] Loading train data from {TRAIN_FILE} ...")
    train_data = load_jsonl(TRAIN_FILE)
    logger.info(f"[TRAIN] Loading val data from {VAL_FILE} ...")
    val_data   = load_jsonl(VAL_FILE)

    logger.info(f"[TRAIN] Train: {len(train_data)} examples | Val: {len(val_data)} examples")

    if args.search:
        logger.info("[TRAIN] Starting Optuna hyperparameter search ...")
        optuna_search(train_data, val_data)
    else:
        logger.info(f"[TRAIN] Single run: model={args.model}, lr={args.lr}, "
                    f"bs={args.batch_size}, epochs={args.epochs}")
        train(
            model_name=args.model,
            train_data=train_data,
            val_data=val_data,
            lr=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            weight_decay=args.weight_decay,
            output_dir=MODEL_DIR,
            use_class_weights=CFG["training"]["use_class_weights"],
            fp16=CFG["training"]["fp16"],
        )


if __name__ == "__main__":
    main()
