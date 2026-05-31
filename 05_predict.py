#!/usr/bin/env python3
"""
05_predict.py
-------------
Sequential / early-detection inference for eRisk Task 2.

For each subject, we feed conversation windows in chronological order
(one per message round) and fire a prediction the first time the model's
P(positive) crosses the confidence_threshold set in config.yaml.

Outputs two files in outputs/:
  predictions_decision.txt   — subject_id  decision  score  round
  predictions_ranking.txt    — subject_id  score  (for ranking-based eval)

The decision file is the main submission artefact. Both files are also
saved as JSON for easier programmatic use.

Usage:
  python scripts/05_predict.py                          # use config.yaml model
  python scripts/05_predict.py --model_path models/xyz  # override model path
  python scripts/05_predict.py --threshold 0.6          # override threshold
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PREPROCESSED  = Path(CFG["paths"]["preprocessed"])
MODEL_NAME    = CFG["model"]["name"]
MODEL_DIR     = Path(CFG["paths"]["model_dir"])
OUTPUT_DIR    = Path(CFG["paths"]["output_dir"])
THRESHOLD     = CFG["inference"]["confidence_threshold"]
MIN_MSGS      = CFG["inference"]["min_messages_before_decision"]
INFER_BS      = CFG["inference"]["batch_size"]
SCORE_EVERY   = CFG["inference"]["score_on_every_message"]
MAX_LENGTH    = CFG["preprocessing"]["max_tokens"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class SingleTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    """Load the fine-tuned model from disk (or a HF hub name for zero-shot)."""
    print(f"[PREDICT] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to(DEVICE)
    return model, tokenizer


@torch.no_grad()
def score_batch(texts: list[str], model, tokenizer) -> list[float]:
    """Return P(positive) for a list of texts."""
    ds     = SingleTextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BS, shuffle=False)
    scores = []
    for batch in loader:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        out    = model(**batch)
        probs  = torch.softmax(out.logits, dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())
    return scores


# ── Load all preprocessed examples grouped by subject ─────────────────────────

def load_and_group(preprocessed_path: Path) -> dict[str, list[dict]]:
    """
    Load the preprocessed JSONL and group by subject_id,
    sorted by round number (ascending = chronological).
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)

    if not preprocessed_path.exists():
        sys.exit(f"[ERROR] {preprocessed_path} not found. Run 02_preprocess.py first.")

    with open(preprocessed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            groups[ex["subject_id"]].append(ex)

    for sid in groups:
        groups[sid].sort(key=lambda e: e["round"])

    return groups


# ── Early-detection logic ─────────────────────────────────────────────────────

def run_early_detection(
    subject_groups: dict[str, list[dict]],
    model,
    tokenizer,
    threshold: float,
    min_msgs: int,
    score_every: bool,
) -> list[dict]:
    """
    Sequentially process each subject's rounds.
    Fire a decision (positive=1) the first time score >= threshold AND
    at least min_msgs messages have been seen.

    If the threshold is never crossed, the final score is used to decide.

    Returns a list of result dicts:
      {subject_id, true_label, predicted_label, score, decision_round,
       num_total_rounds, total_target_msgs}
    """
    results = []

    # Decide which rounds to score
    # score_every=True  → score every message
    # score_every=False → score only rounds where a new TARGET msg appeared
    def should_score(ex: dict, prev_target_count: int) -> bool:
        if score_every:
            return True
        return ex["num_target_msgs"] > prev_target_count

    subject_ids = list(subject_groups.keys())

    for sid in tqdm(subject_ids, desc="Early-detection inference"):
        rounds       = subject_groups[sid]
        true_label   = rounds[-1].get("label")
        num_rounds   = len(rounds)
        total_target = rounds[-1]["num_target_msgs"]

        fired       = False
        final_score = 0.0
        final_pred  = 0
        decision_round = num_rounds  # default: last round

        prev_target_count = 0

        for ex in rounds:
            if not should_score(ex, prev_target_count):
                prev_target_count = ex["num_target_msgs"]
                continue

            prev_target_count = ex["num_target_msgs"]

            # Gate on minimum messages
            if ex["num_total_msgs"] < min_msgs:
                continue

            # Score this window
            score = score_batch([ex["formatted_text"]], model, tokenizer)[0]
            final_score = score

            if score >= threshold and not fired:
                fired          = True
                final_pred     = 1
                decision_round = ex["round"]
                break  # early exit

        # If threshold never crossed, use final score to classify
        if not fired:
            final_pred     = 1 if final_score >= threshold else 0
            decision_round = num_rounds

        results.append({
            "subject_id":     sid,
            "true_label":     true_label,
            "predicted_label": final_pred,
            "score":          final_score,
            "decision_round": decision_round,
            "num_total_rounds": num_rounds,
            "total_target_msgs": total_target,
        })

    return results


# ── Write outputs ─────────────────────────────────────────────────────────────

def write_outputs(results: list[dict], out_dir: Path):
    # 1. Decision file (main submission format)
    decision_txt = out_dir / "predictions_decision.txt"
    with open(decision_txt, "w") as f:
        f.write("subject_id\tdecision\tscore\tdecision_round\n")
        for r in results:
            f.write(f"{r['subject_id']}\t{r['predicted_label']}\t"
                    f"{r['score']:.4f}\t{r['decision_round']}\n")
    print(f"[PREDICT] Decision predictions → {decision_txt}")

    # 2. Ranking file (sorted by descending score)
    ranking_txt = out_dir / "predictions_ranking.txt"
    sorted_r    = sorted(results, key=lambda x: x["score"], reverse=True)
    with open(ranking_txt, "w") as f:
        f.write("subject_id\tscore\n")
        for r in sorted_r:
            f.write(f"{r['subject_id']}\t{r['score']:.4f}\n")
    print(f"[PREDICT] Ranking predictions → {ranking_txt}")

    # 3. Full JSON for evaluation script
    json_out = out_dir / "predictions_full.json"
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[PREDICT] Full predictions JSON → {json_out}")

    # 4. Summary
    labelled = [r for r in results if r["true_label"] is not None]
    if labelled:
        from sklearn.metrics import precision_score, recall_score, f1_score
        true  = [r["true_label"]     for r in labelled]
        preds = [r["predicted_label"] for r in labelled]
        p  = precision_score(true, preds, zero_division=0)
        r  = recall_score(true, preds,    zero_division=0)
        f1 = f1_score(true, preds,        zero_division=0)
        avg_round = sum(r["decision_round"] for r in labelled) / len(labelled)
        print(f"\n[PREDICT] Quick eval on labelled subjects:")
        print(f"  Precision : {p:.3f}")
        print(f"  Recall    : {r:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"  Avg decision round : {avg_round:.1f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Sequential early-detection inference")
    safe_model = MODEL_NAME.replace("/", "_")
    default_model_path = str(MODEL_DIR / safe_model / "best")
    p.add_argument("--model_path", default=default_model_path,
                   help="Path to fine-tuned model directory (or HF model name)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help="Confidence threshold for positive prediction")
    p.add_argument("--min_msgs",  type=int,   default=MIN_MSGS,
                   help="Minimum messages before any decision is fired")
    p.add_argument("--score_every", action="store_true", default=SCORE_EVERY,
                   help="Score on every message (default: only on new TARGET msgs)")
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print(f"[PREDICT] Loading preprocessed data from {PREPROCESSED} ...")
    subject_groups = load_and_group(PREPROCESSED)
    print(f"[PREDICT] {len(subject_groups)} subjects to process.")
    print(f"[PREDICT] Threshold={args.threshold} | Min messages={args.min_msgs}")

    results = run_early_detection(
        subject_groups=subject_groups,
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
        min_msgs=args.min_msgs,
        score_every=args.score_every,
    )

    write_outputs(results, OUTPUT_DIR)
    print(f"\n[PREDICT] Done. {len(results)} subjects processed.")


if __name__ == "__main__":
    main()
