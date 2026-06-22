import argparse
import json
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CFG_PATH = Path(__file__).parent / "config.yaml"

with open(CFG_PATH, encoding="utf-8", errors="replace") as f:
    CFG = yaml.safe_load(f)

PREPROCESSED = Path(
    CFG["paths"].get(
        "val_all_rounds_file",
        Path(CFG["paths"]["preprocessed"]).parent / "val_all_rounds.jsonl"
    )
)

MODEL_NAME = CFG["model"]["name"]
MODEL_DIR = Path(CFG["paths"]["model_dir"])
OUTPUT_DIR = Path(CFG["paths"]["output_dir"])

THRESHOLD = CFG["inference"]["confidence_threshold"]
MIN_MSGS = CFG["inference"]["min_messages_before_decision"]
INFER_BS = CFG["inference"]["batch_size"]
SCORE_EVERY = CFG["inference"]["score_on_every_message"]
MAX_LENGTH = CFG["preprocessing"]["max_tokens"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


def load_model_and_tokenizer(model_path: str):
    print(f"[PREDICT] Loading model from {model_path} ...")
    print(f"[PREDICT] Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()
    model.to(DEVICE)

    return model, tokenizer


@torch.no_grad()
def score_batch(texts: list[str], model, tokenizer) -> list[float]:
    ds = SingleTextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=INFER_BS, shuffle=False)

    scores = []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        probs = torch.softmax(out.logits, dim=-1)[:, 1]
        scores.extend(probs.cpu().tolist())

    return scores


def load_and_group(preprocessed_path: Path) -> dict[str, list[dict]]:
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)

    if not preprocessed_path.exists():
        sys.exit(
            f"[ERROR] {preprocessed_path} not found.\n"
        )

    with open(preprocessed_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            if "subject_id" not in ex:
                continue

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
    results = []

    def should_score(ex: dict, prev_target_count: int) -> bool:
        if score_every:
            return True

        return ex["num_target_msgs"] > prev_target_count

    subject_ids = list(subject_groups.keys())

    for sid in tqdm(subject_ids, desc="Early-detection inference"):
        rounds = subject_groups[sid]

        if not rounds:
            continue

        true_label = rounds[-1].get("label")
        total_rounds = int(rounds[-1]["round"])
        total_target = int(rounds[-1].get("num_target_msgs", total_rounds))

        fired = False
        final_score = 0.0
        final_pred = 0
        decision_round = total_rounds

        prev_target_count = 0

        for ex in rounds:
            if not should_score(ex, prev_target_count):
                prev_target_count = ex["num_target_msgs"]
                continue

            prev_target_count = ex["num_target_msgs"]
            if ex["num_total_msgs"] < min_msgs:
                continue

            score = score_batch([ex["formatted_text"]], model, tokenizer)[0]
            final_score = score

            if score >= threshold:
                fired = True
                final_pred = 1
                decision_round = int(ex["round"])
                break
        if not fired:
            final_pred = 1 if final_score >= threshold else 0
            decision_round = total_rounds

        # This prevents negative speed/latency bugs.
        if decision_round > total_rounds:
            decision_round = total_rounds

        if decision_round < 1:
            decision_round = 1

        results.append({
            "subject_id": sid,
            "true_label": true_label,
            "predicted_label": int(final_pred),
            "score": float(final_score),
            "decision_round": int(decision_round),
            "num_total_rounds": int(total_rounds),
            "total_target_msgs": int(total_target),
        })

    return results


def write_outputs(results: list[dict], out_dir: Path):
    decision_txt = out_dir / "predictions_decision.txt"

    with open(decision_txt, "w", encoding="utf-8", errors="replace") as f:
        f.write("subject_id\tdecision\tscore\tdecision_round\n")

        for r in results:
            f.write(
                f"{r['subject_id']}\t"
                f"{r['predicted_label']}\t"
                f"{r['score']:.4f}\t"
                f"{r['decision_round']}\n"
            )

    print(f"[PREDICT] Decision predictions -> {decision_txt}")

    ranking_txt = out_dir / "predictions_ranking.txt"

    sorted_r = sorted(results, key=lambda x: x["score"], reverse=True)

    with open(ranking_txt, "w", encoding="utf-8", errors="replace") as f:
        f.write("subject_id\tscore\n")

        for r in sorted_r:
            f.write(f"{r['subject_id']}\t{r['score']:.4f}\n")

    print(f"[PREDICT] Ranking predictions -> {ranking_txt}")

    json_out = out_dir / "predictions_full.json"

    with open(json_out, "w", encoding="utf-8", errors="replace") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[PREDICT] Full predictions JSON -> {json_out}")

    labelled = [r for r in results if r["true_label"] is not None]

    if labelled:
        from sklearn.metrics import precision_score, recall_score, f1_score

        true = [int(r["true_label"]) for r in labelled]
        preds = [int(r["predicted_label"]) for r in labelled]

        p = precision_score(true, preds, zero_division=0)
        r = recall_score(true, preds, zero_division=0)
        f1 = f1_score(true, preds, zero_division=0)

        avg_round = sum(r["decision_round"] for r in labelled) / len(labelled)

        print(f"\n[PREDICT] Quick eval on labelled subjects:")
        print(f"  Precision : {p:.3f}")
        print(f"  Recall    : {r:.3f}")
        print(f"  F1        : {f1:.3f}")
        print(f"  Avg decision round : {avg_round:.1f}")


def parse_args():
    p = argparse.ArgumentParser(description="Sequential early-detection inference")

    safe_model = MODEL_NAME.replace("/", "_")
    default_model_path = str(MODEL_DIR / safe_model / "best")

    p.add_argument(
        "--model_path",
        default=default_model_path,
        help="Path to fine-tuned model directory."
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Confidence threshold for positive prediction."
    )

    p.add_argument(
        "--min_msgs",
        type=int,
        default=MIN_MSGS,
        help="Minimum total messages before any decision is fired."
    )

    p.add_argument(
        "--score_every",
        action="store_true",
        default=SCORE_EVERY,
        help="Score on every message. Default is usually only target-user rounds."
    )

    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print(f"[PREDICT] Loading prediction data from {PREPROCESSED} ...")
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