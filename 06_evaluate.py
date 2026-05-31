#!/usr/bin/env python3
"""
06_evaluate.py
--------------
Full eRisk Task 2 evaluation suite.

Metrics implemented:
  ┌──────────────┬─────────────────────────────────────────────────────────┐
  │ ERDE(o)      │ Early Risk Detection Error (penalises late decisions)    │
  │ Flatency     │ F1 weighted by speed (latency-penalised F1)              │
  │ P, R, F1     │ Standard decision metrics                               │
  │ latencyT     │ Normalised latency (messages to decision / total msgs)   │
  │ speed        │ 1 - latencyT                                             │
  │ P@k, NDCG@k  │ Ranking metrics at k=10 and k=100                       │
  └──────────────┴─────────────────────────────────────────────────────────┘

Usage:
  python scripts/06_evaluate.py
  python scripts/06_evaluate.py --preds outputs/predictions_full.json
"""

import argparse
import json
import math
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

OUTPUT_DIR   = Path(CFG["paths"]["output_dir"])
ERDE_O_VALS  = CFG["evaluation"]["erde_o_values"]
NDCG_CUTOFFS = CFG["evaluation"]["ndcg_cutoffs"]
P_AT_K       = CFG["evaluation"]["p_at_k"]


# ── ERDE ──────────────────────────────────────────────────────────────────────

def lc(delay: int, o: int) -> float:
    """Stable latency cost function for ERDE."""
    x = delay - o

    if x >= 50:
        return 1.0

    if x <= -50:
        return 0.0

    return 1.0 / (1.0 + math.exp(-x))


def erde(results: list[dict], o: int) -> float:
    """
    ERDE(o): Early Risk Detection Error with delay penalty o.

    For positive subjects:
      cost = lc(delay, o)          if correctly identified
      cost = 1.0                   if missed (false negative)
    For negative subjects:
      cost = c_fp * 1.0            if wrongly identified (false positive)
      cost = 0.0                   if correctly rejected

    c_fp is the false-positive penalty (standard eRisk: 0.5).
    """
    C_FP      = 0.5
    costs     = []
    labelled  = [r for r in results if r["true_label"] is not None]

    for r in labelled:
        true  = r["true_label"]
        pred  = r["predicted_label"]
        delay = r["decision_round"]  # number of messages seen when decision fired

        if true == 1:
            if pred == 1:
                costs.append(lc(delay, o))
            else:
                costs.append(1.0)   # missed positive
        else:
            if pred == 1:
                costs.append(C_FP)  # false positive
            else:
                costs.append(0.0)   # correct rejection

    return sum(costs) / len(costs) if costs else 0.0


# ── Flatency ──────────────────────────────────────────────────────────────────

def flatency(results: list[dict]) -> tuple[float, float]:
    """
    Latency-weighted F1.
    speed_i = 1 - (decision_round_i / total_rounds_i)
    flatency = F1 * mean_speed
    """
    labelled = [r for r in results if r["true_label"] is not None]
    if not labelled:
        return 0.0, 0.0

    true  = [r["true_label"]      for r in labelled]
    preds = [r["predicted_label"] for r in labelled]

    f1 = f1_score(true, preds, zero_division=0)

    speeds = []
    for r in labelled:
        total = r.get("num_total_rounds") or 1
        dr    = r.get("decision_round", total)
        speeds.append(1.0 - (dr / total))

    mean_speed = float(np.mean(speeds))
    mean_latency = 1.0 - mean_speed
    flat = f1 * mean_speed

    return flat, mean_latency


# ── Ranking metrics (P@k, NDCG@k) ────────────────────────────────────────────

def precision_at_k(sorted_results: list[dict], k: int) -> float:
    """P@k: fraction of top-k ranked subjects that are truly positive."""
    top_k  = sorted_results[:k]
    hits   = sum(1 for r in top_k if r.get("true_label") == 1)
    return hits / k if k > 0 else 0.0


def dcg_at_k(sorted_results: list[dict], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, r in enumerate(sorted_results[:k], start=1):
        rel = float(r.get("true_label") or 0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(sorted_results: list[dict], k: int) -> float:
    """NDCG@k. Ideal ranking puts all positives at the top."""
    # Ideal: sort by true label descending
    ideal = sorted(sorted_results, key=lambda r: r.get("true_label") or 0, reverse=True)
    idcg  = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(sorted_results, k) / idcg


def ranking_metrics(results: list[dict], k_vals: list[int]) -> dict[str, float]:
    """Compute P@k and NDCG@k for all k in k_vals on labelled results."""
    labelled = [r for r in results if r.get("true_label") is not None]
    sorted_r = sorted(labelled, key=lambda r: r["score"], reverse=True)

    out = {}
    for k in k_vals:
        out[f"P@{k}"]    = precision_at_k(sorted_r, k)
        out[f"NDCG@{k}"] = ndcg_at_k(sorted_r, k)
    return out


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate(results: list[dict]) -> dict:
    labelled = [r for r in results if r["true_label"] is not None]
    if not labelled:
        print("[EVAL] No labelled results - skipping decision-based metrics.")
        return {}

    true  = [r["true_label"]      for r in labelled]
    preds = [r["predicted_label"] for r in labelled]

    p  = precision_score(true, preds, zero_division=0)
    r  = recall_score(true, preds,    zero_division=0)
    f1 = f1_score(true, preds,        zero_division=0)

    flat, mean_lat = flatency(results)
    mean_speed     = 1.0 - mean_lat

    erde_scores = {f"ERDE{o}": erde(results, o) for o in ERDE_O_VALS}

    # Ranking (all k_vals combined)
    all_k    = sorted(set(NDCG_CUTOFFS + P_AT_K))
    rank_met = ranking_metrics(results, all_k)

    metrics = {
        "num_subjects":  len(labelled),
        "precision":     p,
        "recall":        r,
        "f1":            f1,
        "flatency":      flat,
        "mean_speed":    mean_speed,
        "mean_latencyT": mean_lat,
        **erde_scores,
        **rank_met,
    }
    return metrics


def print_report(metrics: dict, results: list[dict]):
    labelled = [r for r in results if r["true_label"] is not None]
    true  = [r["true_label"]      for r in labelled]
    preds = [r["predicted_label"] for r in labelled]

    print("\n" + "=" * 60)
    print("  eRisk 2025 Task 2 - Evaluation Report")
    print("=" * 60)

    print(f"\n  Subjects evaluated : {metrics.get('num_subjects', 0)}")

    print("\n  ── Decision-Based Metrics ──────────────────────────────")
    print(f"  Precision   : {metrics.get('precision', 0):.4f}")
    print(f"  Recall      : {metrics.get('recall', 0):.4f}")
    print(f"  F1          : {metrics.get('f1', 0):.4f}")
    print(f"  Flatency    : {metrics.get('flatency', 0):.4f}")
    print(f"  Mean speed  : {metrics.get('mean_speed', 0):.4f}")
    print(f"  Mean latency: {metrics.get('mean_latencyT', 0):.4f}")

    for o in ERDE_O_VALS:
        key = f"ERDE{o}"
        print(f"  {key:<12}: {metrics.get(key, 0):.4f}")

    print("\n  ── Ranking Metrics ─────────────────────────────────────")
    for k in P_AT_K:
        print(f"  P@{k:<10}: {metrics.get(f'P@{k}', 0):.4f}")
    for k in NDCG_CUTOFFS:
        print(f"  NDCG@{k:<8}: {metrics.get(f'NDCG@{k}', 0):.4f}")

    print("\n  ── Classification Report ───────────────────────────────")
    print(classification_report(true, preds,
                                target_names=["Control", "Depression"],
                                zero_division=0))

    # Decision round stats
    pos_results = [r for r in labelled if r["true_label"] == 1]
    if pos_results:
        rounds = [r["decision_round"] for r in pos_results]
        print(f"  ── Decision Round Stats (positive subjects) ──────────")
        print(f"  Mean  : {np.mean(rounds):.1f}")
        print(f"  Median: {np.median(rounds):.1f}")
        print(f"  Min   : {min(rounds)}")
        print(f"  Max   : {max(rounds)}")

    print("=" * 60 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Task 2 predictions")
    default_preds = str(OUTPUT_DIR / "predictions_full.json")
    p.add_argument("--preds", default=default_preds,
                   help="Path to predictions_full.json from 05_predict.py")
    p.add_argument("--out",   default=str(OUTPUT_DIR / "evaluation_results.json"),
                   help="Where to save the JSON results summary")
    return p.parse_args()


def main():
    args = parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        sys.exit(f"[ERROR] Predictions file not found: {preds_path}\n"
                 "Run 05_predict.py first.")

    with open(preds_path, encoding='utf-8', errors='replace') as f:
        results = json.load(f)

    print(f"[EVAL] Loaded {len(results)} subject predictions.")

    metrics = evaluate(results)
    print_report(metrics, results)

    out_path = Path(args.out)
    with open(out_path, "w", encoding='utf-8', errors='replace') as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Evaluation results saved -> {out_path}")


if __name__ == "__main__":
    main()
