#!/usr/bin/env python3
"""
01_explore_data.py
------------------
Exploratory Data Analysis for the eRisk 2025 Task 2 dataset.

Outputs summary statistics, class balance, and message-count distributions
to stdout and saves figures in outputs/.
"""

import json
import os
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ── Config ─────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

DATA_DIR    = Path(CFG["paths"]["data_dir"])
GT_FILE     = Path(CFG["paths"]["ground_truth"])
OUTPUT_DIR  = Path(CFG["paths"]["output_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_ground_truth(gt_path: Path) -> dict[str, int]:
    """Return {subject_id: label} from the space-separated GT file."""
    labels = {}
    with open(gt_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


def load_all_json_files(data_dir: Path) -> list[dict]:
    """Load every .json file in data_dir into a list of raw dicts."""
    records = []
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"[ERROR] No .json files found in {data_dir}. "
                 "Please place your dataset there.")
    for jf in json_files:
        with open(jf, encoding='utf-8', errors='replace') as f:
            try:
                data = json.load(f)
                # Each file may contain a list of thread dicts or a single dict
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Could not parse {jf.name}: {e}")
    return records


def extract_subject_stats(records: list[dict], labels: dict[str, int]) -> pd.DataFrame:
    """
    Iterate over all threads and collect per-subject statistics.
    Returns a DataFrame with one row per subject.
    """
    subject_stats = defaultdict(lambda: {
        "num_threads": 0,
        "total_posts": 0,
        "total_comments": 0,
        "total_words": 0,
        "self_comments": 0,
        "label": None,
    })

    for record in records:
        # Each record is one thread (submission + comments)
        submission = record.get("submission", {})
        comments   = record.get("comments", [])

        # Identify the target subject for this thread
        # Field names may vary - support both schema versions
        target_id = (
            submission.get("targetSubject")
            or submission.get("target_subject")
            or None
        )

        # Also find who is flagged as target=True in comments
        target_from_comments = None
        for c in comments:
            if c.get("target") is True:
                target_from_comments = c.get("user_id") or c.get("author")
                break
        # Submission target flag
        if submission.get("target") is True:
            target_from_comments = submission.get("user_id") or submission.get("author")

        subject_id = target_id or target_from_comments
        if subject_id is None:
            continue

        s = subject_stats[subject_id]
        s["num_threads"] += 1
        s["label"] = labels.get(subject_id)

        # Submission text
        sub_text = " ".join(filter(None, [
            submission.get("title", ""),
            submission.get("body", ""),
        ]))
        s["total_posts"] += 1
        s["total_words"] += len(sub_text.split())

        # Comments
        for c in comments:
            s["total_comments"] += 1
            body = c.get("body", "")
            s["total_words"] += len(body.split())
            author = c.get("user_id") or c.get("author", "")
            if author == subject_id:
                s["self_comments"] += 1

    rows = []
    for sid, stats in subject_stats.items():
        rows.append({"subject_id": sid, **stats})
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, labels: dict):
    print("\n" + "=" * 60)
    print("  eRisk 2025 Task 2 - Dataset Summary")
    print("=" * 60)

    total     = len(labels)
    positive  = sum(v == 1 for v in labels.values())
    negative  = sum(v == 0 for v in labels.values())
    unknown   = total - positive - negative

    print(f"\n  Total labelled subjects : {total}")
    print(f"  Positive (depression)   : {positive}  ({100*positive/total:.1f}%)")
    print(f"  Negative (control)      : {negative}  ({100*negative/total:.1f}%)")
    if unknown:
        print(f"  Unknown label           : {unknown}")

    labelled_df = df[df["label"].notna()].copy()
    labelled_df["label"] = labelled_df["label"].astype(int)

    for grp_name, grp in labelled_df.groupby("label"):
        tag = "POSITIVE" if grp_name == 1 else "NEGATIVE"
        print(f"\n  [{tag}] n={len(grp)}")
        print(f"    Avg threads/subject  : {grp['num_threads'].mean():.2f}")
        print(f"    Avg comments/subject : {grp['total_comments'].mean():.2f}")
        print(f"    Avg words/subject    : {grp['total_words'].mean():.1f}")
        print(f"    Avg self-comments    : {grp['self_comments'].mean():.2f}")

    print("\n")


def plot_distributions(df: pd.DataFrame, out_dir: Path):
    """Save distribution plots for key metrics."""
    labelled = df[df["label"].notna()].copy()
    labelled["label"] = labelled["label"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("eRisk 2025 Task 2 - Dataset Distributions", fontsize=14)

    metrics = [
        ("total_comments", "Total Comments"),
        ("total_words",    "Total Words"),
        ("self_comments",  "Self-Comments"),
        ("num_threads",    "Number of Threads"),
    ]

    colors = {0: "#4C9BE8", 1: "#E84C4C"}
    labels_map = {0: "Negative (control)", 1: "Positive (depression)"}

    for ax, (col, title) in zip(axes.flat, metrics):
        for label_val, group in labelled.groupby("label"):
            vals = group[col].clip(upper=group[col].quantile(0.99))
            ax.hist(vals, bins=30, alpha=0.6,
                    color=colors[label_val],
                    label=labels_map[label_val])
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Subjects")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "eda_distributions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [EDA] Saved plot -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[EDA] Loading ground truth from {GT_FILE} ...")
    if not GT_FILE.exists():
        sys.exit(f"[ERROR] Ground truth file not found: {GT_FILE}")
    labels = load_ground_truth(GT_FILE)
    print(f"[EDA] Loaded labels for {len(labels)} subjects.")

    print(f"[EDA] Loading JSON files from {DATA_DIR} ...")
    records = load_all_json_files(DATA_DIR)
    print(f"[EDA] Loaded {len(records)} thread records.")

    print("[EDA] Extracting per-subject statistics ...")
    df = extract_subject_stats(records, labels)

    print_summary(df, labels)
    plot_distributions(df, OUTPUT_DIR)

    # Save stats CSV for reference
    stats_path = OUTPUT_DIR / "subject_stats.csv"
    df.to_csv(stats_path, index=False)
    print(f"  [EDA] Subject stats saved -> {stats_path}")


if __name__ == "__main__":
    main()
