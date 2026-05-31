#!/usr/bin/env python3
"""
02_preprocess.py
----------------
Parse the raw JSON thread files into a structured JSONL format.

For each subject, we emit ONE record per "message round" (i.e., after each
new message by the target subject appears). Each record contains:

  {
    "subject_id": "subject_XXX",
    "label": 0 or 1 (or null for unlabelled test),
    "round": int,                 # which message round (1-indexed)
    "formatted_text": str,        # [MSG] [USER] TARGET/CONTEXT ... string
    "num_target_msgs": int,       # how many target messages seen so far
    "num_total_msgs": int,        # total messages seen so far
  }

The formatted_text follows the paper's schema:
  [MSG] [USER] TARGET <text> [MSG] [USER] CONTEXT <text> ...

Output: data/preprocessed.jsonl  (one JSON object per line)
"""

import json
import os
import re
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

DATA_DIR       = Path(CFG["paths"]["data_dir"])
GT_FILE        = Path(CFG["paths"]["ground_truth"])
OUT_FILE       = Path(CFG["paths"]["preprocessed"])
MAX_CTX_MSGS   = CFG["preprocessing"]["max_context_messages"]
REMOVE_URLS    = CFG["preprocessing"]["remove_urls"]
REMOVE_BRACKETS= CFG["preprocessing"]["remove_brackets"]
ANONYMIZE      = CFG["preprocessing"]["anonymize_users"]

Out_FILE_parent = OUT_FILE.parent
Out_FILE_parent.mkdir(parents=True, exist_ok=True)


# ── Text Cleaning ──────────────────────────────────────────────────────────────

URL_RE     = re.compile(r"https?://\S+|www\.\S+")
BRACKET_RE = re.compile(r"\[.*?\]")
NEWLINE_RE = re.compile(r"\s+")


def clean_text(text: str, user_id: str = "") -> str:
    """Strip noise from raw Reddit text."""
    if not text:
        return ""
    if REMOVE_URLS:
        text = URL_RE.sub(" ", text)
    if REMOVE_BRACKETS:
        text = BRACKET_RE.sub(" ", text)
    if ANONYMIZE and user_id:
        text = text.replace(user_id, "user")
    text = NEWLINE_RE.sub(" ", text).strip()
    return text


# ── JSON Schema Helpers ────────────────────────────────────────────────────────

def get_submission_author(sub: dict) -> str:
    return sub.get("author") or sub.get("user_id") or ""

def get_submission_target(sub: dict) -> Optional[str]:
    return sub.get("targetSubject") or sub.get("target_subject") or None

def get_submission_text(sub: dict) -> str:
    title = sub.get("title") or ""
    body  = sub.get("body")  or ""
    return f"{title} {body}".strip()

def get_submission_date(sub: dict) -> str:
    return sub.get("date") or sub.get("created_utc") or ""

def get_submission_id(sub: dict) -> str:
    return sub.get("submissionId") or sub.get("submission_id") or ""

def get_comment_author(c: dict) -> str:
    return c.get("author") or c.get("user_id") or ""

def get_comment_id(c: dict) -> str:
    return c.get("commentId") or c.get("comment_id") or ""

def get_comment_parent(c: dict) -> str:
    return c.get("parent") or c.get("parent_id") or ""

def get_comment_date(c: dict) -> str:
    return c.get("date") or c.get("created_utc") or ""

def is_target(node: dict, target_id: str) -> bool:
    """Return True if this submission/comment belongs to the target subject."""
    author = get_submission_author(node) if "submission_id" in node or "submissionId" in node \
             else get_comment_author(node)
    if author == target_id:
        return True
    return node.get("target") is True


# ── Core: Build chronological message list for one thread ──────────────────────

def build_message_sequence(record: dict, target_id: str) -> list[dict]:
    """
    Flatten submission + comments into a chronological list of messages,
    each annotated with whether it is from the target subject.

    Returns: list of dicts with keys: type, author, text, date, is_target
    """
    messages = []

    sub = record.get("submission", {})
    sub_author = get_submission_author(sub)
    sub_text   = get_submission_text(sub)
    sub_date   = get_submission_date(sub)
    sub_id     = get_submission_id(sub)

    if sub_text.strip():
        messages.append({
            "id":        sub_id,
            "parent_id": None,
            "type":      "submission",
            "author":    sub_author,
            "text":      clean_text(sub_text, sub_author),
            "date":      sub_date,
            "is_target": (sub_author == target_id) or sub.get("target", False),
        })

    # Sort comments by date string (ISO 8601 sorts lexicographically)
    comments = sorted(
        record.get("comments", []),
        key=lambda c: get_comment_date(c)
    )
    for c in comments:
        c_author = get_comment_author(c)
        c_text   = c.get("body") or ""
        c_date   = get_comment_date(c)
        c_id     = get_comment_id(c)
        c_parent = get_comment_parent(c)

        if not c_text.strip():
            continue

        messages.append({
            "id":        c_id,
            "parent_id": c_parent,
            "type":      "comment",
            "author":    c_author,
            "text":      clean_text(c_text, c_author),
            "date":      c_date,
            "is_target": (c_author == target_id) or c.get("target", False),
        })

    return messages


# ── Core: Format a window of messages into the [MSG] schema ──────────────────

def format_conversation_window(messages: list[dict]) -> str:
    """
    Convert a list of message dicts into the structured string:
      [MSG] [USER] TARGET <text> [MSG] [USER] CONTEXT <text> ...

    Keeps up to MAX_CTX_MSGS messages total to avoid blowing the token budget.
    If the window is longer, we keep the most recent MAX_CTX_MSGS messages
    but always include all TARGET messages.
    """
    if not messages:
        return ""

    # Separate target vs context messages
    target_msgs  = [m for m in messages if m["is_target"]]
    context_msgs = [m for m in messages if not m["is_target"]]

    # Build the window: always include all target messages;
    # fill remaining slots with the most recent context messages
    slots_for_context = max(0, MAX_CTX_MSGS - len(target_msgs))
    selected_context  = context_msgs[-slots_for_context:] if slots_for_context > 0 else []

    window = sorted(
        target_msgs + selected_context,
        key=lambda m: m["date"]
    )

    parts = []
    for m in window:
        role = "TARGET" if m["is_target"] else "CONTEXT"
        parts.append(f"[MSG] [USER] {role} {m['text']}")

    return " ".join(parts)


# ── Main Processing ────────────────────────────────────────────────────────────

def load_ground_truth(gt_path: Path) -> dict[str, int]:
    labels = {}
    if not gt_path.exists():
        print(f"[WARN] Ground truth not found at {gt_path}. "
              "All labels will be null (inference-only mode).")
        return labels
    with open(gt_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels


def infer_target_id(record: dict) -> Optional[str]:
    """
    Best-effort: find the target subject ID for a thread record.
    The README says targetSubject field; fall back to scanning for target=True.
    """
    sub = record.get("submission", {})

    # Explicit field
    explicit = get_submission_target(sub)
    if explicit:
        return explicit

    # submission itself is flagged target
    if sub.get("target") is True:
        return get_submission_author(sub)

    # scan comments
    for c in record.get("comments", []):
        if c.get("target") is True:
            return get_comment_author(c)

    return None


def process_all_records(records: list[dict], labels: dict[str, int]) -> list[dict]:
    """
    For each thread, build per-round examples (one example per message seen).
    A new example is emitted every time a new message appears (or only on
    TARGET messages, depending on config).

    Returns a flat list of example dicts ready for JSONL serialisation.
    """
    # Group records by target subject - one subject may have multiple threads
    subject_threads: dict[str, list[list[dict]]] = {}

    for record in records:
        target_id = infer_target_id(record)
        if target_id is None:
            continue
        msgs = build_message_sequence(record, target_id)
        if not msgs:
            continue
        subject_threads.setdefault(target_id, []).append(msgs)

    examples = []
    for subject_id, all_thread_msgs in subject_threads.items():
        label = labels.get(subject_id)  # None for unlabelled

        # Merge all threads into one chronological stream
        combined = sorted(
            [m for thread in all_thread_msgs for m in thread],
            key=lambda m: m["date"]
        )

        # Emit one example per message (rolling window)
        for i, _ in enumerate(combined, start=1):
            window  = combined[:i]
            n_target = sum(1 for m in window if m["is_target"])
            fmt_text = format_conversation_window(window)

            if not fmt_text.strip():
                continue

            examples.append({
                "subject_id":      subject_id,
                "label":           label,
                "round":           i,
                "num_target_msgs": n_target,
                "num_total_msgs":  i,
                "formatted_text":  fmt_text,
            })

    return examples


def main():
    print(f"[PREPROCESS] Loading ground truth from {GT_FILE} ...")
    labels = load_ground_truth(GT_FILE)
    print(f"[PREPROCESS] {len(labels)} labelled subjects.")

    print(f"[PREPROCESS] Scanning {DATA_DIR} for JSON files ...")
    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        sys.exit(f"[ERROR] No .json files found in {DATA_DIR}.")
    print(f"[PREPROCESS] Found {len(json_files)} files.")

    all_records = []
    for jf in tqdm(json_files, desc="Loading JSON files"):
        with open(jf, encoding='utf-8', errors='replace') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                else:
                    all_records.append(data)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping {jf.name}: {e}")

    print(f"[PREPROCESS] Loaded {len(all_records)} thread records. Processing ...")
    examples = process_all_records(all_records, labels)

    print(f"[PREPROCESS] Writing {len(examples)} examples to {OUT_FILE} ...")
    with open(OUT_FILE, "w") as out:
        for ex in examples:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Quick stats
    labelled   = [e for e in examples if e["label"] is not None]
    positives  = [e for e in labelled if e["label"] == 1]
    negatives  = [e for e in labelled if e["label"] == 0]
    subjects   = len(set(e["subject_id"] for e in examples))

    print(f"\n[PREPROCESS] Done.")
    print(f"  Total examples  : {len(examples)}")
    print(f"  Unique subjects : {subjects}")
    print(f"  Labelled exs    : {len(labelled)}")
    print(f"    Positive      : {len(positives)}")
    print(f"    Negative      : {len(negatives)}")
    print(f"  Output          : {OUT_FILE}")


if __name__ == "__main__":
    main()
