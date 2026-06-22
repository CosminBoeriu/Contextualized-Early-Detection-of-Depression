import json
import os
import re
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from pathlib import Path
from typing import Optional
from collections import deque
import yaml
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

DATA_DIR = Path(CFG["paths"]["data_dir"])
GT_FILE = Path(CFG["paths"]["ground_truth"])
OUT_FILE = Path(CFG["paths"]["preprocessed"])
MAX_CTX_MSGS = CFG["preprocessing"]["max_context_messages"]
REMOVE_URLS = CFG["preprocessing"]["remove_urls"]
REMOVE_BRACKETS = CFG["preprocessing"]["remove_brackets"]
ANONYMIZE = CFG["preprocessing"]["anonymize_users"]

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


def build_message_sequence(record: dict, target_id: str) -> list[dict]:
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


def format_conversation_window(messages: list[dict]) -> str:
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
    sub = record.get("submission", {})

    explicit = get_submission_target(sub)
    if explicit:
        return explicit

    if sub.get("target") is True:
        return get_submission_author(sub)

    # scan comments
    for c in record.get("comments", []):
        if c.get("target") is True:
            return get_comment_author(c)

    return None


MAX_TARGET_MSGS = CFG["preprocessing"].get("max_target_messages", 20)
MAX_CONTEXT_MSGS = CFG["preprocessing"].get("max_context_messages", 30)

# Useful for reducing training size.
# Keep early rounds + periodic rounds + final round.
MILESTONE_ROUNDS = set(CFG["preprocessing"].get(
    "milestone_rounds",
    [1, 2, 3, 5, 10, 20, 50, 100, 200]
))

USE_MILESTONES_ONLY = CFG["preprocessing"].get("use_milestones_only", False)


def format_message(m: dict) -> str:
    role = "TARGET" if m["is_target"] else "CONTEXT"
    msg_type = m.get("type", "message").upper()
    return f"[MSG] [USER] {role} [{msg_type}] {m['text']}"


def format_fast_window(
    target_buffer: deque,
    context_buffer: deque,
) -> str:
    window = list(target_buffer) + list(context_buffer)
    window.sort(key=lambda m: m["date"])

    return " ".join(format_message(m) for m in window if m.get("text", "").strip())


def process_all_records(records: list[dict], labels: dict[str, int]) -> list[dict]:
    subject_threads: dict[str, list[list[dict]]] = {}

    print("[PREPROCESS] Grouping threads by subject")

    for record in records:
        target_id = infer_target_id(record)
        if target_id is None:
            continue

        msgs = build_message_sequence(record, target_id)
        if not msgs:
            continue

        subject_threads.setdefault(target_id, []).append(msgs)

    examples = []

    print("[PREPROCESS] Creating incremental examples")

    for subject_id, all_thread_msgs in tqdm(subject_threads.items(), desc="Subjects"):
        label = labels.get(subject_id)

        combined = sorted(
            [m for thread in all_thread_msgs for m in thread],
            key=lambda m: m["date"]
        )

        target_buffer = deque(maxlen=MAX_TARGET_MSGS)
        context_buffer = deque(maxlen=MAX_CONTEXT_MSGS)

        n_target = 0
        emitted_for_subject = []

        for total_seen, msg in enumerate(combined, start=1):

            if msg["is_target"]:
                n_target += 1
                target_buffer.append(msg)

                formatted_text = format_fast_window(
                    target_buffer=target_buffer,
                    context_buffer=context_buffer
                )

                if not formatted_text.strip():
                    continue

                example = {
                    "subject_id": subject_id,
                    "label": label,
                    "round": n_target,
                    "num_target_msgs": n_target,
                    "num_total_msgs": total_seen,
                    "formatted_text": formatted_text,
                }

                emitted_for_subject.append(example)

            else:
                context_buffer.append(msg)

        if not emitted_for_subject:
            continue

        if USE_MILESTONES_ONLY:
            final_round = emitted_for_subject[-1]["round"]

            selected = [
                ex for ex in emitted_for_subject
                if ex["round"] in MILESTONE_ROUNDS or ex["round"] == final_round
            ]

            examples.extend(selected)
        else:
            examples.extend(emitted_for_subject)

    return examples


def main():
    print(f"[PREPROCESS] Loading ground truth from {GT_FILE}")
    labels = load_ground_truth(GT_FILE)
    print(f"[PREPROCESS] {len(labels)} labelled subjects.")

    print(f"[PREPROCESS] Scanning {DATA_DIR} for JSON files")
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

    print(f"[PREPROCESS] Loaded {len(all_records)} thread records. Processing")
    examples = process_all_records(all_records, labels)

    print(f"[PREPROCESS] Writing {len(examples)} examples to {OUT_FILE}")
    with open(OUT_FILE, "w", encoding="utf-8", errors="replace") as out:
        for ex in examples:
            out.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Quick stats
    labelled   = [e for e in examples if e["label"] is not None]
    positives  = [e for e in labelled if e["label"] == 1]
    negatives  = [e for e in labelled if e["label"] == 0]
    subjects   = len(set(e["subject_id"] for e in examples))

    print(f"\n[PREPROCESS] Done.")
    print(f"Total examples: {len(examples)}")
    print(f"Unique subjects: {subjects}")
    print(f"Labelled exs: {len(labelled)}")
    print(f"Positive: {len(positives)}")
    print(f"Negative: {len(negatives)}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
