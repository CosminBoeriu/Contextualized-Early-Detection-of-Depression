import json
import random
import sys
import sys; sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
from collections import defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

CFG_PATH = Path(__file__).parent / "config.yaml"
with open(CFG_PATH, encoding='utf-8', errors='replace') as f:
    CFG = yaml.safe_load(f)

IN_FILE = Path(CFG["paths"]["preprocessed"])
TRAIN_FILE = Path(CFG["paths"]["train_file"])
VAL_FILE = Path(CFG["paths"]["val_file"])
VAL_SIZE = CFG["split"]["val_size"]
SEED = CFG["split"]["random_seed"]
STRATIFY = CFG["split"]["stratify"]

random.seed(SEED)

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"[ERROR] {path} not found. Run 02_preprocess.py first.")
    examples = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

def group_by_subject(examples: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[ex["subject_id"]].append(ex)
    # Sort each subject's examples by round number
    for sid in groups:
        groups[sid].sort(key=lambda e: e["round"])
    return groups


def select_training_examples(
    subject_examples: list[dict],
    strategy: str = "last_only",
) -> list[dict]:
    if strategy == "last_only":
        return [subject_examples[-1]]

    elif strategy == "all_rounds":
        return subject_examples

    elif strategy == "milestone":
        milestones = {1, 10, 50, 100, 200}
        selected   = [ex for ex in subject_examples if ex["round"] in milestones]
        selected.append(subject_examples[-1])
        # deduplicate
        seen  = set()
        dedup = []
        for ex in selected:
            if ex["round"] not in seen:
                seen.add(ex["round"])
                dedup.append(ex)
        return dedup

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    TRAIN_STRATEGY = "all_rounds"

    print(f"[SPLIT] Loading {IN_FILE} ...")
    examples = load_jsonl(IN_FILE)
    print(f"[SPLIT] {len(examples)} examples loaded.")

    # Keep only labelled examples for supervised training
    labelled = [e for e in examples if e.get("label") is not None]
    unlabelled = [e for e in examples if e.get("label") is None]
    print(f"[SPLIT] Labelled: {len(labelled)} | Unlabelled (test): {len(unlabelled)}")

    by_subject = group_by_subject(labelled)
    subject_ids = list(by_subject.keys())
    subject_labels = [by_subject[sid][-1]["label"] for sid in subject_ids]

    pos_count = sum(subject_labels)
    neg_count = len(subject_labels) - pos_count
    print(f"[SPLIT] {len(subject_ids)} labelled subjects "
          f"(pos={pos_count}, neg={neg_count})")

    stratify_arg = subject_labels if STRATIFY else None
    train_sids, val_sids = train_test_split(
        subject_ids,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=stratify_arg,
    )
    print(f"[SPLIT] Train subjects: {len(train_sids)} | Val subjects: {len(val_sids)}")

    train_examples = []
    for sid in train_sids:
        train_examples.extend(select_training_examples(by_subject[sid], TRAIN_STRATEGY))

    val_examples = []
    for sid in val_sids:
        val_examples.extend(select_training_examples(by_subject[sid], "last_only"))

    random.shuffle(train_examples)

    def write_jsonl(path: Path, data: list[dict]):
        with open(path, "w", encoding='utf-8', errors='replace') as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(VAL_FILE,   val_examples)

    # Stats
    train_pos = sum(e["label"] == 1 for e in train_examples)
    train_neg = sum(e["label"] == 0 for e in train_examples)
    val_pos = sum(e["label"] == 1 for e in val_examples)
    val_neg = sum(e["label"] == 0 for e in val_examples)

    print(f"\n[SPLIT] Training set  : {len(train_examples)} examples "
          f"(pos={train_pos}, neg={train_neg})")
    print(f"[SPLIT] Validation set: {len(val_examples)} examples "
          f"(pos={val_pos}, neg={val_neg})")
    print(f"[SPLIT] Strategy      : {TRAIN_STRATEGY}")
    print(f"[SPLIT] Saved to {TRAIN_FILE} and {VAL_FILE}")

    test_subjects_file = Path(CFG["paths"]["preprocessed"]).parent / "test_subjects.jsonl"
    if unlabelled:
        test_by_subject = group_by_subject(unlabelled)
        test_examples   = [grp[-1] for grp in test_by_subject.values()]
        write_jsonl(test_subjects_file, test_examples)
        print(f"[SPLIT] Test set (unlabelled): {len(test_examples)} subjects -> {test_subjects_file}")


if __name__ == "__main__":
    main()
