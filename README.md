# eRisk 2025 Task 2 - Contextualized Early Depression Detection Pipeline

## Overview

This pipeline implements the full Task 2 approach based on the SINAI-UJA paper.
It trains transformer-based classifiers on the conversational dataset and performs
early sequential prediction, emitting a decision per subject as soon as the model
is confident enough.

## Project Structure

```
erisk_task2/
├── README.md
├── requirements.txt
├── config.yaml                  # All hyperparameters & paths in one place
│
├── scripts/
│   ├── 01_explore_data.py       # EDA: stats, label balance, message counts
│   ├── 02_preprocess.py         # Parse JSONs → structured conversation format
│   ├── 03_prepare_training.py   # Build train/val splits with the [MSG] format
│   ├── 04_train.py              # Fine-tune transformer (RoBERTa / MentalRoBERTa)
│   ├── 05_predict.py            # Sequential early-detection inference
│   ├── 06_evaluate.py           # ERDE, F1, Flatency, P@10, NDCG metrics
│   └── run_all.sh               # Run the full pipeline end-to-end
│
├── data/                        # Put your dataset files here
│   ├── all_combined/            # ← place your .json files here
│   └── shuffled_ground_truth_labels.txt
│
├── models/                      # Saved checkpoints
└── outputs/                     # Predictions & evaluation results
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit config.yaml - set DATA_DIR, MODEL_NAME, etc.

# 3. Explore your data
python scripts/01_explore_data.py

# 4. Preprocess
python scripts/02_preprocess.py

# 5. Prepare training splits
python scripts/03_prepare_training.py

# 6. Train
python scripts/04_train.py

# 7. Predict (sequential / early detection)
python scripts/05_predict.py

# 8. Evaluate
python scripts/06_evaluate.py
```

## Model Choices (set in config.yaml)

| Model | Notes |
|---|---|
| `roberta-base` | Fast baseline |
| `roberta-large` | Better accuracy |
| `mental_health/mental-roberta-base` | Domain-adapted |
| `mental_health/mental-roberta-large` | Best in paper |
| `mnaylor/mega-base-fastmax` | Long-context option |

## Key Design Decisions

- Conversations are formatted as `[MSG] [USER] TARGET/CONTEXT {text}` chains
- The model sees the growing conversation window at each round
- A confidence threshold controls when to fire an early prediction
- Class imbalance is handled via weighted cross-entropy loss
