#!/usr/bin/env bash
# run_all.sh - Execute the full eRisk Task 2 pipeline end-to-end.
# Usage: bash run_all.sh [--skip-train] [--search]
#
# Steps:
#   1. EDA
#   2. Preprocess
#   3. Prepare train/val splits
#   4. Train (or HP search)
#   5. Predict (sequential early detection)
#   6. Evaluate

set -euo pipefail

SKIP_TRAIN=0
SEARCH_FLAG=""

for arg in "$@"; do
  case $arg in
    --skip-train) SKIP_TRAIN=1 ;;
    --search)     SEARCH_FLAG="--search" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "  eRisk 2025 Task 2 - Full Pipeline"
echo "================================================================"

echo ""
echo "[1/6] Exploratory Data Analysis ..."
python "${SCRIPT_DIR}/01_explore_data.py" 2>&1 | tee "${LOG_DIR}/01_eda.log"

echo ""
echo "[2/6] Preprocessing ..."
python "${SCRIPT_DIR}/02_preprocess.py" 2>&1 | tee "${LOG_DIR}/02_preprocess.log"

echo ""
echo "[3/6] Preparing train/val splits ..."
python "${SCRIPT_DIR}/03_prepare_training.py" 2>&1 | tee "${LOG_DIR}/03_split.log"

if [ "$SKIP_TRAIN" -eq 0 ]; then
  echo ""
  echo "[4/6] Training model ${SEARCH_FLAG} ..."
  python "${SCRIPT_DIR}/04_train.py" ${SEARCH_FLAG} 2>&1 | tee "${LOG_DIR}/04_train.log"
else
  echo ""
  echo "[4/6] Skipping training (--skip-train specified)."
fi

echo ""
echo "[5/6] Running sequential early-detection inference ..."
python "${SCRIPT_DIR}/05_predict.py" 2>&1 | tee "${LOG_DIR}/05_predict.log"

echo ""
echo "[6/6] Evaluating predictions ..."
python "${SCRIPT_DIR}/06_evaluate.py" 2>&1 | tee "${LOG_DIR}/06_evaluate.log"

echo ""
echo "================================================================"
echo "  Pipeline complete. Results in outputs/"
echo "================================================================"