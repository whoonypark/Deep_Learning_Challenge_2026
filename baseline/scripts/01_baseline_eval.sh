#!/usr/bin/env bash
# Zero-shot baseline on the lab GPU: greedy + self-consistency on val,
# then a k=8 leaderboard submission.
#
#   cd ~/private/test/competition/deep-learning-challenge-2026/baseline
#   bash scripts/01_baseline_eval.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh   # data/output dirs + CUDA_VISIBLE_DEVICES=9

python src/prepare_data.py

# 1) greedy floor
python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/val.csv" \
    --k 1 --temperature 0 --out-dir "$DLC_OUTPUT_DIR/val_greedy"

# 2) self-consistency k=8
python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/val.csv" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_sc8"

# 3) leaderboard submission with the better setting (check accuracies first)
python src/infer_vllm.py --input "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/leaderboard_sc8" \
    --submission "$DLC_OUTPUT_DIR/leaderboard_sc8/submission.csv"
