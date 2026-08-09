#!/usr/bin/env bash
# Round 1 of rejection-sampling SFT (STaR-style), all on the lab GPU:
#   generate k solutions per train question -> keep correct ones -> LoRA SFT
#
#   cd ~/private/test/competition/deep-learning-challenge-2026/baseline
#   bash scripts/02_rejection_sampling_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh   # data/output dirs + CUDA_VISIBLE_DEVICES=9

# 1) sample solutions on the train pool (largest step: ~16k questions x k=4)
python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/train_pool.csv" \
    --k 4 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/rs_train"

# 2) keep solutions whose boxed answer matches gold
python src/build_sft_data.py --preds "$DLC_OUTPUT_DIR/rs_train/preds.jsonl" \
    --out "$DLC_OUTPUT_DIR/data_processed/sft.jsonl"

# 3) LoRA SFT
python src/sft_lora.py --data "$DLC_OUTPUT_DIR/data_processed/sft.jsonl" \
    --out "$DLC_OUTPUT_DIR/sft-lora-r1"

# 4) evaluate the adapter on val (greedy + SC), then leaderboard
python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/val.csv" \
    --lora "$DLC_OUTPUT_DIR/sft-lora-r1" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft_greedy"
python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/val.csv" \
    --lora "$DLC_OUTPUT_DIR/sft-lora-r1" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft_sc8"
python src/infer_vllm.py --input "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
    --lora "$DLC_OUTPUT_DIR/sft-lora-r1" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/leaderboard_sft_sc8" \
    --submission "$DLC_OUTPUT_DIR/leaderboard_sft_sc8/submission.csv"
