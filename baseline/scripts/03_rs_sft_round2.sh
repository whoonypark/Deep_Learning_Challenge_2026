#!/usr/bin/env bash
# Round 2 of rejection-sampling SFT (STaR):
#   resample ONLY the questions round 1 never solved, using the round-1
#   adapter as sampler (k=8, T=1.0) -> union SFT data from both rounds ->
#   retrain from the base model -> eval -> k=16 leaderboard submission.
#
# Prerequisite: scripts/02_rejection_sampling_sft.sh finished
#   (needs $DLC_OUTPUT_DIR/rs_train/preds.jsonl and sft-lora-r1).
#
#   cd ~/private/test/competition/deep-learning-challenge-2026/baseline
#   bash scripts/03_rs_sft_round2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"
R1="$DLC_OUTPUT_DIR/rs_train"
R2="$DLC_OUTPUT_DIR/rs_train_r2"

# 1) questions with zero correct samples in round 1
python src/select_unsolved.py --preds "$R1/preds.jsonl" \
    --pool "$PROC/train_pool.csv" --out "$PROC/train_unsolved_r1.csv"

# 2) resample them with the round-1 model (double the budget, hot sampling)
#    NOTE: use the MERGED model, not --lora — vllm's LoRA path needs Triton
#    kernels the CUDA-11.8 driver can't load ("device kernel image is invalid")
python src/infer_vllm.py --input "$PROC/train_unsolved_r1.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r1" --k 8 --temperature 1.0 \
    --out-dir "$R2"

# 3) union SFT data: round-1 correct + round-2 newly-solved
python src/build_sft_data.py --preds "$R1/preds.jsonl" "$R2/preds.jsonl" \
    --out "$PROC/sft_r2.jsonl"

# 4) train round-2 LoRA from the base model on the combined data
#    (training env — see note in 02_rejection_sampling_sft.sh)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/sft_lora.py --data "$PROC/sft_r2.jsonl" \
    --out "$DLC_OUTPUT_DIR/sft-lora-r2"

# 4b) merge the adapter (inference always uses merged models — see step 2 note)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/sft-lora-r2" \
    --out "$DLC_OUTPUT_DIR/merged-r2"

# 5) eval on val (greedy + SC@8) — compare against r1 before trusting it
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r2" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft2_greedy"
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r2" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft2_sc8"

# 6) leaderboard submission with a bigger vote (k=16)
python src/infer_vllm.py --input "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r2" --k 16 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/leaderboard_sft2_sc16" \
    --submission "$DLC_OUTPUT_DIR/leaderboard_sft2_sc16/submission.csv" \
    --no-save-texts
