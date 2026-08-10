#!/usr/bin/env bash
# Round 3: SFT on filtered public external data (OpenMathInstruct-2)
# mixed with our round-2 self-distill data, then merge + eval + submission.
#
# Self-distill saturated after round 1 (val: base .720 -> r1 .726 -> r2 .716
# @SC8); external high-quality solutions are the next lever. Public datasets
# are explicitly allowed; keep provenance for the verification deliverable.
#
#   cd ~/private/test/competition/deep-learning-challenge-2026/baseline
#   bash scripts/04_external_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"

# 1) download + filter external data (training env: has `datasets`), mix in r2 data
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/prepare_external_data.py \
    --max-samples 100000 \
    --include "$PROC/sft_r2.jsonl" \
    --out "$PROC/sft_ext.jsonl"

# 2) LoRA SFT — 1 epoch over ~115k samples (several hours on a 24GB GPU)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/sft_lora.py --data "$PROC/sft_ext.jsonl" \
    --out "$DLC_OUTPUT_DIR/sft-lora-r3" --epochs 1

# 3) merge for vllm inference (LoRA path unusable on the CUDA-11.8 driver)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/sft-lora-r3" \
    --out "$DLC_OUTPUT_DIR/merged-r3"

# 4) val eval — greedy + SC@8 (compare against r1: .686 / .726)
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r3" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft3_greedy"
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r3" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft3_sc8"

# 5) leaderboard submission (submit only if step 4 beats r1 on val!)
python src/infer_vllm.py --input "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r3" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/leaderboard_sft3_sc8" \
    --submission "$DLC_OUTPUT_DIR/leaderboard_sft3_sc8/submission.csv" \
    --no-save-texts
