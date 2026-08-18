#!/usr/bin/env bash
# Round 3b: external SFT, corrected after the r3 failure (LB 0.743 < r2 0.783).
# Changes vs 04: (1) GSM-source rows only — challenge data is word-problem
# style, MATH-style rows caused distribution drift; (2) much smaller external
# share (30k vs 100k) so self-distill data keeps weight; (3) gentler training:
# lr 5e-5, max_len 3072 to avoid truncated solutions.
#
# NO leaderboard submission in this script — evaluate on val first and only
# submit if it beats r2 (val SC@8 0.716 / r1 0.726).
#
#   bash scripts/04b_external_gsm_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"

# 1) re-filter: GSM-style sources only, shorter solutions, 30k cap
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/prepare_external_data.py \
    --sources gsm8k \
    --max-samples 30000 --max-chars 4000 \
    --include "$PROC/sft_r2.jsonl" \
    --out "$PROC/sft_ext_gsm.jsonl"

# 2) gentle LoRA SFT
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/sft_lora.py --data "$PROC/sft_ext_gsm.jsonl" \
    --out "$DLC_OUTPUT_DIR/sft-lora-r3b" \
    --epochs 1 --lr 5e-5 --max-len 3072 --batch-size 2 --grad-accum 16

# 3) merge
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/sft-lora-r3b" \
    --out "$DLC_OUTPUT_DIR/merged-r3b"

# 4) val gate — greedy + SC@8; submit manually ONLY if this beats r1/r2
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r3b" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft3b_greedy"
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r3b" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft3b_sc8"
