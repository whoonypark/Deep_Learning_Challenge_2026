#!/usr/bin/env bash
# Round 5: GRPO (integer-exact-match reward) on top of the best SFT model.
# Prompt set = audited-clean questions with intermediate pass rate (1-7 of 8
# in the rs_audit sampling) — the only ones that give GRPO a gradient.
#
# NOTE: training generation runs through transformers (slow). ~400 steps is
# roughly 1-3 days on a 24GB card. Checkpoints save every 50 steps, so you
# can stop early and evaluate any checkpoint.
#
#   conda activate dlc && bash scripts/08_grpo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"
POLICY="$DLC_OUTPUT_DIR/merged-r4"

# 1) build the GRPO prompt set (fast, no GPU)
python src/prepare_grpo_data.py \
    --preds "$DLC_OUTPUT_DIR/rs_audit/preds.jsonl" \
    --only-ids "$PROC/train_pool_clean.csv" \
    --out "$PROC/grpo_prompts.jsonl"

# 2) GRPO training (training env; the long step)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/grpo_train.py \
    --model "$POLICY" \
    --data "$PROC/grpo_prompts.jsonl" \
    --out "$DLC_OUTPUT_DIR/grpo-lora-r5" \
    --max-steps 400

# 3) merge (note: base = the SFT policy, not the original Qwen!)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --base "$POLICY" \
    --adapter "$DLC_OUTPUT_DIR/grpo-lora-r5" \
    --out "$DLC_OUTPUT_DIR/merged-r5"

# 4) clean-val gate
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r5" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_grpo_greedy"
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r5" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_grpo_sc8"

echo "=== accuracy on audited-clean val (481 q) ==="
python src/score_preds.py --ids "$PROC/val_clean.csv" \
    "$DLC_OUTPUT_DIR/val_sft4_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_sft3b_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_grpo_greedy/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_grpo_sc8/preds.jsonl"
