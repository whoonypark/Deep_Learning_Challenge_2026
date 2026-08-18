#!/usr/bin/env bash
# Round 4: retrain from the BASE model on audited-clean self-distill data.
# Uses the rs_audit sampling (merged-r2, k=8 T=1.0) restricted to questions
# that passed the label audit — suspect-label solutions never enter training.
# Evaluation gate = val_clean (481 questions, audited).
#
#   conda activate dlc && bash scripts/07_clean_sft.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"

# 1) clean SFT data from the audit sampling pass
python src/build_sft_data.py --preds "$DLC_OUTPUT_DIR/rs_audit/preds.jsonl" \
    --only-ids "$PROC/train_pool_clean.csv" \
    --out "$PROC/sft_clean.jsonl"

# 2) LoRA SFT from base (training env)
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/sft_lora.py --data "$PROC/sft_clean.jsonl" \
    --out "$DLC_OUTPUT_DIR/sft-lora-r4"

# 3) merge
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/sft-lora-r4" \
    --out "$DLC_OUTPUT_DIR/merged-r4"

# 4) eval on CLEAN val
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r4" --k 1 --temperature 0 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft4_greedy"
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --model "$DLC_OUTPUT_DIR/merged-r4" --k 8 --temperature 0.8 \
    --out-dir "$DLC_OUTPUT_DIR/val_sft4_sc8"

# 5) apples-to-apples comparison on the clean subset (old runs re-scored)
echo "=== accuracy on audited-clean val (481 q) ==="
python src/score_preds.py --ids "$PROC/val_clean.csv" \
    "$DLC_OUTPUT_DIR/val_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_sft_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_sft2_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_audit/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_sft4_greedy/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_sft4_sc8/preds.jsonl"
