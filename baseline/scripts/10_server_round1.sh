#!/usr/bin/env bash
# Lab-server rebuild, round 1 — same pipeline as 10_colab_round1.sh, adapted to
# the cu118 server: training runs in $DLC_TRAIN_ENV (conda run -p), and
# inference always uses MERGED models (vllm 0.6.1's LoRA path needs Triton
# kernels the CUDA-11.8 driver cannot load).
#
# Pipeline: prepare (community reports auto-applied) -> base k=8 sampling
# -> label audit (train+val) -> clean SFT -> M1 -> merge -> clean-val gate.
#
#   conda activate dlc
#   nohup bash scripts/10_server_round1.sh > "$DLC_OUTPUT_DIR/10_round1.log" 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"

# 0) data prep (drops illposed_623, relabels mislabel_442 when present in math_data)
python src/prepare_data.py

# 1) base-model sampling over the train pool: RS-SFT source + audit signal
python src/infer_vllm.py --input "$PROC/train_pool.csv" \
    --k 8 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/rs_r1"

# 2) label audit (residual errors beyond the community reports)
python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
    --input "$PROC/train_pool.csv" \
    --out-suspects "$DLC_OUTPUT_DIR/audit/train_suspects.csv" \
    --out-clean "$PROC/train_pool_clean.csv"

python src/infer_vllm.py --input "$PROC/val.csv" \
    --k 16 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/val_audit"
python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/val_audit/preds.jsonl" \
    --input "$PROC/val.csv" \
    --out-suspects "$DLC_OUTPUT_DIR/audit/val_suspects.csv" \
    --out-clean "$PROC/val_clean.csv"

# 3) clean SFT data -> train M1 from base (training env)
python src/build_sft_data.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
    --only-ids "$PROC/train_pool_clean.csv" --out "$PROC/sft_m1.jsonl"
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/sft_lora.py --data "$PROC/sft_m1.jsonl" \
    --out "$DLC_OUTPUT_DIR/lora-m1"

# 3b) merge for inference
conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
    python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m1" \
    --out "$DLC_OUTPUT_DIR/merged-m1"

# 4) clean-val gate: base vs M1
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_base_sc8"
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --model "$DLC_OUTPUT_DIR/merged-m1" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m1_sc8"

echo "=== accuracy on audited-clean val ==="
python src/score_preds.py \
    "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl"
