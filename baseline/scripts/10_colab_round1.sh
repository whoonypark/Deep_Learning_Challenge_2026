#!/usr/bin/env bash
# Colab A100 rebuild, round 1 — server artifacts were lost; rebuilding from raw
# data with every lesson from the server runs baked in:
#   * label audit FIRST (4.2% of train labels are provably wrong) -> train on clean data only
#   * single modern env (no cu118 split), vllm serves LoRA adapters directly (no merge step)
#   * everything written to Drive so a session disconnect loses nothing
#
# Colab cell usage:
#   from google.colab import drive; drive.mount('/content/drive')
#   !git clone <your-repo-url> /content/dlc && cp -r /content/drive/MyDrive/dlc2026/data_csvs ...
#   %cd /content/dlc/baseline
#   !pip -q install vllm trl peft datasets accelerate tensorboard pandas
#   !bash scripts/10_colab_round1.sh
#
# Total ~3-4h on an A100. Steps are idempotent-ish: if the session dies, re-run;
# completed outputs already on Drive can be skipped by commenting out.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_colab.sh

PROC="$DLC_OUTPUT_DIR/data_processed"

# 0) data prep (fast)
python src/prepare_data.py

# 1) base-model sampling over the train pool: doubles as RS-SFT source AND audit signal
python src/infer_vllm.py --input "$PROC/train_pool.csv" \
    --k 8 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/rs_r1"

# 2) label audit (consensus vs label) on train and val
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

# 3) clean SFT data -> train M1 from the base model
python src/build_sft_data.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
    --only-ids "$PROC/train_pool_clean.csv" --out "$PROC/sft_m1.jsonl"
python src/sft_lora.py --data "$PROC/sft_m1.jsonl" --out "$DLC_OUTPUT_DIR/lora-m1"

# 4) clean-val gate: base vs M1 (LoRA served directly by vllm on A100)
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_base_sc8"
python src/infer_vllm.py --input "$PROC/val_clean.csv" \
    --lora "$DLC_OUTPUT_DIR/lora-m1" \
    --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m1_sc8"

echo "=== accuracy on audited-clean val ==="
python src/score_preds.py \
    "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl"
