#!/usr/bin/env bash
# Label audit of train_pool AND val using model consensus (merged-r2).
# The k=8 train sampling doubles as round-3 rejection-sampling data.
#
# Outputs:
#   $DLC_OUTPUT_DIR/rs_audit/preds.jsonl        (train, k=8 T=1.0, with texts)
#   $DLC_OUTPUT_DIR/val_audit/preds.jsonl       (val,   k=16 T=1.0)
#   $DLC_OUTPUT_DIR/audit/train_suspects.csv    -> spot-check by hand
#   $DLC_OUTPUT_DIR/audit/val_suspects.csv      -> spot-check by hand
#   $PROC/train_pool_clean.csv, $PROC/val_clean.csv
#
#   conda activate dlc && bash scripts/06_audit_labels.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"
MODEL="$DLC_OUTPUT_DIR/merged-r2"

# 1) big sampling pass on train (keep texts: reused as round-3 SFT source)
python src/infer_vllm.py --input "$PROC/train_pool.csv" \
    --model "$MODEL" --k 8 --temperature 1.0 \
    --out-dir "$DLC_OUTPUT_DIR/rs_audit"

# 2) high-k pass on val (small, makes the val gate trustworthy)
python src/infer_vllm.py --input "$PROC/val.csv" \
    --model "$MODEL" --k 16 --temperature 1.0 \
    --out-dir "$DLC_OUTPUT_DIR/val_audit"

# 3) consensus audit -> suspects + cleaned CSVs
python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/rs_audit/preds.jsonl" \
    --input "$PROC/train_pool.csv" \
    --out-suspects "$DLC_OUTPUT_DIR/audit/train_suspects.csv" \
    --out-clean "$PROC/train_pool_clean.csv"

python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/val_audit/preds.jsonl" \
    --input "$PROC/val.csv" \
    --agree-frac 0.75 \
    --out-suspects "$DLC_OUTPUT_DIR/audit/val_suspects.csv" \
    --out-clean "$PROC/val_clean.csv"
