#!/usr/bin/env bash
# Final-week job for the SECOND GPU (RTX 3090 #8): run with an explicit GPU
# override so it never collides with 12_round2_m2.sh (default GPU 7, A6000):
#
#   conda activate dlc
#   CUDA_VISIBLE_DEVICES=8 nohup bash scripts/13_sweep_and_m3.sh \
#       > "$DLC_OUTPUT_DIR/13_sweep_m3.log" 2>&1 &
#
# Part A — inference sweep on clean val with M1: find the best (k, T) for the
#          final submissions (we never actually completed this; likely free
#          points — top teams are probably voting harder than SC@8).
# Part B — diversity member M3: GSM-source external data (small share, gentle
#          lr) mixed with M1 self-distill data. Slightly weaker solo but its
#          errors differ -> proven ensemble value.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

PROC="$DLC_OUTPUT_DIR/data_processed"
SWEEP="$DLC_OUTPUT_DIR/val_sweep_m1"

step() {
    if [ -e "$2" ]; then echo ">>> [skip] $1"; return 1; fi
    echo ">>> [run ] $1"; return 0
}

# wait for merged-m1 (created early by 12_round2_m2.sh on the other GPU)
while [ ! -e "$DLC_OUTPUT_DIR/merged-m1/config.json" ]; do
    echo "waiting for merged-m1 (12_round2_m2.sh creates it in its first minutes)..."
    sleep 60
done

# ---- Part A: (k, T) sweep on clean val ----
for k in 8 16 32; do
  for t in 0.6 0.8 1.0; do
    if step "sweep k=$k T=$t" "$SWEEP/k${k}_t${t}/preds.jsonl"; then
        python src/infer_vllm.py --input "$PROC/val_clean.csv" \
            --model "$DLC_OUTPUT_DIR/merged-m1" \
            --k "$k" --temperature "$t" \
            --out-dir "$SWEEP/k${k}_t${t}" --no-save-texts
    fi
  done
done

echo
echo "=== sweep results (clean val) ==="
python src/score_preds.py "$SWEEP"/k*/preds.jsonl

# ---- Part B: diversity member M3 ----
if step "prepare GSM external data" "$PROC/sft_m3.jsonl"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/prepare_external_data.py \
        --sources gsm8k --max-samples 20000 --max-chars 4000 \
        --include "$PROC/sft_m1.jsonl" \
        --out "$PROC/sft_m3.jsonl"
fi

if step "train M3" "$DLC_OUTPUT_DIR/lora-m3/adapter_model.safetensors"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/sft_lora.py --data "$PROC/sft_m3.jsonl" \
        --out "$DLC_OUTPUT_DIR/lora-m3" \
        --epochs 1 --lr 5e-5 --max-len 3072 --batch-size 1 --grad-accum 32   # 3090 24GB
fi

if step "merge M3" "$DLC_OUTPUT_DIR/merged-m3/config.json"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m3" \
        --out "$DLC_OUTPUT_DIR/merged-m3"
fi

if step "eval M3 on clean val" "$DLC_OUTPUT_DIR/val_m3_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --model "$DLC_OUTPUT_DIR/merged-m3" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m3_sc8"
fi

echo
echo "=== M3 vs M1 (clean val) ==="
python src/score_preds.py \
    "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m3_sc8/preds.jsonl"
