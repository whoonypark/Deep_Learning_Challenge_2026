#!/usr/bin/env bash
# Final-week round 2 (GPU 8): the recipe that produced our best single model
# (old r2), upgraded — clean+relabeled data, M1 as sampler, k=16 on unsolved,
# 3 solutions kept per question. Resumable: finished steps are skipped.
#
#   conda activate dlc
#   nohup bash scripts/12_round2_m2.sh > "$DLC_OUTPUT_DIR/12_m2.log" 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh   # CUDA_VISIBLE_DEVICES defaults to 7 (A6000 48GB)

PROC="$DLC_OUTPUT_DIR/data_processed"

step() {
    if [ -e "$2" ]; then echo ">>> [skip] $1"; return 1; fi
    echo ">>> [run ] $1"; return 0
}

# -1) M1 adapter was lost in the server wipe -> retrain from sft_m1.jsonl when
#     missing (A6000 48GB: default batch 4 is fine and ~3x faster than the 3090)
if step "train M1 (adapter missing)" "$DLC_OUTPUT_DIR/lora-m1/adapter_model.safetensors"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/sft_lora.py --data "$PROC/sft_m1.jsonl" \
        --out "$DLC_OUTPUT_DIR/lora-m1" \
        --batch-size 4 --grad-accum 8   # A6000 48GB, grad-ckpt ON — the proven config; no-ckpt OOMs
fi

# 0) merge M1 for vllm-0.6.1 inference (LoRA path unusable on the cu118 driver)
if step "merge M1" "$DLC_OUTPUT_DIR/merged-m1/config.json"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m1" \
        --out "$DLC_OUTPUT_DIR/merged-m1"
fi

# 0b) baseline numbers we never got: base vs M1 on clean val
if step "eval base on clean val" "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_base_sc8"
fi
if step "eval M1 on clean val" "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --model "$DLC_OUTPUT_DIR/merged-m1" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m1_sc8"
fi

# 1) questions the base sampler never solved (vs the audited-clean pool)
if step "select unsolved" "$PROC/train_unsolved_m1.csv"; then
    python src/select_unsolved.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
        --pool "$PROC/train_pool_clean.csv" --out "$PROC/train_unsolved_m1.csv"
fi

# 2) resample them with M1, double budget + hot sampling
if step "M1 resampling k=16 on unsolved" "$DLC_OUTPUT_DIR/rs_m2/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/train_unsolved_m1.csv" \
        --model "$DLC_OUTPUT_DIR/merged-m1" \
        --k 16 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/rs_m2"
fi

# 3) union SFT data (base round + M1 round), clean ids only, 3 per question
if step "build sft_m2" "$PROC/sft_m2.jsonl"; then
    python src/build_sft_data.py \
        --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" "$DLC_OUTPUT_DIR/rs_m2/preds.jsonl" \
        --only-ids "$PROC/train_pool_clean.csv" \
        --max-per-question 3 --out "$PROC/sft_m2.jsonl"
fi

# 4) train M2 from the base model, merge
if step "train M2" "$DLC_OUTPUT_DIR/lora-m2/adapter_model.safetensors"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/sft_lora.py --data "$PROC/sft_m2.jsonl" \
        --out "$DLC_OUTPUT_DIR/lora-m2" \
        --batch-size 4 --grad-accum 8   # A6000 48GB, grad-ckpt ON — the proven config; no-ckpt OOMs
fi
if step "merge M2" "$DLC_OUTPUT_DIR/merged-m2/config.json"; then
    conda run --no-capture-output -p "$DLC_TRAIN_ENV" \
        python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m2" \
        --out "$DLC_OUTPUT_DIR/merged-m2"
fi

# 5) clean-val gate
if step "eval M2 on clean val" "$DLC_OUTPUT_DIR/val_m2_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --model "$DLC_OUTPUT_DIR/merged-m2" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m2_sc8"
fi

echo
echo "=== accuracy on audited-clean val ==="
python src/score_preds.py \
    "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m2_sc8/preds.jsonl"
