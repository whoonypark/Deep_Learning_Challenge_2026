#!/usr/bin/env bash
# Colab A100 rebuild, round 1 — rebuilding from raw data with every lesson from
# the earlier runs baked in:
#   * community error reports applied in prepare_data (602 ill-posed dropped,
#     440 labels corrected) + our own consensus audit as a second pass
#   * one modern env, so vllm serves the LoRA adapter directly (no merge step)
#   * every artifact lands on Drive, and each step is SKIPPED if its output is
#     already there -> a Colab disconnect costs only the running step
#
#   !bash scripts/10_colab_round1.sh
#
# ~3-4h total on an A100. To force a step to re-run, delete its output dir.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_colab.sh

PROC="$DLC_OUTPUT_DIR/data_processed"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"

step() {  # step <marker-path> <description>; returns 1 when it should be skipped
    if [ -e "$2" ]; then
        echo ">>> [skip] $1 (found $(basename "$2"))"
        return 1
    fi
    echo ">>> [run ] $1"
    return 0
}

# 0) data prep (community reports auto-applied when present in $DLC_DATA_DIR)
if step "prepare data" "$PROC/train_pool.csv"; then
    python src/prepare_data.py
fi

# 1) base-model sampling over the train pool: RS-SFT source AND audit signal.
#    SHARDED on purpose: vllm only writes preds.jsonl when a run finishes, so a
#    single 2-3h call would lose everything if Colab drops the runtime. Each
#    shard saves independently and is skipped on re-run; the shards are then
#    concatenated into the preds.jsonl the later steps expect.
SHARD=2000
if step "base sampling k=8 over train pool (longest step, sharded)" "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl"; then
    N_ROWS=$(($(wc -l < "$PROC/train_pool.csv") - 1))
    for ((off=0; off<N_ROWS; off+=SHARD)); do
        SHARD_DIR=$(printf "%s/rs_r1_shards/shard_%06d" "$DLC_OUTPUT_DIR" "$off")
        if [ -e "$SHARD_DIR/preds.jsonl" ]; then
            echo "    [skip] shard offset $off"
            continue
        fi
        echo "    [run ] shard offset $off / $N_ROWS"
        python src/infer_vllm.py --input "$PROC/train_pool.csv" \
            --offset "$off" --limit "$SHARD" \
            --k 8 --temperature 1.0 --out-dir "$SHARD_DIR"
    done
    mkdir -p "$DLC_OUTPUT_DIR/rs_r1"
    cat "$DLC_OUTPUT_DIR"/rs_r1_shards/shard_*/preds.jsonl > "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl"
    echo "    merged $(wc -l < "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl") rows into rs_r1/preds.jsonl"
fi

# 2) label audit — residual errors the community list did not catch
if step "audit train labels" "$PROC/train_pool_clean.csv"; then
    python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
        --input "$PROC/train_pool.csv" \
        --out-suspects "$DLC_OUTPUT_DIR/audit/train_suspects.csv" \
        --out-clean "$PROC/train_pool_clean.csv"
fi

if step "val sampling k=16" "$DLC_OUTPUT_DIR/val_audit/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val.csv" \
        --k 16 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/val_audit"
fi

if step "audit val labels" "$PROC/val_clean.csv"; then
    python src/audit_labels.py --preds "$DLC_OUTPUT_DIR/val_audit/preds.jsonl" \
        --input "$PROC/val.csv" \
        --out-suspects "$DLC_OUTPUT_DIR/audit/val_suspects.csv" \
        --out-clean "$PROC/val_clean.csv"
fi

# 3) clean SFT data -> M1 (LoRA, from the base model)
if step "build clean SFT data" "$PROC/sft_m1.jsonl"; then
    python src/build_sft_data.py --preds "$DLC_OUTPUT_DIR/rs_r1/preds.jsonl" \
        --only-ids "$PROC/train_pool_clean.csv" --out "$PROC/sft_m1.jsonl"
fi

if step "LoRA SFT -> M1" "$DLC_OUTPUT_DIR/lora-m1/adapter_model.safetensors"; then
    python src/sft_lora.py --data "$PROC/sft_m1.jsonl" --out "$DLC_OUTPUT_DIR/lora-m1"
fi

# 4) clean-val gate: base vs M1 (vllm serves the adapter directly on A100)
if step "eval base on clean val" "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_base_sc8"
fi

if step "eval M1 on clean val" "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl"; then
    python src/infer_vllm.py --input "$PROC/val_clean.csv" \
        --lora "$DLC_OUTPUT_DIR/lora-m1" \
        --k 8 --temperature 0.8 --out-dir "$DLC_OUTPUT_DIR/val_m1_sc8"
fi

echo
echo "=== accuracy on audited-clean val ==="
python src/score_preds.py \
    "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl" \
    "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl"
