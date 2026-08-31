#!/usr/bin/env bash
# Lab-server environment. Source this (or let scripts/0*.sh source it):
#   source scripts/env_server.sh
#
# Layout on the server (moved to ssd4000, 2026-08):
#   code    ~/private/test/competition/deep-learning-challenge-2026/
#   data    ~/shared/hdd_ext/ssd4000/jihoon/math_data/
#   output  ~/shared/hdd_ext/ssd4000/jihoon/math_output/

_DLC_STORE="$HOME/shared/hdd_ext/ssd4000/jihoon"
export DLC_DATA_DIR="${DLC_DATA_DIR:-$_DLC_STORE/math_data}"
export DLC_OUTPUT_DIR="${DLC_OUTPUT_DIR:-$_DLC_STORE/math_output}"

# Default GPU 7 (A6000 48GB) = main pipeline; GPU 8 (RTX 3090 24GB) is the
# second worker — run those jobs with an explicit CUDA_VISIBLE_DEVICES=8.
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA use the same GPU numbering as
# nvidia-smi; without it CUDA orders mixed GPUs fastest-first, so the number
# can silently land on a different physical card (we saw it pick smi-#6).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

# conda env used ONLY for the TRL training steps (sft_lora.py, merge_lora.py).
# Inference (vllm 0.6.1 cu118) stays in the env you launch the script from.
# Lives on the big shared disk (home quota is tight) -> full path, conda -p.
export DLC_TRAIN_ENV="${DLC_TRAIN_ENV:-$_DLC_STORE/conda/dlc-train}"

# keep temp files off the tight home/root filesystems too
export TMPDIR="${TMPDIR:-$_DLC_STORE/tmp}"
mkdir -p "$TMPDIR"

# the server's global pip config adds pypi.ngc.nvidia.com, which no longer
# resolves (5 DNS retries per package = crawling installs); override it
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"

# keep the ~6 GB HF model cache on the big NVMe instead of the home quota;
# delete these two lines if you prefer the default ~/.cache/huggingface
export HF_HOME="${HF_HOME:-$_DLC_STORE/hf_cache}"
mkdir -p "$DLC_OUTPUT_DIR" "$HF_HOME"

echo "DLC_DATA_DIR         = $DLC_DATA_DIR"
echo "DLC_OUTPUT_DIR       = $DLC_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "HF_HOME              = $HF_HOME"
