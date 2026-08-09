#!/usr/bin/env bash
# Lab-server environment. Source this (or let scripts/0*.sh source it):
#   source scripts/env_server.sh
#
# Layout on the server:
#   code    ~/private/test/competition/deep-learning-challenge-2026/
#   data    ~/shared/hdd_ext/nvme1/jihoonpark22/data/
#   output  ~/shared/hdd_ext/nvme1/jihoonpark22/output/

export DLC_DATA_DIR="${DLC_DATA_DIR:-$HOME/shared/hdd_ext/nvme1/jihoonpark22/data}"
export DLC_OUTPUT_DIR="${DLC_OUTPUT_DIR:-$HOME/shared/hdd_ext/nvme1/jihoonpark22/output}"

# GPU 9 only (shared server — never grab other GPUs).
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA use the same GPU numbering as
# nvidia-smi; without it CUDA orders mixed GPUs fastest-first, so "9" can
# silently land on a different physical card (we saw it pick smi-#6).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-9}"

# keep the ~6 GB HF model cache on the big NVMe instead of the home quota;
# delete these two lines if you prefer the default ~/.cache/huggingface
export HF_HOME="${HF_HOME:-$HOME/shared/hdd_ext/nvme1/jihoonpark22/hf_cache}"
mkdir -p "$DLC_OUTPUT_DIR" "$HF_HOME"

echo "DLC_DATA_DIR         = $DLC_DATA_DIR"
echo "DLC_OUTPUT_DIR       = $DLC_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "HF_HOME              = $HF_HOME"
