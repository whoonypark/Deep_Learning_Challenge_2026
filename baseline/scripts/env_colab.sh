#!/usr/bin/env bash
# Colab (A100) environment. Everything persistent goes to Google Drive so a
# session disconnect never loses work; only the HF model cache is ephemeral
# (re-downloads in minutes on Colab's fast network).
#
#   source scripts/env_colab.sh

export DLC_DATA_DIR="${DLC_DATA_DIR:-/content/drive/MyDrive/data}"
export DLC_OUTPUT_DIR="${DLC_OUTPUT_DIR:-/content/drive/MyDrive/output}"
export HF_HOME="${HF_HOME:-/content/hf_cache}"

mkdir -p "$DLC_OUTPUT_DIR" "$HF_HOME"

echo "DLC_DATA_DIR   = $DLC_DATA_DIR"
echo "DLC_OUTPUT_DIR = $DLC_OUTPUT_DIR"
echo "HF_HOME        = $HF_HOME (ephemeral)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
