#!/usr/bin/env bash
# One-time environment rebuild on the lab server (NVIDIA driver = CUDA 11.8).
# Encodes every lesson from the first setup:
#   * vllm must be the 0.6.1.post2 +cu118 wheel (last release with cu118 builds;
#     newer wheels fail with "undefined symbol: cuTensorMapEncodeTiled")
#   * inference env pins transformers 4.45.2 (matches vllm 0.6.1)
#   * training env lives on the NVMe (home quota is tight) with the verified
#     torch 2.4.1+cu118 / transformers 4.55.4 / trl 0.21 / peft 0.17 combo
#   * TMPDIR on NVMe (pip builds overflow the root disk otherwise)
#
#   bash scripts/00_setup_server_envs.sh
set -euo pipefail

STORE="$HOME/shared/hdd_ext/ssd4000/jihoon"
export TMPDIR="$STORE/tmp"
# dead NGC pip mirror in the global pip config -> override (see env_server.sh)
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
mkdir -p "$TMPDIR" "$STORE/math_output" "$STORE/hf_cache" "$STORE/conda"

# ---- inference env: dlc (home miniconda, name-based) ----
if conda env list | grep -qE "^dlc\s"; then
    echo "[skip] conda env 'dlc' already exists"
else
    conda create -y -n dlc python=3.11
    conda run -n dlc pip install --no-cache-dir \
        "https://github.com/vllm-project/vllm/releases/download/v0.6.1.post2/vllm-0.6.1.post2+cu118-cp311-cp311-manylinux1_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu118
    conda run -n dlc pip install --no-cache-dir "transformers==4.45.2" pandas
fi

# ---- training env: dlc-train (NVMe, path-based) ----
TRAIN_ENV="$STORE/conda/dlc-train"
if [ -d "$TRAIN_ENV" ]; then
    echo "[skip] $TRAIN_ENV already exists"
else
    conda create -y -p "$TRAIN_ENV" python=3.11
    conda run -p "$TRAIN_ENV" pip install --no-cache-dir \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    conda run -p "$TRAIN_ENV" pip install --no-cache-dir \
        "transformers==4.55.4" "trl==0.21.0" "peft==0.17.0" "accelerate==1.10.0" \
        datasets tensorboard pandas
fi

echo "---- verify ----"
conda run -n dlc python -c "from vllm import LLM; import torch; print('dlc OK       :', torch.__version__, torch.version.cuda)"
conda run -p "$TRAIN_ENV" python -c "from trl import SFTTrainer; from peft import LoraConfig; import torch; print('dlc-train OK :', torch.__version__, torch.version.cuda)"
echo "both environments ready"
