#!/usr/bin/env bash
# Colab one-time install. Order matters:
#   1. vllm first — it pins torch/transformers to a combination it can import
#   2. trl/peft after, with --upgrade-strategy only-if-needed so pip does NOT
#      churn transformers back to a version vllm cannot load (that mismatch is
#      what produced the "vllm/config/model.py" ImportError last time)
# RESTART THE RUNTIME after this script, then verify with 01_verify_colab.sh.
#
#   !bash scripts/00_setup_colab.sh
set -euo pipefail

# Colab preinstalls torchaudio built for an older CUDA; vllm upgrades torch and
# the pair then mismatches. transformers imports torchaudio (guarded by
# is_torchaudio_available()) on the path vllm needs, so a stale torchaudio
# breaks `from vllm import LLM` with a CUDA-version RuntimeError. We never use
# audio -> remove it so the guard simply skips.
pip uninstall -y -q torchaudio || true

pip install -q -U vllm
pip install -q --upgrade-strategy only-if-needed trl datasets accelerate tensorboard pandas
# peft separately: trl 1.x does not depend on it, and in a batch install it can
# be silently skipped while pip is busy resolving Colab's preinstalled packages
pip install -q --upgrade-strategy only-if-needed peft

echo
echo "installed versions:"
python - <<'EOF'
for m in ("torch", "transformers", "trl", "peft", "vllm"):
    try:
        mod = __import__(m)
        print(f"  {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:  # vllm often needs the restart before it imports
        print(f"  {m:14s} (import deferred until restart: {type(e).__name__})")
EOF
echo
echo "NOW RESTART THE RUNTIME, then run: bash scripts/01_verify_colab.sh"
