#!/usr/bin/env bash
# Post-restart sanity check: GPU, data files, and the imports that matter.
# Run this BEFORE launching round 1 — it fails in seconds instead of hours.
#
#   !bash scripts/01_verify_colab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_colab.sh

echo
echo "--- data files ---"
for f in deep_chal_math_train.csv train_filtered_ids.csv \
         deep_chal_math_leaderboard_filtered.csv \
         organizer_report_illposed_623.csv organizer_report_mislabel_442.csv; do
    if [ -f "$DLC_DATA_DIR/$f" ]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
done

echo
echo "--- torch/torchaudio CUDA consistency ---"
python - <<'EOF'
import importlib.util
if importlib.util.find_spec("torchaudio") is not None:
    print("  WARNING: torchaudio is installed. If its CUDA build differs from "
          "torch's, `from vllm import LLM` fails. Fix: pip uninstall -y torchaudio")
else:
    print("  OK      torchaudio absent (transformers skips its audio path)")
EOF

echo
echo "--- imports ---"
python -c "import torch; print('  torch       ', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"
python -c "from vllm import LLM, SamplingParams; import vllm; print('  vllm        ', vllm.__version__, 'OK')"
python -c "from trl import SFTTrainer; from peft import LoraConfig; import transformers; print('  trl/peft    OK | transformers', transformers.__version__)"
python tests/test_answer_extraction.py
echo
echo "all checks passed - ready for: bash scripts/10_colab_round1.sh"
