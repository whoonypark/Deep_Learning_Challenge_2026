#!/usr/bin/env bash
# Inference-hyperparameter sweep on VAL ONLY (no submissions): find the best
# (k, temperature) for the current best model. Test-time technique tuning is
# free score — but only trustworthy on val, never on the noisy leaderboard.
#
#   bash scripts/05_val_sweep.sh [model_dir]   (default: merged-r2)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

MODEL="${1:-$DLC_OUTPUT_DIR/merged-r2}"
PROC="$DLC_OUTPUT_DIR/data_processed"
SWEEP="$DLC_OUTPUT_DIR/val_sweep"

for k in 8 16 32; do
  for t in 0.6 0.8 1.0; do
    python src/infer_vllm.py --input "$PROC/val.csv" \
        --model "$MODEL" --k "$k" --temperature "$t" \
        --out-dir "$SWEEP/k${k}_t${t}" --no-save-texts
  done
done

# summary table
python - "$SWEEP" <<'EOF'
import json, pathlib, sys
sweep = pathlib.Path(sys.argv[1])
print(f"\n{'config':12s} accuracy")
for d in sorted(sweep.iterdir()):
    p = d / "preds.jsonl"
    if not p.exists():
        continue
    rs = [json.loads(l) for l in open(p)]
    acc = sum(r["voted"] == r["gold"] for r in rs) / len(rs)
    print(f"{d.name:12s} {acc:.4f}")
EOF
