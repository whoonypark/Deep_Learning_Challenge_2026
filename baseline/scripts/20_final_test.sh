#!/usr/bin/env bash
# FINAL DAY (8/31): one command from the released test csv to submission files.
#
#   conda activate dlc
#   bash scripts/20_final_test.sh /path/to/deep_chal_math_test.csv
#
# Rehearsal (mechanics check on 30 questions of the leaderboard file, ~10 min):
#   LIMIT=30 bash scripts/20_final_test.sh "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv"
#
# Runs all ensemble members across BOTH GPUs in parallel, then writes:
#   final_test/submission_ensemble.csv   <- primary submission
#   final_test/submission_m1_solo.csv    <- single-model backup (M1 @ k32 T0.6)
# Resumable: finished member runs are skipped on re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env_server.sh

TEST_CSV="${1:?usage: bash scripts/20_final_test.sh <test_csv_path>}"

# Final recipe (locked 2026-08-29, [10]/[12] LB + clean-val 0.8390 plateau):
#   ensemble = base@sc16 + base@sc32 + M3@sc16 + M3@sc32, equal weights.
# Zero-weight members (M1_16/M4/M2) are NOT run — saves hours on test day.
WEIGHTS="${WEIGHTS:-1 1 1 1}"

LIM=""
[ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"
OUT="$DLC_OUTPUT_DIR/final_test"
mkdir -p "$OUT"

run() {  # gpu model('BASE'=no adapter) k temperature name [input_csv]
    local gpu="$1" model="$2" k="$3" t="$4" name="$5" input="${6:-$TEST_CSV}"
    if [ -e "$OUT/$name/preds.jsonl" ]; then echo "[skip] $name"; return 0; fi
    echo "[run ] $name (gpu $gpu, k=$k, T=$t)"
    local margs=()
    [ "$model" != "BASE" ] && margs=(--model "$model")
    CUDA_VISIBLE_DEVICES="$gpu" python src/infer_vllm.py --input "$input" \
        "${margs[@]}" --k "$k" --temperature "$t" \
        --out-dir "$OUT/$name" --no-save-texts $LIM
}

# GPU 7 (A6000): heavy k=32 runs + M1 backup | GPU 9 (A6000): the k=16 runs
(
    run 7 BASE 32 0.6 t_base_sc32
    run 7 "$DLC_OUTPUT_DIR/merged-m3" 32 0.6 t_m3_sc32
    run 7 "$DLC_OUTPUT_DIR/merged-m1" 32 0.6 t_m1_sc32
) > "$OUT/gpu7.log" 2>&1 &
P7=$!
(
    run 9 BASE 16 0.8 t_base_sc16
    run 9 "$DLC_OUTPUT_DIR/merged-m3" 16 0.8 t_m3_sc16
) > "$OUT/gpu8.log" 2>&1 &
P8=$!
echo "inference running on both GPUs (logs: $OUT/gpu7.log, $OUT/gpu8.log)..."
wait "$P7" "$P8"

FILES=("$OUT/t_base_sc16/preds.jsonl" "$OUT/t_base_sc32/preds.jsonl"
       "$OUT/t_m3_sc16/preds.jsonl"   "$OUT/t_m3_sc32/preds.jsonl")
W="$WEIGHTS"

# ---- adaptive deep pass (proved +4 questions on the LB: [12]->[13]) ----
python src/select_uncertain.py "${FILES[@]}" --weights $WEIGHTS \
    --questions "$TEST_CSV" \
    --max-share 0.45 --out "$OUT/uncertain.csv"
N_UNC=$(($(wc -l < "$OUT/uncertain.csv") - 1))
if [ "$N_UNC" -gt 0 ]; then
    echo "deep-resampling $N_UNC uncertain questions at k=64 on both GPUs..."
    ( run 7 BASE 64 0.8 t_base_deep64 "$OUT/uncertain.csv" ) > "$OUT/gpu7_deep.log" 2>&1 &
    D7=$!
    ( run 9 "$DLC_OUTPUT_DIR/merged-m3" 64 0.8 t_m3_deep64 "$OUT/uncertain.csv" ) > "$OUT/gpu8_deep.log" 2>&1 &
    D8=$!
    wait "$D7" "$D8"
    FILES+=("$OUT/t_base_deep64/preds.jsonl" "$OUT/t_m3_deep64/preds.jsonl")
    W="$WEIGHTS 1 1"
else
    echo "no uncertain questions found - skipping deep pass"
fi

# primary: weighted ensemble (member order MUST match W)
python src/ensemble_vote.py "${FILES[@]}" \
    --weights $W \
    --submission "$OUT/submission_ensemble.csv"

# backup: best single model at its best setting
python src/ensemble_vote.py "$OUT/t_m1_sc32/preds.jsonl" --weights 1 \
    --submission "$OUT/submission_m1_solo.csv"

echo
echo "=== DONE ==="
echo "primary : $OUT/submission_ensemble.csv"
echo "backup  : $OUT/submission_m1_solo.csv"
head -3 "$OUT/submission_ensemble.csv"
