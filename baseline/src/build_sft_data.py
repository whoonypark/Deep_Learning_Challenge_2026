"""Build rejection-sampling SFT data from train-pool predictions.

Takes preds.jsonl produced by infer_vllm.py on train_pool.csv (with texts
saved), keeps only solutions whose \\boxed{} answer equals the gold answer,
dedups, caps per question, and writes a TRL prompt-completion JSONL:

  {"prompt": [{role,content}...], "completion": [{"role":"assistant","content": ...}]}

Accepts multiple preds files (e.g. round 1 + round 2 of STaR); solutions for
the same question are pooled across files before dedup/capping.

Usage:
  python src/build_sft_data.py --preds rs_train/preds.jsonl rs_train_r2/preds.jsonl \
      --out data_processed/sft_r2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import paths
from answer_extraction import extract_boxed_int
from prompts import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, nargs="+", help="one or more preds.jsonl from train_pool runs")
    ap.add_argument("--out", default=str(paths.SFT_JSONL))
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=6000, help="drop overlong solutions")
    ap.add_argument("--only-ids", default=None,
                    help="CSV with an `id` column: keep ONLY these question ids "
                         "(pass the audited train_pool_clean.csv so solutions that "
                         "matched a suspect label never enter SFT data)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    allowed_ids = None
    if args.only_ids:
        import pandas as pd

        allowed_ids = set(pd.read_csv(args.only_ids)["id"])
        print(f"restricting to {len(allowed_ids)} audited-clean question ids")

    # pool correct solutions per question across all preds files
    by_q: dict = {}  # id -> {"question": str, "good": [texts]}
    n_q = 0
    for preds_file in args.preds:
        with open(preds_file) as f:
            for line in f:
                rec = json.loads(line)
                if allowed_ids is not None and rec["id"] not in allowed_ids:
                    continue
                gold = rec.get("gold")
                texts = rec.get("texts")
                if rec["id"] not in by_q:
                    n_q += 1
                    by_q[rec["id"]] = {"question": rec.get("question"), "good": []}
                if gold is None or not texts:
                    continue
                # strict filter: the boxed value itself must equal gold
                good = [t.strip() for t in texts if extract_boxed_int(t) == gold]
                good = [t for t in good if len(t) <= args.max_chars]
                by_q[rec["id"]]["good"].extend(good)

    records = []
    n_q_covered = 0
    for qid, entry in by_q.items():
        good = list(dict.fromkeys(entry["good"]))  # dedup, keep order
        if not good or entry["question"] is None:
            continue
        n_q_covered += 1
        rng.shuffle(good)
        for t in good[: args.max_per_question]:
            records.append(
                {
                    "prompt": build_messages(entry["question"]),
                    "completion": [{"role": "assistant", "content": t}],
                }
            )

    rng.shuffle(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"questions in preds      : {n_q}")
    print(f"questions with >=1 good : {n_q_covered}  ({n_q_covered / max(n_q,1):.1%})")
    print(f"SFT samples written     : {len(records)} -> {out}")


if __name__ == "__main__":
    main()
