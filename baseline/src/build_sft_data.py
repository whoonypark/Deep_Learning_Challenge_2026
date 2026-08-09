"""Build rejection-sampling SFT data from train-pool predictions.

Takes preds.jsonl produced by infer_vllm.py on train_pool.csv (with texts
saved), keeps only solutions whose \\boxed{} answer equals the gold answer,
dedups, caps per question, and writes a TRL prompt-completion JSONL:

  {"prompt": [{role,content}...], "completion": [{"role":"assistant","content": ...}]}

Usage:
  python src/build_sft_data.py --preds outputs/rs_train/preds.jsonl --out data_processed/sft.jsonl
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
    ap.add_argument("--preds", required=True, help="preds.jsonl from a train_pool run")
    ap.add_argument("--out", default=str(paths.SFT_JSONL))
    ap.add_argument("--max-per-question", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=6000, help="drop overlong solutions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records = []
    n_q = n_q_covered = 0

    with open(args.preds) as f:
        for line in f:
            rec = json.loads(line)
            n_q += 1
            gold = rec.get("gold")
            texts = rec.get("texts")
            if gold is None or not texts:
                continue
            # strict filter: the boxed value itself must equal gold
            good = [t.strip() for t in texts if extract_boxed_int(t) == gold]
            good = [t for t in good if len(t) <= args.max_chars]
            good = list(dict.fromkeys(good))  # dedup, keep order
            if not good:
                continue
            n_q_covered += 1
            rng.shuffle(good)
            for t in good[: args.max_per_question]:
                records.append(
                    {
                        "prompt": build_messages(rec["question"]),
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
