"""Select train-pool questions that got zero correct samples in a preds run.

Used between rejection-sampling rounds (STaR): round 2 only resamples the
questions round 1 failed on, with a stronger sampler / higher k.

  python src/select_unsolved.py --preds $OUT/rs_train/preds.jsonl \
      --pool $OUT/data_processed/train_pool.csv \
      --out  $OUT/data_processed/train_unsolved_r1.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from answer_extraction import extract_boxed_int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, nargs="+", help="preds.jsonl file(s) with texts+gold")
    ap.add_argument("--pool", required=True, help="train_pool.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seen: set = set()
    solved: set = set()
    for preds_file in args.preds:
        with open(preds_file) as f:
            for line in f:
                rec = json.loads(line)
                qid, gold = rec["id"], rec.get("gold")
                seen.add(qid)
                texts = rec.get("texts") or []
                if gold is not None and any(extract_boxed_int(t) == gold for t in texts):
                    solved.add(qid)

    pool = pd.read_csv(args.pool)
    unsolved = pool[pool["id"].isin(seen - solved)].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    unsolved.to_csv(out, index=False)

    print(f"questions in preds : {len(seen)}")
    print(f"solved >=1 time    : {len(solved)}  ({len(solved) / max(len(seen), 1):.1%})")
    print(f"unsolved -> {out}  ({len(unsolved)} rows)")


if __name__ == "__main__":
    main()
