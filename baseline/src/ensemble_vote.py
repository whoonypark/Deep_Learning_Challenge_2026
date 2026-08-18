"""Ensemble majority vote across multiple models' preds.jsonl on the SAME input.

Different SFT rounds make different mistakes; pooling their samples and
re-voting cancels individual-model errors (pure inference-time technique,
explicitly allowed). Needs no GPU — reuses the `extracted` lists already
saved in each leaderboard run.

  python src/ensemble_vote.py \
      outputs/leaderboard_sft_sc16/preds.jsonl \
      outputs/leaderboard_sft2_sc16/preds.jsonl \
      outputs/leaderboard_sft4_sc8/preds.jsonl \
      --submission outputs/leaderboard_ensemble/submission.csv

Optional --weights (one float per file) count each file's samples that many
times, e.g. --weights 1 2 1 to trust the middle model double.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from answer_extraction import majority_vote


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="preds.jsonl files over the same question set")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="per-file sample weights (default: all 1)")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--id-col", default="id")
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.preds)
    if len(weights) != len(args.preds):
        raise SystemExit("--weights must have one value per preds file")

    pool: dict = {}
    order: list = []
    for pf, w in zip(args.preds, weights):
        n_rows = 0
        with open(pf) as f:
            for line in f:
                r = json.loads(line)
                qid = r["id"]
                if qid not in pool:
                    pool[qid] = []
                    order.append(qid)
                votes = [a for a in r.get("extracted", []) if a is not None]
                pool[qid].extend(votes * int(round(w)))
                n_rows += 1
        print(f"{pf}: {n_rows} questions, weight {w}")

    rows = []
    n_none = 0
    for qid in order:
        voted = majority_vote(pool[qid])
        if voted is None:
            voted = 0
            n_none += 1
        rows.append((qid, voted))

    sub = pd.DataFrame(rows, columns=[args.id_col, "answer"])
    Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.submission, index=False)
    print(f"\nensembled {len(rows)} questions ({n_none} with no valid vote -> 0)")
    print(f"wrote {args.submission}")


if __name__ == "__main__":
    main()
