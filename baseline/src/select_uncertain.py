"""Select low-agreement questions from pooled ensemble votes for deep resampling.

Pools the weighted votes exactly like ensemble_vote.py, computes each
question's top-vote share, prints the distribution, and writes the questions
below --max-share (plus any with no valid vote) to a CSV ready for
infer_vllm.py. Deep-resampling only these is the cheapest way left to flip
answers: confident questions won't change, split ones might.

  python src/select_uncertain.py \
      lb_base_sc16/preds.jsonl lb_base_sc32/preds.jsonl \
      lb_m3_sc16/preds.jsonl   lb_m3_sc32/preds.jsonl \
      --weights 1 1 1 1 \
      --questions "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
      --max-share 0.40 --out "$DLC_OUTPUT_DIR/data_processed/lb_uncertain.csv"
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--questions", required=True, help="CSV with id,question")
    ap.add_argument("--max-share", type=float, default=0.40,
                    help="keep questions whose top answer holds <= this share of votes")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = args.weights or [1.0] * len(args.preds)
    pool: dict = {}
    for pf, w in zip(args.preds, weights):
        for line in open(pf):
            r = json.loads(line)
            votes = [a for a in r.get("extracted", []) if a is not None]
            pool.setdefault(r["id"], []).extend(votes * int(round(w)))

    shares = {}
    for qid, votes in pool.items():
        if not votes:
            shares[qid] = 0.0
            continue
        top = Counter(votes).most_common(1)[0][1]
        shares[qid] = top / len(votes)

    s = pd.Series(shares)
    print("top-vote share distribution over", len(s), "questions:")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  p{int(q*100):02d}: {s.quantile(q):.3f}")
    for thr in (0.3, 0.4, 0.5, 0.6):
        print(f"  share <= {thr:.2f}: {(s <= thr).sum()} questions")

    keep = set(s[s <= args.max_share].index)
    dfq = pd.read_csv(args.questions)
    out_df = dfq[dfq["id"].isin(keep)][["id", "question"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote {len(out_df)} uncertain questions (share <= {args.max_share}) -> {args.out}")


if __name__ == "__main__":
    main()
