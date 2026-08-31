"""Grid-search ensemble weights on clean-val preds (files must contain gold).

Pools each member's extracted votes `weight` times, majority-votes, scores
against gold, and prints the top combinations. Use the winning weights with
ensemble_vote.py on the leaderboard/test preds (same file order!).

  python src/tune_weights.py \
      "$DLC_OUTPUT_DIR/val_base_sc8/preds.jsonl" \
      "$DLC_OUTPUT_DIR/val_m1_sc8/preds.jsonl" \
      "$DLC_OUTPUT_DIR/val_m3_sc8/preds.jsonl" \
      "$DLC_OUTPUT_DIR/val_m2_sc8/preds.jsonl"
"""

from __future__ import annotations

import argparse
import itertools
import json

from answer_extraction import majority_vote


def load(pf: str):
    rows = {}
    for line in open(pf):
        r = json.loads(line)
        rows[r["id"]] = ([a for a in r.get("extracted", []) if a is not None],
                         r.get("gold"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="val preds.jsonl files (with gold)")
    ap.add_argument("--max-weight", type=int, default=3)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    data = [load(p) for p in args.preds]
    ids = set(data[0])
    for d in data[1:]:
        ids &= set(d)
    ids = sorted(ids)
    gold = {qid: data[0][qid][1] for qid in ids}
    print(f"{len(ids)} common questions; file order:")
    for i, p in enumerate(args.preds):
        print(f"  w{i+1} = {p}")

    results = []
    for weights in itertools.product(range(args.max_weight + 1), repeat=len(data)):
        if not any(weights):
            continue
        correct = 0
        for qid in ids:
            pool = []
            for w, d in zip(weights, data):
                if w:
                    pool.extend(d[qid][0] * w)
            v = majority_vote(pool)
            if v is not None and v == gold[qid]:
                correct += 1
        results.append((correct, sum(weights), weights))

    # best accuracy first; among ties prefer simpler (smaller total weight)
    results.sort(key=lambda t: (-t[0], t[1]))
    print(f"\ntop {args.top} weight combos (clean-val accuracy):")
    for correct, _, weights in results[: args.top]:
        print(f"  acc={correct/len(ids):.4f} ({correct}/{len(ids)})  weights={list(weights)}")

    best = results[0]
    print(f"\nrecommended --weights {' '.join(str(w) for w in best[2])} "
          f"(apply to leaderboard preds IN THE SAME FILE ORDER)")


if __name__ == "__main__":
    main()
