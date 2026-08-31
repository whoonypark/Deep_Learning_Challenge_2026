"""Answer-level majority vote across submission CSVs (id,answer).

The old models are gone, but their ANSWERS survive in downloaded submission
files — and old-family vs new-family errors are far less correlated than
anything inside one family. Ties are broken by file priority: list the
strongest submission FIRST.

  python src/submission_vote.py sub9.csv sub13.csv sub4.csv sub10.csv sub11.csv \
      --out submission_fused.csv
"""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="submission CSVs, strongest FIRST (tie-break priority)")
    ap.add_argument("--weights", nargs="*", type=int, default=None, help="votes per file (default 1 each)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = args.weights or [1] * len(args.csvs)
    if len(weights) != len(args.csvs):
        raise SystemExit("--weights must match number of csvs")

    subs = []
    for p in args.csvs:
        df = pd.read_csv(p)
        id_col = df.columns[0]
        subs.append(dict(zip(df[id_col], df["answer"])))
        print(f"loaded {p}: {len(df)} rows")

    ids = list(subs[0].keys())
    n_flipped = 0
    rows = []
    for qid in ids:
        votes = Counter()
        for w, s in zip(weights, subs):
            if qid in s:
                votes[s[qid]] += w
        top = votes.most_common()
        best_count = top[0][1]
        tied = [a for a, c in top if c == best_count]
        if len(tied) == 1:
            ans = tied[0]
        else:  # tie -> earliest file whose answer is among the tied
            ans = next(s[qid] for s in subs if s.get(qid) in tied)
        if ans != subs[0][qid]:
            n_flipped += 1
        rows.append((qid, ans))

    out = pd.DataFrame(rows, columns=["id", "answer"])
    out.to_csv(args.out, index=False)
    print(f"\n{len(rows)} questions; {n_flipped} answers differ from the first (strongest) file")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
