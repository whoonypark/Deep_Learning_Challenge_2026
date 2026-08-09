"""Prepare challenge data: filter bad rows, make a fixed train/val split.

Outputs (under --out-dir):
  train_pool.csv  - cleaned training questions with integer answers
  val.csv         - held-out validation set (never train on this)

Usage:
  python src/prepare_data.py   # reads $DLC_DATA_DIR, writes $DLC_PROCESSED_DIR (see src/paths.py)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

import paths

# rows whose question depends on an image / Asymptote figure are unsolvable
# from text and would only add label noise
_UNSOLVABLE_RE = re.compile(r"\[asy\]|https?://|!\[|\.png|\.jpg|\.jpeg", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(paths.DATA_DIR))
    ap.add_argument("--out-dir", default=str(paths.PROCESSED_DIR))
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "deep_chal_math_train.csv")
    bad_ids = set(pd.read_csv(data_dir / "train_filtered_ids.csv")["id"])
    n0 = len(train)

    train = train[~train["id"].isin(bad_ids)].copy()
    n1 = len(train)

    figure_mask = train["question"].str.contains(_UNSOLVABLE_RE)
    train = train[~figure_mask].copy()
    n2 = len(train)

    # answers are guaranteed integers; enforce and store as int
    train["answer"] = pd.to_numeric(train["answer"], errors="raise").astype("int64")

    val = train.sample(n=args.val_size, random_state=args.seed)
    pool = train.drop(index=val.index)

    pool.to_csv(out_dir / "train_pool.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)

    print(f"raw train rows          : {n0}")
    print(f"after bad-id filter     : {n1}  (-{n0 - n1})")
    print(f"after figure filter     : {n2}  (-{n1 - n2})")
    print(f"train_pool / val        : {len(pool)} / {len(val)}")
    print(f"written to              : {out_dir}")


if __name__ == "__main__":
    main()
