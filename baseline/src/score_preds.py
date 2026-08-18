"""Score one or more preds.jsonl files, optionally restricted to an id subset.

Lets us re-score OLD runs on the audited-clean val without re-running the GPU:

  python src/score_preds.py --ids data_processed/val_clean.csv \
      outputs/val_sc8/preds.jsonl outputs/val_sft_sc8/preds.jsonl ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="preds.jsonl files (need gold)")
    ap.add_argument("--ids", default=None, help="CSV with an `id` column: score only these rows")
    args = ap.parse_args()

    keep = None
    if args.ids:
        import pandas as pd

        keep = set(pd.read_csv(args.ids)["id"])

    for pf in args.preds:
        p = Path(pf)
        if not p.exists():
            print(f"{pf}: (missing)")
            continue
        n = c = 0
        for line in open(p):
            r = json.loads(line)
            if keep is not None and r["id"] not in keep:
                continue
            if r.get("gold") is None:
                continue
            n += 1
            c += int(r.get("voted") == r.get("gold"))
        label = str(p.parent.name)
        acc = c / n if n else float("nan")
        print(f"{label:24s} {c:4d}/{n:4d} = {acc:.4f}")


if __name__ == "__main__":
    main()
