"""Select GRPO training prompts with real learning signal.

GRPO's advantage is zero when every sampled completion gets the same reward,
so questions the sampler always solves (8/8) or never solves (0/8) contribute
nothing. We use the rs_audit sampling stats (merged-r2, k=8) to keep only
questions with intermediate pass rates, restricted to audited-clean ids.

Output: JSONL with {"prompt": [messages], "answer": int} for GRPOTrainer.

  python src/prepare_grpo_data.py \
      --preds "$DLC_OUTPUT_DIR/rs_audit/preds.jsonl" \
      --only-ids "$PROC/train_pool_clean.csv" \
      --out "$PROC/grpo_prompts.jsonl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prompts import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="rs_audit preds.jsonl (has question+gold)")
    ap.add_argument("--only-ids", required=True, help="audited-clean CSV (id column)")
    ap.add_argument("--min-correct", type=int, default=1)
    ap.add_argument("--max-correct", type=int, default=7, help="of the k=8 audit samples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allowed = set(pd.read_csv(args.only_ids)["id"])
    kept = 0
    n_seen = 0
    dist: dict = {}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.preds) as f, open(out, "w") as g:
        for line in f:
            r = json.loads(line)
            n_seen += 1
            if r["id"] not in allowed or r.get("gold") is None or not r.get("question"):
                continue
            n_correct = sum(int(a == r["gold"]) for a in r.get("extracted", []))
            dist[n_correct] = dist.get(n_correct, 0) + 1
            if not (args.min_correct <= n_correct <= args.max_correct):
                continue
            kept += 1
            g.write(json.dumps(
                {"prompt": build_messages(r["question"]), "answer": int(r["gold"])},
                ensure_ascii=False) + "\n")

    print(f"scanned {n_seen}, pass-rate distribution (correct/8): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
    print(f"kept {kept} prompts with {args.min_correct}<=correct<={args.max_correct} -> {out}")


if __name__ == "__main__":
    main()
