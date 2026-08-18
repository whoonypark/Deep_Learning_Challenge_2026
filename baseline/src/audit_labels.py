"""Automatic label audit via model consensus (community reports confirm the
challenge data still contains wrong labels after the organizer's filtering).

Method (same as the manual community reports, but automated): sample k
solutions per question with the current best model; when a strong majority
converges on one integer that DISAGREES with the gold label, flag the label
as suspect. Also flag questions matching unsolvable patterns (lost figures,
mathpix URLs, preprocessing残骸).

Inputs : preds.jsonl from infer_vllm.py (id, extracted, gold [, question])
         + the original CSV (id, question, answer)
Outputs: --out-suspects  CSV for manual spot-checking (sorted by agreement)
         --out-clean     input CSV minus suspect/unsolvable rows

  python src/audit_labels.py --preds outputs/rs_audit/preds.jsonl \
      --input data_processed/train_pool.csv \
      --out-suspects outputs/audit/train_suspects.csv \
      --out-clean data_processed/train_pool_clean.csv
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# extends prepare_data.py's filter with patterns from the community reports
_UNSOLVABLE_RE = re.compile(
    r"\[asy\]|https?://|!\[|\.png|\.jpg|\.jpeg|cdn\.mathpix|untransla",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--input", required=True, help="CSV with id,question,answer")
    ap.add_argument("--agree-frac", type=float, default=0.75,
                    help="consensus threshold: fraction of samples agreeing on one "
                         "answer (0.75 = 6/8 or 12/16)")
    ap.add_argument("--min-k", type=int, default=4,
                    help="ignore questions with fewer than this many extracted answers")
    ap.add_argument("--out-suspects", required=True)
    ap.add_argument("--out-clean", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    consensus = {}
    with open(args.preds) as f:
        for line in f:
            r = json.loads(line)
            answers = [a for a in r.get("extracted", []) if a is not None]
            if len(answers) < args.min_k:
                continue
            top, cnt = Counter(answers).most_common(1)[0]
            consensus[r["id"]] = (top, cnt, len(answers), r.get("gold"))

    suspects = []
    for qid, (top, cnt, k, gold) in consensus.items():
        if gold is None:
            continue
        if cnt / k >= args.agree_frac and top != gold:
            suspects.append(
                {"id": qid, "label": gold, "consensus": top,
                 "agree": f"{cnt}/{k}", "agree_frac": cnt / k}
            )

    sus = pd.DataFrame(suspects).sort_values("agree_frac", ascending=False) if suspects else \
        pd.DataFrame(columns=["id", "label", "consensus", "agree", "agree_frac"])
    sus = sus.merge(df[["id", "question"]], on="id", how="left")

    unsolvable_mask = df["question"].str.contains(_UNSOLVABLE_RE)
    suspect_ids = set(sus["id"])
    clean = df[~df["id"].isin(suspect_ids) & ~unsolvable_mask].copy()

    Path(args.out_suspects).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_clean).parent.mkdir(parents=True, exist_ok=True)
    sus.to_csv(args.out_suspects, index=False)
    clean.to_csv(args.out_clean, index=False)

    print(f"questions audited        : {len(consensus)}")
    print(f"suspect labels flagged   : {len(sus)}  ({len(sus)/max(len(df),1):.1%} of input)")
    print(f"unsolvable-pattern rows  : {int(unsolvable_mask.sum())}")
    print(f"clean rows written       : {len(clean)} -> {args.out_clean}")
    print(f"suspects for review      : {args.out_suspects}")
    print("\ntop suspects (verify a few by hand, like the community reports):")
    for _, row in sus.head(10).iterrows():
        q = str(row['question'])[:90].replace("\n", " ")
        print(f"  {row['id']}  label={row['label']}  consensus={row['consensus']} ({row['agree']})  {q}")


if __name__ == "__main__":
    main()
