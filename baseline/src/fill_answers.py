"""Fill the answer column of the organizer's test_submission.csv for upload.

The final submission is the ORIGINAL downloaded file with its answer column
filled with integers (Google Form upload). This script merges our predicted
answers into that file without touching anything else, and refuses to write
unless every row got an integer.

  python src/fill_answers.py \
      --template "$DLC_DATA_DIR/test_submission.csv" \
      --answers  "$DLC_OUTPUT_DIR/final_test/submission_ensemble.csv" \
      --out      "$DLC_OUTPUT_DIR/final_test/test_submission_FILLED.csv"
"""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="organizer's test_submission.csv")
    ap.add_argument("--answers", required=True, help="our id,answer predictions CSV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tpl = pd.read_csv(args.template)
    ans = pd.read_csv(args.answers)
    amap = dict(zip(ans[ans.columns[0]], ans["answer"]))

    missing = [qid for qid in tpl["id"] if qid not in amap]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} template ids have no prediction, e.g. {missing[:5]}")

    filled = [int(amap[qid]) for qid in tpl["id"]]  # int() guards against floats
    tpl["answer"] = filled
    tpl.to_csv(args.out, index=False)

    print(f"template rows          : {len(tpl)}")
    print(f"answers filled         : {len(filled)}  (all integers)")
    print(f"answer stats           : min={min(filled)}, max={max(filled)}, zeros={sum(1 for a in filled if a == 0)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
