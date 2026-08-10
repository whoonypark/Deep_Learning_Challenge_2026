"""Build SFT data from a public external math dataset (rules: free & public OK).

Primary source: nvidia/OpenMathInstruct-2 (Apache-2.0, HuggingFace) —
problem / generated_solution / expected_answer fields. We keep only rows where

  * expected_answer parses as a plain integer (challenge answers are ints)
  * the solution's own \\boxed{} value equals that integer (self-consistent)
  * the problem is text-only (no figures/URLs) and the solution isn't overlong

and convert them to the same TRL prompt-completion JSONL format as
build_sft_data.py, using OUR prompt (prompts.build_messages) so training and
inference always match. Optionally mixes in existing self-distill JSONL files.

Run inside the TRAINING env (needs `datasets`; downloads go to $HF_HOME):

  python src/prepare_external_data.py --max-samples 100000 \
      --include "$DLC_OUTPUT_DIR/data_processed/sft_r2.jsonl" \
      --out "$DLC_OUTPUT_DIR/data_processed/sft_ext.jsonl"

Provenance note (verification deliverable): record dataset name + split +
this filter script in the methods doc.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from answer_extraction import extract_boxed_int
from prompts import build_messages

_UNSOLVABLE_RE = re.compile(r"\[asy\]|https?://|!\[|\.png|\.jpg|\.jpeg", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nvidia/OpenMathInstruct-2")
    ap.add_argument("--split", default="train_1M",
                    help="train_1M is the deduplicated 1M subset; fall back to 'train' if missing")
    ap.add_argument("--max-samples", type=int, default=100_000)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--include", nargs="*", default=[],
                    help="existing prompt-completion JSONL files to mix in (e.g. self-distill data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset

    try:
        ds = load_dataset(args.dataset, split=args.split)
    except ValueError:
        print(f"split {args.split!r} not found, falling back to 'train' (streaming filter)")
        ds = load_dataset(args.dataset, split="train")

    rng = random.Random(args.seed)
    n_seen = n_kept = 0
    seen_problems = set()
    records = []

    for row in ds:
        n_seen += 1
        ans = str(row.get("expected_answer", "")).strip()
        if not _INT_RE.match(ans):
            continue
        problem = (row.get("problem") or "").strip()
        sol = (row.get("generated_solution") or "").strip()
        if not problem or not sol or len(sol) > args.max_chars:
            continue
        if _UNSOLVABLE_RE.search(problem):
            continue
        if extract_boxed_int(sol) != int(ans):
            continue
        key = problem[:300]
        if key in seen_problems:
            continue
        seen_problems.add(key)
        n_kept += 1
        records.append(
            {
                "prompt": build_messages(problem),
                "completion": [{"role": "assistant", "content": sol}],
            }
        )
        # keep memory bounded: reservoir-free early stop at 3x target, then sample
        if len(records) >= args.max_samples * 3:
            break

    rng.shuffle(records)
    records = records[: args.max_samples]

    n_mixed = 0
    for extra in args.include:
        with open(extra) as f:
            for line in f:
                records.append(json.loads(line))
                n_mixed += 1

    rng.shuffle(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"rows scanned            : {n_seen}")
    print(f"external kept (filtered): {n_kept} -> sampled {min(n_kept, args.max_samples)}")
    print(f"mixed-in self-distill   : {n_mixed}")
    print(f"total SFT samples       : {len(records)} -> {out}")


if __name__ == "__main__":
    main()
