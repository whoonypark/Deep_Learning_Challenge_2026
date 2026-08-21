"""Batch inference with vLLM (GPU): self-consistency voting -> metrics/submission.

Runs Qwen2.5-3B-Instruct (optionally with a LoRA adapter) on a CSV of
questions, samples k completions per question, majority-votes the extracted
integers, and writes:
  - preds.jsonl   (per-question: extracted ints, voted answer, raw texts)
  - submission.csv (only when --submission is given)
  - exact-match accuracy on stdout (only when the input CSV has an `answer` col)

All default locations come from src/paths.py (override with DLC_DATA_DIR /
DLC_OUTPUT_DIR env vars, e.g. via scripts/env_server.sh on the lab server).

Examples:
  # zero-shot greedy on validation
  python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/val.csv" --k 1 --temperature 0

  # self-consistency k=8 leaderboard submission
  python src/infer_vllm.py --input "$DLC_DATA_DIR/deep_chal_math_leaderboard_filtered.csv" \
      --k 8 --temperature 0.8 --submission "$DLC_OUTPUT_DIR/leaderboard_sc8/submission.csv"

  # rejection-sampling generation on the train pool (for SFT data)
  python src/infer_vllm.py --input "$DLC_OUTPUT_DIR/data_processed/train_pool.csv" \
      --k 4 --temperature 1.0 --out-dir "$DLC_OUTPUT_DIR/rs_train"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import paths
from answer_extraction import extract_final_int, majority_vote
from prompts import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora", default=None, help="path to a LoRA adapter dir (optional)")
    ap.add_argument("--input", required=True, help="CSV with id,question[,answer]")
    ap.add_argument("--out-dir", default=None, help="default: $DLC_OUTPUT_DIR/<input-stem>")
    ap.add_argument("--submission", default=None, help="write submission CSV here")
    # grader expects lowercase "id" (confirmed 2026-08: "ID" is rejected with
    # "ID column id not found in submission")
    ap.add_argument("--id-col", default="id", help="submission id column header")
    ap.add_argument("--k", type=int, default=8, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N rows; with --limit this selects a shard, so a "
                         "long run can be split into pieces that each save on completion "
                         "(a killed Colab session then costs one shard, not the whole run)")
    ap.add_argument("--limit", type=int, default=None, help="only N rows (after --offset)")
    ap.add_argument("--no-save-texts", action="store_true", help="omit raw texts in preds.jsonl")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported late: GPU-only dependency

    df = pd.read_csv(args.input)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"no rows to process (offset={args.offset}, limit={args.limit}) - nothing to do")
        return
    has_gold = "answer" in df.columns

    out_dir = Path(args.out_dir) if args.out_dir else paths.OUTPUT_DIR / Path(args.input).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        seed=args.seed,
    )
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        llm_kwargs.update(enable_lora=True, max_lora_rank=64)
        lora_request = LoRARequest("adapter", 1, args.lora)

    llm = LLM(**llm_kwargs)

    greedy = args.k == 1 and args.temperature == 0
    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature if not greedy else 0.0,
        top_p=args.top_p if not greedy else 1.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Build prompt strings ourselves instead of llm.chat(): batched chat needs
    # vllm >= 0.6.2, but this works on every version (incl. 0.6.1 cu118 wheels
    # for CUDA-11-driver machines).
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(build_messages(q), tokenize=False, add_generation_prompt=True)
        for q in df["question"]
    ]
    outputs = llm.generate(prompts, sampling_params=sp, lora_request=lora_request, use_tqdm=True)

    n_correct = 0
    n_sample_correct = 0
    n_samples_total = 0
    rows = []
    with open(out_dir / "preds.jsonl", "w") as f:
        for (_, r), out in zip(df.iterrows(), outputs):
            texts = [c.text for c in out.outputs]
            ints = [extract_final_int(t) for t in texts]
            voted = majority_vote(ints)
            rec = {"id": r["id"], "extracted": ints, "voted": voted}
            if has_gold:
                gold = int(r["answer"])
                rec["gold"] = gold
                n_correct += int(voted == gold)
                n_sample_correct += sum(int(a == gold) for a in ints)
                n_samples_total += len(ints)
            if not args.no_save_texts:
                rec["question"] = r["question"]
                rec["texts"] = texts
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append((r["id"], 0 if voted is None else voted))

    print(f"\nwrote {out_dir / 'preds.jsonl'}  ({len(rows)} questions, k={args.k})")
    if has_gold:
        acc = n_correct / len(rows)
        pass1 = n_sample_correct / max(n_samples_total, 1)
        print(f"majority-vote accuracy : {acc:.4f}  ({n_correct}/{len(rows)})")
        print(f"avg per-sample accuracy: {pass1:.4f}")

    if args.submission:
        sub = pd.DataFrame(rows, columns=[args.id_col, "answer"])
        Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(args.submission, index=False)
        print(f"wrote {args.submission}")


if __name__ == "__main__":
    main()
