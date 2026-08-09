"""Local smoke test on the MacBook (MPS) - NOT for full runs.

Runs a handful of validation questions through the base model with plain
transformers so you can sanity-check the prompt + answer extraction pipeline
without a CUDA GPU. Expect ~10-30 s per question on an M4 Pro.

  python src/infer_local_smoke.py --n 3
"""

from __future__ import annotations

import argparse

import pandas as pd

import paths
from answer_extraction import extract_final_int
from prompts import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--input", default=str(paths.VAL_CSV))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    df = pd.read_csv(args.input).head(args.n)
    n_ok = 0
    for _, r in df.iterrows():
        messages = build_messages(r["question"])
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_final_int(text)
        gold = int(r["answer"]) if "answer" in df.columns else None
        ok = pred == gold
        n_ok += int(ok)
        print("=" * 80)
        print(f"[{r['id']}] gold={gold}  pred={pred}  {'OK' if ok else 'MISS'}")
        print(text[-500:])
    print("=" * 80)
    print(f"{n_ok}/{len(df)} correct (smoke test only - not a benchmark)")


if __name__ == "__main__":
    main()
