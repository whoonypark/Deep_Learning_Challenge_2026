"""Merge a LoRA adapter into the base model (for checkpoint submission).

  python src/merge_lora.py --adapter outputs/sft-lora-r1 --out outputs/merged-r1
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import json
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)

    # transformers >= 4.51 saves the chat template to a separate
    # chat_template.jinja, which older transformers (the vllm inference env)
    # cannot read -> embed it back into tokenizer_config.json.
    out = Path(args.out)
    jinja = out / "chat_template.jinja"
    cfg_path = out / "tokenizer_config.json"
    if jinja.exists() and cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if not cfg.get("chat_template"):
            cfg["chat_template"] = jinja.read_text()
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
            print("embedded chat_template.jinja into tokenizer_config.json")

    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
