"""LoRA SFT on rejection-sampled solutions (TRL >= 0.20, single 24GB GPU).

Trains Qwen2.5-3B-Instruct with LoRA on the prompt-completion JSONL from
build_sft_data.py. Loss is applied to the completion only (TRL default for
prompt-completion datasets). Logs to TensorBoard (a recommended deliverable).

  python src/sft_lora.py --data data_processed/sft.jsonl --out outputs/sft-lora-r1
"""

from __future__ import annotations

import argparse

import paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default=str(paths.SFT_JSONL))
    ap.add_argument("--out", default=str(paths.OUTPUT_DIR / "sft-lora-r1"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing: ~2x faster, more VRAM. "
                         "Use on the 48GB A6000; keep checkpointing ON for 24GB cards.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=args.data, split="train")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        bf16=True,
        gradient_checkpointing=not args.no_grad_ckpt,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
