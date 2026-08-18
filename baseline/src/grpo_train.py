"""GRPO with an integer-exact-match reward (TRL >= 0.20, single 24GB GPU).

Policy starts from the best SFT checkpoint (merged model dir). LoRA keeps
memory low and gives a free reference model (adapter-disabled policy) for KL.
Generation runs through transformers (no vllm in the training env — and the
CUDA-11.8 driver rules out vllm's LoRA path anyway), so steps are slow:
expect a few minutes per optimizer step. TensorBoard logs go to --out/runs.

  conda run -p $DLC_TRAIN_ENV python src/grpo_train.py \
      --model "$DLC_OUTPUT_DIR/merged-r4" \
      --data  "$PROC/grpo_prompts.jsonl" \
      --out   "$DLC_OUTPUT_DIR/grpo-lora-r5"
"""

from __future__ import annotations

import argparse
import os

# reduce fragmentation-induced OOM (must be set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from answer_extraction import extract_boxed_int


def _completion_text(comp) -> str:
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        return comp[0].get("content", "")
    return ""


def accuracy_reward(completions, answer, **kwargs):
    """1.0 iff the completion's boxed integer equals the gold answer."""
    return [
        1.0 if extract_boxed_int(_completion_text(c)) == int(a) else 0.0
        for c, a in zip(completions, answer)
    ]


def format_reward(completions, **kwargs):
    """Small bonus for producing any \\boxed{<int>} at all (keeps format stable)."""
    return [
        0.1 if extract_boxed_int(_completion_text(c)) is not None else 0.0
        for c in completions
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged SFT model dir (policy init)")
    ap.add_argument("--data", required=True, help="grpo_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8, help="via grad accumulation")
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="completions per forward/backward pass (memory knob; the "
                         "150k-vocab fp32 logits + activations OOM a 48GB card at 8)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL coefficient; 0 skips the reference-model forward "
                         "entirely (large memory/time saving, common practice)")
    ap.add_argument("--max-prompt-len", type=int, default=768)
    ap.add_argument("--max-completion-len", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("json", data_files=args.data, split="train").shuffle(seed=args.seed)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    total_completions = args.num_generations * args.prompts_per_step
    if total_completions % args.micro_batch:
        raise SystemExit("--micro-batch must divide num_generations * prompts_per_step")

    config = GRPOConfig(
        output_dir=args.out,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=total_completions // args.micro_batch,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        temperature=args.temperature,
        bf16=True,
        # NOTE: gradient checkpointing disabled on purpose — with it enabled the
        # KV cache is turned off during GRPO's generation phase ("Caching is
        # incompatible with gradient checkpointing" spam), making generation
        # ~10-50x slower. 3B + LoRA fits in 24GB without it.
        gradient_checkpointing=False,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        report_to="tensorboard",
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[accuracy_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"saved GRPO LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
