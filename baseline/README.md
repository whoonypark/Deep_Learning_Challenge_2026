# 아주소중한딥러닝챌린지 2026 — Baseline

Fine-tune-free → rejection-sampling-SFT pipeline for the math challenge
(base model fixed: `Qwen/Qwen2.5-3B-Instruct`, metric: integer Exact Match).

## Layout

```
baseline/
├── src/
│   ├── paths.py              # central data/output dirs (DLC_DATA_DIR / DLC_OUTPUT_DIR env vars)
│   ├── prepare_data.py       # filter bad ids + figure-dependent rows, 500-question val split
│   ├── prompts.py            # single source of truth for the chat prompt
│   ├── answer_extraction.py  # \boxed{} / phrase / last-int parsing + majority vote
│   ├── infer_vllm.py         # GPU batch inference, self-consistency, submission.csv
│   ├── infer_local_smoke.py  # Mac (MPS) smoke test on a few questions
│   ├── build_sft_data.py     # rejection sampling -> TRL prompt-completion JSONL
│   ├── sft_lora.py           # LoRA SFT (TRL), TensorBoard logging
│   └── merge_lora.py         # merge adapter for checkpoint submission
├── tests/test_answer_extraction.py
├── scripts/env_server.sh                # lab-server paths + GPU pin (sourced by 0*.sh)
├── scripts/01_baseline_eval.sh          # zero-shot: greedy + SC@8 on val, LB submission
├── scripts/02_rejection_sampling_sft.sh # RS-SFT round 1 + eval
├── requirements.txt          # GPU (lab server, CUDA >= 12)
└── requirements-mac.txt      # Mac (data prep / smoke test only)
```

## Paths

All locations resolve in `src/paths.py` and can be overridden by env vars —
no code edits needed when moving between machines:

| env var          | server value (set by `scripts/env_server.sh`)      | local default        |
|------------------|-----------------------------------------------------|----------------------|
| `DLC_DATA_DIR`   | `~/shared/hdd_ext/nvme1/jihoonpark22/data`          | `<repo>/data`        |
| `DLC_OUTPUT_DIR` | `~/shared/hdd_ext/nvme1/jihoonpark22/output`        | `baseline/outputs`   |

Processed data (`train_pool.csv`, `val.csv`, `sft.jsonl`) goes to
`$DLC_OUTPUT_DIR/data_processed/`. Sanity-check with `python src/paths.py`.

## Setup

**Lab server (primary — code in `~/private/test/competition/deep-learning-challenge-2026/`):**
```bash
# one-time: put the 4 challenge CSVs in ~/shared/hdd_ext/nvme1/jihoonpark22/data/
cd ~/private/test/competition/deep-learning-challenge-2026/baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run on GPU 9 (env_server.sh pins CUDA_VISIBLE_DEVICES=9), survive ssh disconnect:
nohup bash scripts/01_baseline_eval.sh > ~/shared/hdd_ext/nvme1/jihoonpark22/output/01_baseline.log 2>&1 &
tail -f ~/shared/hdd_ext/nvme1/jihoonpark22/output/01_baseline.log

nohup bash scripts/02_rejection_sampling_sft.sh > ~/shared/hdd_ext/nvme1/jihoonpark22/output/02_rs_sft.log 2>&1 &
```
(or run inside `tmux` without `nohup` — same effect)

**Mac (data prep + pipeline dev only):**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt
python src/prepare_data.py    # note: now writes to baseline/outputs/data_processed/
python tests/test_answer_extraction.py
python src/infer_local_smoke.py --n 3   # downloads ~6 GB model once
```

## Baseline strategy

1. **v0 — no training:** zero-shot CoT prompt + self-consistency (k=8, T=0.8)
   + robust integer extraction. Establishes the floor and validates the whole
   submission pipeline end-to-end. (Inference-time techniques are explicitly
   allowed by the rules.)
2. **v1 — rejection-sampling SFT (STaR):** sample k=4 solutions per train
   question with the base model, keep only those whose `\boxed{}` equals the
   gold integer (~uses only organizer data + self-generated text → clean under
   the rules), LoRA-SFT on them, re-evaluate. Iterate (round 2 with the SFT
   model as sampler, higher k on unsolved questions).
3. **Later:** external public math data (allowed; document provenance for
   verification), GRPO with integer-match reward, tuned voting (k, T),
   difficulty-aware sampling budgets.

## Rules checklist (from the challenge docs)

- Base model fixed: `Qwen/Qwen2.5-3B-Instruct` — no other base models.
- No commercial APIs to produce test answers; no fabricated labels; offline
  inference (no internet / external API calls at inference time).
- Public external datasets allowed — keep a provenance list (verification
  deliverable).
- Submission: `submission.csv`, columns `ID,answer`, integers only
  (**check the exact id column name against the Kaggle sample submission**).
- Leaderboard file (2026-08-03+): `deep_chal_math_leaderboard_filtered.csv`
  (831 questions). Final `test.csv` drops 2026-08-31 00:00, submit same day.
- Top finishers must hand in: final checkpoint (3B architecture check),
  full training + inference code, data list, requirements.txt; training logs
  (TensorBoard) recommended → keep `outputs/` runs intact.

## Data facts (2026-08-09 profile)

- train: 17,000 rows → 16,373 after bad-id filter → 16,331 after dropping 42
  figure/URL-dependent rows; split 15,831 train-pool / 500 val (seed 42).
- leaderboard_filtered: 831 questions.
- All answers are integers; 486 negative, 210 zero, 123 with |answer| > 1e6,
  max ≈ 3.4e15 → answer parsing must keep Python ints (no float rounding).
- Question length: p50 ≈ 200 chars, p99 ≈ 760, max 3,560 → `max_model_len=4096`
  with 1,024 generated tokens is safe.
