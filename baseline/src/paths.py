"""Central path configuration — single source of truth for data / output dirs.

Every location can be overridden with an environment variable, so the same
code runs unchanged on the Mac (defaults) and on the lab server:

  DLC_DATA_DIR       raw challenge CSVs (deep_chal_math_train.csv, ...)
                     default: <repo_root>/data
  DLC_OUTPUT_DIR     everything the pipeline generates (preds, adapters,
                     submissions, tensorboard logs, processed data)
                     default: <baseline>/outputs
  DLC_PROCESSED_DIR  train_pool.csv / val.csv / sft.jsonl
                     default: $DLC_OUTPUT_DIR/data_processed

Lab server setup (set in scripts/env_server.sh):
  DLC_DATA_DIR=$HOME/shared/hdd_ext/ssd4000/jihoon/math_data
  DLC_OUTPUT_DIR=$HOME/shared/hdd_ext/ssd4000/jihoon/math_output
"""

from __future__ import annotations

import os
from pathlib import Path

_BASELINE_DIR = Path(__file__).resolve().parents[1]   # .../baseline
_REPO_ROOT = _BASELINE_DIR.parent                     # repo root


def _env_path(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


DATA_DIR = _env_path("DLC_DATA_DIR", _REPO_ROOT / "data")
OUTPUT_DIR = _env_path("DLC_OUTPUT_DIR", _BASELINE_DIR / "outputs")
PROCESSED_DIR = _env_path("DLC_PROCESSED_DIR", OUTPUT_DIR / "data_processed")

# frequently used files
TRAIN_CSV = DATA_DIR / "deep_chal_math_train.csv"
BAD_IDS_CSV = DATA_DIR / "train_filtered_ids.csv"
LEADERBOARD_CSV = DATA_DIR / "deep_chal_math_leaderboard_filtered.csv"
TRAIN_POOL_CSV = PROCESSED_DIR / "train_pool.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
SFT_JSONL = PROCESSED_DIR / "sft.jsonl"


if __name__ == "__main__":  # quick sanity check: python src/paths.py
    for name in ("DATA_DIR", "OUTPUT_DIR", "PROCESSED_DIR"):
        p = globals()[name]
        print(f"{name:14s} = {p}   (exists: {p.exists()})")
