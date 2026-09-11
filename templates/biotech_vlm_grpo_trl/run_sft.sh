#!/usr/bin/env bash
# TRL SFT + Ray Train skeleton on fake slide-tile QA data.
#
#   bash run_sft.sh                 # MODE=vlm: Qwen3-VL-2B on tiles + QA, 4 GPUs, 20 steps
#   MODE=llm bash run_sft.sh        # text-only twin with Qwen2.5-0.5B-Instruct
#   NUM_GPUS=2 MAX_STEPS=50 EVAL_STEPS=25 bash run_sft.sh
#
# Generates the fake JSONL dataset on first run (needs the SkyRL arm's NCT-CRC
# val.parquet for the tiles; see make_sft_data.py).
set -eo pipefail
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$HOME/.workspacerc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.workspacerc"
fi
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HOME="${HF_HOME:-/mnt/cluster_storage/hf_cache}"
: "${SFT_DATA_DIR:=/mnt/cluster_storage/data/pathai_sft_fake}"
export SFT_DATA_DIR
cd "$TEMPLATE_DIR"
if [ ! -f "$SFT_DATA_DIR/train.jsonl" ]; then
  echo "=== generating fake SFT data in $SFT_DATA_DIR ==="
  uv run --frozen python make_sft_data.py --output_dir "$SFT_DATA_DIR"
fi
exec uv run --frozen python train_sft_trl.py --ray "$@"
