#!/usr/bin/env bash
# Vanilla TRL GRPO (HF generate rollouts, no vLLM) on NCT-CRC-HE, launched through
# ray.train.torch.TorchTrainer onto the cluster's GPU worker.
#
#   bash run_trl.sh                       # 4 GPUs, 10 steps, the smoke test
#   NUM_GPUS=2 MAX_STEPS=30 bash run_trl.sh
#   MAX_COMPLETION=512 PER_DEVICE_BS=8 bash run_trl.sh
#
# Same cluster facts as ../biotech_vlm_grpo/run_pathvlm.sh: the head node is CPU-only
# and has no torch, so the environment is a uv project and Ray's uv runtime-env hook
# rebuilds it on the worker (working_dir = this directory, py_executable = uv run).
# Everything the workers read or write lives on /mnt/cluster_storage.
set -eo pipefail

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HF_TOKEN lives in ~/.workspacerc on this workspace (not needed for the ungated model
# and dataset, but it silences Hub rate-limit warnings).
if [ -f "$HOME/.workspacerc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.workspacerc"
fi

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HOME="${HF_HOME:-/mnt/cluster_storage/hf_cache}"
: "${DATA_DIR:=/mnt/cluster_storage/data/nct_crc}"
export DATA_DIR

cd "$TEMPLATE_DIR"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== $DATA_DIR/train.parquet missing; generating with the SkyRL arm's script ==="
  uv run --frozen python ../biotech_vlm_grpo/nct_crc_dataset.py --output_dir "$DATA_DIR"
fi

exec uv run --frozen python train_grpo_trl.py --ray "$@"
