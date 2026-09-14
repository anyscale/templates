#!/usr/bin/env bash
# Vanilla TRL GRPO (HF generate rollouts, no vLLM) on NCT-CRC-HE, launched through
# ray.train.torch.TorchTrainer onto the cluster's GPU worker.
#
#   bash run_trl.sh                                        # configs/grpo_smoke.yaml
#   bash run_trl.sh --max_steps 30 --num_gpus 2            # override any config key
#   bash run_trl.sh --config configs/my_run.yaml           # a different config
#
# Cluster facts: the head node is CPU-only
# and has no torch, so the environment is a uv project and Ray's uv runtime-env hook
# rebuilds it on the worker (working_dir = this directory, py_executable = uv run).
set -eo pipefail
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEMPLATE_DIR"

# HF_TOKEN lives in ~/.workspacerc on this workspace (not needed for the ungated model
# and dataset, but it silences Hub rate-limit warnings). Absent on a job cluster.
if [ -f "$HOME/.workspacerc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.workspacerc"
fi
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HOME="${HF_HOME:-/mnt/cluster_storage/hf_cache}"
# One shared uv env per node (not a 7 GB .venv inside every uploaded working_dir copy).
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/biotech_vlm_grpo_trl}"

# Default config unless the caller passed one.
case " $* " in *" --config "*) ;; *) set -- --config configs/grpo_smoke.yaml "$@" ;; esac

# Build the dataset parquet on first run.
DATA_DIR=$(python3 - "$@" <<'PY'
import sys, yaml
argv = sys.argv[1:]
if "--data_dir" in argv: print(argv[argv.index("--data_dir") + 1]); sys.exit()
print(yaml.safe_load(open(argv[argv.index("--config") + 1])).get("data_dir", "/mnt/cluster_storage/data/nct_crc"))
PY
)
if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== $DATA_DIR/train.parquet missing; generating with nct_crc_dataset.py ==="
  uv run --frozen python nct_crc_dataset.py --output_dir "$DATA_DIR"
fi

# Ray Data preprocessing, once: TRL-ready parquet so workers skip the per-rank image decode.
if [ ! -d "$DATA_DIR/trl/train" ]; then
  echo "=== preparing TRL-ready parquet with Ray Data ==="
  uv run --frozen python prepare_data.py --data_dir "$DATA_DIR"
fi

# vLLM server mode needs the vllm extra in the environment (both driver and workers).
EXTRA=()
if python3 - "$@" <<'PY'
import sys, yaml
argv = sys.argv[1:]
v = argv[argv.index("--use_vllm") + 1].lower() if "--use_vllm" in argv else str(yaml.safe_load(open(argv[argv.index("--config") + 1])).get("use_vllm", False)).lower()
sys.exit(0 if v == "true" else 1)
PY
then EXTRA=(--extra vllm); fi

exec uv run --frozen "${EXTRA[@]}" python train_grpo_trl.py --ray "$@"
