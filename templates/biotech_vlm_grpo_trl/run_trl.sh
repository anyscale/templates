#!/usr/bin/env bash
# Vanilla TRL GRPO (HF generate rollouts, no vLLM) on NCT-CRC-HE, launched through
# ray.train.torch.TorchTrainer onto the cluster's GPU worker.
#
#   bash run_trl.sh                                        # configs/grpo_smoke.yaml
#   bash run_trl.sh --max_steps 30 --num_gpus 2            # override any config key
#   bash run_trl.sh --config configs/my_run.yaml           # a different config
#
# Same cluster facts as ../biotech_vlm_grpo/run_pathvlm.sh: the head node is CPU-only
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

# Default config unless the caller passed one.
case " $* " in *" --config "*) ;; *) set -- --config configs/grpo_smoke.yaml "$@" ;; esac

# The dataset is the SkyRL arm's parquet; regenerate it if missing (workspace only:
# the sibling directory is not part of a job's working_dir).
DATA_DIR=$(python3 - "$@" <<'PY'
import sys, yaml
argv = sys.argv[1:]
if "--data_dir" in argv: print(argv[argv.index("--data_dir") + 1]); sys.exit()
print(yaml.safe_load(open(argv[argv.index("--config") + 1])).get("data_dir", "/mnt/cluster_storage/data/nct_crc"))
PY
)
if [ ! -f "$DATA_DIR/train.parquet" ] && [ -f ../biotech_vlm_grpo/nct_crc_dataset.py ]; then
  echo "=== $DATA_DIR/train.parquet missing; generating with the SkyRL arm's script ==="
  uv run --frozen python ../biotech_vlm_grpo/nct_crc_dataset.py --output_dir "$DATA_DIR"
fi

exec uv run --frozen python train_grpo_trl.py --ray "$@"
