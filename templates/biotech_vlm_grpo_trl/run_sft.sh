#!/usr/bin/env bash
# TRL SFT + Ray Train skeleton on fake slide-tile QA data.
#
#   bash run_sft.sh                                        # configs/sft_vlm_smoke.yaml
#   bash run_sft.sh --config configs/sft_llm_smoke.yaml    # text-only twin
#   bash run_sft.sh --max_steps 50 --eval_steps 25         # override any config key
#
# Generates the fake JSONL dataset on first run (needs the SkyRL arm's NCT-CRC
# val.parquet for the tiles; see make_sft_data.py).
set -eo pipefail
TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEMPLATE_DIR"
if [ -f "$HOME/.workspacerc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.workspacerc"
fi
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export HF_HOME="${HF_HOME:-/mnt/cluster_storage/hf_cache}"

case " $* " in *" --config "*) ;; *) set -- --config configs/sft_vlm_smoke.yaml "$@" ;; esac

DATA_DIR=$(python3 - "$@" <<'PY'
import sys, yaml
argv = sys.argv[1:]
if "--data_dir" in argv: print(argv[argv.index("--data_dir") + 1]); sys.exit()
print(yaml.safe_load(open(argv[argv.index("--config") + 1])).get("data_dir", "/mnt/cluster_storage/data/pathai_sft_fake"))
PY
)
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "=== generating fake SFT data in $DATA_DIR ==="
  uv run --frozen python make_sft_data.py --output_dir "$DATA_DIR"
fi

exec uv run --frozen python train_sft_trl.py --ray "$@"
