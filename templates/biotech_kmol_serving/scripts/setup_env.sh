#!/usr/bin/env bash
# Build the validated kMoL + Ray Serve environment (CPU-capable). See ENVIRONMENT.md.
# Verified on an Anyscale workspace: produces a py3.9 conda env "kmol" in which the
# molecule ensemble serving path imports and runs with no compiled extensions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KMOL_REF="${KMOL_REF:-c7f8833}"

cd "$ROOT"
[ -d kmol ] || { git clone https://github.com/elix-tech/kmol.git kmol; git -C kmol checkout "$KMOL_REF"; }

echo "==> creating conda env 'kmol' from kmol/environment.yml (~5GB, a few min)"
mamba env create -f kmol/environment.yml || echo "(env may already exist)"

BASE="$(conda info --base)"
PY="$BASE/envs/kmol/bin/python"
PIP="$BASE/envs/kmol/bin/pip"

echo "==> installing kMoL light pip deps + Ray 2.51.2 (newest py3.9 wheel)"
$PIP install --no-input \
  "disklist==0.4.0" "filelock==3.12.4" "boxsdk[jwt]==3.6.1" \
  "torch-lr-finder==0.2.1" "ml-collections==0.1.0" "proDy" "biopython==1.81" \
  "ray[serve]==2.51.2" requests

echo "==> verifying import (with stubs on PYTHONPATH)"
export PYTHONPATH="$ROOT/stubs:$ROOT:$ROOT/kmol/src"
$PY -c "from kmol.model.executors import Predictor; from ray import serve; print('IMPORT_OK — ray', __import__('ray').__version__)"

cat <<EOF

Environment ready. To serve locally (CPU):
  export PYTHONPATH="$ROOT/stubs:$ROOT:$ROOT/kmol/src"
  export KMOL_CONFIG_PATH=configs/ensemble_serve.example.json KMOL_NUM_GPUS=0
  python scripts/make_synthetic_checkpoints.py \$KMOL_CONFIG_PATH   # if no real weights
  $BASE/envs/kmol/bin/serve run serve_app:app

In an Anyscale workspace, first: unset RAY_RUNTIME_ENV_HOOK RAY_ADDRESS
and NEVER run 'ray stop' (it kills the managed cluster).
EOF
