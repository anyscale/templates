#!/usr/bin/env bash
set -euxo pipefail

# Install papermill via `uv pip` (not the workspace's default pip). The default
# /home/ray/anaconda3/bin/pip carries Anyscale's `snapshot_util` hook, which
# auto-captures installed packages into the Ray runtime_env shipped to workers.
# A plain `pip install papermill` would inject an UNPINNED papermill next to the
# hash-pinned python_depset.lock in the runtime_env, tripping pip's
# --require-hashes mode ("all requirements must be pinned with =="). uv bypasses
# the hook, so papermill stays on the head (where it belongs) and out of workers.
uv pip install -q --system papermill
papermill README.ipynb /tmp/fintech_fraud_risk.out.ipynb --log-output --kernel python3 --cwd .
