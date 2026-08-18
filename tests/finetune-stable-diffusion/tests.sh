#!/usr/bin/env bash
set -euxo pipefail

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill
papermill README.ipynb /tmp/finetune-stable-diffusion.out.ipynb --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
