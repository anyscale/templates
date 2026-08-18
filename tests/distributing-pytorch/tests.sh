#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/distributing-pytorch.out.ipynb --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
