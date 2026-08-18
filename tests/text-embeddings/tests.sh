#!/usr/bin/env bash
set -euo pipefail
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill
papermill README.ipynb output.ipynb -k python3 --log-output

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
