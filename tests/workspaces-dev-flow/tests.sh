#!/usr/bin/env bash
set -euo pipefail
pip install papermill
uv pip install -q --system emoji
papermill README.ipynb output.ipynb -k python3 --log-output

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
