#!/bin/bash

set -euxo pipefail

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -e . --system

jupyter execute e2e_timeseries/01-Distributed-Training.ipynb e2e_timeseries/02-Validation.ipynb e2e_timeseries/03-Serving.ipynb

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
