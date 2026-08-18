#!/usr/bin/env bash
set -euo pipefail

# Install the locked closure the scripts ship to Ray.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

echo "=== Running preprocess.py validation ==="
python scripts/preprocess.py --limit 5 --resolution 512 --no-visualize-output

# NOTE: Skipping standalone train.py test - it requires preprocessed S3 data which uses
# SD v2's CLIP encoder (1024-dim), but we use SD v1.4 (768-dim) to avoid HF auth issues.
# The end_to_end.py test below covers both preprocessing and training code paths.

echo "=== Running end_to_end.py validation ==="
python scripts/end_to_end.py --experiment-name sd-e2e-v254-$(date +%s) --limit 5 --max-steps 10 --checkpoint-every-n-steps 100 --val-check-interval 100

echo "=== All validations passed ==="

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
