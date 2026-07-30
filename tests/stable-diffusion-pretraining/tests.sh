#!/usr/bin/env bash
set -euo pipefail

# Install the lock rather than a hand-typed subset: it is the closure the scripts hand
# to Ray via runtime_env, and it carries the pins (numpy, pyarrow, torch+cu128) that the
# loose list dropped.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

echo "=== Running preprocess.py validation ==="
python scripts/preprocess.py --limit 5 --resolution 512 --no-visualize-output

# NOTE: Skipping standalone train.py test - it requires preprocessed S3 data which uses
# SD v2's CLIP encoder (1024-dim), but we use SD v1.4 (768-dim) to avoid HF auth issues.
# The end_to_end.py test below covers both preprocessing and training code paths.

echo "=== Running end_to_end.py validation ==="
python scripts/end_to_end.py --experiment-name sd-e2e-v254-$(date +%s) --limit 5 --max-steps 10 --checkpoint-every-n-steps 100 --val-check-interval 100

echo "=== All validations passed ==="
