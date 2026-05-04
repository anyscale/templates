#!/usr/bin/env bash
set -euo pipefail

# Pin versions to avoid CUDA driver incompatibility (image has CUDA 12.8)
pip install papermill diffusers==0.34.0 transformers==4.47.0 accelerate==1.2.1 lightning==2.6.1 typer s3fs matplotlib Pillow 'torch>=2.5,<2.6' 'torchvision>=0.20,<0.21'

echo "=== Running preprocess.py validation ==="
python scripts/preprocess.py --limit 5 --resolution 512 --no-visualize-output

# NOTE: Skipping standalone train.py test - it requires preprocessed S3 data which uses
# SD v2's CLIP encoder (1024-dim), but we use SD v1.4 (768-dim) to avoid HF auth issues.
# The end_to_end.py test below covers both preprocessing and training code paths.

echo "=== Running end_to_end.py validation ==="
python scripts/end_to_end.py --experiment-name sd-e2e-v254-$(date +%s) --limit 5 --max-steps 10 --checkpoint-every-n-steps 100 --val-check-interval 100

echo "=== All validations passed ==="
