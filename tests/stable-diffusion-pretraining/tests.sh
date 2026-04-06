#!/usr/bin/env bash
set -euo pipefail

pip install papermill diffusers transformers accelerate lightning typer s3fs matplotlib Pillow torchvision

echo "=== Running preprocess.py validation ==="
python scripts/preprocess.py --limit 5 --resolution 512 --no-visualize-output

echo "=== Running train.py validation ==="
python scripts/train.py --max-steps 10 --checkpoint-every-n-steps 100 --val-check-interval 100

echo "=== Running end_to_end.py validation ==="
python scripts/end_to_end.py --limit 5 --max-steps 10 --checkpoint-every-n-steps 100 --val-check-interval 100

echo "=== All validations passed ==="
