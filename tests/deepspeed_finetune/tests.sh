#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/deepspeed_finetune.out.ipynb --log-output --kernel python3 --cwd .

# The standalone training script (separately exercised; --debug_steps caps to ~30 steps).
python train.py --debug_steps 30
