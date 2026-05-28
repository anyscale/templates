#!/usr/bin/env bash
set -euo pipefail

echo "=== SkyRL Template Test ==="

# Mirror the template README: clone + pin to the commit the BYOD image is built for.
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/
git checkout acbc21c

# CI-only patch: switch `uv run --isolated` to `--frozen` in run_gsm8k.sh.
# SkyRL's top-level `requires-python = ">=3.11"` makes uv's universal resolver
# consider Python 3.14 + arm64-macOS, where ray==2.51.1 has no wheels, so
# `--isolated` fails resolution. `--frozen` uses the shipped uv.lock and skips
# resolution — equivalent runtime for our (linux x86_64 / py3.12) CI box.
sed -i 's/uv run --isolated/uv run --frozen/g' examples/train/gsm8k/run_gsm8k.sh

# run_gsm8k.sh reads DATA_DIR to build data.train_data/data.val_data, so the
# training step reads exactly where the dataset step writes.
export DATA_DIR=/mnt/cluster_storage/data/gsm8k

# Data prep. --max_train_dataset_length truncates train so 1 epoch is a few steps.
uv run --frozen examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR" --max_train_dataset_length 16

# Eval is disabled below; keep validation tiny as insurance.
uv run --frozen python -c "
import pyarrow.parquet as pq
p = '$DATA_DIR/validation.parquet'
pq.write_table(pq.read_table(p).slice(0, 8), p)
"

if [[ ! -f "$DATA_DIR/train.parquet" ]] || [[ ! -f "$DATA_DIR/validation.parquet" ]]; then
    echo "FAILED: Dataset files not found"
    exit 1
fi
echo "SUCCESS: Dataset preparation completed"

# Minimal real GRPO smoke via the template's own run_gsm8k.sh. Trailing args override
# the script's hardcoded values (main_base uses OmegaConf.from_cli, which is last-wins).
# Batch sizes are the smallest that pass SkyRL's divisibility asserts at dp_size=4
# (train==mini==4, micro=1, n_samples=4); 16 train rows / batch 4 = ~4 steps.
SKYRL_RAY_PG_TIMEOUT_IN_S=90 LOGGER=console bash examples/train/gsm8k/run_gsm8k.sh \
  trainer.epochs=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=256 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10000 \
  trainer.ckpt_interval=10000 \
  trainer.ckpt_path=/mnt/cluster_storage/ckpts/gsm8k_smoke \
  2>&1 | tee /tmp/skyrl_train.log

# ConsoleLogger emits "Step N:" per training step; confirms the GRPO loop ran.
if ! grep -Eq "Step [0-9]+:" /tmp/skyrl_train.log; then
    echo "FAILED: no training step logged"
    exit 1
fi

echo "=== SkyRL Template Test PASSED ==="
