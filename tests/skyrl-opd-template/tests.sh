#!/usr/bin/env bash
set -euo pipefail

echo "=== OPD (On-Policy Distillation) Template Test — ~20 min smoke ==="

# rayapp flattens the template + test files side by side at the workspace root,
# so the companion files (run_opd_gsm8k.sh, opd_trainer.py) sit next to this script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/
git checkout 94d53895

cp "$SCRIPT_DIR/run_opd_gsm8k.sh" "$SCRIPT_DIR/opd_trainer.py" .

# CI-only patch: --isolated resolves from scratch against Pythons the GPU stack ships
# no wheels for; --frozen uses the shipped uv.lock (Ray pinned to the BYOD image).
sed -i 's/uv run --isolated/uv run --frozen/g' run_opd_gsm8k.sh

export DATA_DIR=/mnt/cluster_storage/data/gsm8k

# Small slice: 48 train prompts => ~6 steps at batch 8; 50 val for a quick baseline eval.
uv run --frozen examples/train/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR" --max_train_dataset_length 48
uv run --frozen python -c "
import pyarrow.parquet as pq
p = '$DATA_DIR/validation.parquet'
pq.write_table(pq.read_table(p).slice(0, 50), p)
"

if [[ ! -f "$DATA_DIR/train.parquet" ]] || [[ ! -f "$DATA_DIR/validation.parquet" ]]; then
    echo "FAILED: Dataset files not found"
    exit 1
fi
echo "SUCCESS: Dataset preparation completed"

# ~6 real OPD steps with a small teacher (Qwen3-1.7B) to keep the run ~20 min while
# exercising the real path: teacher in the ref slot + per-token reverse-KL reward.
# eval_before_train logs a baseline; we don't gate on accuracy (too noisy this short).
SKYRL_RAY_PG_TIMEOUT_IN_S=90 \
LOGGER=console \
STUDENT_MODEL=Qwen/Qwen3-0.6B-Base \
TEACHER_MODEL=Qwen/Qwen3-1.7B \
bash run_opd_gsm8k.sh \
  trainer.epochs=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=384 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.eval_before_train=true \
  trainer.eval_interval=100000 \
  trainer.eval_batch_size=50 \
  trainer.ckpt_interval=100000 \
  trainer.ckpt_path=/mnt/cluster_storage/ckpts/opd_smoke \
  2>&1 | tee /tmp/opd_train.log

# ConsoleLogger emits "Step N:" per training step; confirms the OPD loop ran.
if ! grep -Eq "Step [0-9]+:" /tmp/opd_train.log; then
    echo "FAILED: no training step logged"
    exit 1
fi

echo "=== OPD Template Test PASSED ==="
