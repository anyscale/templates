#!/usr/bin/env bash
# On-Policy Distillation (OPD) for GSM8K on a single 4xL4 node.
#
# A small STUDENT (Qwen3-0.6B-Base) learns to solve GSM8K word problems by
# distilling from a larger TEACHER (Qwen3-4B) *on the student's own rollouts*:
# the teacher scores every token the student generates, and the student is
# pulled toward the teacher token-by-token. The OPD logic lives in opd_trainer.py
# (copied in next to this script); this file only supplies the GSM8K + 4xL4
# config and launches it.
#
# Run from the SkyRL repo root, after preparing data + setting a teacher:
#   uv run --isolated examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   export WANDB_API_KEY=<key>          # or set LOGGER=console below
#   bash run_opd_gsm8k.sh
set -x

: "${DATA_DIR:=$HOME/data/gsm8k}"
: "${NUM_GPUS:=4}"                              # one 4xL4 node
: "${LOGGER:=wandb}"                            # "console" to print to stdout
: "${STUDENT_MODEL:=Qwen/Qwen3-0.6B-Base}"     # trained
: "${TEACHER_MODEL:=Qwen/Qwen3-4B}"            # frozen; MUST share the student's tokenizer (Qwen3 family)

ARGS=(
  data.train_data="['$DATA_DIR/train.parquet']"
  data.val_data="['$DATA_DIR/validation.parquet']"

  # ---- OPD wiring: the ONLY real differences from the GRPO gsm8k recipe ----
  trainer.policy.model.path="$STUDENT_MODEL"
  trainer.ref.model.path="$TEACHER_MODEL"          # teacher goes in the reference-model slot
  trainer.algorithm.advantage_estimator=no_op      # advantage = the per-token reward, passed through (no GRPO group baseline)
  trainer.algorithm.policy_loss_type=importance_sampling
  trainer.algorithm.use_kl_in_reward=true          # reward_token = teacher_logprob - student_logprob  (reverse-KL, k1)
  trainer.algorithm.use_kl_loss=false              # don't ALSO add KL as a separate loss term (it's already the reward)

  # ---- placement: one colocated 4xL4 node (policy + teacher + vLLM time-slice the 4 GPUs) ----
  trainer.placement.colocate_all=true
  trainer.strategy=fsdp
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS
  generator.inference_engine.num_engines=$NUM_GPUS
  generator.inference_engine.tensor_parallel_size=1
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.gpu_memory_utilization=0.45   # TUNABLE: leave headroom for the 4B teacher + student weights

  # ---- L4-sized batches / lengths (GSM8K answers are short) — all TUNABLE on the real node ----
  trainer.epochs=1
  trainer.train_batch_size=64
  trainer.policy_mini_batch_size=64
  trainer.micro_forward_batch_size_per_gpu=4
  trainer.micro_train_batch_size_per_gpu=4
  trainer.max_prompt_length=512
  generator.sampling_params.max_generate_length=1024
  generator.n_samples_per_prompt=8

  # ---- eval: measure GSM8K accuracy BEFORE training (baseline) and every few steps ----
  trainer.eval_before_train=true
  trainer.eval_interval=4
  trainer.eval_batch_size=1024

  trainer.policy.optimizer_config.lr=1e-5
  environment.env_class=gsm8k
  trainer.logger="$LOGGER"
  trainer.project_name=gsm8k_opd
  trainer.run_name=opd_gsm8k_0.6b_from_4b
  trainer.ckpt_path="$HOME/ckpts/opd_gsm8k"
  trainer.resume_mode=null
  trainer.log_path=/tmp/skyrl-logs
)

uv run --isolated --extra fsdp opd_trainer.py "${ARGS[@]}" "$@"
