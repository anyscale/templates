set -x

# Single-turn VLM GRPO on NCT-CRC-HE colorectal pathology patches.
# Modeled on examples/train/geometry3k/run_geometry3k.sh (not the LoRA variant).
#
#   bash run_pathvlm.sh
#
# max_model_len MUST be capped. Qwen3-VL-2B-Instruct advertises a 262,144-token
# context; vLLM sizes its KV cache from that and refuses to start on a 24GB
# A10G ("28.0 GiB KV cache is needed ... available 11.66 GiB"). SkyRL has no
# first-class max_model_len field -- engine_init_kwargs is the pass-through, and
# it is applied last onto the vLLM arg namespace. 4096 comfortably covers
# max_prompt_length(1024) + max_generate_length(1024).
#
# Nothing is copied into the SkyRL checkout -- this directory is the only home
# for the spike code. Two constraints shape how that is achieved:
#
#  1. Ray's uv integration sets runtime_env["working_dir"] = os.getcwd() and then
#     validates that pyproject.toml lives inside it, so the launch directory must
#     be the SkyRL repo root. Scripts are therefore invoked by absolute path from
#     there rather than by copying them in.
#  2. SkyRL's entrypoint is a Ray task and the head node has CPU=0 schedulable,
#     so it always runs on a worker -- where this directory is not on sys.path.
#     pathvlm_entrypoint.py handles that by registering the env class by value
#     (cloudpickle) instead of by dotted import path. See its docstring.

set -e

: "${SKYRL_HOME:="$HOME/default/SkyRL"}"
: "${DATA_DIR:="/mnt/cluster_storage/data/nct_crc"}"
: "${CKPT_DIR:="/mnt/cluster_storage/ckpts/pathvlm_2b"}"
: "${EXPORT_PATH:="/mnt/cluster_storage/exports/pathvlm_2b"}"
: "${NUM_GPUS:=4}"
: "${MODEL:="Qwen/Qwen3-VL-2B-Instruct"}"
: "${LOGGER:=tensorboard}"
: "${EPOCHS:=1}"
: "${EVAL_BEFORE_TRAIN:=true}"
: "${TENSORBOARD_DIR:="/mnt/cluster_storage/tensorboard/pathvlm_2b"}"
export TENSORBOARD_DIR

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# HF_TOKEN lives in ~/.workspacerc on this workspace (workspace_v2 --env would
# require terminating the workspace to set it properly).
if [ -f "$HOME/.workspacerc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.workspacerc"
fi

# Required for Ray to replicate the uv environment onto the GPU worker nodes.
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# The g5.12xlarge worker autoscales from cold; the 180s default is not enough.
: "${SKYRL_RAY_PG_TIMEOUT_IN_S:=1800}"
export SKYRL_RAY_PG_TIMEOUT_IN_S

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating NCT-CRC-HE dataset ==="
  (cd "$SKYRL_HOME" && uv run --isolated --extra fsdp \
    python "$TEMPLATE_DIR/nct_crc_dataset.py" --output_dir "$DATA_DIR")
fi

cd "$SKYRL_HOME"

uv run --isolated --extra fsdp \
  python "$TEMPLATE_DIR/pathvlm_entrypoint.py" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=200 \
  trainer.eval_before_train=$EVAL_BEFORE_TRAIN \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=25 \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=0.8 \
  generator.max_turns=1 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=nct_crc \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.inference_engine.engine_init_kwargs.max_model_len=4096 \
  trainer.logger="$LOGGER" \
  trainer.project_name="biotech_vlm_grpo" \
  trainer.run_name="pathvlm_2b" \
  trainer.resume_mode=null \
  trainer.log_path="/mnt/cluster_storage/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  "$@"
