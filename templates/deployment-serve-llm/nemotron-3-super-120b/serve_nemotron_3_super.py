# serve_nemotron_3_super.py
import os

from ray.serve.llm import LLMConfig, build_openai_app

# NVIDIA-Nemotron-3-Super-120B-A12B-FP8
# ---------------------------------------------------------------------------
# 120B-total / 12B-active LatentMoE (Mamba-2 + MoE + Attention hybrid) with a
# built-in Multi-Token-Prediction (MTP) head. Ships as an FP8 checkpoint.
#
# NVIDIA's validated H100 recipe uses tensor_parallel_size=4 + expert
# parallelism at a 256k context. The engine_kwargs below are the snake_case
# translation of the `vllm serve` flags from the model card.
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="nvidia/nemotron-3-super",
        model_source="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    ),
    accelerator_type="H100",
    deployment_config=dict(
        autoscaling_config=dict(
            # Start with 1 replica (4 H100s via tensor_parallel_size=4). Raise
            # max_replicas for more concurrency, or set min_replicas=0 for
            # production scale-to-zero (first request incurs a cold start).
            min_replicas=1,
            max_replicas=1,
        )
    ),
    ### Uncomment if you load the model from a gated/private source needing a token.
    # runtime_env=dict(env_vars={"HF_TOKEN": os.environ.get("HF_TOKEN")}),
    engine_kwargs=dict(
        # --- Parallelism (single node, NVLink) -----------------------------
        tensor_parallel_size=4,
        enable_expert_parallel=True,  # MoE expert parallelism
        # --- Context ------------------------------------------------------
        # Model supports up to 1M tokens. To go beyond this, set env var
        # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and raise max_model_len to 1048576.
        max_model_len=262144,  # 256k
        # --- Memory / precision -------------------------------------------
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",  # stability for the Mamba-2 SSM cache
        gpu_memory_utilization=0.9,
        trust_remote_code=True,  # required: custom "nemotron_h" architecture
        # --- Scheduling / throughput --------------------------------------
        enable_chunked_prefill=True,
        # NOTE: NVIDIA's model card (written for vLLM 0.18.1) also passes
        # `swap_space=0`, which is INVALID on vLLM 0.25.1 -- the V1 engine
        # removed CPU KV-swap, so Ray raises "Unknown engine argument:
        # swap_space". It is therefore removed. The card's
        # `async_scheduling=True` and `max_cudagraph_capture_size=128` ARE
        # valid in 0.25.1 and can be re-added as perf tweaks; they're left off
        # here to keep the first deploy minimal.
        # --- Reasoning ----------------------------------------------------
        # Splits <think>...</think> traces from the final answer.
        reasoning_parser="nemotron_v3",
        # --- Tool / function calling (OPTIONAL — uncomment for agents) -----
        # This model is built for agentic/tool-use workloads. Enable both
        # flags below to serve OpenAI-style tool calls.
        # enable_auto_tool_choice=True,
        # tool_call_parser="qwen3_coder",
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
