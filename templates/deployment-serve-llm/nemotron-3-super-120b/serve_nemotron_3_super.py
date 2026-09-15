# serve_nemotron_3_super.py
import os

from ray.serve.llm import LLMConfig, build_openai_app

# NVIDIA-Nemotron-3-Super-120B-A12B-FP8
# ---------------------------------------------------------------------------
# 120B-total / 12B-active LatentMoE (Mamba-2 + MoE + Attention hybrid) with a
# built-in Multi-Token-Prediction (MTP) head. Ships as an FP8 checkpoint.
#
# vLLM's recipes serve this model at tensor_parallel_size=8 on a single H100
# node, with MTP speculative decoding and no expert parallelism. The
# engine_kwargs below follow that layout, validated for the Ray 2.58.0 /
# vLLM 0.26.0 image this template ships on and cross-checked against NVIDIA's
# model card and vLLM's recipes (recipes.vllm.ai).
llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="nvidia/nemotron-3-super",
        model_source="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    ),
    accelerator_type="H100",
    deployment_config=dict(
        autoscaling_config=dict(
            # Start with 1 replica (8 H100s via tensor_parallel_size=8). Raise
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
        tensor_parallel_size=8,
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
        # Caps the CUDA-graph capture size (default: min(max_num_seqs*2, 512))
        # to cut warmup memory; part of NVIDIA's validated command.
        max_cudagraph_capture_size=128,
        # --- Reasoning ----------------------------------------------------
        # Splits <think>...</think> traces from the final answer.
        reasoning_parser="nemotron_v3",
        # --- Tool / function calling --------------------------------------
        # This model is built for agentic/tool-use workloads, so tool calling
        # is on by default; both flags are inert for requests that pass no
        # `tools`. `qwen3_xml` is the streaming-capable parser for this model
        # family.
        enable_auto_tool_choice=True,
        tool_call_parser="qwen3_xml",
        # --- Speculative decoding -----------------------------------------
        # Uses the checkpoint's built-in MTP head for faster decode on
        # low-entropy output (code, summarization). Three caveats:
        #  1. Structured output (`response_format`) returns INVALID JSON while
        #     this is on -- the grammar FSM mis-advances in the spec-decode
        #     path (vllm#34650), reproduced on vLLM 0.25.1. Use tool calling
        #     instead, or drop this kwarg. See README Troubleshooting.
        #  2. `min_p` and `logit_bias` are silently ignored.
        #  3. Draft state costs KV cache: ~19.1M vs ~25.7M tokens at TP=8.
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
