# Deploy NVIDIA Nemotron-3-Super-120B-A12B-FP8

<div align="left">
<a target="_blank" href="https://console.anyscale.com/template-preview/deployment-serve-llm?file=%252Ffiles%252Fnemotron-3-super-120b"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
<a href="https://github.com/anyscale/templates/tree/main/templates/deployment-serve-llm/nemotron-3-super-120b" role="button"><img src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 15 min

This tutorial deploys [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8) with Ray Serve LLM on H100 GPUs. It's a 120&nbsp;B-total / 12&nbsp;B-active hybrid model built for agentic workflows, long-context reasoning, and high-volume workloads.

---

## Configure Ray Serve LLM

Ray Serve LLM builds an OpenAI-compatible app from an [`LLMConfig`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html) via [`build_openai_app`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.build_openai_app.html). The `engine_kwargs` below are the snake_case translation of NVIDIA's validated `vllm serve` flags from the model card.


```python
# serve_nemotron_3_super.py
import os

from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="nvidia/nemotron-3-super",
        model_source="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
    ),
    accelerator_type="H100",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=1,
        )
    ),
    # runtime_env=dict(env_vars={"HF_TOKEN": os.environ.get("HF_TOKEN")}),
    engine_kwargs=dict(
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        max_model_len=262144,  # 256k
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float32",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        # NOTE: the model card's `swap_space=0` is INVALID on vLLM 0.25.1
        # (the V1 engine removed CPU KV-swap) and is omitted. Its
        # `async_scheduling=True` / `max_cudagraph_capture_size=128` ARE valid
        # 0.25.1 perf tweaks -- uncomment to try them:
        # async_scheduling=True,
        # max_cudagraph_capture_size=128,
        reasoning_parser="nemotron_v3",
        # enable_auto_tool_choice=True,   # uncomment for agents/tool use
        # tool_call_parser="qwen3_coder", # uncomment for agents/tool use
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})
```

**Note:** Before moving to a production setup, migrate to a [Serve config file](https://docs.ray.io/en/latest/serve/production-guide/config.html) to make your deployment version-controlled, reproducible, and easier to maintain for CI/CD pipelines. For an example, see [Serving LLMs - Quickstart Examples: Production Guide](https://docs.ray.io/en/latest/serve/llm/quick-start.html#production-deployment).

**Key settings explained:**

- `tensor_parallel_size=4` + `enable_expert_parallel=True` — NVIDIA's validated H100 layout; the 4 GPUs must be on the same node (NVLink).
- `kv_cache_dtype="fp8"` and `mamba_ssm_cache_dtype="float32"` — FP8 KV cache to save memory; float32 for the Mamba-2 SSM cache for numerical stability.
- `trust_remote_code=True` — required for the custom `nemotron_h` architecture.
- `reasoning_parser="nemotron_v3"` — separates `<think>` reasoning traces from the final answer.
- `max_model_len=262144` — 256k context. For up to 1M, set env var `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and raise to `1048576` (needs more GPU memory).

### Optional features

- **Tool / function calling** (agentic use): uncomment `enable_auto_tool_choice=True` and `tool_call_parser="qwen3_coder"`. This model was designed for tool use — enable it if you're building agents.
- **Speculative decoding** via the built-in MTP head — faster decode for low-entropy outputs (code, summarization). See the [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode.html); NVIDIA also ships an updated [MTPv2 checkpoint](https://huggingface.co/nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2).
- **Structured output**: works at request time via `response_format` — no engine config needed. Always bound schema fields and pass `max_tokens`.

---

## Deploy locally

**Prerequisites**

- Access to 4× H100-80&nbsp;GB GPUs on one node.

**Dependencies:** this template targets **Ray 2.57.0** and **vLLM 0.25.1** (matching the `anyscale/ray-llm:2.57.0` image; the model requires vLLM ≥ 0.18.1).


```python
!pip install "ray[serve,llm]==2.57.0"
!pip install "vllm==0.25.1"
```

**Beware:** this is an expensive deployment.

### Launch

Define your app in `serve_nemotron_3_super.py`, then run:


```python
!serve run serve_nemotron_3_super:app --non-blocking
```

Startup takes several minutes: the cluster is provisioned, the vLLM engine starts, and the ~120&nbsp;GB FP8 checkpoint is downloaded and loaded.

### Send requests

The endpoint is available locally at `http://localhost:8000` (use a placeholder token like `"FAKE_KEY"`).


```python
# client.py
from urllib.parse import urljoin
from openai import OpenAI

client = OpenAI(base_url=urljoin("http://localhost:8000", "v1"), api_key="FAKE_KEY")

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super",
    messages=[{"role": "user", "content": "What is the sum of all even numbers between 1 and 100?"}],
    temperature=1.0,
    top_p=0.95,
)
# This model returns the reasoning trace in the `reasoning` field (older vLLM
# builds use `reasoning_content`); read via model_extra since the typed OpenAI
# SDK doesn't surface either field directly.
message = response.choices[0].message
extra = getattr(message, "model_extra", None) or {}
reasoning = extra.get("reasoning") or extra.get("reasoning_content")
print(f"Reasoning:\n{reasoning}\n")
print(f"Answer:\n{message.content}")
```

The repo also includes `client_streaming.py`, which streams the reasoning trace, then the final answer. Run it with:


```python
!python client_streaming.py
```

### Shutdown


```python
!serve shutdown -y
```

---

## Deploy to production with Anyscale Services

For production deployment, use Anyscale services to deploy the Ray Serve app to a dedicated cluster without modifying the code. Anyscale ensures scalability, fault tolerance, and load balancing, keeping the service resilient against node failures, high traffic, and rolling updates. For more details, see [Serve LLMs with Anyscale](https://docs.anyscale.com/llm/serving).

This template runs on **H100 GPUs**. Because `tensor_parallel_size=4`, all 4 GPUs of a replica must stay on one node (over NVLink); `auto_select_worker_config: true` lets Anyscale pick a suitable multi-GPU node (for example an 8× H100 `p5.48xlarge`) for you.

### Launch the service

Anyscale provides out-of-the-box images (`anyscale/ray-llm`) pre-loaded with Ray Serve LLM, vLLM, and all required GPU/runtime dependencies. Create your Anyscale Service configuration in a `service.yaml` file:

```yaml
# service.yaml
name: deploy-nemotron-3-super
image_uri: anyscale/ray-llm:2.57.0-py312-cu130 # Anyscale Ray Serve LLM image. To build an image from a custom Dockerfile, set `containerfile: ./Dockerfile`
compute_config:
  auto_select_worker_config: true
working_dir: .
cloud:
applications:
  # Point to your app in your Python module
  - import_path: serve_nemotron_3_super:app
```

Deploy your service:


```python
!anyscale service deploy -f service.yaml
```

If your model source is gated/private, pass a token: `--env HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>`.

**Custom Dockerfile:** to customize the image, reference `containerfile: ./Dockerfile` in `service.yaml` instead of `image_uri`. See `Dockerfile` in this folder and the [Anyscale base images](https://docs.anyscale.com/reference/base-images).

### Send requests

The `anyscale service deploy` output prints the endpoint and auth token:

```console
(anyscale +3.9s) curl -H "Authorization: Bearer <YOUR-TOKEN>" <YOUR-ENDPOINT>
```

You can also retrieve both from the service page in the Anyscale Console (**Query** button). Point `client.py` at your production endpoint by setting the base URL to `<YOUR-ENDPOINT>` and the API key to `<YOUR-TOKEN>`.

### Access the Serve LLM Dashboard

See [Enable LLM monitoring](#enable-llm-monitoring) below to turn on LLM-specific logging. To open the dashboard from an Anyscale Service:

1. In the Anyscale Console, go to your **Service**.
2. Navigate to the **Metrics** tab.
3. Click **View in Grafana** and click **Serve LLM Dashboard**.

### Shutdown


```python
!anyscale service terminate -n deploy-nemotron-3-super
```

---

## Enable LLM monitoring

Set `log_engine_metrics: true` in your LLM config to enable the Serve LLM Dashboard (token throughput, TTFT/TPOT, KV cache utilization). Open it via **Metrics → View in Grafana → Serve LLM Dashboard** in the Anyscale Console.

---

## Improve concurrency

vLLM logs the maximum concurrency it can support for your config. To increase it:

- **Reduce** `max_model_len` — less KV-cache memory per request (avoid for agent workloads that need long context).
- **Increase** `tensor_parallel_size` — e.g. TP=8 on a full 8× H100 node aggregates more memory/bandwidth for longer context.
- **Scale replicas** — raise `max_replicas` for more concurrent capacity under bursty traffic.

See [Choose a GPU for LLM serving](https://docs.anyscale.com/llm/serving/gpu-guidance), [Performance optimization](https://docs.anyscale.com/llm/serving/performance-optimization), and [Parameter tuning](https://docs.anyscale.com/llm/serving/parameter-tuning).

---

## Troubleshooting

**Out-of-memory (OOM) errors** — the most common failure with large/long-context models. Lower `max_model_len`, raise `tensor_parallel_size`, or use a larger node. Note: CUDA-graph-capture OOM (a ~1&nbsp;GiB alloc during warmup) is independent of `max_model_len` — it needs more VRAM. See the [Troubleshooting guide](https://docs.anyscale.com/llm/serving/troubleshooting).

**`trust_remote_code` / unknown architecture** — ensure `trust_remote_code=True` is set and you're on the `anyscale/ray-llm:2.57.0` image (vLLM 0.25.1) or newer.

**`ValueError: Unknown engine argument: <name>`** — Ray Serve LLM validates every `engine_kwarg` against the installed vLLM's engine args. The NVIDIA model card was written for vLLM 0.18.1; on 0.25.1 (the V1 engine) `swap_space` was removed and raises this error, so it's omitted here. If you hit this for another key, remove or rename that flag for your vLLM version (the card's `async_scheduling` and `max_cudagraph_capture_size` are still valid in 0.25.1).

**Service stuck starting / no GPUs available** — if the cluster can't acquire a 4× H100 node (capacity exhausted or quota limits in your cloud), the service stays in a starting state. Confirm your cloud has H100 quota/availability, or set `accelerator_type` to another GPU your cloud offers.

**Reasoning content appears empty when streaming** — the OpenAI SDK doesn't surface `reasoning`/`reasoning_content` as typed fields; read them off `delta.model_extra` as shown in `client_streaming.py`.

---

## Summary

You deployed `NVIDIA-Nemotron-3-Super-120B-A12B-FP8` with Ray Serve LLM on 4× H100 GPUs using tensor + expert parallelism, FP8 weights and KV cache, a 256k context, and reasoning-trace parsing — from local development to a production Anyscale Service.
