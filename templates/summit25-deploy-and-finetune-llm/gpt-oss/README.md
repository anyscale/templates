# Deploy gpt-oss

<div align="left">
<a target="_blank" href="https://console.anyscale.com/template-preview/deployment-serve-llm?file=%252Ffiles%252Fgpt-oss"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
<a href="https://github.com/ray-project/ray/tree/master/doc/source/serve/tutorials/deployment-serve-llm/gpt-oss" role="button"><img src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
</div>

*gpt-oss* is a family of open-source models designed for general-purpose language understanding and generation. The 20B parameter variant (`gpt-oss-20b`) offers strong reasoning capabilities with lower latency. This makes it well-suited for local or specialized use cases. The larger 120B parameter variant (`gpt-oss-120b`) is designed for production-scale, high-reasoning workloads.

For more information, see the [gpt-oss collection](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4).

---

## Configure Ray Serve LLM

Ray Serve LLM provides multiple [Python APIs](https://docs.ray.io/en/latest/serve/api/index.html#llm-api) for defining your application. Use [`build_openai_app`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.build_openai_app.html#ray.serve.llm.build_openai_app) to build a full application from your [`LLMConfig`](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig) object.

To deploy a small-sized model such as gpt-oss-20b, a single GPU is sufficient:


```python
# serve_gpt_oss.py
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-gpt-oss",
        model_source="s3://llm-guide/data/ray-serve-llm/hf_repo/gpt-oss-20b", # also support huggingface repo syntax like openai/gpt-oss-20b
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, # avoid cold starts by keeping at least 1 replica always on
            max_replicas=2, # limit max replicas to control cost
        )
    ),
    engine_kwargs=dict(
        max_model_len=32768
    ),
    log_engine_metrics= True,
)

app = build_openai_app({"llm_configs": [llm_config]})
```

**Note:** Before moving to a production setup, migrate to using a [Serve config file](https://docs.ray.io/en/latest/serve/production-guide/config.html) to make your deployment version-controlled, reproducible, and easier to maintain for CI/CD pipelines. For an example, see [Serving LLMs - Quickstart Examples: Production Guide](https://docs.ray.io/en/latest/serve/llm/quick-start.html#production-deployment).

---

## Deploy locally

### Prerequisites

* Access to GPU compute.

### Dependencies

gpt-oss integration is available starting from `ray[serve,llm]>=2.50.0`.

---

### Launch the service

Follow the instructions in [Configure Ray Serve LLM](#configure-ray-serve-llm), and define your app in a Python module `serve_gpt_oss.py`.

In a terminal, run:


```bash
%%bash
serve run serve_gpt_oss:app --non-blocking
```

Deployment typically takes a few minutes as Ray provisions the cluster, the vLLM server starts, and Ray Serve downloads the model.

---

### Send requests

Your endpoint is available locally at `http://localhost:8000`. You can use a placeholder authentication token for the OpenAI client, for example `"FAKE_KEY"`.

#### Example Python


```python
#client.py
from urllib.parse import urljoin
from openai import OpenAI

api_key = "FAKE_KEY"
base_url = "http://localhost:8000"

client = OpenAI(base_url=urljoin(base_url, "v1"), api_key=api_key)

# Example query
response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        {"role": "user", "content": "How many r's in strawberry"}
    ],
    stream=True
)

# Stream
for chunk in response:
    # Stream reasoning content
    if hasattr(chunk.choices[0].delta, "reasoning_content"):
        data_reasoning = chunk.choices[0].delta.reasoning_content
        if data_reasoning:
            print(data_reasoning, end="", flush=True)
    # Later, stream the final answer
    if hasattr(chunk.choices[0].delta, "content"):
        data_content = chunk.choices[0].delta.content
        if data_content:
            print(data_content, end="", flush=True)
```


---

### Shut down the service

To shutdown your LLM service: 


```bash
%%bash
serve shutdown -y
```


---

## Enable LLM monitoring

The *Serve LLM Dashboard* offers deep visibility into model performance, latency, and system behavior, including:

- Token throughput (tokens/sec).
- Latency metrics: Time To First Token (TTFT), Time Per Output Token (TPOT).
- KV cache utilization.

To enable these metrics, go to your LLM config and set `log_engine_metrics: true`:

```yaml
applications:
- ...
  args:
    llm_configs:
      - ...
        log_engine_metrics: true
```

---

## Improve concurrency

Ray Serve LLM uses [vLLM](https://docs.vllm.ai/en/stable/) as its backend engine, which logs the *maximum concurrency* it can support based on your configuration.

Example log for gpt-oss-20b with 1xL4:
```console
INFO 09-08 17:34:28 [kv_cache_utils.py:1017] Maximum concurrency for 32,768 tokens per request: 5.22x
```

To improve concurrency for gpt-oss models, see [Deploy a small-sized LLM: Improve concurrency](https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm/small-size-llm/README.html#improve-concurrency) for small-sized models such as `gpt-oss-20b`, and [Deploy a medium-sized LLM: Improve concurrency](https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm/medium-size-llm/README.html#improve-concurrency) for medium-sized models such as `gpt-oss-120b`.

**Note:** Some example guides recommend using quantization to boost concurrency. `gpt-oss` weights are already 4-bit by default, so further quantization typically isn’t applicable.  

For broader guidance, also see [Choose a GPU for LLM serving](https://docs.anyscale.com/llm/serving/gpu-guidance) and [Optimize performance for Ray Serve LLM](https://docs.anyscale.com/llm/serving/performance-optimization).

---

## Reasoning configuration (with gpt-oss)

You don’t need a custom reasoning parser when deploying `gpt-oss` with Ray Serve LLM, you can access the reasoning content in the model's response directly. You can also control the reasoning effort of the model in the request.

---

### Access reasoning output

The reasoning content is available directly in the `reasoning_content` field of the response:

```python
response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        ...
    ]
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

---

### Control reasoning effort

`gpt-oss` supports [three reasoning levels](https://huggingface.co/openai/gpt-oss-20b#reasoning-levels): **low**, **medium**, and **high**. The default level is **medium**.

You can control reasoning with the `reasoning_effort` request parameter:  
```python
response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        {"role": "user", "content": "What are the three main touristic spots to see in Paris?"}
    ],
    reasoning_effort="low" # Or "medium", "high"
)
```

You can also set a level explicitly in the system prompt:  
```python
response = client.chat.completions.create(
    model="my-gpt-oss",
    messages=[
        {"role": "system", "content": "Reasoning: low. You are an AI travel assistant."},
        {"role": "user", "content": "What are the three main touristic spots to see in Paris?"}
    ]
)
```

**Note:** There's no reliable way to completely disable reasoning.

---

## Summary

In this tutorial, you learned how to deploy `gpt-oss` models with Ray Serve LLM, from development to production. You learned how to configure Ray Serve LLM, deploy your service on a Ray cluster, send requests, and monitor your service.
