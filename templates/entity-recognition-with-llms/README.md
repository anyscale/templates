# LLM training and inference

<a href="https://console.anyscale.com/register/ha?render_flow=ray&utm_source=ray_docs&utm_medium=docs&utm_campaign=entity-recognition-with-llms&redirectTo=/v2/template-preview/entity-recognition-with-llms\">
<img src="https://raw.githubusercontent.com/ray-project/ray/c34b74c22a9390aa89baf80815ede59397786d2e/doc/source/_static/img/run-on-anyscale.svg" alt=\"Run on Anyscale\">
</a>
<br></br>
<div align="left">
<a href="https://github.com/anyscale/e2e-llm-workflows" role="button"><img src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
</div>

This end-to-end tutorial **fine-tunes** an LLM to perform **batch inference** and **online serving** at scale. While entity recognition (NER) is the main task in this tutorial, you can easily extend these end-to-end workflows to any use case.

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/e2e_llm.png" width=800>

**Note**: The intent of this tutorial is to show how you can use Ray to implement end-to-end LLM workflows that can extend to any use case, including multimodal.

This tutorial uses the [Ray library](https://github.com/ray-project/ray) to implement these workflows, namely the LLM APIs:

[`ray.data.llm`](https://docs.ray.io/en/latest/data/working-with-llms.html):
- Batch inference over distributed datasets
- Streaming and async execution for throughput
- Built-in metrics and tracing, including observability
- Zero-copy GPU data transfer
- Composable with preprocessing and postprocessing steps

[`ray.serve.llm`](https://docs.ray.io/en/latest/serve/llm/index.html):
- Automatic scaling and load balancing
- Unified multi-node multi-model deployment
- Multi-LoRA support with shared base models
- Deep integration with inference engines, vLLM to start
- Composable multi-model LLM pipelines

And all of these workloads come with all the observability views you need to debug and tune them to **maximize throughput/latency**.

## Set up

### Compute
This [Anyscale Workspace](https://docs.anyscale.com/platform/workspaces/) automatically provisions and autoscales the compute your workloads need. If you're not on Anyscale, then you need to provision the appropriate compute (L4) for this tutorial.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/compute.png" width=500>

### Dependencies
Start by downloading the dependencies required for this tutorial. Notice in your [`containerfile`](https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/containerfile) you have a base image [`anyscale/ray-llm:latest-py311-cu124`](https://hub.docker.com/layers/anyscale/ray-llm/latest-py311-cu124/images/sha256-5a1c55f7f416d2d2eb5f4cdd13afeda25d4f7383406cfee1f1f60da495d1b50f) followed by a list of pip packages. If you're not on [Anyscale](https://console.anyscale.com/), you can pull this Docker image yourself and install the dependencies.



```bash
%%bash
# Install dependencies
pip install -q \
    "pynvml==12.0.0" \
    "hf_transfer==0.1.9" \
    "tensorboard==2.19.0" \
    "llamafactory@git+https://github.com/hiyouga/LLaMA-Factory.git" \
    "transformers>=4.56.0,<5" \
    "peft>=0.18" \
    "trl>=0.18" \
    "omegaconf" \
    "torchdata"
```

## Data ingestion


```python
import json
import textwrap
from IPython.display import Code, Image, display
```

Start by downloading the data from cloud storage to local shared storage. 


```bash
%%bash
rm -rf /mnt/cluster_storage/viggo  # clean up
mkdir /mnt/cluster_storage/viggo
wget https://viggo-ds.s3.amazonaws.com/train.jsonl -O /mnt/cluster_storage/viggo/train.jsonl
wget https://viggo-ds.s3.amazonaws.com/val.jsonl -O /mnt/cluster_storage/viggo/val.jsonl
wget https://viggo-ds.s3.amazonaws.com/test.jsonl -O /mnt/cluster_storage/viggo/test.jsonl
wget https://viggo-ds.s3.amazonaws.com/dataset_info.json -O /mnt/cluster_storage/viggo/dataset_info.json
```


```bash
%%bash
head -n 1 /mnt/cluster_storage/viggo/train.jsonl | python3 -m json.tool
```


```python
with open("/mnt/cluster_storage/viggo/train.jsonl", "r") as fp:
    first_line = fp.readline()
    item = json.loads(first_line)
system_content = item["instruction"]
print(textwrap.fill(system_content, width=80))
```

You also have an info file that identifies the datasets and format (Alpaca and ShareGPT formats) to use for post training.


```python
display(Code(filename="/mnt/cluster_storage/viggo/dataset_info.json", language="json"))
```

## Fine-tuning

Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to perform fine-tuning on a GPU worker via Ray. Find the parameters for the training workload, post-training method, dataset location, train/val details, etc. in the `lora_sft_ray.yaml` config file. See the recipes for even more post-training methods, like SFT, pretraining, PPO, DPO, KTO, etc. [on GitHub](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples).

**Note**: Ray also supports using other tools like [axolotl](https://axolotl-ai-cloud.github.io/axolotl/docs/ray-integration.html) or even [Ray Train + HF Accelerate + FSDP/DeepSpeed](https://docs.ray.io/en/latest/train/huggingface-accelerate.html) directly for complete control of your post-training workloads.

### `config`


```python
import os
from pathlib import Path
import yaml
```


```python
display(Code(filename="lora_sft_ray.yaml", language="yaml"))
```


```python
model_id = "ft-model"  # call it whatever you want
model_source = yaml.safe_load(open("lora_sft_ray.yaml"))["model_name_or_path"]  # HF model ID, S3 mirror config, or GCS mirror config
print (model_source)
```

### Training

Use `run_training.py` to launch LLaMA-Factory training on a GPU worker node via Ray remote task. The training config is read from shared storage so the GPU worker can access it.

<div class="alert alert-block alert"> <b>Ray Train</b> 

Using [Ray Train](https://docs.ray.io/en/latest/train/train.html) has several advantages:
- it automatically handles **multi-node, multi-GPU** setup with no manual SSH setup or `hostfile` configs. 
- you can define **per-worker fractional resource requirements**, for example, 2 CPUs and 0.5 GPU per worker.
- you can run on **heterogeneous machines** and scale flexibly, for example, CPU for preprocessing and GPU for training.
- it has built-in **fault tolerance** through retry of failed workers, and continue from last checkpoint.
- it supports Data Parallel, Model Parallel, Parameter Server, and even custom strategies.
- [Ray Compiled graphs](https://docs.ray.io/en/latest/ray-core/compiled-graph/ray-compiled-graph.html) allow you to even define different parallelism for jointly optimizing multiple models. Megatron, DeepSpeed, and similar frameworks only allow for one global setting.

[RayTurbo Train](https://docs.anyscale.com/rayturbo/rayturbo-train) offers even more improvement to the price-performance ratio, performance monitoring, and more:
- **elastic training** to scale to a dynamic number of workers, and continue training on fewer resources, even on spot instances.
- **purpose-built dashboard** designed to streamline the debugging of Ray Train workloads:
    - Monitoring: View the status of training runs and train workers.
    - Metrics: See insights on training throughput and training system operation time.
    - Profiling: Investigate bottlenecks, hangs, or errors from individual training worker processes.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/train_dashboard.png" width=700>


```bash
%%bash
# Copy config to shared storage and run training on a GPU worker
cp lora_sft_ray.yaml /mnt/cluster_storage/viggo/lora_sft_ray.yaml
python run_training.py --config /mnt/cluster_storage/viggo/lora_sft_ray.yaml
```


```python
display(Code(filename="/mnt/cluster_storage/viggo/outputs/all_results.json", language="json"))
```

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/loss.png" width=500>

### Observability

<div class="alert alert-block alert"> <b> 🔎 Monitoring and debugging with Ray</b> 


OSS Ray offers an extensive [observability suite](https://docs.ray.io/en/latest/ray-observability/index.html) with logs and an observability dashboard that you can use to monitor and debug. The dashboard includes a lot of different components such as:

-  memory, utilization, etc., of the tasks running in the [cluster](https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-node-view)

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/cluster_util.png" width=700>

- views to see all running tasks, utilization across instance types, autoscaling, etc.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/observability_views.png" width=1000>


<div class="alert alert-block alert"> <b> 🔎➕➕ Monitoring and debugging on Anyscale</b> 

OSS Ray comes with an extensive observability suite, and Anyscale takes it many steps further to make monitoring and debugging your workloads even easier and faster with:

- [unified log viewer](https://docs.anyscale.com/monitoring/accessing-logs/) to see logs from *all* driver and worker processes
- Ray workload specific dashboard, like Data, Train, etc., that can breakdown the tasks. For example, you can observe the preceding training workload live through the Train specific Ray Workloads dashboard:

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/train_dashboard.png" width=700>




### Save to cloud storage

<div class="alert alert-block alert"> <b> 🗂️ Storage on Anyscale</b> 

You can always store to data inside [any storage buckets](https://docs.anyscale.com/configuration/storage/#private-storage-buckets) but Anyscale offers a [default storage bucket](https://docs.anyscale.com/configuration/storage/#anyscale-default-storage-bucket) to make things even easier. You also have plenty of other [storage options](https://docs.anyscale.com/configuration/storage/) as well, shared at the cluster, user, and cloud levels.


```bash
%%bash
# Anyscale default storage bucket.
echo $ANYSCALE_ARTIFACT_STORAGE
```


```bash
%%bash
# Save fine-tuning artifacts to cloud storage.
STORAGE_PATH="$ANYSCALE_ARTIFACT_STORAGE/viggo"
LOCAL_OUTPUTS_PATH="/mnt/cluster_storage/viggo/outputs"

# AWS S3 operations.
if [[ "$STORAGE_PATH" == s3://* ]]; then
    if aws s3 ls "$STORAGE_PATH" > /dev/null 2>&1; then
        aws s3 rm "$STORAGE_PATH" --recursive --quiet
    fi
    aws s3 cp "$LOCAL_OUTPUTS_PATH" "$STORAGE_PATH/outputs" --recursive --quiet

# Google Cloud Storage operations.
elif [[ "$STORAGE_PATH" == gs://* ]]; then
    if gsutil ls "$STORAGE_PATH" > /dev/null 2>&1; then
        gsutil -m -q rm -r "$STORAGE_PATH"
    fi
    gsutil -m -q cp -r "$LOCAL_OUTPUTS_PATH" "$STORAGE_PATH/outputs"

else
    echo "Unsupported storage protocol: $STORAGE_PATH"
    exit 1
fi
```


```bash
%%bash
ls /mnt/cluster_storage/viggo/outputs/
```


```python
# LoRA paths.
lora_path = "/mnt/cluster_storage/viggo/outputs"
cloud_lora_path = os.path.join(os.getenv("ANYSCALE_ARTIFACT_STORAGE"), "viggo/outputs")
dynamic_lora_path = cloud_lora_path
print(lora_path)
print(cloud_lora_path)
print(dynamic_lora_path)
```


```bash
%%bash
ls /mnt/cluster_storage/viggo/outputs/
```

## Batch inference 
[`Overview`](https://docs.ray.io/en/latest/data/working-with-llms.html) |  [`API reference`](https://docs.ray.io/en/latest/data/api/llm.html)

The `ray.data.llm` module integrates with key large language model (LLM) inference engines and deployed models to enable LLM batch inference. These LLM modules use [Ray Data](https://docs.ray.io/en/latest/data/data.html) under the hood, which makes it extremely easy to distribute workloads but also ensures that they happen:
- **efficiently**: minimizing CPU/GPU idle time with heterogeneous resource scheduling.
- **at scale**: with streaming execution to petabyte-scale datasets, especially when [working with LLMs](https://docs.ray.io/en/latest/data/working-with-llms.html).
- **reliably** by checkpointing processes, especially when running workloads on spot instances with on-demand fallback.
- **flexibly**: connecting to data from any source, applying transformations, and saving to any format and location for your next workload.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/ray_data_solution.png" width=800>

[RayTurbo Data](https://docs.anyscale.com/rayturbo/rayturbo-data) has more features on top of Ray Data:
- **accelerated metadata fetching** to improve reading first time from large datasets 
- **optimized autoscaling** where Jobs can kick off before waiting for the entire cluster to start
- **high reliability** where entire failed jobs, like head node, cluster, uncaptured exceptions, etc., can resume from checkpoints. OSS Ray can only recover from worker node failures.

Start by defining the [vLLM engine processor config](https://docs.ray.io/en/latest/data/api/doc/ray.data.llm.vLLMEngineProcessorConfig.html#ray.data.llm.vLLMEngineProcessorConfig) where you can select the model to use and the [engine behavior](https://docs.vllm.ai/en/stable/serving/engine_args.html). The model can come from [Hugging Face (HF) Hub](https://huggingface.co/models) or a local model path `/path/to/your/model`. Anyscale supports GPTQ, GGUF, or LoRA model formats.

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/data_llm.png" width=800>

### vLLM engine processor


```python
import os
import ray
from ray.data.llm import vLLMEngineProcessorConfig
```


```python
config = vLLMEngineProcessorConfig(
    model_source=model_source,
    runtime_env={
        "env_vars": {
            "VLLM_USE_V1": "0",  # v1 doesn't support lora adapters yet
            # "HF_TOKEN": os.environ.get("HF_TOKEN"),
        },
    },
    engine_kwargs={
        "enable_lora": True,
        "max_lora_rank": 8,
        "max_loras": 1,
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,
        "max_model_len": 4096,  # or increase KV cache size
        # complete list: https://docs.vllm.ai/en/stable/serving/engine_args.html
    },
    concurrency=1,
    batch_size=16,
    accelerator_type="L4",
)
```

### LLM processor

Next, pass the config to an [LLM processor](https://docs.ray.io/en/master/data/api/doc/ray.data.llm.build_processor.html#ray.data.llm.build_processor) where you can define the preprocessing and postprocessing steps around inference. With your base model defined in the processor config, you can define the LoRA adapter layers as part of the preprocessing step of the LLM processor itself.


```python
from ray.data.llm import build_processor
```


```python
processor = build_processor(
    config,
    preprocess=lambda row: dict(
        model=lora_path,  # REMOVE this line if doing inference with just the base model
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": row["input"]}
        ],
        sampling_params={
            "temperature": 0.3,
            "max_tokens": 250,
            # complete list: https://docs.vllm.ai/en/stable/api/inference_params.html
        },
    ),
    postprocess=lambda row: {
        **row,  # all contents
        "generated_output": row["generated_text"],
        # add additional outputs
    },
)
```


```python
# Evaluation on test dataset
ds = ray.data.read_json("/mnt/cluster_storage/viggo/test.jsonl")  # complete list: https://docs.ray.io/en/latest/data/api/input_output.html
ds = processor(ds)
results = ds.take_all()
results[0]
```


```python
# Exact match (strict!)
matches = 0
for item in results:
    if item["output"] == item["generated_output"]:
        matches += 1
matches / float(len(results))
```

**Note**: The objective of fine-tuning here isn't to create the most performant model but to show that you can leverage it for downstream workloads, like batch inference and online serving at scale. However, you can increase `num_train_epochs` if you want to.

Observe the individual steps in the batch inference workload through the Anyscale Ray Data dashboard:

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/data_dashboard.png" width=1000>

<div class="alert alert-info">

💡 For more advanced guides on topics like optimized model loading, multi-LoRA, OpenAI-compatible endpoints, etc., see [more examples](https://docs.ray.io/en/latest/data/working-with-llms.html) and the [API reference](https://docs.ray.io/en/latest/data/api/llm.html).

</div>

## Online serving
[`Overview`](https://docs.ray.io/en/latest/serve/llm/index.html) | [`API reference`](https://docs.ray.io/en/latest/serve/api/index.html#llm-api)

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/ray_serve.png" width=600>

`ray.serve.llm` APIs allow users to deploy multiple LLM models together with a familiar Ray Serve API, while providing compatibility with the OpenAI API.

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/serve_llm.png" width=500>

Ray Serve LLM is designed with the following features:
- Automatic scaling and load balancing
- Unified multi-node multi-model deployment
- OpenAI compatibility
- Multi-LoRA support with shared base models
- Deep integration with inference engines, vLLM to start
- Composable multi-model LLM pipelines

[RayTurbo Serve](https://docs.anyscale.com/rayturbo/rayturbo-serve) on Anyscale has more features on top of Ray Serve:
- **fast autoscaling and model loading** to get services up and running even faster: [5x improvements](https://www.anyscale.com/blog/autoscale-large-ai-models-faster) even for LLMs
- 54% **higher QPS** and up-to 3x **streaming tokens per second** for high traffic serving use-cases
- **replica compaction** into fewer nodes where possible to reduce resource fragmentation and improve hardware utilization
- **zero-downtime** [incremental rollouts](https://docs.anyscale.com/platform/services/update-a-service/#resource-constrained-updates) so your service is never interrupted
- [**different environments**](https://docs.anyscale.com/platform/services/multi-app/#multiple-applications-in-different-containers) for each service in a multi-serve application
- **multi availability-zone** aware scheduling of Ray Serve replicas to provide higher redundancy to availability zone failures


### LLM serve config


```python
import os
from openai import OpenAI  # to use openai api format
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
```

Define an [LLM config](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.llm.LLMConfig.html#ray.serve.llm.LLMConfig) where you can define where the model comes from, its [autoscaling behavior](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling), what hardware to use and [engine arguments](https://docs.vllm.ai/en/stable/serving/engine_args.html).

**Note**: If you're using AWS S3, replace `AWS_REGION` in the `runtime_env`'s `env_vars` below with the cloud storage and respective region you saved your model artifacts to. Do the same if using other cloud storage options as well.


```python
# Define config.
llm_config = LLMConfig(
    model_loading_config={
        "model_id": model_id,
        "model_source": model_source
    },
    lora_config={  # REMOVE this section if you're only using a base model.
        "dynamic_lora_loading_path": dynamic_lora_path,
        "max_num_adapters_per_replica": 16,  # You only have 1.
    },
    runtime_env={"env_vars": {"AWS_REGION": "us-west-2"}},
    # runtime_env={"env_vars": {"HF_TOKEN": os.environ.get("HF_TOKEN")}},
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 2,
            # complete list: https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling
        }
    },
    accelerator_type="L4",
    engine_kwargs={
        "max_model_len": 4096,  # Or increase KV cache size.
        "tensor_parallel_size": 1,
        "enable_lora": True,
        # complete list: https://docs.vllm.ai/en/stable/serving/engine_args.html
    },
)
```

Now deploy the LLM config as an application. And because this application is all built on top of [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), you can have advanced service logic around composing models together, deploying multiple applications, model multiplexing, observability, etc.


```python
# Deploy.
app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app)
```

### Service request


```python
# Initialize client.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="fake-key")
response = client.chat.completions.create(
    model=f"{model_id}:checkpoint-125",  # Use checkpoint subdirectory as LoRA ID
    messages=[
        {"role": "system", "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"},
        {"role": "user", "content": "Blizzard North is mostly an okay developer, but they released Diablo II for the Mac and so that pushes the game from okay to good in my view."},
    ],
    stream=True
)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

And of course, you can observe the running service, the deployments, and metrics like QPS, latency, etc., through the [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html)'s [Serve view](https://docs.ray.io/en/latest/ray-observability/getting-started.html#dash-serve-view):

<img src="https://raw.githubusercontent.com/anyscale/e2e-llm-workflows/refs/heads/main/images/serve_dashboard.png" width=1000>

<div class="alert alert-info">

💡 See [more examples](https://docs.ray.io/en/latest/serve/llm/examples.html) and the [API reference](https://docs.ray.io/en/latest/serve/api/index.html#llm-api) for advanced guides on topics like structured outputs (like JSON), vision LMs, multi-LoRA on shared base models, using other inference engines (like `sglang`), fast model loading, etc.

</div>

## Production

Seamlessly integrate with your existing CI/CD pipelines by leveraging the Anyscale [CLI](https://docs.anyscale.com/reference/quickstart-cli) or [SDK](https://docs.anyscale.com/reference/quickstart-sdk) to run [reliable batch jobs](https://docs.anyscale.com/platform/jobs) and deploy [highly available services](https://docs.anyscale.com/platform/services). Given you've been developing in an environment that's almost identical to production with a multi-node cluster, this integration should drastically speed up your dev to prod velocity.

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/cicd.png" width=600>

### Jobs

[Anyscale Jobs](https://docs.anyscale.com/platform/jobs/) ([API ref](https://docs.anyscale.com/reference/job-api/)) allows you to execute discrete workloads in production such as batch inference, embeddings generation, or model fine-tuning.
- [define and manage](https://docs.anyscale.com/platform/jobs/manage-jobs) your Jobs in many different ways, like CLI and Python SDK
- set up [queues](https://docs.anyscale.com/platform/jobs/job-queues) and [schedules](https://docs.anyscale.com/platform/jobs/schedules)
- set up all the [observability, alerting, etc.](https://docs.anyscale.com/platform/jobs/monitoring-and-debugging) around your Jobs

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/job_result.png" width=700>

### Services

[Anyscale Services](https://docs.anyscale.com/platform/services/) ([API ref](https://docs.anyscale.com/reference/service-api/)) offers an extremely fault tolerant, scalable, and optimized way to serve your Ray Serve applications:
- you can [rollout and update](https://docs.anyscale.com/platform/services/update-a-service) services with canary deployment with zero-downtime upgrades
- [monitor](https://docs.anyscale.com/platform/services/monitoring) your Services through a dedicated Service page, unified log viewer, tracing, set up alerts, etc.
- scale a service (`num_replicas=auto`) and utilize replica compaction to consolidate nodes that are fractionally utilized
- [head node fault tolerance](https://docs.anyscale.com/platform/services/production-best-practices#head-node-ft) because OSS Ray recovers from failed workers and replicas but not head node crashes
- serving [multiple applications](https://docs.anyscale.com/platform/services/multi-app) in a single Service

<img src="https://raw.githubusercontent.com/anyscale/foundational-ray-app/refs/heads/main/images/canary.png" width=700>



```bash
%%bash
# clean up
rm -rf /mnt/cluster_storage/viggo
STORAGE_PATH="$ANYSCALE_ARTIFACT_STORAGE/viggo"
if [[ "$STORAGE_PATH" == s3://* ]]; then
    aws s3 rm "$STORAGE_PATH" --recursive --quiet
elif [[ "$STORAGE_PATH" == gs://* ]]; then
    gsutil -m -q rm -r "$STORAGE_PATH"
fi
```
