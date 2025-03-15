# LLM offline batch inference with Ray Data LLM APIs

**⏱️ Time to complete**: 10 min


<!-- TODO: add a link for the API reference -->
This template shows you how to run batch inference for LLMs using Ray Data LLM.

**Note:** This tutorial runs within a workspace. Review the `Introduction to Workspaces` template before this tutorial.


### How to decide between online vs offline inference for LLM
Online LLM inference (e.g. Anyscale Endpoint) should be used when you want to get real-time response for prompt or to interact with the LLM. Use online inference when you want to optimize latency of inference to be as quick as possible.

On the other hand, offline LLM inference (also referred to as batch inference) should be used when you want to get reponses for a large number of prompts within some time frame, but not required to be real-time (minutes to hours granularity). Use offline inference when you want to:
1. Scale your workload to large-scale datasets
2. Optimize inference throughput and resource usage (for example, maximizing GPU utilization).

In this tutorial, we will focus on the latter, using offline LLM inference for a summarization task using real-world news articles.


## Step 1: Set up the workload

Ray Data LLM is a library for running batch inference for LLMs. It uses Ray Data for data processing and provides an easy and flexible interface for the user to define their own workload. In this tutorial, we will implement a workload based on the [`CNNDailyMail`](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset, which is a collection of news articles. And we will summarize each article with our batch inferencing pipeline. We will cover more details on how to customize the workload in the later sections.



```python
import ray 
import datasets

# Load the dataset from Hugging Face into Ray Data. If you're using your own dataset,
# refer to Ray Data APIs https://docs.ray.io/en/latest/data/api/input_output.html to load it.
# For example, you can use ray.data.read_json(dataset_file) to load dataset in JSONL.

df = datasets.load_dataset("cnn_dailymail", "3.0.0")
ds = ray.data.from_huggingface(df["train"])

```

## Step 2: Define the processor config for vLLM engine

We will also need to define the model configs for the LLM engine, which configures the model and compute resources needed for inference. 

Some models will require you to input your [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens). This will be used to authenticate/download the model and **is required for official LLaMA, Mistral, and Gemma models**. You can use one of the other models which don't require a token if you don't have access to this model (for example, `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`).

Run the following cell to start the authentication flow. A VS Code overlay will appear and prompt you to enter your Hugging Face token if your selected model requires authentication. If you are using a model that does not require a token, you can skip this step. For this example, we will be using the `meta-llama/Meta-Llama-3.1-8B-Instruct` model, which requires a token.



```python
# Prompts the user for Hugging Face token if required by the model.
from util.utils import prompt_for_hugging_face_token
HF_TOKEN = prompt_for_hugging_face_token("meta-llama/Meta-Llama-3.1-8B-Instruct")
```

In this example, we will be using the `meta-llama/Meta-Llama-3.1-8B-Instruct` model.
We will also need to define a configuration associated with the model we want to use to configure the compute resources, engine arguments and other inference engine specific parameters. For more details on the the model configs, see the [API doc](https://docs.anyscale.com/llms/serving/guides/bring_any_model/) on bringing your own models.


```python
from ray.data.llm import vLLMEngineProcessorConfig
from util.utils import is_on_gcp_cloud

# There's no a10g on GCP.
accelerator_type = "L4" if is_on_gcp_cloud() else "A10G"

processor_config = vLLMEngineProcessorConfig(
    model_source="meta-llama/Meta-Llama-3.1-8B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
    ),
    # Override Ray's runtime env to include the Hugging Face token. Ray is being used under the hood to orchestrate the inference pipeline.
    runtime_env=dict(
        env_vars=dict(
            HF_TOKEN=HF_TOKEN,
        ),
    ),
    batch_size=16,
    accelerator_type=accelerator_type,
)

```

## Step 3: Define the preprocess and postprocess lambda


We will need to define the preprocess lambda to convert input dataset to format that `vLLMEngineProcessor` can consume, and also postprocessor
lambda that filter out the uninterested fields from vLLM engine.


```python
preprocess = lambda row: dict(
    messages=[
        {
            "role": "system",
            "content": "You are a commentator. Your task is to "
            "summarize highlights from article.",
        },
        {
            "role": "user",
            "content": f"# Article:\n{row['article']}\n\n"
            "#Instructions:\nIn clear and concise language, "
            "summarize the highlights presented in the article.",
        },
    ],
    sampling_params=dict(
        temperature=0.3,
        max_tokens=150,
        detokenize=False,
    ),
)
postprocess = lambda row: {
    "resp": row["generated_text"],
}
```

## Step 4: Build and run the processor


With the workload and configs defined, we can now build then run the processor


```python
from ray.data.llm import build_llm_processor

processor = build_llm_processor(
    processor_config,
    preprocess=preprocess,
    postprocess=postprocess,
)

ds = processor(ds)
ds = ds.materialize()


# Peak the first 3 entries. 
sampled = ds.take(3)
print("==================GENERATED OUTPUT===============")
print('\n'.join(sampled))
```


### Monitoring the execution

RayLLM-Batch uses Ray Data to implement the execution of the batch inference pipeline, and one can use the Ray Dashboard to monitor the execution. In the Ray Dashboard tab, navigate to the Job page and open the "Ray Data Overview" section. Click on the link for the running job, and open the "Ray Data Overview" section to view the details of the batch inference execution:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/batch-llm/assets/ray-data-jobs.png" width=900px />

### Handling GPU out-of-memory failures
If you run into CUDA out of memory, your batch size is likely too large. Set an explicit small batch size or use a smaller model (or a larger GPU).

## Summary

This notebook:
- Created a custom workload for the CNN/DailyMail summarization task.
- Defined the model configs for the Meta Llama 3.1 8B model.
- Ran the batch inference through RayLLM-Batch and monitored the execution.
