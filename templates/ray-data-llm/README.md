# LLM offline batch inference with Ray Data LLM APIs

**⏱️ Time to complete**: 10 min


This notebook shows you how to run batch inference for LLMs using [Ray Data LLM](https://docs.ray.io/en/latest/data/api/llm.html).

**Note:** This tutorial runs within a workspace. Review the [Introduction to Workspaces](https://docs.anyscale.com/examples/intro-workspaces) template before this tutorial.


### Deciding between online vs offline inference for LLM
Use online LLM inference (e.g., Anyscale Endpoints) to get real-time responses for prompts or to interact with the LLM. Use online inference when you want to optimize latency of inference.

On the other hand, use offline LLM inference, also referred to as batch inference, when you want to get responses for a large number of prompts within some time frame, but not necessarily in real-time, for example in minutes to hours. Use offline inference when you want to:
1. Process large-scale datasets.
2. Optimize inference throughput and resource usage. For example, maximizing GPU utilization.

This tutorial focuses on the latter, using offline LLM inference for a summarization task using real-world news articles.


## Step 1: Prepare a Ray Data dataset

Ray Data LLM runs batch inference for LLMs on Ray Data datasets. This tutorial runs batch inference with an LLM that summarizes news articles from [`CNNDailyMail`](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset, which is a collection of news articles. It also summarizes each article with your batch inferencing pipeline. It covers more details on how to customize the pipeline in the later sections.



```python
# Install datasets library.
!pip install datasets
```


```python
import ray 
import datasets

# Load the dataset from Hugging Face into Ray Data. Refer to Ray Data APIs
# https://docs.ray.io/en/latest/data/api/input_output.html for details.
# For example, you can use ray.data.read_json(dataset_file) to load dataset in JSONL.

df = datasets.load_dataset("cnn_dailymail", "3.0.0")
ds = ray.data.from_huggingface(df["train"])

```

## Step 2: Define the processor config for the vLLM engine

You also need to define the model configs for the LLM engine, which configures the model and compute resources needed for inference. 

Make sure to provide your [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens). Ray uses this token to authenticate and download the model and Hugging Face **requires the token for official LLaMA, Mistral, and Gemma models**.


```python
HF_TOKEN = "Insert your Hugging Face token here"
```

This example uses the `meta-llama/Meta-Llama-3.1-8B-Instruct` model.
You also need to define a configuration associated with the model you want to use to configure the compute resources, engine arguments, and other inference engine specific parameters. For more details on the configs you can pass to vLLM engine, see [vLLM doc](https://docs.vllm.ai/en/latest/serving/engine_args.html).


```python
from ray.data.llm import vLLMEngineProcessorConfig


processor_config = vLLMEngineProcessorConfig(
    model_source="unsloth/Llama-3.1-8B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=16384,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
    ),
    # Override Ray's runtime env to include the Hugging Face token. Ray Data uses Ray under the hood to orchestrate the inference pipeline.
    runtime_env=dict(
        env_vars=dict(
            HF_TOKEN=HF_TOKEN,
        ),
    ),
    batch_size=16,
    accelerator_type="L4",
    concurrency=1,
)

```

## Step 3: Define the preprocess and postprocess functions


Define the preprocess function to prepare `messages` and `sampling_params` for vLLM engine, and the postprocessor function to consume `generated_text`.


```python
from typing import Any

# Preprocess function prepares `messages` and `sampling_params` for vLLM engine.
# It ignores all other fields.
def preprocess(row: dict[str, Any]) -> dict[str, Any]:
    return dict(
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

# Input row of postprocess function will have `generated_text`. Alse `**row` syntax
# can be used to return all the original columns in the input dataset.
def postprocess(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "resp": row["generated_text"],
        **row,  # This will return all the original columns in the dataset.
    }
```

## Step 4: Build and run the processor


With the processors and configs defined, you can now build then run the processor


```python
from ray.data.llm import build_llm_processor

processor = build_llm_processor(
    processor_config,
    preprocess=preprocess,
    postprocess=postprocess,
)

processed_ds = processor(ds)
# Materialize the dataset to memory. User can also use writing APIs like
# `write_parquet`(https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_parquet.html#ray.data.Dataset.write_parquet)
# `write_csv`(https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_csv.html#ray.data.Dataset.write_csv)
# to persist the dataset.
processed_ds = processed_ds.materialize()


# Peak the first 3 entries. 
sampled = processed_ds.take(3)
print("==================GENERATED OUTPUT===============")
print('\n'.join(sampled))
```

### Monitoring the execution

Use the Ray Dashboard to monitor the execution. In the **Ray Dashboard** tab, navigate to the **Job** page and open the **Ray Data Overview** section. Click the link for the running job, and open the **Ray Data Overview** section to view the details of the batch inference execution:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/batch-llm/assets/ray-data-jobs.png" width=900px />

### Handling GPU out-of-memory failures
If you run into CUDA out of memory, your batch size is likely too large. Set an explicit small batch size or use a smaller model, or a larger GPU.

## Summary

This notebook:
- Created a custom processor for the CNN/DailyMail summarization task.
- Defined the model configs for the Meta Llama 3.1 8B model.
- Ran the batch inference through Ray Data LLM API and monitored the execution.
