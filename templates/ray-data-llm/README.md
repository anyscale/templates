# LLM offline batch inference with Ray Data LLM APIs

**⏱️ Time to complete**: 10 min


This notebook shows you how to run batch inference for LLMs using [Ray Data LLM](https://docs.ray.io/en/latest/data/api/llm.html).

**Note:** This tutorial runs within a workspace. Review the [Introduction to Workspaces](https://docs.anyscale.com/examples/intro-workspaces) template before this tutorial.


### Deciding between online vs offline inference for LLM
Use online LLM inference to get real-time responses for prompts or to interact with the LLM. Use online inference when you want to optimize latency of inference.

On the other hand, use offline LLM inference, also referred to as batch inference, when you want to get responses for a large number of prompts within some time frame, but not necessarily in real-time, for example in minutes to hours. Use offline inference when you want to:
1. Process large-scale datasets.
2. Optimize inference throughput and resource usage. For example, maximizing GPU utilization.

This tutorial focuses on the latter, using offline LLM inference for a summarization task using real-world news articles.


## Step 1: Prepare a Ray Data dataset

Ray Data LLM runs batch inference for LLMs on Ray Data datasets. In this tutorial, we will perform batch inference with an LLM to reformat dates. Our source is a 2-million-row CSV file containing sample customer data.
First, we load the data from a remote URL. Then, to ensure the workload can be distributed across multiple GPUs, we repartition the dataset. This step is crucial for achieving parallelism.


```python
import ray

# Define the path to the sample CSV file hosted on S3.
# This dataset contains 2 million rows of synthetic customer data.
path = "https://llm-guide.s3.us-west-2.amazonaws.com/data/ray-data-llm/customers-2000000.csv"

# Load the CSV file into a Ray Dataset.
print("Loading dataset from remote URL...")
ds = ray.data.read_csv(path)

# You can inspect the dataset schema and a few rows to verify it loaded correctly.
# print(ds.schema())
# ds.show(limit=2)

# For this example, we'll limit the dataset to 100,000 rows for faster processing.
print("Limiting dataset to 100,000 rows.")
ds = ds.limit(100000)

# Repartition the dataset to enable parallelism across multiple workers (e.g., GPUs).
# By default, a large remote file might be read into a single block. Repartitioning
# splits the data into a specified number of blocks, allowing Ray to process them
# in parallel. 
num_partitions = 128
print(f"Repartitioning dataset into {num_partitions} blocks for parallelism...")
ds = ds.repartition(num_blocks=num_partitions)

```

## Step 2: Define the processor config for the vLLM engine

You also need to define the model configs for the LLM engine, which configures the model and compute resources needed for inference.

This example uses the `unsloth/Llama-3.1-8B-Instruct` model.
You also need to define a configuration associated with the model you want to use to configure the compute resources, engine arguments, and other inference engine specific parameters. For more details on the configs you can pass to vLLM engine, see [vLLM doc](https://docs.vllm.ai/en/latest/serving/engine_args.html).

Note that because our input prompts and expected output token lengths are small, we have set `batch_size=256` in this case. However, depending on your workload, a large batch size can lead to increased idle GPU time when decoding long sequences. Be sure to adjust this value to find the optimal trade-off between throughput and latency.


```python
from ray.data.llm import vLLMEngineProcessorConfig


processor_config = vLLMEngineProcessorConfig(
    model_source="unsloth/Llama-3.1-8B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.85,
    ),
    batch_size=256,
    accelerator_type="L4",
    concurrency=4,
)

```

## Step 3: Define the preprocess and postprocess functions

The task is to format the `Subscription Date`as the format `MM-DD-YYYY` using LLM. 

Define the preprocess function to prepare `messages` and `sampling_params` for vLLM engine, and the postprocessor function to consume `generated_text`.


```python
from typing import Any

# Preprocess function prepares `messages` and `sampling_params` for vLLM engine, and
# all other fields will be ignored.
def preprocess(row: dict[str, Any]) -> dict[str, Any]:
    return dict(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Convert this date:\n{row['Subscription Date']}\n\n as the format:MM-DD-YYYY"
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )

# Input row of postprocess function will have `generated_text`. Also `**row` syntax
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
from pprint import pprint

processor = build_llm_processor(
    processor_config,
    preprocess=preprocess,
    postprocess=postprocess,
)

processed_ds = processor(ds.limit(10_000))
# Materialize the dataset to memory. User can also use writing APIs like
# `write_parquet`(https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_parquet.html#ray.data.Dataset.write_parquet)
# `write_csv`(https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_csv.html#ray.data.Dataset.write_csv)
# to persist the dataset.
processed_ds = processed_ds.materialize()


# Peek the first 3 entries.
sampled = processed_ds.take(3)
print("==================GENERATED OUTPUT===============")

pprint(sampled)
```

### Monitoring the execution

Use the Ray Dashboard to monitor the execution. In the **Ray Dashboard** tab, navigate to the **Job** page and open the **Ray Data Overview** section. Click the link for the running job, and open the **Ray Data Overview** section to view the details of the batch inference execution:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/batch-llm/assets/ray-data-jobs.png" width=900px />

### Handling GPU out-of-memory failures
If you run into CUDA out of memory, your batch size is likely too large. Set an explicit small batch size or use a smaller model, or a larger GPU.

## Advanced: Image query with a vision language model

Ray Data LLM also supports running batch inference with vision language models. This example shows how
to prepare a dataset with images and run batch inference with a vision language model.

We applied 2 adjustments on top of the previous example:
* set `has_image=True` in `vLLMEngineProcessorConfig`
* prepare image input inside preprocessor

### ⚠️ Before continuing, restart your Anyscale Workspace

LLM batch inference + Ray Data is not optimized for execution via Jupyter notebook. To free up GPU memory held by the previously loaded model and prevent out-of-memory (OOM) errors, please restart your workspace before running your next job.


```python
# Install datasets library.
!pip install "datasets<4"
```


```python
import ray 
import datasets
from io import BytesIO
from PIL import Image
from ray.data.llm import vLLMEngineProcessorConfig

# Load "LMMs-Eval-Lite" dataset from Hugging Face.
vision_dataset_llms_lite = datasets.load_dataset("lmms-lab/LMMs-Eval-Lite", "coco2017_cap_val")
vision_dataset = ray.data.from_huggingface(vision_dataset_llms_lite["lite"])

vision_processor_config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    engine_kwargs=dict(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_model_len=4096,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
    ),
    runtime_env=dict(
        env_vars=dict(
            VLLM_USE_V1="1",
        ),
    ),
    batch_size=16,
    accelerator_type="L4",
    concurrency=4,
    has_image=True,
)

def vision_preprocess(row: dict) -> dict:
    choice_indices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    return dict(
        messages=[
            {
                "role": "system",
                "content": """Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> is:
- The single correct letter choice A, B, C, D, E, F, etc. when options are provided. Only include the letter.
- Your direct answer if no options are given, as a single phrase or number.
- If your answer is a number, only include the number without any unit.
- If your answer is a word or phrase, do not paraphrase or reformat the text you see in the image.
- You cannot answer that the question is unanswerable. You must either pick an option or provide a direct answer.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>.""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": row["question"] + "\n\n"
                    },
                    {
                        "type": "image",
                        # Ray Data accepts PIL Image or image URL.
                        "image": Image.open(BytesIO(row["image"]["bytes"]))
                    },
                    {
                        "type": "text",
                        "text": "\n\nChoices:\n" + "\n".join([f"{choice_indices[i]}. {choice}" for i, choice in enumerate(row["answer"])])
                    }
                ]
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=150,
            detokenize=False,
        ),
    )

def vision_postprocess(row: dict) -> dict:
    return {
        "resp": row["generated_text"],
    }

```

Similar to previous example, build and run the processor.


```python
from ray.data.llm import build_llm_processor

vision_processor = build_llm_processor(
    vision_processor_config,
    preprocess=vision_preprocess,
    postprocess=vision_postprocess,
)

vision_processed_ds = vision_processor(vision_dataset).materialize()
```

Similar to previous example, peek the first 3 entries.


```python
from pprint import pprint
# Peek the first 3 entries.
vision_sampled = vision_processed_ds.take(3)
print("==================GENERATED OUTPUT===============")
pprint(vision_sampled)


```

## Summary

This notebook:
- Created a custom processor for the CNN/DailyMail summarization task.
- Defined the model configs for the Meta Llama 3.1 8B model.
- Ran the batch inference through Ray Data LLM API and monitored the execution.
- As an advanced usage, ran the batch vision query through Ray Data LLM API
  * Constructed vision understanding task with COCO dataset
  * Using Qwen2.5-VL-3B-Instruct model

