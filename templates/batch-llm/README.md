# LLM offline batch inference with Ray Data and vLLM

**⏱️ Time to complete**: 10 min

This template shows you how to:
1. Read in data from in-memory samples or files on cloud storage. 
2. Use Ray Data and vLLM to run batch inference of a LLM.
3. Write the inference outputs to cloud storage.

For a Python script version of the code in this workspace template, refer to `main.py`.

**Note:** This tutorial is run within a workspace. Review the `Introduction to Workspaces` template before this tutorial.

### How to decide between online vs offline inference for LLM
Online LLM inference (e.g. Anyscale Endpoint) should be used when you want to get real-time response for prompt or to interact with the LLM. Use online inference when you want to optimize latency of inference to be as quick as possible.

On the other hand, offline LLM inference (also referred to as batch inference) should be used when you want to get reponses for a large number of prompts within some time frame, but not required to be real-time (minutes to hours granularity). Use offline inference when you want to:
1. Scale your workload to large-scale datasets
2. Optimize inference throughput and resource usage (for example, maximizing GPU utilization).

In this tutorial, we will focus on the latter, using offline LLM inference for a sentence completion task.

## Step 1: Set up model configs

First, import the dependencies used in this template.


```python
import os
from typing import Dict

import numpy as np
import ray
from vllm import LLM, SamplingParams

from util.utils import generate_output_path, get_a10g_or_equivalent_accelerator_type
```

Set up values that will be used in the batch inference workflow:
* The model to use for inference ([see the list of vLLM models](https://docs.vllm.ai/en/latest/models/supported_models.html)).
    * This workspace template has been tested and verified with the following models:
        * [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
        * [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
        * [`mistralai/Mistral-7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
        * [`google/gemma-7b-it`](https://huggingface.co/google/gemma-7b-it)
        * [`mlabonne/NeuralHermes-2.5-Mistral-7B`](https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B)
    * Support for the following larger models are actively a work-in-progress, and will be supported very soon:
        * [`meta-llama/Meta-Llama-3-70B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
        * [`meta-llama/Llama-2-13b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
        * [`mistralai/Mixtral-8x7B-Instruct-v0.1`](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
        * [`meta-llama/Llama-2-70b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
        * [`codellama/CodeLlama-70b-Instruct-hf`](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf)
* The [sampling parameters object](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) used by vLLM.
* The output path where results will be written as parquet files.

*Note*: Some models will require you to input your [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens). This will be used to authenticate/download the model and **is required for official LLaMA, Mistral, and Gemma models**. You can use one of the other models which don't require a token if you don't have access to this model (for example, `mlabonne/NeuralHermes-2.5-Mistral-7B`).


```python
# Set to the name of the Hugging Face model that you wish to use from the preceding list.
# Note that using the Llama models will prompt you to set your Hugging Face user token.
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Create a sampling params object.
sampling_params = SamplingParams(n=1, temperature=0, max_tokens=2048, stop=["<|eot_id|>", "<|end_of_text|>"])

# Output path to write output result. You can also change this to any cloud storage path,
# e.g. a specific S3 bucket.
output_path = generate_output_path(
    # `ANYSCALE_ARTIFACT_STORAGE` is the URI to the pre-generated folder for storing
    # your artifacts while keeping them separate them from Anyscale-generated ones.
    # See: https://docs.anyscale.com/workspaces/storage#object-storage-s3-or-gcs-buckets
    os.environ.get("ANYSCALE_ARTIFACT_STORAGE"),
    HF_MODEL,
)

# If your chosen model requires a user token for authentication, set the following
# variable to `True` to trigger authentication.
REQUIRE_HF_TOKEN = False

hf_token_cache_path = "/home/ray/.cache/huggingface/token"
try:
    LLM(model=HF_MODEL)
except OSError:
    # Model requires HF token to access. Get the token, either from
    # cached token or from user input.
    if not os.path.isfile(hf_token_cache_path):
        import huggingface_hub
        # Starts authentication through VSCode overlay. 
        # Token saved to `hf_token_cache_path`
        huggingface_hub.interpreter_login()
    
    with open(hf_token_cache_path, "r") as file:
        os.environ["HF_TOKEN"] = file.read()
```

Start up Ray, using the Hugging Face token as an environment variable so that it's made available to all nodes in the cluster.


```python
if ray.is_initialized():
    ray.shutdown()
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
    }
)
```

## Step 2: Read input data with Ray Data
Use Ray Data to read in your input data from some sample prompts.


```python
# Create some sample sentences, and use Ray Data to create a dataset for it.
prompts = [
    "I always wanted to be a ...",
    "The best way to learn a new language is ...",
    "The biggest challenge facing our society today is ...",
    "One thing I would change about my past is ...",
    "The key to a happy life is ...",
]
ds = ray.data.from_items(prompts)

# View one row of the Dataset.
ds.take(1)
```

Construct the input prompts for your model using the format required by the specific model. Run the cell below to apply this prompt construction to each row in the Dataset with Ray Data's [`map`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html) method.


```python
model_name_to_input_prompt_format = {
    "meta-llama/Llama-2-7b-chat-hf": "[INST] {} [/INST]",
    "mistralai/Mistral-7B-Instruct-v0.1": "[INST] {} [/INST]",
    "google/gemma-7b-it": "<start_of_turn>model\n{}<end_of_turn>\n",
    "mlabonne/NeuralHermes-2.5-Mistral-7B": "<|im_start|>system\nYou are a helpful assistant that will complete the sentence in the given input prompt.<|im_end|>\n<|im_start|>user{}<|im_end|>\n<|im_start|>assistant",
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Complete the given prompt in several concise sentences.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}

def construct_input_prompt(row, text_column):
    """Given the input row with raw text in `text_column` column,
    construct the input prompt for the model."""
    prompt_format = model_name_to_input_prompt_format.get(HF_MODEL)
    if prompt_format:
        row[text_column] = prompt_format.format(row[text_column])
    return row

ds = ds.map(construct_input_prompt, fn_kwargs={"text_column": "item"})
```

So far, we have defined two operations of the Dataset (`from_items()`, `map()`), but have not executed the Dataset yet and don't see any results. Why is that?

Ray Data uses [lazy, streaming execution](https://docs.ray.io/en/latest/data/data-internals.html#execution) by default, which means that:
- Datasets and any associated transformations are not executed until you call a consuming operation such as [`ds.take()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.take.html), [`ds.take_all()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.take_all.html), [`ds.iter_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.DataIterator.iter_batches.html), or [`Dataset.write_parquet()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_parquet.html).
- The entire Dataset is not stored in memory, but rather, the Dataset is executed incrementally on parts of data while overlapping execution of various operations in the Dataset. This allows Ray Data to execute batch transformations without needing to load the entire dataset into memory and overlap data preprocessing and model training steps during ML training.

We will trigger Dataset execution after the next step, which is applying the vLLM model to the formatted input prompts.

## Step 3: Run Batch Inference with vLLM

Create a class to define batch inference logic.


```python
# Mapping of model name to max_model_len supported by model.
model_name_to_args = {
    "mistralai/Mistral-7B-Instruct-v0.1": {"max_model_len": 16832},
    "google/gemma-7b-it": {"max_model_len": 2432},
    "mlabonne/NeuralHermes-2.5-Mistral-7B": {"max_model_len": 16800},
}

class LLMPredictor:
    def __init__(self, text_column):
        # Name of column containing the input text.
        self.text_column = text_column

        # Create an LLM.
        self.llm = LLM(
            model=HF_MODEL,
            **model_name_to_args.get(HF_MODEL, {}),
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.text_column], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }
```

### Scaling with GPUs

Next, apply batch inference for all input data with the Ray Data [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) method. When using vLLM, LLM instances require GPUs; here, we will demonstrate how to configure Ray Data to scale the number of LLM instances and GPUs needed.

To use GPUs for inference in the Workspace, we can specify `num_gpus` and `concurrency` in the `ds.map_batches()` call below to indicate the number of LLM instances and the number of GPUs per LLM instance, respectively. For example, if we want to use 4 LLM instances, with each requiring 1 GPU, we would set `concurrency=4` and `num_gpus=1`, requiring 4 total GPUs.

Smaller models, such as `Meta-Llama-3-8B-Instruct` and `Mistral-7B-Instruct-v0.1`, typically require 1 GPU per instance. Larger models, such as `Mixtral-8x7B-Instruct-v0.1` and `meta-llama/Meta-Llama-3-70B-Instruct`, typically require multiple GPUs per instance. You should configure these parameters according to the compute needed by the model.


```python
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=4,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=1,
    # Specify the batch size for inference. Set the batch size to as large as possible without running out of memory.
    # If you encounter out-of-memory errors, decreasing batch_size may help.
    batch_size=5,
    # Pass keyword arguments for the LLMPredictor class.
    fn_constructor_kwargs={"text_column": "item"},
    # Select the accelerator type; A10G or L4.
    accelerator_type=get_a10g_or_equivalent_accelerator_type(),
)
```

Finally, make sure to either enable *Auto-select worker nodes* or configure your workspace cluster to have the appropriate GPU worker nodes (A10G or L4):

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/batch-llm/assets/ray-data-gpu.png"/>

Run the following cell to start dataset execution and view the results!



```python
ds.take_all()
```

### Scaling to a larger dataset
In the example above, we performed batch inference for Ray Dataset with 5 example prompts. Next, let's explore how to scale to a larger dataset based on files stored in cloud storage.

Run the following cell to create a Dataset from a text file stored on S3. This Dataset has 100 rows, with each row containing a single prompt in the `text` column.


```python
ds = ray.data.read_text("s3://anonymous@air-example-data/prompts_100.txt")
ds.take(1)
```

Similar to before, we apply batch inference for all input data with the Ray Data [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) method.


```python
ds = ds.map(construct_input_prompt, fn_kwargs={"text_column": "text"})
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=4,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=1,
    # Specify the batch size for inference. Set the batch size to as large possible without running out of memory.
    # If you encounter CUDA out-of-memory errors, decreasing batch_size may help.
    batch_size=5,
    # Pass keyword arguments for the LLMPredictor class.
    fn_constructor_kwargs={"text_column": "text"},
    # Select the accelerator type; A10G or L4.
    accelerator_type=get_a10g_or_equivalent_accelerator_type(),
)
```

### Output Results
Finally, write the inference output data out to Parquet files on S3. 

Running the following cell will trigger execution for the full Dataset, which will execute all of the operations (`read_text()`, `map_batches(LLMPredictor)`, `write_parquet()`) at once:


```python
ds.write_parquet(output_path)
print(f"Batch inference result is written into {output_path}.")
```

### Monitoring Dataset execution
We can use the Ray Dashboard to monitor the Dataset execution. In the Ray Dashboard tab, navigate to the Job page and open the "Ray Data Overview" section. Click on the link for the running job, and open the "Ray Data Overview" section to view the details of the batch inference execution:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/batch-llm/assets/ray-data-jobs.png" width=900px/>

### Handling GPU out-of-memory failures
If you run into CUDA out of memory, your batch size is likely too large. Decrease the batch size as described above.

If your batch size is already set to 1, then use either a smaller model or GPU devices with more memory.

For advanced users working with large models, you can use model parallelism to shard the model across multiple GPUs.

### Reading back results
We can also use Ray Data to read back the output files to ensure the results are as expected.


```python
ds_output = ray.data.read_parquet(output_path)
ds_output.take(5)
```

### Submitting to Anyscale Jobs

The script in `main.py` has the same code as this notebook; you can use `ray job submit` to submit the app in that file to Anyscale Jobs. Refer to [Introduction to Jobs](https://docs.endpoints.anyscale.com/preview/examples/intro-jobs/) for more details.


After modifying the configurations at the top of `main.py` (model name, input/output path, input text column), run the following cell to submit a job:


```python
!ray job submit -- python main.py
```

## Summary

This notebook:
- Read in data from in-memory samples or input files from cloud storage. 
- Used Ray Data and vLLM to run offline batch inference of a LLM.
- Wrote the inference outputs to cloud storage and read back the results.


