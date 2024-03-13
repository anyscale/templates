## LLM offline batch inference with Ray Data and vLLM

**⏱️ Time to complete**: 10 min

This template shows you how to:
1. Read in data from in-memory samples or files on cloud storage. 
2. Use Ray Data and vLLM to run batch inference of a LLM.
3. Write the inference outputs to cloud storage.

For a Python script version of the `.ipynb` notebook used for the workspace template, refer to `examples/main.py`.

**Note:** For a more general introduction to batch inference with Ray Data, check out the `Batch Inference Basics` workspace template.

### How to decide between online vs offline inference for LLM
Online LLM inference (e.g. Anyscale Endpoint) should be used when you want to get real-time response for prompt. Use online inference when you want to optimize latency of inference to be as quick as possible.

On the other hand, offline LLM inference should be used when you want to get reponses for a large number of prompts within an end-to-end time requirement (e.g. minutes to hours granularity). Use offline inference when you want to optimize throughput of inference to use resource (e.g. GPU) as much as possible on large-scale input data.

### Step 1: Install Python dependencies
Install additional required dependencies using `pip`.


```python
!pip install -q vllm==0.3.3 && echo 'Install complete!'
```

Next, import the dependencies used in this template.


```python
import os
from typing import Dict

import numpy as np
import ray
from vllm import LLM, SamplingParams

from util.utils import generate_output_path
```

### Step 2: Set up model defaults
Set up default values that will be used in the batch inference workflow:
* Your [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens). This will be used to download the model.
* The model to use for inference ([the list of supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)).
* The [sampling parameters object](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py) used by vLLM.
* The output path where results will be written as parquet files.


```python
# Set the Hugging Face token. Replace the following with your token.
HF_TOKEN = "<REPLACE_WITH_YOUR_HUGGING_FACE_USER_TOKEN>"
# Set to the model that you wish to use. Note that using the llama models will require a hugging face token to be set.
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=4096)
# Output path to write output result. You can also change this to any cloud storage path, e.g. a specific S3 bucket.
output_path = generate_output_path(os.environ.get("ANYSCALE_ARTIFACT_STORAGE"), HF_MODEL)
```

Start up Ray, using the Hugging Face token as an environment variable so that it's made available to all nodes in the cluster.


```python
if ray.is_initialized():
    ray.shutdown()
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    }
)
```

### Step 3: Read input data with Ray Data
Use Ray Data to read in your input data from some sample prompts.


```python
# Create some sample prompts, and use Ray Data to create a dataset for it.
prompts = [
"""
I always wanted to be a ...
""",
"""
The best way to learn a new language is ...
""",
"""
The biggest challenge facing our society today is ...
""",
"""
One thing I would change about my past is ...
""",
"""
The key to a happy life is ...
""",
]
ds = ray.data.from_items(prompts)

# View one row of the Dataset.
ds.take(1)
```

### Scaling to a larger dataset
In the cell above, we created a Ray Dataset with 5 example prompts. Next, let's explore how to scale to a larger dataset based on files stored in cloud storage.

Run the following cell to create a Dataset from a text file stored on S3.


```python
ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
print(ds)
```

This Dataset has 5800 rows, each row containing a single prompt in the `text` column. For the purposes of this workspace template, we will only run inference on the first 100 rows, which we can achieve using the [`limit`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.limit.html)  method.


```python
ds = ds.limit(100)
ds.take_all()
```

### Step 4: Run Batch Inference with vLLM

Create a class to define batch inference logic.


```python
class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model=HF_MODEL)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["text"], sampling_params)
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

Apply batch inference for all input data with the Ray Data [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) method. Here, you can easily configure Ray Data to scale the number of LLM instances and compute (number of GPUs to use).


```python
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=4,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=1,
    # Specify the batch size for inference.
    batch_size=10,
)
```

Time to execute and view the results!


```python
ds.take_all()
```

Finally, write the inference output data out to Parquet files on S3.


```python
ds.write_parquet(output_path)
print(f"Batch inference result is written into {output_path}.")
```

### Summary

This notebook:
- Read in data from in-memory samples or input files from cloud storage. 
- Used Ray Data and vLLM to run offline batch inference of a LLM.
- Wrote the inference outputs to cloud storage.


