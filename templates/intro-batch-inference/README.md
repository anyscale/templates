# Batch Inference Basics

Offline batch inference is a process for generating model predictions on a fixed set of input data. [Ray Data](https://docs.ray.io/en/latest/data/data.html) offers a scalable solution for batch inference, providing optimized inference performance for deep learning applications.

In this tutorial, you will learn:
1. How to set up and run a basic batch inference job in Anyscale using Ray Data + HuggingFace.
2. Features of the Ray Data parallelization API.
3. Tips and tricks for improving performance and avoiding out of memory errors.

**Note**: This tutorial is run within a workspace. Please overview the ``Introduction to Workspaces`` template first before this tutorial.

## Overview

Using Ray Data for offline inference involves four basic steps:

- **Step 1:** Load your data into a Ray Dataset. Ray Data supports [most common formats](https://docs.ray.io/en/latest/data/loading-data.html).
- **Step 2:** Define a Python class to load the pre-trained model.
- **Step 3:** Transform your dataset using the pre-trained model by calling [ds.map_batches()](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html).
- **Step 4:** Get the final predictions by either iterating through the output or saving the results.

Run the following cell for GPT-2 inference against a toy in-memory dataset with two records. The model will run on CPU:


```python
from typing import Dict
import numpy as np

import ray

# Step 1: Create a Ray Dataset from in-memory Numpy arrays.
# You can also create a Ray Dataset from many other sources and file
# formats.
ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

# Step 2: Define a Predictor class for inference.
# Use a class to initialize the model just once in `__init__`
# and re-use it for inference across multiple batches.
class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline
        # Initialize a pre-trained GPT2 Huggingface pipeline.
        self.model = pipeline("text-generation", model="gpt2")

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Get the predictions from the input batch.
        predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
        # `predictions` is a list of length-one lists. For example:
        # [[{'generated_text': 'output_1'}], ..., [{'generated_text': 'output_2'}]]
        # Modify the output to get it into the following format instead:
        # ['output_1', 'output_2']
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch

# Step 2: Map the Predictor over the Dataset to get predictions.
# Use 2 parallel actors for inference. Each actor predicts on a
# different partition of data.
predictions = ds.map_batches(HuggingFacePredictor, concurrency=2)
# Step 3: Show one prediction output.
predictions.show(limit=1)
```

### Interpreting the results

You should see as output something like this: ``"{'data': 'Complete this', 'output': 'Complete this order to ensure the best possible service for your business.\n\nWe take great pride in'}``

Note that above we called ``ds.show()`` in order to print the results to the console. Typically, results are saved to storage using a [write call](https://docs.ray.io/en/latest/data/saving-data.html).

In the Ray Dashboard tab, navigate to the Job page and open the "Ray Data Overview" section to view the details of the batch inference execution:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-batch-inference/assets/ray-data-job.png" width=800px/>



## Scaling to a larger dataset

Let's explore how to scale the above to a larger dataset, which will run on a cluster. Run the following cell to generate completions for 10000 rows with a concurrency of 20 actors. Ensure *Auto-select worker nodes* is checked in the cluster sidebar, and Anyscale will automatically add worker nodes to the cluster as needed:


```python
# Create a 10k row dataset.
ds = ray.data.from_numpy(np.asarray(["Today's weather"] * 10000))

# Ensure the dataset has enough blocks to be executed in parallel on many actors.
ds = ds.repartition(1000)

# Execute the batch inference.
predictions = ds.map_batches(HuggingFacePredictor, concurrency=20)
predictions.show()
```

## Scaling with GPUs

To use GPUs for inference, make the following changes to your code:

1. Update the class implementation to move the model and data to and from GPU.
2. Specify ``num_gpus=1`` in the ``ds.map_batches()`` call to indicate that each actor should use 1 GPU.
3. Specify a ``batch_size`` for inference. We'll cover how to optimize batch size in the next section.

The remaining is the same as in the code we ran above. To test this out, first make sure to either enable *Auto-select worker nodes* or configure your workspace cluster to have GPU worker nodes:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-batch-inference/assets/ray-data-gpu.png" width=300px/>

Run the below cell to test out the new code using GPUs:


```python
from typing import Dict
import numpy as np

import ray

ds = ray.data.from_numpy(np.asarray(["Large language models", "Text completion models"]))

class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline
        # Set "cuda:0" as the device so the Huggingface pipeline uses GPU.
        self.model = pipeline("text-generation", model="gpt2", device=0)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch

# Use 2 actors, each actor using 1 GPU. 2 GPUs total.
predictions = ds.map_batches(
    HuggingFacePredictor,
    num_gpus=1,
    # Specify the batch size for inference.
    # Increase this for larger datasets.
    batch_size=1,
    # Set the concurrency to the number of GPUs in your cluster.
    concurrency=2,
    )
predictions.show(limit=1)
```

### Configuring batch_size

Configure the size of the input batch that’s passed to ``__call__`` by setting the batch_size argument for ``ds.map_batches()``.

Increasing batch size results in faster execution because inference is a vectorized operation. For GPU inference, increasing batch size increases GPU utilization. Set the batch size to as large possible without running out of memory. If you encounter out-of-memory errors, decreasing ``batch_size`` may help.

**Caution:** The default batch_size of 4096 may be too large for datasets with large rows (for example, tables with many columns or a collection of large images).





```python
import numpy as np

import ray

ds = ray.data.from_numpy(np.ones((10, 100)))

def assert_batch(batch: Dict[str, np.ndarray]):
    assert len(batch["data"]) == 2, batch
    return batch

# Specify that each input batch should be of size 2.
ds.map_batches(assert_batch, batch_size=2).show(limit=1)
```

### Handling GPU out-of-memory failures

If you run into CUDA out-of-memory issues, your batch size is likely too large. Decrease the batch size as described above.

If your batch size is already set to 1, then use either a smaller model or GPU devices with more memory.

For advanced users working with large models, you can use model parallelism to shard the model across multiple GPUs.

## More tips and tricks

### Optimizing expensive CPU preprocessing

If your workload involves expensive CPU preprocessing in addition to model inference, you can optimize throughput by separating the preprocessing and inference logic into separate stages. This separation allows inference on batch 
``N`` to execute concurrently with preprocessing on batch ``N + 1``.

For an example where preprocessing is done in a separate map call, see [Image Classification Batch Inference with PyTorch ResNet18](https://docs.ray.io/en/latest/data/examples/pytorch_resnet_batch_prediction.html).



### Handling CPU out-of-memory failures

If you run out of CPU RAM, you likely that you have too many model replicas that are running concurrently on the same node. For example, if a model uses 5 GB of RAM when created / run, and a machine has 16 GB of RAM total, then no more than three of these models can be run at the same time. The default resource assignments of one CPU per task/actor might lead to ``OutOfMemoryError`` from Ray in this situation.

Suppose your cluster has 4 nodes, each with 16 CPUs. To limit to at most 3 of these actors per node, you can override the CPU or memory:



```python
from typing import Dict
import numpy as np

import ray

ds = ray.data.from_numpy(np.asarray(["Complete this", "for me"]))

class HuggingFacePredictor:
    def __init__(self):
        from transformers import pipeline
        self.model = pipeline("text-generation", model="gpt2")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        predictions = self.model(list(batch["data"]), max_length=20, num_return_sequences=1)
        batch["output"] = [sequences[0]["generated_text"] for sequences in predictions]
        return batch

predictions = ds.map_batches(
    HuggingFacePredictor,
    # Require 5 CPUs per actor (so at most 3 can fit per 16 CPU node).
    num_cpus=5,
    # 3 actors per node, with 4 nodes in the cluster means concurrency of 12.
    concurrency=12,
    )
predictions.show(limit=1)
```

This concludes our batch inference tutorial. To learn more about Ray Data and how you can use it to scale workloads such as batch inference, check out the [Ray Data docs](https://docs.ray.io/en/latest/data/data.html).

## Summary

This notebook:
- Run a basic batch inference job using Ray Data + HuggingFace.
- Showed how to configure Ray Data's parallelization options.
- Overviewed common performance tips and how to avoid out of memory errors.


