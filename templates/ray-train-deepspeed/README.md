# Getting Started with DeepSpeed ZeRO and Ray Train

This template demonstrates how to combine DeepSpeed ZeRO with Ray Train to efficiently scale PyTorch training across GPUs and nodes while minimizing memory usage.

DeepSpeed is a deep learning optimization library designed for scalability and efficiency. Its ZeRO (Zero Redundancy Optimizer) family partitions model states, gradients, and optimizer states across workers to drastically reduce memory consumption while preserving data-parallel semantics.

This tutorial provides a step-by-step guide to integrating DeepSpeed ZeRO with Ray Train. Specifically, it covers:
- A hands-on example of fine-tuning a LLM
- Checkpoint saving and resuming with Ray Train
- Launching a distributed training job
- Configuring DeepSpeed for memory and performance (stages, mixed precision, CPU offload)

Note: This template is optimized for the Anyscale platform. When running on open source Ray, you must configure a Ray cluster, install dependencies on all nodes, and set up storage for checkpoints.

**Anyscale Specific Configuration**

Note: This tutorial is optimized for the Anyscale platform. When running on open source Ray, additional configuration is required. For example, you will need to manually:

- **Configure your Ray Cluster**: Set up your multi-node environment and manage resource allocation without Anyscale's automation.
- **Manage Dependencies**: Manually install and manage dependencies on each node.
- **Set Up Storage**: Configure your own distributed or shared storage system for model checkpointing.

## Step by Step Guide

In this example, we will demonstrate how to fine-tune an LLM with Ray Train and DeepSpeed in a multi-GPU, multi-node environment.
Before writing a Python script for fine-tuning, install the required dependencies:

```bash
%%bash
pip install torch torchvision
pip install transformers datasets==3.6.0 trl
pip install deepspeed
```

### 1. Import Packages

We start by importing the required libraries. These include Ray Train APIs for distributed training, PyTorch utilities for model and data handling, Transformers and Datasets from Hugging Face, and DeepSpeed.

```python
import os
import tempfile
import uuid
import logging

import argparse
from typing import Dict, Any

import ray
import ray.train
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, Checkpoint

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DownloadConfig

import deepspeed

logger = logging.getLogger(__name__)
```


### 2. Set up dataloader

We now define a dataset and a dataloader. The function below:

1. Downloads a tokenizer from the Hugging Face Hub (AutoTokenizer).
1. Loads a dataset using Hugging Face’s load_dataset.
1. Applies tokenization with padding and truncation using map.
1. Converts the dataset into a PyTorch DataLoader, which handles batching and shuffling.
1. Finally, use ray.train.torch.prepare_data_loader to make the dataloader distributed-ready.


```python
def setup_dataloader(model_name: str, dataset_name: str, seq_length: int, batch_size: int) -> DataLoader:
    # (1) Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # (2) Load dataset
    dataset = load_dataset(dataset_name, split="train[:100%]")

    # (3) Apply tokenization
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', max_length=seq_length, truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=1, keep_in_memory=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # (4) Create DataLoader
    data_loader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size,
        shuffle=True
    )

    # (5) Use prepare_data_loader for distributed training
    return ray.train.torch.prepare_data_loader(data_loader)
```

**Making the dataloader distributed-ready with Ray:**

When training across multiple GPUs, the most common strategy is data parallelism:
- Each GPU worker gets a shard of the dataset.
- Workers process their batches independently.
- After each step, gradients are synchronized across workers to keep model parameters aligned.

Normally, you’d need to manually configure PyTorch’s DistributedSampler for this. Ray’s prepare_data_loader automates that setup:
- Ensures each worker only sees its own shard.
- Avoids overlapping samples across GPUs.
- Handles epoch boundaries automatically.
This makes distributed training easier, while still relying on familiar PyTorch APIs.


### 3. Model and optimizer initialization

We now set up the model and optimizer. The function below:

1. Downloads a pretrained model from the Hugging Face Hub (AutoModelForCausalLM).
1. Defines the optimizer (AdamW).
1. Wraps the model and optimizer with DeepSpeed’s initialize, which applies ZeRO optimizations and returns a DeepSpeedEngine.


```python
def setup_model_and_optimizer(model_name: str, learning_rate: float, ds_config: Dict[str, Any]) -> deepspeed.runtime.engine.DeepSpeedEngine:
    # (1) Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # (2) Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # (3) Initialize with DeepSpeed (distributed + memory optimizations)
    ds_engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    return ds_engine
```

**Making the model distributed-ready with Ray and DeepSpeed**

In distributed training, every worker needs its own copy of the model and optimizer, but memory can quickly become a bottleneck.
DeepSpeed’s `initialize` always partitions the optimizer states across workers (ZeRO Stage 1). Depending on the chosen stage, it can also partition gradients (Stage 2) and model parameters/weights (Stage 3). This staged approach lets you balance memory savings with communication overhead while still applying additional optimizations for performance. We will describe these ZeRO stages in more detail [later in the tutorial](#deepspeed-zero-stages).

## 4. Checkpointing and Loading

Checkpointing is crucial for fault tolerance and for resuming training after interruptions. The functions below:

1. Create a temporary directory for storing checkpoints.
1. Save the partitioned model and optimizer states with DeepSpeed’s `save_checkpoint`.
1. Synchronize all workers with `torch.distributed.barrier` to ensure every process finishes saving.
1. Report metrics and checkpoint location to Ray with `ray.train.report`.
1. Restore a previously saved checkpoint into the DeepSpeed engine using `load_checkpoint`.


```python
def report_metrics_and_save_checkpoint(
    ds_engine: deepspeed.runtime.engine.DeepSpeedEngine,
    metrics: Dict[str, Any]
) -> None:
    # (1) Create temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_epoch = os.path.join(tmp, "epoch")
        os.makedirs(tmp_epoch, exist_ok=True)

        # (2) Save checkpoint (partitioned across workers)
        ds_engine.save_checkpoint(tmp_epoch)

        # (3) Synchronize workers
        torch.distributed.barrier()

        # (4) Report metrics and checkpoint to Ray
        ray.train.report(metrics, checkpoint=Checkpoint.from_directory(tmp))


def load_checkpoint(ds_engine: deepspeed.runtime.engine.DeepSpeedEngine, ckpt: ray.train.Checkpoint):
    try:
        # (5) Restore checkpoint into DeepSpeed engine
        with ckpt.as_directory() as checkpoint_dir:
            ds_engine.load_checkpoint(checkpoint_dir)
    except Exception as e:
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e
```

**Making checkpoints distributed-ready with Ray and DeepSpeed**

DeepSpeed saves model and optimizer states in a partitioned format, where each worker stores only its shard. This requires synchronization across processes, so it’s important to ensure that all workers reach the same checkpointing point before proceeding. We use `torch.distributed.barrier()` to guarantee that every worker finishes saving before moving on.

Finally, `ray.train.report` both reports training metrics and saves the checkpoint to persistent storage, making it accessible for resuming training later.


### 5. Training Loop

In Ray Train, we define a training loop function that orchestrates the entire process on each GPU worker. The function below:

1. Restores training from a checkpoint if one is available.
1. Sets up the dataloader with setup_dataloader.
1. Initializes the model and optimizer with DeepSpeed.
1. Gets the device assigned to this worker.
1. Iterates through the specified number of epochs.
1. For multi-GPU training, ensures each worker sees a unique data shard each epoch.
1. For each batch:
   - Moves inputs to the device.
   - Runs the forward pass to compute loss.
   - Logs the loss.
1. Performs backward pass and optimizer step with DeepSpeed.
1. Aggregates average loss and reports metrics, saving a checkpoint at the end of each epoch.

```python
def train_loop(config: Dict[str, Any]) -> None:

    # (1) Load checkpoint if exists
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        load_checkpoint(ds_engine, ckpt)

    # (2) Set up dataloader
    train_loader = setup_dataloader(config["model_name"], config["seq_length"], config["batch_size"])

    # (3) Initialize model + optimizer with DeepSpeed
    ds_engine = setup_model_and_optimizer(config["model_name"], config["learning_rate"], config["ds_config"])

    # (4) Get device for this worker
    device = ray.train.torch.get_device()

    for epoch in range(config["epochs"]):
        # (6) Ensure unique shard per worker when using multiple GPUs
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        num_batches = 0

        # (7) Iterate over batches
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = ds_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                use_cache=False
            )
            loss = outputs.loss
            print(f"step {step} loss: {loss.item()}")

            # Backward pass + optimizer step
            ds_engine.backward(loss)
            ds_engine.step()

            running_loss += loss.item()
            num_batches += 1

        # (8) Report metrics + save checkpoint
        report_metrics_and_save_checkpoint(ds_engine, {"loss": running_loss / num_batches, "epoch": epoch})
```

**Coordinating distributed training with Ray and DeepSpeed**

Ray launches this training loop on each GPU worker, while DeepSpeed handles partitioning and optimization under the hood. Each worker processes a unique shard of the data (data parallelism), computes local gradients, and synchronizes with others.
By combining Ray’s orchestration with DeepSpeed’s memory-efficient engine, you get distributed training that scales smoothly across multiple GPUs and nodes — with automatic checkpointing and metric reporting built in.

## 6. Configure DeepSpeed and Launch Trainer


The final step is to configure parameters and launch the distributed training job with Ray’s TorchTrainer. The function below:
1. Parses command-line arguments for training and model settings.
1. Defines the Ray scaling configuration (e.g., number of workers, GPU usage).
1. Builds the DeepSpeed configuration dictionary (ds_config).
1. Prepares the training loop configuration with hyperparameters and model details.
1. Sets up the Ray RunConfig to manage storage and experiment metadata.
1. Creates a TorchTrainer that launches the training loop on multiple GPU workers.
1. Starts training with trainer.fit() and prints the result.

```python
def main():
    # (1) Parse arguments
    args = get_args()
    print(args)

    # (2) Ray scaling configuration
    scaling_config = ScalingConfig(num_workers=2, use_gpu=True)

    # (3) DeepSpeed configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "bf16": {"enabled": True},
        "grad_accum_dtype": "bf16",
        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
    }

    # (4) Training loop configuration
    train_loop_config = {
        "epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "ds_config": ds_config,
        "model_name": args.model_name,
        "seq_length": args.seq_length,
        "dataset_name": args.dataset_name,
    }

    # (5) Ray run configuration
    run_config = RunConfig(
        storage_path="/mnt/cluster_storage/",
        name=f"deepspeed_sample_{uuid.uuid4().hex[:8]}",
    )

    # (6) Create trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=run_config,
    )

    # (7) Launch training
    result = trainer.fit()
    print(f"Training finished. Result: {result}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="MiniLLM/MiniPLM-Qwen-500M")
    parser.add_argument("--dataset_name", type=str, default="ag_news")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--zero_stage", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
```

The complete working script is available as `train.py`.  


**Launching distributed training with Ray and DeepSpeed**

Ray’s TorchTrainer automatically launches multiple workers (one per GPU) and runs the `train_loop` on each. The scaling configuration controls how many workers to start, while the run configuration handles logging, storage, and experiment tracking.

DeepSpeed’s `ds_config` ensures that the right ZeRO stage and optimizations are applied inside each worker. Together, this setup makes it easy to scale from a single GPU to a multi-node cluster without changing your training loop code.


## Advanced Usage

DeepSpeed has many other configuration options to tune performance and memory usage.
Here we introduce some of the most commonly used options.
Please refer to the [DeepSpeed documentation](https://www.deepspeed.ai/docs/config-json/) for more details.


### DeepSpeed ZeRO Stages

DeepSpeed ZeRO has three stages, each providing different levels of memory optimization and performance trade-offs.

- **Stage 1**: This stage focuses on optimizer state partitioning. It reduces memory usage by partitioning the optimizer states across data parallel workers. This is the least aggressive stage and is suitable for most models without significant changes.
- **Stage 2**: In addition to optimizer state partitioning, this stage also partitions the gradients. This further reduces memory usage but may introduce some communication overhead. It's a good choice for larger models that can benefit from additional memory savings.
- **Stage 3**: This is the most aggressive stage, which partitions both the optimizer states and the model parameters. It provides the highest memory savings but may require more careful tuning of the training process. This stage is recommended for very large models that cannot fit into the memory of a single GPU.

The higher the stage, the more memory savings you get, but it may also introduce more communication overhead and complexity in training.
You can select the desired ZeRO stage by setting the `zero_stage` parameter in the DeepSpeed configuration dictionary passed to `deepspeed.initialize`.

```python
ds_config = {
    "zero_optimization": {
        "stage": 2,  # or 1 or 3
...
    },
}
```


### Mixed Precision Training

Mixed precision training is a technique that uses both 16-bit and 32-bit floating-point types in a single network. This can lead to faster training times and reduced memory usage. DeepSpeed has built-in support for mixed precision training using either FP16 or BF16.

To enable mixed precision training, you can set the `bf16` or `fp16` parameters in the DeepSpeed configuration dictionary. For example:

```python
ds_config = {
    "bf16": {"enabled": True}, # or "fp16": {"enabled": True}
}
```

Note that these options keep the clone of weights/gradients and optimizer states in 32-bit precision to maintain numerical stability.


### CPU Offloading

DeepSpeed supports offloading model states and optimizer states to CPU memory.
Offloading these causes a certain amount of overhead due to data transfer between CPU and GPU, but it significantly reduces GPU memory usage, which can be beneficial when training very large models that do not fit into GPU memory.

To enable CPU offloading, you can set the `offload` parameters in the DeepSpeed configuration dictionary. For example:

```python
ds_config = {
    "offload_param": {
        "device": "cpu",
        "pin_memory": True,
    }
}
```

You can also offload only optimizer states similarly by using the `offload_optimizer` parameter.

```python
ds_config = {
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True,
    }
}
```

### Convert Checkpoint for Inference

As the checkpoint of DeepSpeed ZeRO Stage 3 is partitioned across multiple GPUs, it cannot be directly used for inference. To convert a ZeRO Stage 3 checkpoint to a standard model checkpoint that can be loaded for inference, you can use  `get_fp32_state_dict_from_zero_checkpoint` API.

```python
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# do the training and checkpoint saving
state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir)
torch.save(state_dict, "model_fp32.pt")
```
