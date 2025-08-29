# Getting Started with PyTorch Fully Sharded Data Parallel (FSDP2) and Ray Train

**Time to complete:** 30 min

This template shows you how to unlock the memory and performance improvements of integrating PyTorch's Fully Sharded Data Parallel with Ray Train. 

PyTorch's Fully Sharded Data Parallel (FSDP2) enables model sharding across nodes, allowing distributed training of large models with a significantly smaller memory footprint compared to standard Distributed Data Parallel (DDP). For a more detailed overview of FSDP2, see [PyTorch's official documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#getting-started-with-fully-sharded-data-parallel-fsdp2). 

This tutorial provides a comprehensive, step-by-step guide on integrating PyTorch FSDP2 with Ray Train. Specifically, this guide covers: 

- A hands-on example of training an image classification model
- Model checkpoint saving and loading with PyTorch Distributed Checkpoint (DCP)
- Configuring FSDP2 to mitigate out-of-memory (OOM) errors using mixed precision, CPU offloading, sharding granularity, and more
- GPU memory profiling with PyTorch Profiler
- Loading a distributed model for inference

**Note:** This notebook uses FSDP2's `fully_sharded` API. If you're currently using FSDP1's `FullyShardedDataParallel`, consider migrating to FSDP2 for improved performance and features. 

<div class="alert alert-block alert-warning">

**Anyscale Specific Configuration**

Note: This tutorial is optimized for the Anyscale platform. When running on open source Ray, additional configuration is required. For example, you’ll need to manually:

- **Configure your Ray Cluster**: Set up your multi-node environment and manage resource allocation without Anyscale's automation.
- **Manage Dependencies**: Manually install and manage dependencies on each node.
- **Set Up Storage**: Configure your own distributed or shared storage system for model checkpointing.

</div>

## Example Overview

For demonstration purposes, we will integrate Ray Train with FSDP using a **Vision Transformer (ViT)** trained on the FashionMNIST dataset. We chose ViT because it has clear, repeatable block structures (transformer blocks) that are ideal for demonstrating FSDP's sharding capabilities. 

While this is a relatively simple example, FSDP's complexity can lead to common challenges during training, such as out-of-memory (OOM) errors. Throughout this guide, we'll address these common issues and provide practical tips for improving performance and reducing memory utilization based on your specific use case. 

## 1. Package Setup

Install the required dependencies for this tutorial:


```bash
%%bash
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install matplotlib
```

    Requirement already satisfied: matplotlib in /home/ray/anaconda3/lib/python3.11/site-packages (3.10.5)
    Requirement already satisfied: contourpy>=1.0.1 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (4.59.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.4.9)
    Requirement already satisfied: numpy>=1.23 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (23.0)
    Requirement already satisfied: pillow>=8 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /home/ray/anaconda3/lib/python3.11/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in /home/ray/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    [92mSuccessfully registered `matplotlib` package to be installed on all cluster nodes.[0m
    [92mView and update dependencies here: https://console.anyscale.com/cld_kvedZWag2qA8i5BjxUevf5i7/prj_cz951f43jjdybtzkx1s5sjgz99/workspaces/expwrk_nktjw7a3j2l5c7af9rh3n5rskw?workspace-tab=dependencies[0m



```python
# Enable Ray Train V2 for the latest train APIs
import os
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"

# Ray Train imports
import ray
import ray.train
import ray.train.torch

# PyTorch core and FSDP imports
import torch
from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)

# PyTorch Distributed Checkpoint (DCP) imports
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions
)
from torch.distributed.device_mesh import init_device_mesh 
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed.checkpoint as dcp

# PyTorch training components
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

# Computer vision components
from torchvision.models import VisionTransformer
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor, Normalize, Compose

# Profiling and utilities
import torch.profiler
import tempfile
import uuid
import logging

# Set up logging
logger = logging.getLogger(__name__)
```

## 2. Define the Training Function

Below is the main training function that orchestrates the FSDP training process. In the following sections, we'll implement each of the helper functions used within this training loop.

### 2a. GPU Memory Profiling

GPU memory profiling is an useful tool for monitoring and analyzing memory usage during model training. It helps identify bottlenecks, optimize resource allocation, and prevent out-of-memory errors. We configure PyTorch's GPU memory profiler within the training function.

In this demo, we configure the profiler to generate a profiling file for each worker accessible from the Anyscale Files tab under cluster storage. To inspect a worker's memory profile, download the corresponding HTML file and open it in your browser. The profiler configuration and export path can be customized within the training function.  For more details on PyTorch's memory profiler, see the [PyTorch blog](https://pytorch.org/blog/understanding-gpu-memory-1/).

<div style="display: flex; gap: 40px; align-items: flex-start;">
  <div style="text-align: center;">
    <h3>Example Memory Profile</h3>
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray-train-fsdp/images/gpu_memory_profile.png" width="600"/>
  </div>
</div>

### 2b. Storage Configuration

In this demo, we use cluster storage to allow for quick iteration and development, but this may not be suitable in production environments or at high scale. In such cases, object storage should be used instead. For more information about how to select your storage type, see the [Anyscale storage configuration docs](https://docs.anyscale.com/configuration/storage).


```python
def train_func(config):
    """Main training function that integrates FSDP2 with Ray Train.
    
    Args:
        config: Training configuration dictionary containing hyperparameters
    """
    # Step 1: Initialize the model
    model = init_model()

    # Configure device and move model to GPU
    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    model.to(device)

    # Step 2: Apply FSDP sharding to the model
    shard_model(model)

    # Step 3: Initialize loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.get('learning_rate', 0.001))

    # Step 4: Load from checkpoint if available (for resuming training)
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        load_fsdp_checkpoint(model, optimizer, loaded_checkpoint)

    # Step 5: Prepare training data
    transform = Compose([
        ToTensor(), 
        Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(tempfile.gettempdir(), "data")
    train_data = FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=config.get('batch_size', 128), 
        shuffle=True
    )
    # Prepare data loader for distributed training
    train_loader = ray.train.torch.prepare_data_loader(train_loader)

    world_rank = ray.train.get_context().get_world_rank()

    # Step 6: Setup PyTorch Profiler for memory monitoring
    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
   ) as prof:

        # Step 7: Main training loop
        running_loss = 0.0
        num_batches = 0
        epochs = config.get('epochs', 5)
        
        for epoch in range(epochs):
            # Set epoch for distributed sampler to ensure proper shuffling
            if ray.train.get_context().get_world_size() > 1:
                train_loader.sampler.set_epoch(epoch)

            for images, labels in train_loader:
                # Note: Data is automatically moved to the correct device by prepare_data_loader
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Standard training step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update profiler
                prof.step()
                
                # Track metrics
                running_loss += loss.item()
                num_batches += 1

            # Step 8: Report metrics and save checkpoint after each epoch
            avg_loss = running_loss / num_batches
            metrics = {"loss": avg_loss, "epoch": epoch}
            report_metrics_and_save_fsdp_checkpoint(model, optimizer, metrics)

            # Log metrics from rank 0 only to avoid duplicate outputs
            if world_rank == 0:
                print(metrics)
    
    # Step 9: Export memory profiling results
    run_name = ray.train.get_context().get_experiment_name()
    prof.export_memory_timeline(
        f"/mnt/cluster_storage/{run_name}/rank{world_rank}_memory_profile.html"
    )

    # Step 10: Save the final model for inference
    save_model_for_inference(model, world_rank)
```

## 3. Model Initialization

The following function initializes a Vision Transformer (ViT) model configured for the FashionMNIST dataset:


```python
def init_model() -> torch.nn.Module:
    """Initialize a Vision Transformer model for FashionMNIST classification.
    
    Returns:
        torch.nn.Module: Configured ViT model
    """
    logger.info("Initializing Vision Transformer model...")

    # Create a ViT model with architecture suitable for 28x28 images
    model = VisionTransformer(
        image_size=28,        # FashionMNIST image size
        patch_size=7,         # Divide 28x28 into 4x4 patches of 7x7 pixels each
        num_layers=4,         # Number of transformer encoder layers
        num_heads=2,          # Number of attention heads per layer
        hidden_dim=64,        # Hidden dimension size
        mlp_dim=128,          # MLP dimension in transformer blocks
        num_classes=10,       # FashionMNIST has 10 classes
    )

    # Modify the patch embedding layer for grayscale images (1 channel instead of 3)
    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,        # FashionMNIST is grayscale (1 channel)
        out_channels=64,      # Match the hidden_dim
        kernel_size=7,        # Match patch_size
        stride=7,             # Non-overlapping patches
    )

    return model
```

## 4. Model Sharding with FSDP2

PyTorch's `fully_shard` enables sharding at various granularities. At the most granular level, every layer can be sharded, but this increases communication costs between Ray Train workers. Experiment with different sharding granularities to find the optimal balance for your use case. In this demo, we shard only the encoder blocks—the largest layers in the Vision Transformer.

Beyond sharding granularity, FSDP2 offers several configuration options to optimize performance and mitigate out-of-memory (OOM) errors:

### 4a. CPU Offloading and `reshard_after_forward`

CPU offloading reduces GPU memory footprint by storing model components in the CPU. However, this comes with the trade-off of increased data transfer overhead between CPU and GPU during computation.

**How CPU offloading works:**
- Sharded parameters, gradients, and optimizer states are stored on CPU
- Sharded parameters are copied to GPU during forward/backward computation and freed after use
- Computed gradients are copied to the CPU where the optimizer step is computed

**`reshard_after_forward` optimization:**
When enabled, all-gathered model weights are freed immediately after the forward pass. This reduces peak GPU memory usage but increases communication overhead during backward pass as parameters need to be all-gathered again.

**When to use CPU offloading:**
- When GPU memory is constrained
- For very large models that don't fit in GPU memory

**When not to use CPU offloading:**
- When CPU memory is limited (can cause CPU crashes)
- When training speed is more important than memory usage

<div style="display: flex; gap: 40px; align-items: flex-start;">
  <div style="text-align: center;">
    <h3>Without CPU Offloading</h3>
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray-train-fsdp/images/gpu_memory_profile.png" width="600"/>
  </div>
  <div style="text-align: center;">
    <h3>With CPU Offloading</h3>
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray-train-fsdp/images/cpu_offload_profile.png" width="600"/>
  </div>
</div>

Learn more about CPU offloading on the [PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy).

### 4b. Mixed Precision

Enabling mixed precision accelerates training and reduces GPU memory usage with minimal accuracy impact. Unlike other distributed approaches like DDP, FSDP already maintains high-precision copies of sharded parameters, so mixed precision requires no additional memory overhead.

**Benefits of mixed precision with FSDP:**
- Reduced memory usage for activations and intermediate computations
- Faster computation on modern GPUs
- Maintained numerical stability through selective precision

<div style="display: flex; gap: 40px; align-items: flex-start;">
  <div style="text-align: center;">
    <h3>Without Mixed Precision</h3>
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray-train-fsdp/images/gpu_memory_profile.png" width="600"/>
  </div>
  <div style="text-align: center;">
    <h3>With Mixed Precision</h3>
    <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray-train-fsdp/images/mixed_precision_profile.png" width="600"/>
  </div>
</div>

Learn more about mixed precision configuration on the [PyTorch documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy).

### 4c. Device Mesh Configuration

`init_device_mesh` configures a `DeviceMesh` that describes the training run's device topology. This demo uses a simple 1D mesh for data parallelism, but `DeviceMesh` also supports multi-dimensional parallelism approaches including tensor parallelism and pipeline parallelism.

For advanced multi-dimensional parallelism configurations, see the [PyTorch device mesh documentation](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html).


```python
def shard_model(model: torch.nn.Module): 
    """Apply FSDP2 sharding to the model with optimized configuration.
    
    Args:
        model: The PyTorch model to shard
    """
    logger.info("Applying FSDP2 sharding to model...")

    # Step 1: Create 1D device mesh for data parallel sharding
    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda", 
        mesh_shape=(world_size,), 
        mesh_dim_names=("data_parallel",)
    )

    # Step 2: Configure CPU offloading policy (optional)
    offload_policy = CPUOffloadPolicy()

    # Step 3: Configure mixed precision policy (optional)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float16,    # Store parameters in half precision
        reduce_dtype=torch.float16,   # Use half precision for gradient reduction
    )

    # Step 4: Apply sharding to each transformer encoder block
    for encoder_block in model.encoder.layers.children():
        fully_shard(
            encoder_block, 
            mesh=mesh, 
            reshard_after_forward=True,  # Free memory after forward pass
            offload_policy=offload_policy, 
            mp_policy=mp_policy
        )

    # Step 5: Apply sharding to the root model
    # This wraps the entire model and enables top-level FSDP functionality
    fully_shard(
        model, 
        mesh=mesh, 
        reshard_after_forward=True, 
        offload_policy=offload_policy, 
        mp_policy=mp_policy
    )
    
```

## 5. Distributed Checkpoint Wrapper Setup

We create a checkpointing wrapper using PyTorch's `Stateful` API to simplify distributed checkpoint management. This wrapper handles the complexities of saving and loading FSDP model states across multiple workers.


```python
class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, PyTorch DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to handle calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

```

## 6. Loading FSDP Model from Checkpoint

Distributed checkpoints can be loaded using `dcp.load`, which automatically handles resharding when the number of workers changes between training runs. This flexibility allows you to resume training with different resource configurations. 


```python
def load_fsdp_checkpoint(model: FSDPModule, optimizer: torch.optim.Optimizer, ckpt: ray.train.Checkpoint):
    """Load an FSDP checkpoint into the model and optimizer.
    
    This function handles distributed checkpoint loading with automatic resharding
    support. It can restore checkpoints even when the number of workers differs
    from the original training run.
    
    Args:
        model: The FSDP-wrapped model to load state into
        optimizer: The optimizer to load state into
        ckpt: Ray Train checkpoint containing the saved state
    """
    logger.info("Loading FSDP checkpoint for resuming training...")
    
    try:
        with ckpt.as_directory() as checkpoint_dir:
            # Create state wrapper for DCP loading
            state_dict = {"app": AppState(model, optimizer)}
            
            # Load the distributed checkpoint
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir
            )
            
        logger.info("Successfully loaded FSDP checkpoint")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}") from e
```

## 7. Saving Model Checkpoints

The following function handles periodic checkpoint saving during training, combining metrics reporting with distributed checkpoint storage:


```python
def report_metrics_and_save_fsdp_checkpoint(
    model: FSDPModule, optimizer: torch.optim.Optimizer, metrics: dict
) -> None:
    """Report training metrics and save an FSDP checkpoint.
    
    This function performs two critical operations:
    1. Saves the current model and optimizer state using distributed checkpointing
    2. Reports metrics to Ray Train for tracking
    
    Args:
        model: The FSDP-wrapped model to checkpoint
        optimizer: The optimizer to checkpoint
        metrics: Dictionary of metrics to report (e.g., loss, accuracy)
    """
    logger.info("Saving checkpoint and reporting metrics...")
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        # Perform a distributed checkpoint with DCP
        state_dict = {"app": AppState(model, optimizer)}
        dcp.save(state_dict=state_dict, checkpoint_id=temp_checkpoint_dir)

        # Report each checkpoint shard from all workers
        # This saves the checkpoint to shared cluster storage for persistence
        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)
        
    logger.info(f"Checkpoint saved successfully. Metrics: {metrics}")
```

## 8. Saving Model for Inference

After training completes, we need to save the final model in an unsharded format for inference. This differs from regular checkpointing as it consolidates the model checkpoint into one file that is compatible with `torch.load`. The `get_model_state_dict` function performs an all-gather operation to reconstruct the complete model state on rank 0, which then saves and reports the full model checkpoint.


```python
def save_model_for_inference(model: FSDPModule, world_rank: int) -> None:
    """Save the complete unsharded model for inference.
    
    This function consolidates the distributed model weights into a single
    checkpoint file that can be used for inference without FSDP.
    
    Args:
        model: The FSDP-wrapped model to save
        world_rank: The rank of the current worker
    """
    logger.info("Preparing model for inference...")
    
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        # Step 1: All-gather the model state across all ranks
        # This reconstructs the complete model from distributed shards
        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,    # Reconstruct full model
                cpu_offload=True,        # Move to CPU to save GPU memory
            )
        )

        logger.info("Successfully retrieved complete model state dict")
        checkpoint = None

        # Step 2: Save the complete model (rank 0 only)
        if world_rank == 0: 
            torch.save(model_state_dict, save_file)
            logger.info(f"Saved complete model to {save_file}")

            # Create checkpoint for shared storage
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        # Step 3: Report the final checkpoint to Ray Train
        ray.train.report(
            {}, 
            checkpoint=checkpoint, 
            checkpoint_dir_name="full_model"
        )
```

## 9. Launching the Distributed Training Job

Now we'll configure and launch the distributed training job using Ray Train's TorchTrainer:


```python
# Configure distributed training resources
scaling_config = ray.train.ScalingConfig(
    num_workers=2,      # Number of distributed workers
    use_gpu=True        # Enable GPU training
)

# Configure training parameters
train_loop_config = {
    "epochs": 5,
    "learning_rate": 0.001,
    "batch_size": 128,
}

# Configure run settings and storage
run_config = ray.train.RunConfig(
    # Persistent storage path accessible across all worker nodes
    storage_path="/mnt/cluster_storage/",
    # Unique experiment name (use consistent name to resume from checkpoints)
    name=f"fsdp_mnist_{uuid.uuid4().hex[:8]}",
    # Fault tolerance configuration
    failure_config=ray.train.FailureConfig(max_failures=1),
)

# Initialize and launch the distributed training job
trainer = ray.train.torch.TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
    train_loop_config=train_loop_config,
    run_config=run_config,
)

print("Starting FSDP training job...")
result = trainer.fit()
print("Training completed successfully!")

```

    Starting FSDP training job...


    2025-08-29 12:58:31,501	INFO worker.py:1768 -- Connecting to existing Ray cluster at address: 10.0.32.71:6379...
    2025-08-29 12:58:31,513	INFO worker.py:1939 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-2tq4lu3kdll2ayzdmkgw6lzt3y.i.anyscaleuserdata.com [39m[22m
    2025-08-29 12:58:31,714	INFO packaging.py:380 -- Pushing file package 'gcs://_ray_pkg_c9f407640d38da9de67604a90e9e88004763278a.zip' (82.15MiB) to Ray cluster...
    2025-08-29 12:58:32,113	INFO packaging.py:393 -- Successfully pushed file package 'gcs://_ray_pkg_c9f407640d38da9de67604a90e9e88004763278a.zip'.
    /home/ray/anaconda3/lib/python3.11/site-packages/ray/_private/worker.py:1987: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
      warnings.warn(
    [36m(TrainController pid=33120)[0m [State Transition] INITIALIZING -> SCHEDULING.
    [36m(TrainController pid=33120)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2
    [36m(TrainController pid=33120)[0m Using blocking ray.get inside async actor. This blocks the event loop. Please use `await` on object ref with asyncio.gather if you want to yield execution to the event loop instead.
    [36m(TrainController pid=33120)[0m [FailurePolicy] Decision: FailureDecision.RETRY, Error source: controller, Error count / maximum errors allowed: 1/inf, Error: Training failed due to controller error:
    [36m(TrainController pid=33120)[0m The worker group startup timed out after 30.0 seconds waiting for 2 workers. Potential causes include: (1) temporary insufficient cluster resources while waiting for autoscaling (ignore this warning in this case), (2) infeasible resource request where the provided `ScalingConfig` cannot be satisfied), and (3) transient network issues. Set the RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S environment variable to increase the timeout.
    [36m(TrainController pid=33120)[0m [State Transition] SCHEDULING -> RESCHEDULING.
    [36m(TrainController pid=33120)[0m [State Transition] RESCHEDULING -> SCHEDULING.
    [36m(TrainController pid=33120)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2
    [36m(TrainController pid=33120)[0m [FailurePolicy] Decision: FailureDecision.RETRY, Error source: controller, Error count / maximum errors allowed: 2/inf, Error: Training failed due to controller error:
    [36m(TrainController pid=33120)[0m The worker group startup timed out after 30.0 seconds waiting for 2 workers. Potential causes include: (1) temporary insufficient cluster resources while waiting for autoscaling (ignore this warning in this case), (2) infeasible resource request where the provided `ScalingConfig` cannot be satisfied), and (3) transient network issues. Set the RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S environment variable to increase the timeout.
    [36m(TrainController pid=33120)[0m [State Transition] SCHEDULING -> RESCHEDULING.
    [36m(TrainController pid=33120)[0m [State Transition] RESCHEDULING -> SCHEDULING.
    [36m(TrainController pid=33120)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Setting up process group for: env:// [rank=0, world_size=2]
    [36m(TrainController pid=33120)[0m Started training worker group of size 2: 
    [36m(TrainController pid=33120)[0m - (ip=10.0.24.158, pid=11421) world_rank=0, local_rank=0, node_rank=0
    [36m(TrainController pid=33120)[0m - (ip=10.0.63.54, pid=10437) world_rank=1, local_rank=0, node_rank=1
    [36m(TrainController pid=33120)[0m [State Transition] SCHEDULING -> RUNNING.
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Initializing Vision Transformer model...
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Applying FSDP2 sharding to model...
      0%|          | 0.00/26.4M [00:00<?, ?B/s]54)[0m 
      0%|          | 32.8k/26.4M [00:00<01:51, 238kB/s]
      0%|          | 65.5k/26.4M [00:00<01:51, 237kB/s]
      0%|          | 98.3k/26.4M [00:00<01:51, 237kB/s]
    100%|██████████| 26.4M/26.4M [00:02<00:00, 11.9MB/s]
    100%|██████████| 29.5k/29.5k [00:00<00:00, 214kB/s] 
    100%|██████████| 29.5k/29.5k [00:00<00:00, 214kB/s]
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Initializing Vision Transformer model...
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Applying FSDP2 sharding to model...
      0%|          | 0.00/4.42M [00:00<?, ?B/s][32m [repeated 5x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
      2%|▏         | 98.3k/4.42M [00:00<00:18, 234kB/s][32m [repeated 41x across cluster][0m
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m /tmp/ray/session_2025-08-29_11-18-58_966078_2389/runtime_resources/pip/2dc646f4cea92923d9b211dd039da1e14fd3e129/virtualenv/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m   warn("Profiler won't be using warmup, this can skew profiler results")
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m ERROR: External init callback must run in same thread as registerClient (-1282410944 != 1599129408)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Saving checkpoint and reporting metrics...
    100%|██████████| 5.15k/5.15k [00:00<00:00, 42.6MB/s][32m [repeated 2x across cluster][0m
    100%|██████████| 4.42M/4.42M [00:01<00:00, 3.15MB/s][32m [repeated 9x across cluster][0m
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m /tmp/ray/session_2025-08-29_11-18-58_966078_2389/runtime_resources/pip/2dc646f4cea92923d9b211dd039da1e14fd3e129/virtualenv/lib/python3.11/site-packages/torch/profiler/profiler.py:509: UserWarning: Profiler won't be using warmup, this can skew profiler results
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m   warn("Profiler won't be using warmup, this can skew profiler results")
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m ERROR: External init callback must run in same thread as registerClient (2139088448 != 843249472)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-00-50.598902)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint saved successfully. Metrics: {'loss': 0.9074613530585106, 'epoch': 0}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m {'loss': 0.9220723902925532, 'epoch': 0}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Saving checkpoint and reporting metrics...[32m [repeated 2x across cluster][0m
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-00-50.598902)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint saved successfully. Metrics: {'loss': 0.9220723902925532, 'epoch': 0}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-01-13.436430)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint saved successfully. Metrics: {'loss': 0.6794438788231383, 'epoch': 1}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m {'loss': 0.6911060089760638, 'epoch': 1}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Saving checkpoint and reporting metrics...[32m [repeated 2x across cluster][0m
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-01-13.436430)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint saved successfully. Metrics: {'loss': 0.6911060089760638, 'epoch': 1}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-01-37.358961)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint saved successfully. Metrics: {'loss': 0.5813783036901595, 'epoch': 2}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m {'loss': 0.5903209496897163, 'epoch': 2}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Saving checkpoint and reporting metrics...[32m [repeated 2x across cluster][0m
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-01-37.358961)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint saved successfully. Metrics: {'loss': 0.5903209496897163, 'epoch': 2}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-02-00.819530)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint saved successfully. Metrics: {'loss': 0.5233754259474734, 'epoch': 3}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m {'loss': 0.5298319065824468, 'epoch': 3}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Saving checkpoint and reporting metrics...[32m [repeated 2x across cluster][0m
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-02-00.819530)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint saved successfully. Metrics: {'loss': 0.5298319065824468, 'epoch': 3}
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-02-23.764725)
    [36m(RayTrainWorker pid=10437, ip=10.0.63.54)[0m Checkpoint saved successfully. Metrics: {'loss': 0.4832368891289894, 'epoch': 4}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m {'loss': 0.4901455493683511, 'epoch': 4}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m generated new fontManager
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Saving checkpoint and reporting metrics...
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/checkpoint_2025-08-29_13-02-23.764725)
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint saved successfully. Metrics: {'loss': 0.4901455493683511, 'epoch': 4}
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Preparing model for inference...
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Successfully retrieved complete model state dict
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Saved complete model to /tmp/tmp019hes64/full-model.pt
    [36m(RayTrainWorker pid=11421, ip=10.0.24.158)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/fsdp_mnist_ac7573f9/full_model)
    [36m(TrainController pid=33120)[0m [State Transition] RUNNING -> FINISHED.


    Training completed successfully!


    [36m(autoscaler +1h49m20s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.


## 10. Loading the Trained Model for Inference

After training completes, we can load the saved model for inference on new data. The model is loaded in its unsharded form, ready for standard PyTorch inference.


```python
# Update this path to match your trained model location
# The path follows the pattern: /mnt/cluster_storage/{experiment_name}/full_model/full-model.pt
PATH_TO_FULL_MODEL = "/mnt/cluster_storage/fsdp_mnist_16b0e0c2/full_model/full-model.pt"
```


```python
# Initialize the same model architecture for inference
print("Loading trained model for inference...")
model = init_model()

# Load the trained weights 
state_dict = torch.load(PATH_TO_FULL_MODEL, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
```

    Loading trained model for inference...





    VisionTransformer(
      (conv_proj): Conv2d(1, 64, kernel_size=(7, 7), stride=(7, 7))
      (encoder): Encoder(
        (dropout): Dropout(p=0.0, inplace=False)
        (layers): Sequential(
          (encoder_layer_0): EncoderBlock(
            (ln_1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=64, out_features=128, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=128, out_features=64, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_1): EncoderBlock(
            (ln_1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=64, out_features=128, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=128, out_features=64, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_2): EncoderBlock(
            (ln_1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=64, out_features=128, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=128, out_features=64, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
          (encoder_layer_3): EncoderBlock(
            (ln_1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (self_attention): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (ln_2): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            (mlp): MLPBlock(
              (0): Linear(in_features=64, out_features=128, bias=True)
              (1): GELU(approximate='none')
              (2): Dropout(p=0.0, inplace=False)
              (3): Linear(in_features=128, out_features=64, bias=True)
              (4): Dropout(p=0.0, inplace=False)
            )
          )
        )
        (ln): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      )
      (heads): Sequential(
        (head): Linear(in_features=64, out_features=10, bias=True)
      )
    )




```python
# Load the test data
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
test_data = FashionMNIST(
    root=".", train=False, download=True, transform=transform
)
test_data
```




    Dataset FashionMNIST
        Number of datapoints: 10000
        Root location: .
        Split: Test
        StandardTransform
    Transform: Compose(
                   ToTensor()
                   Normalize(mean=(0.5,), std=(0.5,))
               )




```python
# Test model inference
model.eval()
with torch.no_grad():
    out = model(test_data.data[0].reshape(1, 1, 28, 28).float())
    predicted_label = out.argmax().item()
    test_label = test_data.targets[0].item()
    print(f"{predicted_label=} {test_label=}")
```

    predicted_label=8 test_label=9


## Summary

In this tutorial, you: 

- Trained an image classification model using FSDP and Ray Train
- Learned how to load and save distributed checkpoints with PyTorch DCP
- Gained insight on configuring FSDP to balance training performance and memory usage
- Unlocked multi-node GPU memory observability with PyTorch Profiler
