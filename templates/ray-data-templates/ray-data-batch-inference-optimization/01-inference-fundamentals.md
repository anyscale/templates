# Part 1: Inference Fundamentals

**⏱️ Time to complete**: 20 min

**[← Back to Overview](README.md)** | **[Continue to Part 2 →](02-advanced-optimization.md)**

---

## What You'll Learn

In this part, you'll understand the fundamentals of batch inference optimization by comparing inefficient and efficient approaches:
- How to set up Ray Data for accelerated inference (CPU or GPU)
- Why naive inference patterns create performance bottlenecks
- How Ray Data's actor-based pattern solves these problems
- How to implement optimized inference with proper resource allocation for both CPU and GPU

## Table of Contents

1. [Introduction and Setup](#introduction-and-setup)
2. [The Wrong Way: Inefficient Batch Inference](#the-wrong-way-inefficient-batch-inference)
3. [Why the Naive Approach Fails](#why-the-naive-approach-fails)
4. [The Right Way: Optimized with Ray Data](#the-right-way-optimized-with-ray-data)

---

## Introduction and Setup

Batch inference is the process of running ML model predictions on large batches of data. While this sounds straightforward, naive implementations create severe performance bottlenecks that prevent production deployment. This part shows you the difference between inefficient and optimized approaches using real-world examples.

### What You'll Learn

By comparing inefficient and optimized implementations, you'll understand:
- **Why** repeated model loading destroys performance
- **How** Ray Data's actor pattern solves the problem
- **When** to apply specific optimization techniques
- **What** parameters to tune for your workload

### Initial Setup

```python
import ray
import torch
import numpy as np
from PIL import Image
import time

# Initialize Ray for distributed processing
ray.init()

# Configure Ray Data for optimal performance monitoring
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

print("Ray cluster initialized for batch inference optimization")
print(f"Available resources: {ray.cluster_resources()}")

# Detect hardware availability
HAS_GPU = torch.cuda.is_available()
device = torch.device("cuda" if HAS_GPU else "cpu")

print(f"Using device: {device}")
if HAS_GPU:
    print(f"GPU count: {torch.cuda.device_count()}")
    print("✅ GPU detected - examples will use GPU acceleration")
else:
    print("ℹ️  No GPU detected - examples will run on CPU")
    print("   (All patterns work identically on CPU, just use num_cpus instead of num_gpus)")
```

:::note CPU and GPU Compatibility
**This template works on both CPU-only and GPU clusters!**

- **GPU clusters**: Code automatically detects GPUs and uses `num_gpus=1` for acceleration
- **CPU clusters**: Code falls back to CPU and uses `num_cpus=2` for parallelism

All optimization concepts (actor-based loading, batching, concurrency) apply equally to both environments.
:::

### Load Demo Dataset

For this demonstration, you'll use the Imagenette dataset, which provides a realistic subset of ImageNet with 10 classes.

```python
# Load real ImageNet dataset for batch inference demonstration
dataset = ray.data.read_images(
    "s3://ray-benchmark-data/imagenette2/train/",
    mode="RGB",  # Ensure consistent RGB color format
    num_cpus=0.05
).limit(1000)  # Use 1K images for focused performance comparison

print("Loaded ImageNet dataset for batch inference demo")
print("Sample dataset:")
sample_batch = dataset.take_batch(3)
print(f"Batch contains {len(sample_batch['image'])} images")
print(f"Image shape: {sample_batch['image'][0].shape}")
```

---

## The Wrong Way: Inefficient Batch Inference

This section demonstrates a common anti-pattern in ML inference systems. Understanding why this approach fails is essential before learning the optimized solution.

When models are loaded repeatedly for each batch, the initialization overhead dominates processing time. This pattern is unfortunately common in production systems where developers haven't considered the cost of model loading operations.

```python
def inefficient_inference(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """INEFFICIENT: Loads model for every single batch.
    
    Anti-pattern demonstration - DO NOT use this approach in production!
    This function intentionally shows bad practices to highlight optimization opportunities.
    
    Args:
        batch: Dictionary containing 'image' key with list of PIL Images
        
    Returns:
        List of prediction dictionaries with 'prediction' and 'confidence' keys
    """
    import time
    from transformers import pipeline
    
    # ❌ BAD: Model loading happens inside function - repeats for every batch!
    print("Loading model... (this happens for every batch!)")
    start_load = time.time()
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    load_time = time.time() - start_load
    print(f"Model loading took: {load_time:.2f} seconds")
    
    # ❌ BAD: Processing images one by one instead of batched inference
    results = []
    for image in batch["image"]:
        prediction = classifier(image)
        results.append({
            "prediction": prediction[0]["label"],
            "confidence": prediction[0]["score"]
        })
    
    return results

print("Testing inefficient approach...")
print("💡 Watch Ray Dashboard to see the performance problems")

# Run inefficient batch inference with small batches
inefficient_results = dataset.limit(100).map_batches(
    inefficient_inference,
    batch_size=4,
    concurrency=2
).take(20)

print("✅ Inefficient approach completed")
print("   Problems: repeated model loading, poor batching, wasted resources")
```

**Expected issues:**
```
Loading model... (this happens for every batch!)
Model loading took: 3.45 seconds
Loading model... (this happens for every batch!)
Model loading took: 3.52 seconds
[repeated 25 times...]
```

:::caution Performance Anti-Pattern
❌ Model loads 25 times (one per batch)  
❌ Each load takes 3+ seconds = 87.5 seconds wasted  
❌ CPU/GPU mostly idle waiting for model loading  
❌ Total throughput: ~1 image/second (unacceptable)
:::

---

## Why the Naive Approach Fails

Now that you've seen the inefficient implementation, you can understand exactly why it performs poorly.

### Performance Bottlenecks Explained

The inefficient approach suffers from three critical problems:

#### Problem 1: Repeated Model Loading

**What happens**: The model loads from scratch for every batch of 4 images.

**Why it's expensive**:
- Model weights file: 100-500 MB download per load
- Neural network initialization: 2-5 seconds of setup
- GPU memory allocation: Repeated allocation/deallocation cycles
- Wasted overhead: Model loading time >> actual inference time

**Impact**: If model loading takes 3 seconds and inference takes 0.1 seconds, you're spending 97% of time on overhead!

#### Problem 2: Poor Batch Utilization

**What happens**: Processing only 4 images at a time with individual processing.

**Why it's inefficient**:
- GPU underutilization: Modern GPUs can handle 32-128 images simultaneously
- Memory waste: Using <10% of available GPU memory
- No vectorization: Processing images one-by-one instead of batched tensors
- Task overhead: Creating many small tasks instead of fewer large ones

**Impact**: GPU sits idle 90% of the time waiting for data.

#### Problem 3: Inefficient Resource Allocation

**What happens**: Low concurrency with default settings.

**Why it creates bottlenecks**:
- Limited parallelism: Only 2 concurrent workers
- Unbalanced pipeline: Preprocessing can't keep up with potential GPU throughput
- Resource waste: CPU cores sit idle while waiting for model loading

**Impact**: Cluster resources are underutilized, extending total processing time.

### Performance Anti-pattern Summary

| Anti-Pattern | Why It's Bad | Typical Impact |
|--------------|-------------|----------------|
| **Model loading per batch** | Initialization overhead >> inference time | 10-100x slower |
| **Small batch sizes** | GPU memory underutilized | 5-10x slower |
| **Sequential processing** | No vectorization benefits | 3-5x slower |
| **Low concurrency** | Limited parallelism | 2-4x slower |

**Combined effect**: These anti-patterns compound, making the inefficient approach significantly slower than optimized implementations.

---

## The Right Way: Optimized with Ray Data

Now you can see how Ray Data solves these problems with actor-based inference, proper batching, and optimized resource allocation.

Ray Data solves the model loading problem by letting you run stateful, class-based `map_batches` with an actor pool strategy. Each worker loads the model once and reuses it across many batches, eliminating repeated initialization overhead.

```python
from typing import Dict, Any, List

# Efficient: Use Ray Data class-based map_batches with optimized actor configuration

class InferenceWorker:
    """Stateful worker that loads the model once and reuses it.
    
    This is the CORRECT pattern for batch inference - model loads once in __init__
    and is reused across all batches processed by this worker.
    
    Works on both CPU and GPU - automatically detects hardware.
    """
    
    def __init__(self):
        """Initialize worker - called once per actor, not per batch!"""
        from transformers import pipeline
        import torch
        
        # Automatically use GPU if available, otherwise CPU
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "image-classification",
            model="microsoft/resnet-50",
            device=device,
        )
        print(f"✅ Model loaded on: {'GPU' if device >= 0 else 'CPU'}")

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a batch of images - called many times, reuses loaded model.
        
        Args:
            batch: Dictionary containing 'image' key with list of PIL Images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in batch["image"]:
            pred = self.classifier(image)
            results.append({
                "prediction": pred[0]["label"],
                "confidence": pred[0]["score"],
            })
        return results

print("Running optimized Ray Data inference with stateful workers...")

# Best practice: Use the new concurrency parameter for actor-based processing
# Resource allocation adapts to available hardware
try:
    inference_results = dataset.limit(100).map_batches(
        InferenceWorker,
        concurrency=2,      # Number of parallel actors
        num_gpus=1 if HAS_GPU else 0,  # Allocate GPU if available
        num_cpus=2 if not HAS_GPU else 1,  # Use more CPU cores if no GPU
        batch_size=16,      # Optimal batch size for resource utilization
    ).take(20)
    
    print("✅ Optimized approach completed successfully!")
    print("   Improvements: single model load per worker, better batching, efficient resource use")
    print(f"   Processed {len(inference_results)} images")
    
except Exception as e:
    print(f"❌ Error during inference: {e}")
    print("   Check that transformers and torch are installed")
    raise
```

**What's Better:**
- Model loads only once per worker via Ray Data `ActorPoolStrategy`
- Larger batch sizes for better resource utilization
- Proper resource allocation with `num_gpus=1` (GPU) or `num_cpus=2` (CPU)
- Ray Data manages distribution across workers
- **Works identically on CPU and GPU clusters**

:::tip Resource Allocation Patterns
**GPU clusters**: Use `num_gpus=1` to allocate one GPU per actor
```python
.map_batches(InferenceWorker, num_gpus=1, concurrency=2)  # 2 GPUs used
```

**CPU clusters**: Use `num_cpus=2` to allocate CPU cores per actor
```python
.map_batches(InferenceWorker, num_cpus=2, concurrency=4)  # 8 CPU cores used
```

The patterns are identical - Ray Data abstracts away the hardware differences!
:::

### Performance Expectations: CPU vs GPU

:::tip Performance Scaling
**Both CPU and GPU deployments benefit from Ray Data optimizations!**

**GPU clusters** (when available):
- **Throughput**: 500-2000 images/second (model dependent)
- **Best for**: Large models, high-volume inference
- **Resource**: Use `num_gpus=1` per actor

**CPU clusters** (always available):
- **Throughput**: 50-200 images/second (model dependent)
- **Best for**: Development, smaller models, cost-sensitive workloads
- **Resource**: Use `num_cpus=2-4` per actor

**Key insight**: The optimization patterns (actor-based loading, batching, concurrency) provide similar relative speedups on both CPU and GPU!
:::

```python
# Example: Adaptive resource allocation based on available hardware
def create_inference_config():
    """Generate optimal configuration for available hardware."""
    import torch
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        return {
            "num_gpus": 1,
            "concurrency": torch.cuda.device_count(),
            "batch_size": 32,  # Larger batches for GPU
            "expected_throughput": "500-2000 images/sec"
        }
    else:
        return {
            "num_cpus": 2,
            "concurrency": ray.available_resources()["CPU"] // 2,
            "batch_size": 16,  # Smaller batches for CPU
            "expected_throughput": "50-200 images/sec"
        }

config = create_inference_config()
print(f"Optimized configuration for your cluster: {config}")
```

---

## Key Takeaways from Part 1

You've learned the fundamentals of batch inference optimization:
- - Identified common anti-patterns that destroy performance
- - Understood why repeated model loading is catastrophic
- - Implemented class-based actors for stateful model loading
- - Used proper resource allocation with `num_gpus` and `concurrency`

## Next Steps

Now that you understand the fundamentals, you're ready to learn systematic optimization techniques.

**[Continue to Part 2: Advanced Optimization →](02-advanced-optimization.md)**

In Part 2, you'll learn:
- Systematic decision frameworks for choosing optimization techniques
- Multi-model ensemble inference patterns
- Performance monitoring and diagnostics
- Production deployment best practices

**Or skip ahead to Part 3** for a deep dive into Ray Data's architecture:

**[Jump to Part 3: Ray Data Architecture →](03-ray-data-architecture.md)**

In Part 3, you'll learn:
- How streaming execution enables unlimited dataset processing
- How blocks and memory management affect optimization
- How operator fusion and backpressure work
- How to calculate optimal parameters from architectural principles

**[Return to overview](README.md)** to see all available parts.

---

## Cleanup

```python
# Clean up Ray resources
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```

