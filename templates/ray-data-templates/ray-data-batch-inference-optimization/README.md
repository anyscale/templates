# ML batch inference optimization with Ray Data

**Time to complete**: 20 min | **Difficulty**: Intermediate | **Prerequisites**: Basic ML knowledge, Python experience

Learn how to optimize ML batch inference by comparing inefficient and efficient approaches. This template shows common pitfalls and how Ray Data solves them.

## Table of Contents

1. [Setup and Data](#setup) (3 min)
2. [Inefficient Approach](#inefficient-approach) (5 min) 
3. [Optimized with Ray Data](#optimized-approach) (8 min)
4. [Performance Comparison](#performance-comparison) (4 min)

## Learning Objectives

**Ray Data batch processing**: Learn how [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) - Ray Data's core transformation function - improves inference efficiency by processing data in optimized chunks  
**Actor-based optimization**: Use [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html) - stateful distributed processes - to avoid repeated model loading across batches  
**Batch size tuning**: Understand how [batch size configuration](https://docs.ray.io/en/latest/data/performance-tips.html#batch-size) affects performance and resource utilization  
**GPU utilization**: Optimize for distributed GPU workers using Ray's [resource specification system](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)

## Overview

**The Problem**: Naive batch inference approaches suffer from inefficient model loading, poor batching, and resource waste. Traditional approaches often load models repeatedly, use suboptimal batch sizes, and fail to leverage distributed computing resources effectively.

**The Solution**: Ray Data provides distributed batch processing capabilities that address these common inefficiencies. Through its [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) function and integration with [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html), Ray Data enables efficient model loading, optimal batching, and distributed resource utilization.

**Key Ray Data Features for Inference**: Ray Data's [batch inference capabilities](https://docs.ray.io/en/latest/data/batch_inference.html) are specifically designed for ML workloads, providing automatic optimization and scalability that traditional data processing frameworks lack.

## Setup

```python
import ray
import time
import numpy as np

# Initialize Ray cluster
ray.init()

print("Ray cluster initialized")
print(f"Available resources: {ray.cluster_resources()}")
print("Use Ray Dashboard to monitor performance")
```

### Load Demo Dataset

Ray Data provides native support for [reading images](https://docs.ray.io/en/latest/data/loading-data.html#loading-images) directly from cloud storage. We'll use the Imagenette dataset, which is a subset of ImageNet with 10 classes, perfect for demonstrating batch inference optimization patterns.

```python
# Load real ImageNet dataset using Ray Data's native image reader
dataset = ray.data.read_images(
    "s3://ray-benchmark-data/imagenette2/train/",
    mode="RGB"  # Load as RGB color images
).limit(1000)  # 1K images for demo

print("Loaded ImageNet dataset for batch inference demo")
print("Sample dataset:")
sample_batch = dataset.take_batch(3)
print(f"Batch contains {len(sample_batch['image'])} images")
print(f"Image shape: {sample_batch['image'][0].shape}")

# Ray Data automatically handles image loading and format conversion
# See Ray Data documentation: https://docs.ray.io/en/latest/data/working-with-images.html
```

## Inefficient Approach

### The Wrong Way: Loading Model in Every Batch

This approach demonstrates a common anti-pattern in batch inference that leads to severe performance problems. Understanding why this approach fails helps illustrate the value of Ray Data's optimization features.

**The Core Problem**: Loading machine learning models is expensive - it involves reading large files from disk, initializing neural networks, and allocating GPU memory. When this happens repeatedly for every batch, it creates massive overhead that dominates inference time.

```python
def inefficient_inference(batch):
    """INEFFICIENT: Loads model for every single batch."""
    # This is very slow - model loads repeatedly!
    from transformers import pipeline
import time

    print("Loading model... (this happens for every batch!)")
    start_load = time.time()
    
    # Model loading happens for every batch - very inefficient
    classifier = pipeline("image-classification", 
                         model="microsoft/resnet-50")
    
    load_time = time.time() - start_load
    print(f"Model loading took: {load_time:.2f} seconds")
    
    # Process images one by one (also inefficient)
    results = []
    for image in batch["image"]:
        prediction = classifier(image)
        results.append({
            "prediction": prediction[0]["label"],
            "confidence": prediction[0]["score"]
        })
    
    return results

print("Testing inefficient approach...")
print("Watch Ray Dashboard to see the performance problems")

start_time = time.time()

# Run inefficient batch inference with small batches
# map_batches: Ray Data's core transformation function for applying operations to data chunks
# See: https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html
inefficient_results = dataset.limit(100).map_batches(
    inefficient_inference,
    batch_size=4,  # Small batch size demonstrates poor GPU utilization
    concurrency=2  # Concurrency: number of parallel tasks processing batches simultaneously
).take(20)  # take(): executes the lazy dataset and retrieves specified number of results

inefficient_time = time.time() - start_time
print(f"\nInefficient approach completed in: {inefficient_time:.2f} seconds")
print("Problems: Model loads repeatedly, poor batching, wasted resources")
```

**What's wrong here:**
- Model loads for every batch (extremely slow)
- Small batch sizes don't utilize resources efficiently
- Images processed individually instead of in batches
- Poor resource utilization across distributed workers

## Optimized Approach

### The Right Way: Actor-Based Model Loading

Ray Data solves the model loading problem through integration with [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html). Actors are stateful processes that can load a model once and reuse it across many batches, eliminating the repeated loading overhead.

**Actor Benefits for Batch Inference**: Ray actors provide persistent state across multiple function calls, making them ideal for batch inference where you want to load expensive models once and reuse them. The [Ray Data batch inference guide](https://docs.ray.io/en/latest/data/batch_inference.html) explains these patterns in detail.

**GPU Resource Allocation**: Ray's [resource allocation system](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) allows you to specify GPU requirements for actors using the `num_gpus` parameter, ensuring efficient GPU utilization across distributed workers.

```python
# EFFICIENT: Use Ray actors to load model once per worker
# @ray.remote: decorator that converts a class into a distributed Ray actor
# num_gpus=1: resource specification allocating one GPU per actor instance
# See: https://docs.ray.io/en/latest/ray-core/actors.html
@ray.remote(num_gpus=1)  # Allocate GPU to actor
class OptimizedInferenceActor:
    """Stateful actor that loads model once and reuses it."""
    
    def __init__(self):
        """Load model once when actor starts."""
        from transformers import pipeline
        import torch
        
        print("Loading model once per actor...")
        
        # Load model with GPU support if available
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("image-classification",
                                  model="microsoft/resnet-50",
                                  device=device)
        print("Model loaded and ready for efficient inference")
    
    def predict_batch(self, images):
        """Process entire batch of images efficiently."""
        # Process all images in batch (much faster than one-by-one)
        predictions = []
        for image in images:
            result = self.classifier(image)
            predictions.append({
                "prediction": result[0]["label"],
                "confidence": result[0]["score"]
            })
        
        return predictions

# Create actors for distributed inference
print("Creating optimized inference actors...")
# Actor pool: multiple actor instances for parallel processing
# .remote(): creates a new actor instance on a distributed worker
num_actors = 2
actors = [OptimizedInferenceActor.remote() for _ in range(num_actors)]

def optimized_inference(batch):
    """Efficient inference using pre-loaded actors."""
    # Distribute work across available actors
    actor_idx = hash(str(batch)) % len(actors)
    actor = actors[actor_idx]
    
    # Process entire batch efficiently using actor method
    # ray.get(): retrieves results from remote actor function calls
    # .remote(): executes actor method on the distributed worker
    # See: https://docs.ray.io/en/latest/ray-core/actors.html#calling-actor-methods
    results = ray.get(actor.predict_batch.remote(batch["image"]))
    return results

print("Testing optimized approach...")
print("Watch Ray Dashboard to see improved performance")
    
    start_time = time.time()
    
# Run optimized batch inference with larger batches
# Lazy execution: Ray Data builds execution plan without immediate processing
# See: https://docs.ray.io/en/latest/data/key-concepts.html#lazy-execution
optimized_results = dataset.limit(100).map_batches(
    optimized_inference,
    batch_size=16,  # Larger batch size for better GPU utilization
    concurrency=4   # Higher concurrency for distributed parallelism
).take(20)  # take(): triggers execution and retrieves results

optimized_time = time.time() - start_time
print(f"\nOptimized approach completed in: {optimized_time:.2f} seconds")
print("Improvements: Model loads once, better batching, efficient resource use")
```

**What's better:**
- Model loads only once per actor (much faster)
- Larger batch sizes for better resource utilization
- Proper GPU allocation with `num_gpus=1`
- Distributed processing across multiple workers

### GPU Acceleration for Data Preprocessing

:::tip NVIDIA RAPIDS cuDF for Pandas Operations
If your batch inference includes complex pandas data preprocessing, you can accelerate it with **NVIDIA RAPIDS cuDF**. Simply replace `import pandas as pd` with `import cudf as pd` in your `map_batches` functions to leverage GPU acceleration for DataFrame operations.
:::

```python
# Example: Batch inference with GPU-accelerated preprocessing
def gpu_accelerated_preprocessing(batch):
    """Image preprocessing with optional cuDF acceleration.
    
    For GPU acceleration, replace 'import pandas as pd' with 'import cudf as pd'
    to speed up complex DataFrame operations.
    """
    import pandas as pd  # or 'import cudf as pd' for GPU acceleration
    
    # Convert batch to DataFrame for preprocessing
    df = pd.DataFrame(batch)
    
    # Complex preprocessing that benefits from GPU acceleration
    df['image_processed'] = True
    df['batch_id'] = range(len(df))
    
    return df.to_dict('records')

print("GPU acceleration available for complex pandas preprocessing")
```

## Performance Comparison

### Analyzing the Results

```python
# Compare performance between approaches
print("Performance Comparison:")
print(f"Inefficient approach: {inefficient_time:.2f} seconds")
print(f"Optimized approach: {optimized_time:.2f} seconds")

if inefficient_time > 0 and optimized_time > 0:
    speedup = inefficient_time / optimized_time
    print(f"Speedup achieved: {speedup:.1f}x faster")
else:
    print("Run both approaches to see performance comparison")

print("\nKey optimizations:")
print("- Ray actors for efficient model loading")
print("- Larger batch sizes for better throughput")
print("- GPU resource allocation for inference workers")
print("- Distributed processing across cluster")
```

### Optimized Batch Size Configuration

Batch size is a critical parameter in Ray Data batch inference that affects both performance and resource utilization. The [performance tips guide](https://docs.ray.io/en/latest/data/performance-tips.html#batch-size) provides detailed guidance on choosing optimal batch sizes.

**Batch Size Best Practice**: For image classification models, batch sizes of 16-32 typically provide good GPU utilization while avoiding memory issues. Ray Data's [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) function handles batching automatically once you specify the size.

```python
# Demonstrate optimal batch size configuration
print("Using optimized batch size for better performance")

# Use batch_size=32 for optimal GPU utilization
# limit(): creates a new dataset with only the first N elements (lazy operation)
# See: https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.limit.html
optimized_batch_results = dataset.limit(200).map_batches(
    optimized_inference,
    batch_size=32,  # Optimal batch size for GPU memory and throughput balance
    concurrency=4   # Higher concurrency leverages multiple distributed workers
).take(100)  # take(): executes the pipeline and returns results

print("Optimized batch processing completed")
print("Check Ray Dashboard to see improved task execution and GPU utilization")
```

### Ray Dashboard Monitoring

The [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html) provides comprehensive monitoring capabilities specifically designed for distributed Ray workloads. For batch inference, the dashboard is invaluable for understanding performance characteristics and identifying optimization opportunities.

**Key Dashboard Features for Batch Inference**: The [Ray Data section](https://docs.ray.io/en/latest/ray-observability/user-guides/add-app-metrics.html) of the dashboard shows execution plans, block processing statistics, and operator performance. The [Jobs page](https://docs.ray.io/en/latest/ray-observability/getting-started.html#jobs-view) displays task execution timelines and resource utilization across workers.

```python
# Ray Dashboard provides comprehensive monitoring
print("\nRay Dashboard Monitoring:")
print("The Ray Dashboard shows detailed performance metrics:")
print("- Task execution timelines and worker utilization")
print("- GPU utilization across distributed workers (when available)")
print("- Memory usage and object store statistics")
print("- Ray Data execution plans and block processing optimization")
print("\nAccess your dashboard to see real-time performance insights")
print("Dashboard URL available via ray.get_dashboard_url()")

# The dashboard automatically tracks all the metrics you need
# No custom monitoring code required - Ray handles everything!
```

## Key Takeaways

**Actor pattern is essential**: Ray's [stateful actors](https://docs.ray.io/en/latest/ray-core/actors.html) enable loading models once per worker, providing significant performance improvements over repeated model loading. This pattern is fundamental to efficient distributed inference.

**Batch size optimization**: Proper batch sizing affects both performance and resource utilization. The [Ray Data performance guide](https://docs.ray.io/en/latest/data/performance-tips.html#batch-size) provides detailed guidance on choosing optimal batch sizes for your workload.

**Resource allocation**: Ray's [resource specification system](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) allows precise control over GPU allocation using parameters like `num_gpus=1`, ensuring efficient resource utilization across distributed workers.

**Ray Dashboard monitoring**: The [Ray observability system](https://docs.ray.io/en/latest/ray-observability/getting-started.html) provides comprehensive monitoring without requiring custom performance tracking code.

## Action Items

1. **Apply actor patterns**: Use Ray actors for your own model loading scenarios
2. **Experiment with batch sizes**: Test different batch sizes to find optimal performance
3. **Monitor with Ray Dashboard**: Use the dashboard to understand performance characteristics
4. **Add GPU acceleration**: Use RAPIDS cuDF for complex pandas preprocessing when needed

## Next Steps

**Advanced Ray Data features**: Explore Ray Data's [streaming execution](https://docs.ray.io/en/latest/data/performance-tips.html#streaming-execution) and [pipeline optimization](https://docs.ray.io/en/latest/data/performance-tips.html#optimizing-performance) capabilities for even better performance.

**Ray Serve integration**: Learn [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for real-time model serving and online inference applications that complement batch processing workflows.

**Multi-GPU scaling**: Scale inference across multiple GPU workers using Ray's [resource management](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) for larger workloads and more complex models.

**Production deployment**: Explore [Ray clustering](https://docs.ray.io/en/latest/cluster/getting-started.html) options for deploying optimized batch inference pipelines at scale.

---

*The [Ray Dashboard](https://docs.ray.io/en/latest/ray-observability/getting-started.html) provides all the performance monitoring you need - focus on optimizing your Ray Data usage patterns.*
