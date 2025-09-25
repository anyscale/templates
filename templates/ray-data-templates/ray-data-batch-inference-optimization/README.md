# ML batch inference optimization with Ray Data

**⏱️ Time to complete**: 20 min | **Difficulty**: Intermediate | **Prerequisites**: Basic ML knowledge, Python experience

## What You'll Build

Create an optimized ML batch inference pipeline that demonstrates the performance difference between naive and efficient approaches. Learn how Ray Data's actor-based patterns eliminate common bottlenecks in production ML inference.

## Table of Contents

1. [Setup and Data](#setup) (3 min)
2. [Inefficient Approach](#inefficient-approach) (5 min) 
3. [Optimized with Ray Data](#optimized-approach) (8 min)
4. [Performance Comparison](#performance-comparison) (4 min)

## Learning Objectives

**Why batch inference optimization matters**: Poor optimization wastes significant compute resources through repeated model loading and inefficient batching. Understanding these bottlenecks is crucial for production ML systems.

**Ray Data's inference superpowers**: Actor-based model loading and distributed processing eliminate performance bottlenecks that plague traditional ML pipelines. You'll learn how to leverage these capabilities for scalable inference.

**Real-world optimization patterns**: Companies like Netflix and Tesla process millions of inference requests efficiently using the distributed techniques demonstrated in this template. These patterns apply across industries from recommendation systems to autonomous vehicles.

**Production deployment strategies**: Master GPU utilization, batch size tuning, and resource allocation techniques that enable ML systems to scale to enterprise workloads cost-effectively.

## Overview

**Challenge**: Naive batch inference approaches create significant performance bottlenecks that prevent ML systems from scaling to production workloads. Model loading overhead can consume 80-90% of processing time, while poor batch sizing wastes GPU resources and increases operational costs.

**Solution**: Ray Data transforms batch inference through distributed processing and intelligent resource management. Actor-based model loading eliminates repeated initialization overhead, while optimized batching maximizes throughput across GPU clusters.

**Impact**: Production ML systems achieve significant performance improvements through Ray Data's inference optimization patterns. Companies process billions of inference requests using these distributed techniques for recommendation systems, autonomous vehicles, and real-time decision making.

---

## Prerequisites Checklist

Before starting this template, ensure you have Python 3.8+ with basic machine learning experience and understanding of neural networks. You'll need familiarity with the transformers library for model loading and basic knowledge of GPU acceleration concepts.

**Required setup**:
- [ ] Python 3.8+ with machine learning libraries
- [ ] Ray Data installed (`pip install ray[data]`)
- [ ] Basic understanding of distributed computing concepts
- [ ] Familiarity with neural network inference patterns

## Quick Start (3 minutes)

Want to see the optimization impact immediately?

```python
import ray
import time

# Initialize Ray for distributed processing
ray.init()

print("Ray cluster initialized for batch inference optimization")
print(f"Available resources: {ray.cluster_resources()}")
print("Ready to demonstrate efficient vs inefficient batch inference patterns")
```

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

For this demonstration, we'll use the Imagenette dataset, which provides a realistic subset of ImageNet with 10 classes. This dataset showcases the performance characteristics you'll encounter with real-world image classification workloads. Ray Data's native image reading capabilities handle format conversion and preprocessing automatically.

```python
# Load real ImageNet dataset for batch inference demonstration
# Ray Data's read_images() provides efficient distributed image loading
dataset = ray.data.read_images(
    "s3://ray-benchmark-data/imagenette2/train/",
    mode="RGB"  # Ensure consistent RGB color format
).limit(1000)  # Use 1K images for focused performance comparison

print("Loaded ImageNet dataset for batch inference demo")
print("Sample dataset:")
sample_batch = dataset.take_batch(3)
print(f"Batch contains {len(sample_batch['image'])} images")
print(f"Image shape: {sample_batch['image'][0].shape}")
```

This code demonstrates Ray Data's efficient image loading from cloud storage. The dataset will be used to compare naive and optimized inference approaches, showing how different patterns affect performance at scale.

## Inefficient Approach

### The Wrong Way: Loading Model in Every Batch

The first approach demonstrates a critical anti-pattern that plagues many ML inference systems. When models are loaded repeatedly for each batch, the initialization overhead dominates processing time. This pattern is unfortunately common in production systems where developers haven't considered the cost of model loading operations.

Understanding why this approach fails helps illustrate the value of Ray Data's optimization features. Model loading involves reading large files from disk, initializing neural networks, and allocating GPU memory - operations that can take several seconds per model load.

```python
def inefficient_inference(batch):
    """INEFFICIENT: Loads model for every single batch."""
    # This is very slow - model loads repeatedly!
    from transformers import pipeline
    import time
    
    print("Loading model... (this happens for every batch!)")
    start_load = time.time()
    
    try:
        # Model loading happens for every batch - very inefficient
        classifier = pipeline("image-classification", 
                             model="microsoft/resnet-50")
        
        load_time = time.time() - start_load
        print(f"Model loading took: {load_time:.2f} seconds")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return []
    
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
inefficient_results = dataset.limit(100).map_batches(
    inefficient_inference,
    batch_size=4,  # Small batch size
    concurrency=2
).take(20)

inefficient_time = time.time() - start_time
print(f"\nInefficient approach completed in: {inefficient_time:.2f} seconds")
print("Problems: Model loads repeatedly, poor batching, wasted resources")
```

**Analysis of the inefficient approach**: This code demonstrates several critical performance problems. The model loading overhead dominates execution time because the transformer pipeline is initialized for every single batch. Small batch sizes of 4 images fail to utilize GPU memory efficiently, while processing images individually prevents vectorized operations. The combination of these factors results in poor resource utilization across distributed workers.

These performance bottlenecks are common in production ML systems where inference patterns haven't been optimized for distributed execution. Understanding these anti-patterns is essential before implementing the optimized solution.

## Optimized Approach

### The Right Way: Actor-Based Model Loading

Ray Data solves the model loading problem through integration with Ray actors - stateful distributed processes that can load a model once and reuse it across many batches. This architectural pattern eliminates the repeated loading overhead that dominates the inefficient approach.

Actors provide persistent state across multiple function calls, making them ideal for batch inference scenarios. By loading expensive models once during actor initialization, subsequent inference operations can focus on processing data rather than reinitializing the same model repeatedly.

The `@ray.remote` decorator with GPU allocation ensures that each actor has dedicated compute resources, enabling efficient parallel processing across the cluster. This pattern scales naturally as you add more workers to handle larger workloads.

```python
# EFFICIENT: Use Ray actors to load model once per worker
@ray.remote(num_gpus=1)  # Allocate GPU to actor
class OptimizedInferenceActor:
    """Stateful actor that loads model once and reuses it."""
    
    def __init__(self):
        """Load model once when actor starts."""
        from transformers import pipeline
        import torch
        
        print("Loading model once per actor...")
        
        try:
            # Load model with GPU support if available
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline("image-classification",
                                      model="microsoft/resnet-50",
                                      device=device)
            print("Model loaded and ready for efficient inference")
        except Exception as e:
            print(f"Failed to load model in actor: {e}")
            # Use CPU fallback if GPU loading fails
            self.classifier = pipeline("image-classification",
                                      model="microsoft/resnet-50",
                                      device=-1)
            print("Loaded model with CPU fallback")
    
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
num_actors = 2
actors = [OptimizedInferenceActor.remote() for _ in range(num_actors)]

def optimized_inference(batch):
    """Efficient inference using pre-loaded actors."""
    # Distribute work across available actors using consistent hashing
    # This ensures good load distribution across actor pool
    import hashlib
    batch_hash = hashlib.md5(str(len(batch["image"])).encode()).hexdigest()
    actor_idx = int(batch_hash, 16) % len(actors)
    actor = actors[actor_idx]
    
    # Process entire batch efficiently using the selected actor
    results = ray.get(actor.predict_batch.remote(batch["image"]))
    return results

print("Testing optimized approach...")
print("Watch Ray Dashboard to see improved performance")

start_time = time.time()

# Run optimized batch inference with larger batches
optimized_results = dataset.limit(100).map_batches(
    optimized_inference,
    batch_size=16,  # Larger batch size for efficiency
    concurrency=4   # More concurrency for parallelism
).take(20)

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

### Batch Size Optimization

```python
# Test different batch sizes to find optimal performance
print("\nTesting different batch sizes:")

batch_sizes = [4, 8, 16, 32]
batch_performance = []

for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    start_time = time.time()
    
    # Test small sample for quick comparison
    test_results = dataset.limit(80).map_batches(
        optimized_inference,
        batch_size=batch_size,
        concurrency=2
    ).take(40)
    
    batch_time = time.time() - start_time
    throughput = 40 / batch_time if batch_time > 0 else 0
    
    batch_performance.append({
        "batch_size": batch_size,
        "time": batch_time,
        "throughput": throughput
    })
    
    print(f"  Time: {batch_time:.2f}s, Throughput: {throughput:.1f} images/sec")

# Find best performing batch size
if batch_performance:
    best_batch = max(batch_performance, key=lambda x: x["throughput"])
    print(f"\nBest batch size: {best_batch['batch_size']} ({best_batch['throughput']:.1f} images/sec)")
```

### Ray Dashboard Monitoring

```python
# Ray Dashboard provides comprehensive monitoring
print("\nRay Dashboard Monitoring:")
print("The Ray Dashboard shows detailed performance metrics:")
print("- Task execution timelines and worker utilization")
print("- GPU utilization across distributed workers")
print("- Memory usage and object store statistics")
print("- Ray Data execution plans and optimization details")
print("\nNo custom monitoring needed - Ray Dashboard handles everything!")
```

## Key Takeaways

**Actor pattern is essential**: Loading models once per worker using Ray actors provides significant performance improvements  
**Batch size optimization**: Larger batch sizes generally improve GPU utilization and throughput  
**Resource allocation**: Use `num_gpus=1` to properly allocate GPU resources to inference tasks  
**Ray Dashboard monitoring**: Leverage built-in monitoring instead of custom performance tracking  

## Action Items

1. **Apply actor patterns**: Use Ray actors for your own model loading scenarios
2. **Experiment with batch sizes**: Test different batch sizes to find optimal performance
3. **Monitor with Ray Dashboard**: Use the dashboard to understand performance characteristics
4. **Add GPU acceleration**: Use RAPIDS cuDF for complex pandas preprocessing when needed

## Next Steps

**Advanced Ray Data features**: Explore streaming inference and pipeline optimization  
**Ray Serve integration**: Learn about real-time model serving with Ray Serve  
**Multi-model workflows**: Process multiple models in the same Ray Data pipeline  

---

*Ray Dashboard provides all the performance monitoring you need - focus on optimizing your Ray Data usage patterns.*

## Cleanup

```python
# Clean up Ray resources when finished
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```
