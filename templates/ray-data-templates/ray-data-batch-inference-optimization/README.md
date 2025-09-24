# ML batch inference optimization with Ray Data

**Time to complete**: 20 min | **Difficulty**: Intermediate | **Prerequisites**: Basic ML knowledge, Python experience

Learn how to optimize ML batch inference by comparing inefficient and efficient approaches. This template shows common pitfalls and how Ray Data solves them.

## Table of Contents

1. [Setup and Data](#setup) (3 min)
2. [Inefficient Approach](#inefficient-approach) (5 min) 
3. [Optimized with Ray Data](#optimized-approach) (8 min)
4. [Performance Comparison](#performance-comparison) (4 min)

## Learning Objectives

**Ray Data batch processing**: Learn how `map_batches` improves inference efficiency  
**Actor-based optimization**: Use Ray actors to avoid repeated model loading  
**Batch size tuning**: Understand how batch size affects performance  
**GPU utilization**: Optimize for distributed GPU workers in Ray clusters

## Overview

**The Problem**: Naive batch inference approaches suffer from inefficient model loading, poor batching, and resource waste.

**The Solution**: Ray Data provides distributed batch processing that optimizes model loading, batching, and resource utilization automatically.

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

```python
# Load real ImageNet dataset for batch inference
dataset = ray.data.read_images(
    "s3://ray-benchmark-data/imagenette2/train/",
    mode="RGB"
).limit(1000)  # 1K images for demo

print("Loaded ImageNet dataset for batch inference demo")
print("Sample dataset:")
sample_batch = dataset.take_batch(3)
print(f"Batch contains {len(sample_batch['image'])} images")
print(f"Image shape: {sample_batch['image'][0].shape}")
```

## Inefficient Approach

### The Wrong Way: Loading Model in Every Batch

This approach demonstrates common mistakes that lead to poor performance:

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
inefficient_results = dataset.limit(100).map_batches(
    inefficient_inference,
    batch_size=4,  # Small batch size
    concurrency=2
).take(20)

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
num_actors = 2
actors = [OptimizedInferenceActor.remote() for _ in range(num_actors)]

def optimized_inference(batch):
    """Efficient inference using pre-loaded actors."""
    # Distribute work across available actors
    actor_idx = hash(str(batch)) % len(actors)
    actor = actors[actor_idx]
    
    # Process entire batch efficiently
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
