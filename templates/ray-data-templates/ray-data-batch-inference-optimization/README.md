# ML batch inference optimization with Ray Data

**Time to complete**: 20 min | **Difficulty**: Intermediate | **Prerequisites**: Basic ML knowledge, Python experience

## What You'll Build

Create an optimized ML batch inference pipeline that demonstrates the performance difference between naive and efficient approaches. Learn how Ray Data's actor-based patterns eliminate common bottlenecks in production ML inference.

## Table of Contents

1. [Setup and Data](#setup) (3 min)
2. [Inefficient Approach](#inefficient-approach) (5 min) 
3. [Optimized with Ray Data](#optimized-approach) (8 min)
4. [Performance Comparison](#performance-comparison) (4 min)

## Learning Objectives

**Why batch inference optimization matters**: Poor optimization wastes significant compute resources through repeated model loading and inefficient batching. Understanding these bottlenecks is crucial for production ML systems.

**Ray Data's inference capabilities**: Stateful per-worker model loading (via Ray Data) and distributed processing eliminate performance bottlenecks that plague traditional ML pipelines. You'll learn how to leverage these capabilities for scalable inference.

**Real-world optimization patterns**: Netflix processes 200+ million inference requests daily for personalized recommendations using distributed ML systems. Tesla's autonomous driving systems perform real-time inference on sensor data from millions of vehicles. Google Search executes billions of ML inference operations for ranking and relevance. These patterns apply across industries from recommendation systems to autonomous vehicles, search engines, and fraud detection systems.

**Production deployment strategies**: Master GPU utilization, batch size tuning, and resource allocation techniques that enable ML systems to scale to enterprise workloads cost-effectively.

## Overview

**Challenge**: Naive batch inference approaches create significant performance bottlenecks that prevent ML systems from scaling to production workloads. Model loading overhead can consume significant processing time, while poor batch sizing wastes GPU resources and increases operational costs.

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
    # Model loading happens for every batch - very inefficient
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
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

# Run inefficient batch inference with small batches
inefficient_results = dataset.limit(100).map_batches(
    inefficient_inference,
    batch_size=4,
    concurrency=2
).take(20)

print("Inefficient approach completed. Problems: repeated model loading, poor batching, wasted resources")
```

**Analysis of the inefficient approach**: This code demonstrates several critical performance problems. The model loading overhead dominates execution time because the transformer pipeline is initialized for every single batch. Small batch sizes of 4 images fail to utilize GPU memory efficiently, while processing images individually prevents vectorized operations. The combination of these factors results in poor resource utilization across distributed workers.

These performance bottlenecks are common in production ML systems where inference patterns haven't been optimized for distributed execution. Understanding these anti-patterns is essential before implementing the optimized solution.

## Optimized Approach

### The Right Way: Stateful per-worker model loading with Ray Data

Ray Data solves the model loading problem by letting you run stateful, class-based `map_batches` with an actor pool strategy. Each worker loads the model once and reuses it across many batches, eliminating repeated initialization overhead.

```python
# EFFICIENT: Use Ray Data class-based map_batches with optimized actor configuration

class InferenceWorker:
    """Stateful worker that loads the model once and reuses it."""
    def __init__(self):
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "image-classification",
            model="microsoft/resnet-50",
            device=device,
        )

    def __call__(self, batch):
        results = []
        for image in batch["image"]:
            pred = self.classifier(image)
            results.append({
                "prediction": pred[0]["label"],
                "confidence": pred[0]["score"],
            })
        return results

print("Running optimized Ray Data inference with stateful workers...")

# BEST PRACTICE: Use the new concurrency parameter for actor-based processing
inference_results = dataset.limit(100).map_batches(
    InferenceWorker,
    concurrency=2,      # Use concurrency instead of deprecated compute parameter
    num_gpus=1,         # Allocate one GPU per worker
    batch_size=16,      # Optimal batch size for GPU utilization
).take(20)

print("Optimized approach completed. Improvements: single model load per worker, better batching, efficient resource use")
```

**What's better:**
- Model loads only once per worker via Ray Data `ActorPoolStrategy`
- Larger batch sizes for better resource utilization
- Proper GPU allocation with `num_gpus=1`
- Ray Data manages distribution across workers

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

## Evaluation and monitoring

- Use Ray Dashboard to examine task timelines, GPU utilization, and worker load.
- Compare inefficient vs optimized execution plans to understand where time is spent.
- Adjust `batch_size`, `concurrency`, and `num_gpus` based on your cluster resources and dataset.

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

## Performance Comparison Results

| Approach | Model Loading | Batch Size | GPU Utilization | Overall Efficiency |
|----------|---------------|------------|-----------------|-------------------|
| **Inefficient** | Every batch | 4 images | Poor | Very slow |
| **Optimized** | Once per worker | 16 images | Excellent | Fast |

### Optimization Impact

:::tip Key Performance Improvements
The optimized approach delivers significant improvements through:
- **Stateful model loading** eliminates repeated initialization overhead
- **Larger batch sizes** maximize GPU memory utilization  
- **Proper resource allocation** ensures efficient worker distribution
:::

## Implementation Checklist

### Immediate Actions (Next 2 weeks)
- [ ] Use actor-based `map_batches()` for stateful model loading
- [ ] Set optimal `batch_size` for your GPU memory (start with 16-32)
- [ ] Configure `concurrency` based on available resources (2-4 workers initially)
- [ ] Monitor performance with Ray Dashboard for bottleneck identification
- [ ] Implement proper resource cleanup with `ray.shutdown()`

### Production Optimizations (Next 1-2 months)
- [ ] Experiment with different batch sizes for your specific models
- [ ] Add GPU acceleration for data preprocessing pipelines
- [ ] Implement multi-model inference pipelines for A/B testing
- [ ] Integrate with Ray Serve for real-time serving capabilities
- [ ] Set up automated model deployment and versioning
- [ ] Implement comprehensive monitoring and alerting

### Enterprise Scale (Next 3-6 months)
- [ ] Deploy across multi-node GPU clusters for massive throughput
- [ ] Implement auto-scaling based on inference demand
- [ ] Add model performance tracking and drift detection
- [ ] Integrate with MLOps pipelines for continuous deployment
- [ ] Implement cost optimization strategies for GPU utilization  

---

*Ray Dashboard provides all the performance monitoring you need - focus on optimizing your Ray Data usage patterns.*

## Cleanup

```python
# Clean up Ray resources when finished
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```
