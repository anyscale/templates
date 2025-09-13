# Batch Inference Performance Optimization with Ray Data

**⏱ Time to complete**: 30 min | **Difficulty**: Advanced | **Prerequisites**: ML inference experience, understanding of distributed systems

## What You'll Build

Learn to optimize ML inference pipelines by seeing common mistakes and their fixes. You'll transform a slow, inefficient inference pipeline into a high-performance system.

## Table of Contents

1. [Performance Baseline](#step-1-the-slow-way-common-mistakes) (8 min)
2. [Model Optimization](#step-2-optimizing-model-loading) (8 min)
3. [Resource Tuning](#step-3-resource-and-batch-optimization) (8 min)
4. [Final Optimization](#step-4-putting-it-all-together) (6 min)

## Learning Objectives

By completing this tutorial, you'll understand:

- **Why performance optimization matters**: The difference between fast and slow inference at scale
- **Common Ray Data mistakes**: Anti-patterns that kill performance and how to avoid them
- **Optimization techniques**: Proven methods to maximize throughput and minimize costs
- **Performance debugging**: How to identify and fix bottlenecks in inference pipelines

## Overview

**The Challenge**: ML inference can be deceptively slow. Small configuration mistakes can significantly impact pipeline performance, wasting time and money.

**The Learning Approach**: We'll start with a deliberately inefficient pipeline, then systematically fix each issue to show you exactly what makes the difference.

**Real-world Impact**:
- **Cost Savings**: Optimized inference reduces cloud costs through better resource utilization
- **Speed Gains**: Process data efficiently with proper configuration
- **Scalability**: Handle larger workloads through distributed processing
- **Resource Efficiency**: Maximize GPU utilization and minimize waste

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Experience with ML model inference
- [ ] Understanding of GPU/CPU resource management
- [ ] Familiarity with batch processing concepts
- [ ] Knowledge of performance optimization principles

## Quick Start (3 minutes)

Want to see the performance difference immediately?

```python
import ray
import time

# Slow way (don't do this!)
start = time.time()
ds = ray.data.from_items([{"x": i} for i in range(1000)])
result_slow = ds.map(lambda x: x["x"] * 2)  # Inefficient
print(f"⏰ Slow approach: {time.time() - start:.2f} seconds")

# Fast way (Ray Data optimized!)
start = time.time()
result_fast = ds.map_batches(lambda batch: {"x": [x * 2 for x in batch["x"]]})
print(f" Fast approach: {time.time() - start:.2f} seconds")
```

## The Performance Problem

**Why Inference Optimization Matters**:
- **Scale**: Production ML systems process millions of inputs daily
- **Cost**: GPU time is expensive - optimization saves thousands monthly
- **Latency**: Faster inference enables real-time applications
- **Throughput**: Optimized systems handle 10-100x more requests

## Overview

### **Ray Data Batch Inference: From Slow to Fast**

Batch inference is one of the most common Ray Data use cases, yet it's surprisingly easy to get wrong. Small configuration mistakes can lead to significantly slower performance, wasted GPU resources, and frustrated data scientists.

**Why This Matters for ML Engineers:**
- **Performance Impact**: Poor configuration can make inference much slower than necessary
- **Resource Waste**: Incorrect settings waste expensive GPU time and cluster resources
- **Scalability Issues**: Bad patterns don't scale and become bottlenecks in production
- **Development Friction**: Slow inference slows down experimentation and iteration cycles

This template takes a unique approach: we'll first build a batch inference pipeline using common anti-patterns and mistakes, measure its poor performance, then systematically fix each issue to achieve optimal speed.

### **Common Ray Data Batch Inference Mistakes**

Most performance issues in Ray Data batch inference stem from these fundamental misunderstandings:

1. **Model Loading Anti-Pattern**: Loading models inside the inference function instead of initialization
2. **Resource Misconfiguration**: Wrong `num_cpus`, `num_gpus`, or `concurrency` settings  
3. **Block Size Problems**: Inefficient `override_num_blocks` causing too many small tasks or too few large tasks
4. **Batch Size Issues**: Using batch sizes that are too small (underutilizing GPU) or too large (causing OOM)
5. **Data Format Inefficiency**: Using inefficient formats like JSON/CSV instead of Parquet for large outputs
6. **Memory Management**: Not understanding Ray Data's memory model and causing OOM errors
7. **Incorrect Compute Strategy**: Using `compute="tasks"` when `compute="actors"` is needed for model persistence
8. **Wrong Return Format**: Returning list of objects instead of proper batch dictionary format
9. **Data Transfer Overhead**: Inefficient data serialization and transfer between workers
10. **Preprocessing Inefficiency**: Doing expensive preprocessing inside inference instead of separate stage
11. **API Version Issues**: Using deprecated Ray Data API patterns that cause errors

### **The Learning Journey: Broken → Fixed**

This template follows a clear progression:

**Phase 1: The "Wrong Way"**
- Build inference pipeline with common mistakes
- Identify bottlenecks and anti-patterns
- Understand why each mistake creates problems

**Phase 2: Understanding Ray Data Architecture**
- Learn how Ray Data blocks and tasks work
- Understand resource allocation and parallelism
- Master configuration parameters and their effects

**Phase 3: The "Right Way"**  
- Fix each issue systematically
- Apply Ray Data best practices and optimizations
- Build an optimized inference pipeline

**Phase 4: Comparison and Analysis**
- Compare before/after implementations
- Understand the impact of each optimization
- Learn to identify and debug performance issues

### **Memory Management Best Practices** (rule #254)

Ray Data provides streaming operations for large datasets to avoid memory issues:

```python
# Memory-efficient processing configuration
MEMORY_CONFIG = {
    "preserve_order": False,  # Allow reordering for better memory usage
    "local_shuffle_buffer_size": 1000,  # Limit shuffle buffer size
    "target_max_block_size": 512 * 1024 * 1024,  # 512MB max block size
}

# Example: Memory-efficient inference pipeline
def memory_efficient_inference(dataset):
    """Demonstrate memory-efficient inference patterns."""
    return dataset.map_batches(
        inference_function,
        batch_size=16,  # Smaller batches for memory efficiency
        concurrency=2,  # Limit concurrent workers
        **MEMORY_CONFIG
    )
```

## Learning Objectives

By the end of this template, you'll understand:
- Common Ray Data batch inference anti-patterns and their performance impact
- Ray Data's block-based architecture and how it affects performance
- Proper resource configuration for GPU-accelerated batch inference
- Optimal batch sizing and memory management strategies
- How to measure, profile, and optimize Ray Data pipelines
- Production-ready patterns for scalable batch inference

## Use Case: ImageNet Classification at Scale

### **Real-World ML Inference Challenge**

We'll use a realistic computer vision scenario: classifying thousands of ImageNet images using a pre-trained ResNet model. This represents a common production workload where:

**Dataset Characteristics**
- **Volume**: 50,000+ high-resolution images from ImageNet validation set
- **Size**: ~6GB of image data requiring efficient loading and preprocessing
- **Format**: JPEG images with varying dimensions requiring standardization
- **Distribution**: Images stored in distributed storage (S3-compatible format)

**Model Requirements**
- **Architecture**: ResNet-50 pre-trained on ImageNet (25MB model)
- **Input**: 224x224 RGB images with ImageNet normalization
- **Output**: 1,000-class probability distributions
- **Hardware**: GPU acceleration required for reasonable inference speed

**Performance Expectations**
- **Throughput**: Efficient processing of large image batches
- **Latency**: Reasonable per-batch inference times
- **Resource Utilization**: Good GPU utilization during inference
- **Scalability**: Proper scaling with additional GPU workers

This scenario mirrors real production ML workloads where performance optimization directly impacts business metrics like cost, user experience, and system capacity.

## 5-Minute Quick Start

Let's start by running the "wrong way" implementation to see common mistakes in action:

```python
import ray
import ray.data
import torch
import torchvision.transforms as transforms
from torchvision import models
import time

# Ray cluster is already running on Anyscale
print('Connected to Anyscale Ray cluster!')
print(f'Available resources: {ray.cluster_resources()}')

# Load ImageNet data (publicly available)
# Using a larger sample to demonstrate real-world scenarios
image_dataset = ray.data.read_images(
    "s3://anonymous@air-example-data-2/imagenette2/train/",
    mode="RGB"
).limit(5000)  # Larger sample for realistic demonstration

print(f"Loaded {image_dataset.count()} images for inference")

# WRONG WAY: Common mistakes that kill performance
class SlowInference:
    """Example of what NOT to do - loads model in __call__ method."""
    
    def __call__(self, batch):
        # MISTAKE 1: Loading model inside inference function
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # MISTAKE 2: Not using GPU even when available
        # model = model.cuda()  # Commented out - missing GPU usage
        
        predictions = []
        
        # MISTAKE 3: Processing images one by one instead of batching
        for image_path in batch["path"]:
            # Inefficient single-image processing
            predictions.append({"prediction": "demo_class", "confidence": 0.95})
        
        return predictions

# MISTAKE 4: Wrong resource configuration
slow_results = image_dataset.map_batches(
    SlowInference,
    # MISTAKE 5: Bad batch size and concurrency
    batch_size=1,      # Too small - no GPU utilization
    concurrency=1,     # Too low - no parallelism
    # MISTAKE 6: Missing GPU allocation
    # num_gpus=1       # Commented out - not using available GPU
)

start_time = time.time()
predictions = slow_results.take_all()
slow_time = time.time() - start_time

print(f"Slow implementation: {len(predictions)} predictions in {slow_time:.2f}s")
print(f"Throughput: {len(predictions)/slow_time:.1f} images/second")
```

**Expected Output (Slow Implementation):**
```
Slow implementation: 5000 predictions in [time varies]
Processing completed with multiple performance issues
```

Notice the various inefficiencies in this implementation! The next sections will show you why this happens and how to fix it.

## Complete Tutorial

### **Phase 1: The "Wrong Way" - Common Mistakes**

Let's build a complete batch inference pipeline using all the common anti-patterns. This will help you recognize these mistakes in real code and understand their performance impact.

**Step 1: Data Loading with Poor Configuration**

```python
import ray
import ray.data
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import time
from typing import Dict, List, Any

# Load ImageNet validation data (publicly available subset)
def load_imagenet_data():
    """Load ImageNet data with suboptimal configuration."""
    
    # MISTAKE 1: No block size optimization
    # Using default blocks which may be too small or too large
    dataset = ray.data.read_images(
        "s3://anonymous@air-example-data-2/imagenette2/",
        mode="RGB",
        # MISTAKE: Not specifying override_num_blocks
        # This can lead to inefficient task distribution
    ).limit(10000)  # 10K images for realistic demonstration
    
    print(f"Dataset blocks: {dataset.num_blocks()}")  # Likely suboptimal
    print(f"Estimated dataset size: {dataset.size_bytes() / 1024 / 1024:.1f} MB")
    
    return dataset

imagenet_data = load_imagenet_data()
```

**Step 2: Inefficient Inference Class (All the Wrong Patterns)**

```python
class BadInferenceClass:
    """
    Example of what NOT to do in Ray Data batch inference.
    This class demonstrates every common anti-pattern.
    """
    
    def __init__(self):
        # MISTAKE 2: Minimal initialization
        # Not loading the model here where it should be loaded once
        self.transform = None
        print("BadInferenceClass initialized (but model not loaded)")
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch with all the wrong patterns."""
        
        # MISTAKE 3: Loading model INSIDE the inference function
        # This means we reload the 25MB model for every single batch!
        print("Loading ResNet model (this should only happen once!)")
        model = models.resnet50(pretrained=True)
        model.eval()
        
        # MISTAKE 4: Not using GPU even when available
        device = torch.device('cpu')  # Forcing CPU instead of GPU
        model = model.to(device)
        
        # MISTAKE 5: Creating transform every time instead of reusing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        results = []
        
        # MISTAKE 6: Processing images one by one instead of true batching
        for i, image_array in enumerate(batch["image"]):
            try:
                # MISTAKE 7: Inefficient single-image processing
                # Converting numpy -> PIL -> tensor for each image individually
                image_tensor = transform(image_array).unsqueeze(0)
                image_tensor = image_tensor.to(device)
                
                # MISTAKE 8: Individual forward passes instead of batch inference
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, predicted_class = torch.max(probabilities, 0)
                
                results.append({
                    "image_id": i,
                    "predicted_class": int(predicted_class),
                    "confidence": float(confidence),
                    "processing_time": time.time()  # Unnecessary overhead
                })
                
                # MISTAKE 9: Unnecessary logging/printing in hot path
                if i % 10 == 0:
                    print(f"Processed image {i}")
                    
            except Exception as e:
                # MISTAKE 10: Poor error handling that slows down processing
                print(f"Error processing image {i}: {e}")
                results.append({
                    "image_id": i,
                    "predicted_class": -1,
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # MISTAKE 11: Wrong return format for Ray Data
        # Returning a list of objects instead of proper batch format
        # This violates Ray Data's batch output requirements
        return results  # ERROR: Ray 2.5+ requires wrapped format
```

**Step 3: Running with Wrong Configuration**

```python
def run_bad_inference():
    """Run inference with all the wrong configuration settings."""
    
    print("="*60)
    print("RUNNING BAD INFERENCE PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # MISTAKE 12: Terrible resource configuration
    bad_results = imagenet_data.map_batches(
        BadInferenceClass,
        
        # MISTAKE 13: Tiny batch size that doesn't utilize GPU
        batch_size=2,  # Way too small for GPU efficiency
        
        # MISTAKE 14: Wrong concurrency setting
        concurrency=8,  # Too high for GPU tasks, causes resource contention
        
        # MISTAKE 15: Not specifying GPU resources
        # num_gpus=0.1,  # Commented out - not using available GPU
        
        # MISTAKE 16: Wrong compute strategy
        compute="tasks",  # Should be "actors" for model loading
        
        # MISTAKE 17: No resource limits
        # max_concurrency not set - can overwhelm cluster
    )
    
    # MISTAKE 18: This will fail with Ray 2.5+ due to wrong return format
    # The error message will be:
    # "Returning a list of objects from `map_batches` is not allowed"
    
    # MISTAKE 19: Inefficient data collection (if it worked)
    # all_predictions = bad_results.take_all()  # Would load everything into memory
    
    # MISTAKE 20: Inefficient output format (if data collection worked)
    # Would save to JSON instead of efficient format like Parquet
    # import json
    # with open("/tmp/bad_results.json", "w") as f:
    #     json.dump(all_predictions, f)  # Slow serialization
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nBAD IMPLEMENTATION RESULTS:")
    print(f"Total images processed: {len(all_predictions)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {len(all_predictions)/total_time:.1f} images/second")
    print(f"GPU utilization: Very low (not using GPU)")
    print(f"Model loading overhead: Significant (model reloaded many times)")
    
    return {
        'total_time': total_time,
        'throughput': len(all_predictions)/total_time,
        'predictions': all_predictions[:5]  # Sample results
    }

# Run the bad implementation
bad_performance = run_bad_inference()
```

**Understanding Why This is Inefficient**

The bad implementation above demonstrates many common inefficiencies. In fact, this code will actually fail with a Ray Data error:

```
ValueError: Returning a list of objects from `map_batches` is not allowed in Ray 2.5. 
To return Python objects, wrap them in a named dict field, e.g., return `{'results': objects}` 
instead of just `objects`.
```

This error is extremely common when migrating from older Ray Data versions or when following outdated examples. Here's why each mistake creates problems:

1. **Model Reloading**: Loading the 25MB ResNet model for every batch adds significant overhead
2. **No GPU Usage**: CPU inference is much slower than GPU for neural networks
3. **Tiny Batches**: Batch size of 2 means poor GPU utilization and maximum overhead
4. **High Concurrency**: 8 concurrent tasks compete for the same resources
5. **Individual Processing**: Processing images one-by-one prevents vectorized operations
6. **Memory Inefficiency**: Creating transforms and tensors repeatedly wastes memory
7. **Wrong Return Format**: Returning list of objects violates Ray Data's batch output requirements
8. **Data Format Issues**: JSON serialization is slower than binary formats
9. **Resource Competition**: Tasks competing for limited GPU memory
10. **API Misuse**: Using deprecated or incorrect Ray Data API patterns
11. **Return Format Error**: The list return format causes Ray Data validation errors

### **Phase 2: Understanding Ray Data Architecture**

Before we fix the problems, let's understand how Ray Data works internally. This knowledge is crucial for writing efficient batch inference pipelines.

**Ray Data Block Architecture**

Ray Data organizes datasets into blocks, which are the fundamental units of parallel processing:

```python
def understand_ray_data_blocks():
    """Explore Ray Data's block-based architecture."""
    
    print("="*60)
    print("RAY DATA ARCHITECTURE DEEP DIVE")
    print("="*60)
    
    # Load dataset and examine its structure
    dataset = ray.data.read_images(
        "s3://anonymous@air-example-data-2/imagenette2/train/", 
        mode="RGB"
    ).limit(500)
    
    print(f"Dataset Statistics:")
    print(f"  Total records: {dataset.count()}")
    print(f"  Number of blocks: {dataset.num_blocks()}")
    print(f"  Dataset size: {dataset.size_bytes() / 1024 / 1024:.1f} MB")
    
    # Understanding block distribution
    block_sizes = []
    for i in range(min(5, dataset.num_blocks())):
        block = dataset.get_internal_block_refs()[i]
        # This would show block size distribution in real implementation
        print(f"  Block {i}: ~{dataset.size_bytes() // dataset.num_blocks() / 1024:.0f} KB")
    
    print(f"\nKey Insights:")
    print(f"  • Each block becomes a separate task")
    print(f"  • Tasks run in parallel across Ray workers")
    print(f"  • Block size affects memory usage and parallelism")
    print(f"  • Too many small blocks = high overhead")
    print(f"  • Too few large blocks = poor parallelism")
    
    return dataset

# Explore the architecture
sample_dataset = understand_ray_data_blocks()
```

**Resource Allocation and Task Scheduling**

Understanding how Ray Data schedules tasks is critical for performance:

```python
def explain_resource_allocation():
    """Explain Ray Data resource allocation patterns."""
    
    print("\n" + "="*60)
    print("RESOURCE ALLOCATION PATTERNS")
    print("="*60)
    
    cluster_resources = ray.cluster_resources()
    print(f"Available Cluster Resources:")
    for resource, amount in cluster_resources.items():
        print(f"  {resource}: {amount}")
    
    print(f"\nTask Scheduling Rules:")
    print(f"  • map_batches() creates one task per block")
    print(f"  • Each task requests specified resources (CPU/GPU)")
    print(f"  • Tasks wait in queue if resources unavailable")
    print(f"  • Higher concurrency ≠ better performance")
    
    print(f"\nGPU Allocation Best Practices:")
    print(f"  • Use num_gpus=1.0 for dedicated GPU per task")
    print(f"  • Use num_gpus=0.5 to share GPU between 2 tasks")
    print(f"  • Match concurrency to available GPUs")
    print(f"  • Avoid fractional GPU allocation unless memory-constrained")
    
    print(f"\nMemory Management:")
    print(f"  • Ray Data streams blocks through memory")
    print(f"  • Large batches use more memory but improve GPU utilization")
    print(f"  • Object store provides automatic memory management")
    
explain_resource_allocation()
```

**Optimal Configuration Guidelines**

Here are the key principles for configuring Ray Data batch inference:

```python
def configuration_guidelines():
    """Provide guidelines for optimal Ray Data configuration."""
    
    print("\n" + "="*60)
    print("CONFIGURATION OPTIMIZATION GUIDELINES")
    print("="*60)
    
    print("1. BATCH SIZE OPTIMIZATION")
    print("   • GPU Memory Available: Use largest batch that fits")
    print("   • CPU-only: 32-128 samples per batch")
    print("   • GPU: 64-512 samples per batch (depends on model size)")
    print("   • Rule of thumb: Start with 128, increase until OOM")
    
    print("\n2. CONCURRENCY SETTINGS")
    print("   • GPU tasks: concurrency = number of GPUs")
    print("   • CPU tasks: concurrency = number of CPU cores / 2")
    print("   • Never exceed available hardware resources")
    
    print("\n3. BLOCK SIZE TUNING")
    print("   • Target: 100-500 MB per block")
    print("   • Use override_num_blocks to control block count")
    print("   • Formula: num_blocks = dataset_size_mb / target_block_size_mb")
    
    print("\n4. RESOURCE ALLOCATION")
    print("   • Model loading: Use compute='actors' for persistence")
    print("   • Stateless operations: Use compute='tasks'")
    print("   • GPU allocation: Match to actual hardware")
    
    print("\n5. OUTPUT FORMAT OPTIMIZATION")
    print("   • Parquet: Best for structured data, analytics")
    print("   • JSON: Good for small results, debugging")
    print("   • Avoid: CSV for large datasets, Python pickle")

configuration_guidelines()
```

### **Phase 3: The "Right Way" - Optimized Implementation**

Now let's fix every mistake and build a properly optimized batch inference pipeline:

**Step 1: Optimized Data Loading**

```python
def load_imagenet_data_optimized():
    """Load ImageNet data with optimal configuration."""
    
    print("="*60)
    print("OPTIMIZED DATA LOADING")
    print("="*60)
    
    # OPTIMIZATION 1: Proper block size configuration
    # Calculate optimal number of blocks based on cluster resources
    cluster_cpus = int(ray.cluster_resources().get('CPU', 4))
    target_blocks = cluster_cpus * 2  # 2x CPU cores for good parallelism
    
    dataset = ray.data.read_images(
        "s3://anonymous@air-example-data-2/imagenette2/",
        mode="RGB",
        # OPTIMIZATION: Specify optimal block count
        override_num_blocks=target_blocks
    ).limit(10000)  # Same 10K images for fair comparison
    
    print(f"Optimized Dataset Configuration:")
    print(f"  Total records: {dataset.count()}")
    print(f"  Number of blocks: {dataset.num_blocks()}")
    print(f"  Records per block: ~{dataset.count() // dataset.num_blocks()}")
    print(f"  Dataset size: {dataset.size_bytes() / 1024 / 1024:.1f} MB")
    print(f"  Average block size: ~{dataset.size_bytes() / dataset.num_blocks() / 1024 / 1024:.1f} MB")
    
    return dataset

optimized_data = load_imagenet_data_optimized()
```

**Step 2: Proper Inference Class (All the Right Patterns)**

```python
class OptimizedInferenceClass:
    """
    Example of how to do Ray Data batch inference correctly.
    This class demonstrates all the best practices.
    """
    
    def __init__(self):
        """OPTIMIZATION 2: Load model once during initialization."""
        print("Loading ResNet model once during actor initialization...")
        
        # Load model during initialization (happens once per actor)
        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        
        # OPTIMIZATION 3: Use GPU when available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
        
        # OPTIMIZATION 4: Create transform once and reuse
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("OptimizedInferenceClass fully initialized")
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, List[Any]]:
        """Process batch efficiently with proper batching."""
        
        # OPTIMIZATION 5: True batch processing
        batch_size = len(batch["image"])
        
        # OPTIMIZATION 6: Vectorized preprocessing
        # Process all images in the batch together
        processed_images = []
        for image_array in batch["image"]:
            image_tensor = self.transform(image_array)
            processed_images.append(image_tensor)
        
        # OPTIMIZATION 7: Stack into proper batch tensor
        batch_tensor = torch.stack(processed_images).to(self.device)
        
        # OPTIMIZATION 8: Single forward pass for entire batch
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted_classes = torch.max(probabilities, 1)
        
        # OPTIMIZATION 9: Correct Ray Data batch format
        # Return dictionary with arrays, not list of objects
        return {
            "predicted_class": predicted_classes.cpu().numpy().tolist(),
            "confidence": confidences.cpu().numpy().tolist(),
            "batch_size": [batch_size] * batch_size  # Metadata for analysis
        }
```

**Step 3: Optimal Configuration and Execution**

```python
def run_optimized_inference():
    """Run inference with optimal configuration settings."""
    
    print("="*60)
    print("RUNNING OPTIMIZED INFERENCE PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # OPTIMIZATION 10: Proper resource configuration
    optimized_results = optimized_data.map_batches(
        OptimizedInferenceClass,
        
        # OPTIMIZATION 11: Optimal batch size for GPU utilization
        batch_size=64,  # Large enough to utilize GPU effectively
        
        # OPTIMIZATION 12: Appropriate concurrency
        concurrency=2,  # Match available GPU resources
        
        # OPTIMIZATION 13: Proper GPU allocation
        num_gpus=1,  # Use available GPU
        
        # OPTIMIZATION 14: Use actors for model persistence
        compute=ray.data.ActorPoolStrategy(size=2),  # Keep model loaded in memory
        
        # OPTIMIZATION 15: Additional optimizations
        max_concurrency=2,  # Prevent resource oversubscription
    )
    
    # OPTIMIZATION 16: Efficient output format
    # Save to Parquet for efficient storage and future processing
    output_path = "/mnt/cluster_storage/inference_results"
    optimized_results.write_parquet(output_path)
    
    # Get sample results for display
    sample_results = optimized_results.take(100)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nOPTIMIZED IMPLEMENTATION RESULTS:")
    print(f"Total images processed: {len(sample_results)}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {len(sample_results)/total_time:.1f} images/second")
    print(f"GPU utilization: Much higher (efficient GPU usage)")
    print(f"Model loading overhead: Minimal (loaded once per actor)")
    
    return {
        'total_time': total_time,
        'throughput': len(sample_results)/total_time,
        'predictions': sample_results[:5]  # Sample results
    }

# Run the optimized implementation
optimized_performance = run_optimized_inference()
```

### **Phase 4: Performance Comparison and Analysis**

Let's compare the performance improvements and understand the impact of each optimization:

```python
def implementation_comparison():
    """Compare the two implementations and their key differences."""
    
    print("="*60)
    print("IMPLEMENTATION COMPARISON: BEFORE vs AFTER")
    print("="*60)
    
    print(f"KEY DIFFERENCES:")
    print(f"  Bad Implementation:")
    print(f"    • Model reloaded for every batch")
    print(f"    • CPU-only inference")
    print(f"    • Tiny batch sizes (batch_size=2)")
    print(f"    • High concurrency causing resource contention")
    print(f"    • Individual image processing")
    print(f"    • JSON output format")
    print(f"    • No resource optimization")
    
    print(f"\n  Optimized Implementation:")
    print(f"    • Model loaded once per actor")
    print(f"    • GPU acceleration when available")
    print(f"    • Proper batch sizes (batch_size=64)")
    print(f"    • Appropriate concurrency settings")
    print(f"    • Vectorized batch processing")
    print(f"    • Efficient Parquet output")
    print(f"    • Optimized resource allocation")
    
    print(f"\nKEY OPTIMIZATION IMPACTS:")
    print(f"  1. Model Loading Once:    Eliminated reload overhead")
    print(f"  2. GPU Acceleration:      Efficient GPU-based inference")
    print(f"  3. Proper Batching:       Better resource utilization")
    print(f"  4. Optimal Configuration: Improved parallelism")
    print(f"  5. Efficient I/O:         Faster data serialization")

implementation_comparison()
```

## Ray Data Architecture Deep Dive

Understanding Ray Data's internal architecture is crucial for optimization. Let's explore the key concepts:

### **Block-Based Parallelism**

Ray Data organizes datasets into blocks, which are the fundamental units of parallel processing:

**How Blocks Work:**
- Each dataset is divided into multiple blocks (similar to Spark partitions)
- Each block contains a subset of the data (typically 100-500MB)
- Operations like `map_batches()` create one task per block
- Tasks execute in parallel across Ray workers

**Block Size Impact on Performance:**
- **Too Many Small Blocks**: High task overhead, poor GPU utilization
- **Too Few Large Blocks**: Poor parallelism, memory pressure
- **Optimal Block Size**: Balance parallelism with resource efficiency

```python
# Example: Understanding block distribution
dataset = ray.data.read_images("path/to/images")
print(f"Blocks: {dataset.num_blocks()}")
print(f"Records per block: ~{dataset.count() // dataset.num_blocks()}")

# Control block size with override_num_blocks
optimized_dataset = ray.data.read_images(
    "path/to/images",
    override_num_blocks=ray.cluster_resources()['CPU'] * 2
)
```

### **Resource Allocation Model**

Ray Data's resource allocation determines how tasks are scheduled and executed:

**CPU vs GPU Tasks:**
- **CPU Tasks**: Lightweight, many can run concurrently
- **GPU Tasks**: Resource-intensive, limited by available GPUs
- **Mixed Workloads**: Balance CPU preprocessing with GPU inference

**Concurrency Patterns:**
- `concurrency=1`: Sequential processing, lowest resource usage
- `concurrency=num_gpus`: One task per GPU, optimal for GPU workloads  
- `concurrency=num_cpus`: Maximum parallelism for CPU workloads

**Memory Management:**
- Ray Data streams blocks through memory automatically
- Large batches improve GPU utilization but use more memory
- Object store provides distributed memory management

### **Task vs Actor Compute**

Choosing between tasks and actors affects performance significantly:

**Tasks (`compute="tasks"`):**
- **Best for**: Stateless operations, simple transformations
- **Characteristics**: Fast startup, no state persistence
- **Use case**: Data preprocessing, simple computations

**Actors (`compute="actors"`):**
- **Best for**: Stateful operations, model inference
- **Characteristics**: Slower startup, persistent state
- **Use case**: ML model inference, database connections

```python
# Task-based: Model loaded for every batch (slow)
results = dataset.map_batches(
    InferenceClass,
    compute="tasks"  # Model reloaded each time
)

# Actor-based: Model loaded once per worker (fast)
results = dataset.map_batches(
    InferenceClass,
    compute="actors"  # Model persists in memory
)
```

## Configuration Optimization Guide

### **Batch Size Optimization**

Batch size is the most critical parameter for inference performance:

**GPU Memory Constraints:**
- Start with batch_size=64 for most models
- Increase until you hit GPU OOM errors
- Monitor GPU memory usage with `nvidia-smi`
- Reduce batch size if you see memory errors

**Performance vs Memory Trade-off:**
- **Larger batches**: Better GPU utilization, higher throughput
- **Smaller batches**: Lower memory usage, more stable
- **Sweet spot**: Usually 64-256 for vision models

```python
# Find optimal batch size through experimentation
for batch_size in [16, 32, 64, 128, 256]:
    try:
        results = dataset.map_batches(
            OptimizedInference,
            batch_size=batch_size,
            num_gpus=1
        )
        throughput = measure_throughput(results)
        print(f"Batch size {batch_size}: {throughput:.1f} imgs/sec")
    except Exception as e:
        print(f"Batch size {batch_size}: OOM error")
        break
```

### **Concurrency Tuning**

Concurrency controls how many parallel tasks execute simultaneously:

**GPU Workloads:**
- Set `concurrency` equal to number of available GPUs
- Never exceed available GPU resources
- Use fractional GPU allocation for memory-constrained models

**CPU Workloads:**
- Start with `concurrency = num_cpus // 2`
- Increase if CPU utilization is low
- Consider memory constraints for large models

```python
# GPU-optimized concurrency
num_gpus = int(ray.cluster_resources().get('GPU', 1))
results = dataset.map_batches(
    OptimizedInference,
    concurrency=num_gpus,  # One task per GPU
    num_gpus=1,
    batch_size=128
)
```

### **Block Size Tuning**

Control dataset partitioning for optimal parallelism:

**Target Block Size:**
- Aim for 100-500MB per block
- Balance parallelism with task overhead
- Consider downstream processing requirements

**Calculation Formula:**
```python
# Calculate optimal blocks
dataset_size_mb = dataset.size_bytes() / 1024 / 1024
target_block_size_mb = 200  # Target 200MB blocks
optimal_blocks = max(1, int(dataset_size_mb / target_block_size_mb))

# Apply optimization
optimized_dataset = dataset.repartition(optimal_blocks)
```

## Troubleshooting Performance Issues

### **Common Performance Problems**

**1. Low GPU Utilization**
- **Symptoms**: `nvidia-smi` shows <50% GPU usage
- **Causes**: Batch size too small, CPU bottleneck, wrong concurrency
- **Solutions**: Increase batch_size, optimize preprocessing, check resource allocation

**2. Out of Memory Errors**
- **Symptoms**: CUDA OOM, Ray object store full
- **Causes**: Batch size too large, too many concurrent tasks
- **Solutions**: Reduce batch_size, lower concurrency, add memory

**3. Slow Throughput**
- **Symptoms**: Much slower than expected performance
- **Causes**: Model reloading, wrong compute mode, inefficient batching
- **Solutions**: Use actors, optimize batch processing, profile bottlenecks

### **Performance Profiling**

Use Ray's built-in profiling tools to identify bottlenecks:

```python
# Enable Ray Data stats
from ray.data.context import DataContext
ctx = DataContext.get_current()
ctx.enable_progress_bars = True

# Profile execution
import time
start_time = time.time()
results = dataset.map_batches(OptimizedInference, batch_size=64)
results.take_all()
end_time = time.time()

print(f"Total time: {end_time - start_time:.2f}s")
print("Check Ray dashboard for detailed metrics")
```

### **Memory Management**

Monitor and optimize memory usage:

```python
# Check object store usage
print(f"Object store memory: {ray.cluster_resources()}")

# Monitor during execution
def memory_efficient_inference():
    results = dataset.map_batches(
        OptimizedInference,
        batch_size=64,
        # Limit memory usage
        max_concurrency=2
    )
    
    # Process in chunks to avoid memory buildup
    for batch in results.iter_batches(batch_size=1000):
        process_batch(batch)
```

## Implementation Comparison

The optimizations demonstrate several key improvements:

### **Configuration Improvements**

**Resource Allocation:**
- **Before**: No GPU usage, inefficient CPU processing
- **After**: Proper GPU allocation and utilization

**Batch Processing:**
- **Before**: Tiny batches (size=2) with individual processing
- **After**: Optimal batch sizes (size=64) with vectorized operations

**Model Management:**
- **Before**: Model reloaded for every batch
- **After**: Model loaded once per actor and reused

### **Data Processing Improvements**

**Memory Usage:**
- **Before**: Inefficient memory patterns, repeated allocations
- **After**: Optimized memory usage with proper batching

**I/O Efficiency:**
- **Before**: Slow JSON serialization for outputs
- **After**: Fast Parquet format for structured data

**Resource Utilization:**
- **Before**: Poor resource utilization, task contention
- **After**: Balanced resource allocation and parallelism

## Ray Data Best Practices Summary

### **Essential Optimization Checklist**

** Model Loading**
- [ ] Load models in `__init__`, not `__call__`
- [ ] Use `compute="actors"` for model persistence
- [ ] Initialize transforms once per actor

** Resource Configuration**  
- [ ] Set `num_gpus=1` for GPU inference
- [ ] Match `concurrency` to available GPUs
- [ ] Use appropriate `batch_size` for GPU memory

** Data Processing**
- [ ] Process batches, not individual samples
- [ ] Use vectorized operations when possible
- [ ] Optimize block size with `override_num_blocks`

** Output Optimization**
- [ ] Use Parquet for structured output
- [ ] Avoid memory-intensive formats (pickle)
- [ ] Stream results when possible

** Monitoring**
- [ ] Monitor GPU utilization (`nvidia-smi`)
- [ ] Check Ray dashboard for task metrics
- [ ] Profile memory usage and bottlenecks

### **Common Anti-Patterns to Avoid**

**🚫 Never Do This:**
- Loading models inside inference functions
- Using CPU when GPU is available
- Processing samples individually in batches
- Setting concurrency higher than available resources
- Ignoring batch size optimization
- Using inefficient output formats

** Always Do This:**
- Initialize expensive resources once per actor
- Use GPU acceleration when available
- Leverage vectorized batch processing
- Match concurrency to hardware resources
- Optimize batch size for memory and throughput
- Use efficient data formats (Parquet)

### **Production Deployment Tips**

**Cluster Configuration:**
- Size clusters based on inference requirements
- Use GPU instances for neural network inference
- Monitor resource utilization and scale accordingly

**Monitoring and Alerting:**
- Set up throughput monitoring
- Alert on performance degradation
- Track cost per inference for optimization

**Scaling Strategies:**
- Start with single-node optimization
- Scale horizontally with additional GPU nodes
- Use Ray autoscaling for variable workloads

This comprehensive guide should help you avoid common pitfalls and achieve optimal performance with Ray Data batch inference pipelines. The key is understanding the underlying architecture and applying systematic optimization techniques.
