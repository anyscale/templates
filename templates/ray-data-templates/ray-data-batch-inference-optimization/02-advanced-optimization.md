# Part 2: Advanced Optimization

**⏱️ Time to complete**: 20 min

**[← Back to Part 1](01-inference-fundamentals.md)** | **[Return to Overview](README.md)**

---

## What You'll Learn

In this part, you'll master systematic optimization techniques for production ML inference:
- Decision frameworks for choosing the right optimization (CPU and GPU)
- Multi-model ensemble inference patterns (works on both CPU and GPU)
- Systematic parameter tuning approaches for any hardware
- Production deployment best practices for CPU-only and GPU clusters

## Prerequisites

Complete [Part 1: Inference Fundamentals](01-inference-fundamentals.md) before starting this part.

## Table of Contents

1. [Optimization Decision Framework](#optimization-decision-framework)
2. [Advanced Optimization Techniques](#advanced-optimization-techniques)
3. [Performance Monitoring](#performance-monitoring)
4. [Production Best Practices](#production-best-practices)

---

## Optimization Decision Framework

### Optimization Priority Levels

| Level | Tool | When to Use | Impact | Complexity |
|-------|------|-------------|---------|------------|
| **1** | `num_cpus`/`num_gpus` parameter | Primary tool for all performance issues | **High** - Controls parallelism | **Low** - Simple parameter |
| **2** | `batch_size` parameter | Memory issues (CPU or GPU) | **Medium** - Affects resource utilization | **Medium** - Requires testing |
| **3** | `concurrency` parameter | Stateful operations (actors) | **High** - Controls actor pool size | **Low** - Simple parameter |
| **4** | Block/memory configs | Out of memory errors only | **Medium** - Affects memory patterns | **High** - Deep knowledge needed |

### Quick Decision Tree

```
Performance Issue
├── Resource underutilized (CPU/GPU <50%)
│   └── Solution: Increase preprocessing parallelism (reduce num_cpus to 0.025-0.5)
│
├── Out of memory (CPU or GPU)
│   └── Solution: Reduce batch_size progressively (64→32→16→8)
│
├── Slow preprocessing
│   └── Solution: Adjust num_cpus for CPU stages (try 0.25-0.5)
│
└── Workers getting killed
    └── Solution: Increase num_cpus/num_gpus to reduce parallelism (try 2.0-4.0)
```

:::note CPU and GPU Decision Framework
**The same decision framework applies to both CPU and GPU clusters!**

- Replace "GPU utilization" with "CPU utilization" for CPU clusters
- Same optimization patterns, just different resource parameters
- Monitor Ray Dashboard to see CPU or GPU utilization in real-time
:::

### Resource Allocation Quick Reference

| Stage Type | Resource Allocation | Reasoning |
|-----------|-------------------|-----------|
| **Image loading** | `num_cpus=0.025-0.05` | I/O bound, high concurrency needed |
| **CPU preprocessing** | `num_cpus=0.25-0.5` | Light compute, benefit from parallelism |
| **GPU inference** | `num_gpus=1` | One model per GPU |
| **CPU inference** | `num_cpus=2-4` | Heavier CPU allocation for model execution |
| **Post-processing** | `num_cpus=0.25-0.5` | Light compute |

---

## Advanced Optimization Techniques

### Multi-Model Ensemble Inference

```python
class EnsembleInferenceWorker:
    """Advanced worker that uses multiple models for ensemble predictions.
    
    Works on both CPU and GPU - automatically detects hardware.
    """
    
    def __init__(self):
        from transformers import pipeline
        import torch
        
        # Auto-detect GPU availability
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device >= 0 else "CPU"
        
        # Load multiple models for ensemble
        self.resnet = pipeline("image-classification", model="microsoft/resnet-50", device=self.device)
        self.vit = pipeline("image-classification", model="google/vit-base-patch16-224", device=self.device)
        
        print(f"Ensemble models loaded on {device_name}: ResNet-50 + ViT")
    
    def __call__(self, batch):
        """Run ensemble inference with multiple models."""
        results = []
        
        for image in batch["image"]:
            # Get predictions from both models
            resnet_pred = self.resnet(image)
            vit_pred = self.vit(image)
            
            # Choose prediction with higher confidence
            if resnet_pred[0]["score"] > vit_pred[0]["score"]:
                results.append({
                    "prediction": resnet_pred[0]["label"],
                    "confidence": resnet_pred[0]["score"],
                    "model": "ResNet-50"
                })
            else:
                results.append({
                    "prediction": vit_pred[0]["label"],
                    "confidence": vit_pred[0]["score"],
                    "model": "ViT"
                })
        
        return results

# Detect GPU availability for resource allocation
import torch
HAS_GPU = torch.cuda.is_available()

# Run ensemble inference with adaptive resource allocation
ensemble_results = dataset.limit(50).map_batches(
    EnsembleInferenceWorker,
    concurrency=1,  # Single worker for memory management
    num_gpus=1 if HAS_GPU else 0,  # GPU if available
    num_cpus=4 if not HAS_GPU else 1,  # More CPU cores if no GPU
    batch_size=8,   # Smaller batches for multiple models
)

print("Ensemble inference completed")
print(ensemble_results.take(5))
```

### Systematic Batch Size Optimization

```python
def find_optimal_batch_size(model_worker_class, test_dataset):
    """Systematically find optimal batch size for inference (CPU or GPU)."""
    import torch
    
    batch_sizes_to_test = [4, 8, 16, 32, 64, 128]
    results = {}
    HAS_GPU = torch.cuda.is_available()
    
    for batch_size in batch_sizes_to_test:
        print(f"Testing batch_size={batch_size}")
        
        try:
            test_start = time.time()
            
            # Adaptive resource allocation
            test_results = test_dataset.limit(100).map_batches(
                model_worker_class,
                num_gpus=1 if HAS_GPU else 0,
                num_cpus=2 if not HAS_GPU else 1,
                concurrency=1,
                batch_size=batch_size
            )
            
            output = test_results.take_all()
            test_duration = time.time() - test_start
            throughput = len(output) / test_duration
            
            results[batch_size] = {
                "throughput": throughput,
                "time": test_duration,
                "success": True
            }
            
            print(f"  Success: {throughput:.1f} images/sec")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  GPU OOM at batch_size={batch_size}")
                break
            raise
    
    # Find optimal batch size
    optimal = max((k for k in results if results[k]["success"]), 
                  key=lambda k: results[k]["throughput"])
    
    print(f"\nOptimal batch_size: {optimal}")
    print(f"Best throughput: {results[optimal]['throughput']:.1f} images/sec")
    
    return optimal

# Usage
optimal_bs = find_optimal_batch_size(InferenceWorker, dataset)
```

### Systematic Concurrency Optimization

```python
def find_optimal_concurrency(model_worker_class, test_dataset):
    """Find optimal number of concurrent GPU workers."""
    
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus == 0:
        print("No GPUs available")
        return 1
    
    concurrency_levels = [1, 2, 4, 8]
    concurrency_levels = [c for c in concurrency_levels if c <= available_gpus]
    
    best_concurrency = 1
    best_throughput = 0
    
    for concurrency in concurrency_levels:
        print(f"Testing concurrency={concurrency}")
        
        test_start = time.time()
        
        test_results = test_dataset.limit(200).map_batches(
            model_worker_class,
            num_gpus=1,
            concurrency=concurrency,
            batch_size=32
        )
        
        output = test_results.take_all()
        test_duration = time.time() - test_start
        throughput = len(output) / test_duration
        
        print(f"  Throughput: {throughput:.1f} images/sec")
        
        if throughput > best_throughput:
            best_concurrency = concurrency
            best_throughput = throughput
    
    print(f"\nOptimal concurrency: {best_concurrency}")
    print(f"Best throughput: {best_throughput:.1f} images/sec")
    
    return best_concurrency

# Usage
optimal_conc = find_optimal_concurrency(InferenceWorker, dataset)
```

---

## Performance Monitoring

### Enable Comprehensive Monitoring

```python
# Setup detailed monitoring for optimization decisions
def setup_inference_monitoring():
    """Enable all monitoring features for optimization."""
    
    ctx = ray.data.DataContext.get_current()
    
    # Progress tracking
    ctx.enable_progress_bars = True
    ctx.enable_operator_progress_bars = True
    
    # Statistics logging
    ctx.enable_auto_log_stats = True
    ctx.verbose_stats_logs = True
    
    print("Monitoring enabled - use these indicators:")
    print("  1. Progress bars show relative stage speeds")
    print("  2. Ray Dashboard shows GPU utilization")
    print("  3. Ray Dashboard shows CPU utilization per node")
    print("\nLook for these problems:")
    print("  - GPU utilization < 80% → Upstream bottleneck")
    print("  - CPU utilization < 80% → Scheduling issue")
    print("  - One stage much slower → Imbalanced num_cpus")

setup_inference_monitoring()
```

### Performance Visualization

```python
# Visualize batch inference performance improvements
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Throughput comparison
configs = ['Inefficient\n(batch=4)', 'Basic\n(batch=16)', 'Optimized\n(batch=32)', 'Best\n(batch=64)']
throughput = [12, 45, 85, 120]  # images/sec
colors = ['red', 'orange', 'lightblue', 'green']

axes[0].bar(configs, throughput, color=colors, alpha=0.7)
axes[0].set_title('GPU Inference Throughput Comparison', fontweight='bold')
axes[0].set_ylabel('Throughput (images/sec)')

# 2. GPU utilization
batch_sizes = [4, 8, 16, 32, 64]
gpu_util = [15, 28, 52, 78, 92]
axes[1].plot(batch_sizes, gpu_util, 'o-', linewidth=2, markersize=8, color='purple')
axes[1].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Target: 80%')
axes[1].set_title('GPU Utilization vs Batch Size', fontweight='bold')
axes[1].set_xlabel('Batch Size')
axes[1].set_ylabel('GPU Utilization (%)')
axes[1].legend()

# 3. Concurrency impact
concurrency = [1, 2, 4, 8]
latency = [850, 480, 280, 220]  # ms per image
axes[2].bar(concurrency, latency, color='coral', alpha=0.7)
axes[2].set_title('Latency vs Concurrency', fontweight='bold')
axes[2].set_xlabel('Number of Workers')
axes[2].set_ylabel('Latency (ms/image)')

plt.tight_layout()
plt.savefig('inference_performance.png', dpi=150, bbox_inches='tight')
print("Performance visualization saved")
```

---

## Production Best Practices

### Resource Allocation by Cluster Size

**Single GPU (development)**:
```python
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1,
    concurrency=1,
    batch_size=32
)
```

**2-4 GPUs (small production)**:
```python
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1,
    concurrency=2,
    batch_size=64
)
```

**8+ GPUs (large production)**:
```python
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1,
    concurrency=4,
    batch_size=128
)
```

### Pipeline Design Best Practices

```python
# Recommended: Clear separation of CPU and GPU work
inference_pipeline = (
    # Stage 1: I/O (CPU-only, high concurrency)
    ray.data.read_images(path, mode="RGB", num_cpus=0.025)
    
    # Stage 2: Preprocessing (CPU-only, medium concurrency)
    .map_batches(cpu_preprocessing, num_cpus=0.5, batch_format="pandas")
    
    # Stage 3: Inference (GPU, actor-based)
    .map_batches(
        GPUInferenceWorker,
        num_cpus=1.0,
        num_gpus=1.0,
        concurrency=2,
        batch_size=32
    )
    
    # Stage 4: Post-processing (CPU-only, high concurrency)
    .map_batches(postprocessing, num_cpus=0.25, batch_format="pandas")
    
    # Stage 5: Output (I/O, moderate concurrency)
    .write_parquet("/tmp/results/", num_cpus=0.1)
)
```

---

---

## Key Takeaways

**What you learned:**
- ✅ Systematic optimization framework (start with `num_cpus`)
- ✅ Multi-model ensemble inference patterns
- ✅ Batch size and concurrency tuning strategies
- ✅ Resource allocation by cluster size

**Production-ready patterns:**
- Decision trees for performance issues
- Resource allocation tables
- Performance monitoring dashboards
- Troubleshooting guides

:::tip Next Steps
**[Continue to Part 3: Ray Data Architecture →](03-ray-data-architecture.md)**

Understand the internals:
- How streaming execution enables efficiency
- Memory management and backpressure
- Operator fusion and pipelining
- Design pipelines based on architectural insights
:::

---

## Troubleshooting Quick Guide

### Issue 1: GPU Showing 0% Utilization

**Solution**:
```python
# Increase upstream parallelism
dataset = ray.data.read_images(path, mode="RGB", num_cpus=0.025)  # More parallel reads

# Increase preprocessing concurrency
preprocessed = dataset.map_batches(preprocess, num_cpus=0.25, batch_format="pandas")

# Increase GPU worker concurrency
results = preprocessed.map_batches(InferenceWorker, num_gpus=1, concurrency=4, batch_size=32)
```

### Issue 2: GPU Out of Memory

**Solution**:
```python
# Progressive batch_size reduction
for batch_size in [64, 32, 16, 8]:
    try:
        results = dataset.limit(10).map_batches(
            InferenceWorker,
            num_gpus=1,
            batch_size=batch_size
        )
        results.take(5)
        print(f"Success with batch_size={batch_size}")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            continue
        raise
```

### Issue 3: Workers Getting Killed

**Solution**:
```python
# Increase num_cpus to reduce parallelism
results = dataset.map_batches(
    InferenceWorker,
    num_cpus=2.0,  # Reduce concurrent workers
    num_gpus=1,
    batch_size=32
)
```

---

## Results and Key Takeaways

### Performance Comparison Summary

| Approach | Model Loading | Batch Size | GPU Utilization | Characteristics |
|----------|---------------|------------|-----------------|----------------|
| **Inefficient** | Every batch | 4 images | Poor (<20%) | Anti-pattern to avoid |
| **Basic Optimized** | Once per worker | 16 images | Good (60-70%) | Production ready |
| **Advanced Optimized** | Once per worker | 32 images | Excellent (>80%) | Recommended |

### What You Learned

**From inefficient to optimized**:
1. **Don't**: Load models inside `map_batches` functions
   - **Do**: Use class-based workers with `__init__` for model loading

2. **Don't**: Use tiny batch sizes (4-8 images)
   - **Do**: Start with batch_size=32, optimize based on GPU memory

3. **Don't**: Process images individually in loops
   - **Do**: Use vectorized batch processing with tensors

4. **Don't**: Use low concurrency by default
   - **Do**: Set concurrency based on available GPUs (2-4 workers)

### Implementation Checklist

**Immediate actions (next 2 weeks)**:
- [ ] Use class-based actors for stateful model loading
- [ ] Test batch_size values (start with 32)
- [ ] Configure concurrency based on available GPUs
- [ ] Monitor improvements in Ray Dashboard

**Production optimizations (next 1-2 months)**:
- [ ] Systematic parameter tuning
- [ ] Multi-model ensembles for improved accuracy
- [ ] CPU/GPU stage separation
- [ ] Result analysis pipelines

---

## Summary

You've mastered batch inference optimization with Ray Data:

**Phase 1: The Problem** (Part 1)
- Identified common anti-patterns
- Understood why repeated model loading fails
- Learned proper resource allocation

**Phase 2: The Solution** (Part 2)
- Systematic optimization frameworks
- Advanced multi-model patterns
- Production deployment strategies

**Key Optimization Principles:**
1. Always measure baseline before optimizing
2. Change ONE parameter at a time
3. Use Ray Dashboard to validate improvements
4. Document successful configurations

---

## Next Steps

Apply these patterns to your own inference workloads:
1. Start with the optimized approach (class-based workers)
2. Use the decision framework to identify bottlenecks
3. Apply systematic parameter tuning
4. Monitor with Ray Dashboard
5. Scale to production with documented best practices

**[← Back to Part 1](01-inference-fundamentals.md)** | **[Return to Overview](README.md)**

---

## Cleanup

```python
# Clean up Ray resources when finished
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```

---

## Next Steps

You've learned advanced optimization techniques for batch inference. Continue to Part 3 to understand the Ray Data architecture that makes these optimizations possible.

**[Continue to Part 3: Ray Data Architecture →](03-ray-data-architecture.md)**

In Part 3, you'll learn:
- How streaming execution enables unlimited dataset processing
- How blocks and memory management affect your optimization choices
- How operator fusion and backpressure work under the hood
- How to calculate optimal parameters from architectural constraints

Or **[return to the overview](README.md)** to see all available parts.

---
