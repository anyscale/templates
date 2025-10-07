# CPU and GPU Usage Guide for Batch Inference

**⏱ Time to complete**: 10 min | **Purpose**: Run templates on any hardware

---

## Overview

**All batch inference templates work identically on CPU-only and GPU clusters!**

This guide shows you how to adapt examples for your available hardware without changing the core optimization patterns.

---

## Quick Start

### Detect Your Hardware

```python
import torch
import ray

ray.init()

# Detect available hardware
HAS_GPU = torch.cuda.is_available()

if HAS_GPU:
    print(f"✅ GPU detected: {torch.cuda.device_count()} GPU(s) available")
    print(f"   GPU type: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️  No GPU detected - running on CPU")
    print(f"   Available CPU cores: {ray.available_resources()['CPU']}")

print("\nAll examples will adapt automatically to your hardware!")
```

---

## Resource Allocation Patterns

### The Universal Pattern

```python
import torch

# Detect hardware
HAS_GPU = torch.cuda.is_available()

# Adaptive resource allocation
resource_config = {
    "num_gpus": 1 if HAS_GPU else 0,
    "num_cpus": 1 if HAS_GPU else 2,  # More CPU cores when no GPU
    "batch_size": 32 if HAS_GPU else 16,  # Larger batches for GPU
    "concurrency": torch.cuda.device_count() if HAS_GPU else (ray.available_resources()["CPU"] // 2)
}

# Apply to map_batches
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=resource_config["num_gpus"],
    num_cpus=resource_config["num_cpus"],
    batch_size=resource_config["batch_size"],
    concurrency=int(resource_config["concurrency"])
)
```

---

## Code Adaptation Examples

### Example 1: Basic Inference

**Original (GPU-focused)**:
```python
# GPU-only version
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1,
    batch_size=32
)
```

**Adapted (CPU+GPU compatible)**:
```python
import torch
HAS_GPU = torch.cuda.is_available()

# Works on both CPU and GPU
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1 if HAS_GPU else 0,
    num_cpus=1 if HAS_GPU else 2,
    batch_size=32 if HAS_GPU else 16
)
```

---

### Example 2: Model Loading

**Original (GPU-focused)**:
```python
class InferenceWorker:
    def __init__(self):
        from transformers import pipeline
        # Assumes GPU available
        self.model = pipeline("image-classification", model="resnet-50", device=0)
```

**Adapted (CPU+GPU compatible)**:
```python
class InferenceWorker:
    def __init__(self):
        from transformers import pipeline
        import torch
        
        # Auto-detect hardware
        device = 0 if torch.cuda.is_available() else -1
        self.model = pipeline("image-classification", model="resnet-50", device=device)
        
        print(f"Model loaded on: {'GPU' if device >= 0 else 'CPU'}")
```

---

### Example 3: Ensemble Inference

**Original (GPU-focused)**:
```python
# GPU-only version
ensemble_results = dataset.map_batches(
    EnsembleWorker,
    num_gpus=1,
    concurrency=2,
    batch_size=8
)
```

**Adapted (CPU+GPU compatible)**:
```python
import torch
HAS_GPU = torch.cuda.is_available()

# Adaptive configuration
ensemble_results = dataset.map_batches(
    EnsembleWorker,
    num_gpus=1 if HAS_GPU else 0,
    num_cpus=4 if not HAS_GPU else 1,  # More CPU cores for ensemble on CPU
    concurrency=2 if HAS_GPU else 1,  # Fewer parallel workers on CPU
    batch_size=8 if HAS_GPU else 4  # Smaller batches on CPU
)
```

---

## Configuration Guidelines

### CPU-Only Clusters

| Resource | Recommended Value | Reasoning |
|----------|------------------|-----------|
| **num_cpus** | 2-4 per actor | Model execution needs CPU resources |
| **num_gpus** | 0 | No GPUs available |
| **batch_size** | 8-16 | Smaller batches to avoid CPU memory pressure |
| **concurrency** | CPU_cores // 2 | Don't oversubscribe CPUs |

**Example CPU configuration**:
```python
# Optimized for CPU-only cluster
results = dataset.map_batches(
    InferenceWorker,
    num_cpus=2,        # 2 cores per actor
    batch_size=16,     # Conservative batch size
    concurrency=4      # 4 parallel actors (8 total cores)
)
```

---

### GPU Clusters

| Resource | Recommended Value | Reasoning |
|----------|------------------|-----------|
| **num_gpus** | 1 per actor | One model per GPU |
| **num_cpus** | 1 per actor | Minimal CPU for GPU coordination |
| **batch_size** | 32-128 | Larger batches leverage GPU parallelism |
| **concurrency** | = GPU_count | One actor per GPU |

**Example GPU configuration**:
```python
import torch

# Optimized for GPU cluster
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1,        # 1 GPU per actor
    num_cpus=1,        # Minimal CPU
    batch_size=32,     # Larger batch for GPU
    concurrency=torch.cuda.device_count()  # One actor per GPU
)
```

---

## Performance Expectations

### Throughput Comparison

| Hardware | Small Model (ResNet-50) | Large Model (ViT-Large) | Notes |
|----------|------------------------|------------------------|-------|
| **CPU (16 cores)** | 50-150 images/sec | 20-50 images/sec | Varies by CPU type |
| **GPU (1x A10)** | 500-1000 images/sec | 200-400 images/sec | Good for most workloads |
| **GPU (8x A10)** | 4000-8000 images/sec | 1600-3200 images/sec | Production scale |

:::tip Performance Insights
**Key insight**: Both CPU and GPU get similar **relative** speedups from Ray Data optimizations!

- **Naive approach**: 1x baseline (slow)
- **With Ray Data optimization**: 10-50x faster
- **Hardware difference**: GPU is 5-10x faster than CPU for same code

**Example**: If naive CPU inference is 5 images/sec, optimized Ray Data on CPU achieves 50-200 images/sec. With GPU, you get 500-2000 images/sec - but the optimization patterns are identical!
:::

---

## Common Patterns

### Pattern 1: Conditional Resource Allocation

```python
import torch

def get_optimal_config():
    """Generate optimal configuration based on available hardware."""
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        return {
            "resource": {"num_gpus": 1, "num_cpus": 1},
            "batch_size": 32,
            "concurrency": torch.cuda.device_count(),
            "expected_throughput": "500-2000 images/sec"
        }
    else:
        import ray
        cpu_cores = int(ray.available_resources()["CPU"])
        return {
            "resource": {"num_cpus": 2},
            "batch_size": 16,
            "concurrency": cpu_cores // 2,
            "expected_throughput": "50-200 images/sec"
        }

config = get_optimal_config()
print(f"Optimal config: {config}")

# Apply configuration
results = dataset.map_batches(
    InferenceWorker,
    **config["resource"],
    batch_size=config["batch_size"],
    concurrency=config["concurrency"]
)
```

---

### Pattern 2: Helper Function for Model Loading

```python
def load_model_adaptive(model_name):
    """Load model on available hardware (CPU or GPU)."""
    import torch
    from transformers import pipeline
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device >= 0 else "CPU"
    
    print(f"Loading {model_name} on {device_name}...")
    
    model = pipeline(
        "image-classification",
        model=model_name,
        device=device
    )
    
    print(f"✅ {model_name} ready on {device_name}")
    return model

# Use in your InferenceWorker
class InferenceWorker:
    def __init__(self):
        self.model = load_model_adaptive("microsoft/resnet-50")
```

---

### Pattern 3: Adaptive Batch Size Selection

```python
def find_optimal_batch_size_adaptive(worker_class, test_dataset):
    """Find optimal batch size for available hardware."""
    import torch
    
    has_gpu = torch.cuda.is_available()
    
    # Different ranges for CPU vs GPU
    if has_gpu:
        sizes_to_test = [128, 64, 32, 16, 8]
    else:
        sizes_to_test = [32, 16, 8, 4]
    
    for batch_size in sizes_to_test:
        print(f"Testing batch_size={batch_size} on {'GPU' if has_gpu else 'CPU'}...")
        try:
            results = test_dataset.limit(10).map_batches(
                worker_class,
                num_gpus=1 if has_gpu else 0,
                num_cpus=1 if has_gpu else 2,
                batch_size=batch_size
            )
            results.take(5)
            print(f"✅ Optimal batch_size={batch_size}")
            return batch_size
        except (RuntimeError, MemoryError) as e:
            if "memory" in str(e).lower():
                print(f"❌ batch_size={batch_size} OOM, trying smaller...")
                continue
            raise
    
    return 4  # Fallback minimum
```

---

## Troubleshooting

### Issue: Code Only Works on GPU

**Problem**: Template example fails with "CUDA not available" or similar

**Solution**: Add hardware detection:
```python
import torch

# Before any GPU-specific code
if not torch.cuda.is_available():
    print("GPU not available - running on CPU instead")
    # Use CPU-specific configuration
```

---

### Issue: CPU Performance Too Slow

**Problem**: CPU inference much slower than expected

**Solutions**:

1. **Increase CPU allocation**:
```python
# Give each actor more CPU cores
results = dataset.map_batches(
    InferenceWorker,
    num_cpus=4,  # Increased from 2
    concurrency=2  # Fewer parallel actors
)
```

2. **Reduce batch size** (if memory bound):
```python
results = dataset.map_batches(
    InferenceWorker,
    num_cpus=2,
    batch_size=8  # Reduced from 16
)
```

3. **Use smaller/faster models**:
```python
# Instead of large models, use distilled versions on CPU
# e.g., "mobilenet" instead of "resnet-152"
```

---

### Issue: GPU Out of Memory

**Problem**: GPU OOM errors with template examples

**Solution**: Reduce batch size and add error handling:
```python
import torch

if torch.cuda.is_available():
    for batch_size in [32, 16, 8, 4]:
        try:
            results = dataset.limit(10).map_batches(
                InferenceWorker,
                num_gpus=1,
                batch_size=batch_size
            )
            results.take(5)
            print(f"Using batch_size={batch_size}")
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"batch_size={batch_size} too large, trying smaller...")
                torch.cuda.empty_cache()
                continue
            raise
```

---

## Best Practices

### ✅ DO

1. **Always detect hardware availability**:
```python
import torch
HAS_GPU = torch.cuda.is_available()
```

2. **Use adaptive resource allocation**:
```python
num_gpus=1 if HAS_GPU else 0,
num_cpus=1 if HAS_GPU else 2
```

3. **Print hardware information for debugging**:
```python
print(f"Running on: {'GPU' if HAS_GPU else 'CPU'}")
```

4. **Test with small datasets first**:
```python
dataset.limit(100).map_batches(...)  # Test before full run
```

---

### ❌ DON'T

1. **Don't assume GPU availability**:
```python
# ❌ BAD
device = 0  # Crashes on CPU-only clusters

# ✅ GOOD
device = 0 if torch.cuda.is_available() else -1
```

2. **Don't use fixed batch sizes**:
```python
# ❌ BAD
batch_size = 128  # May OOM on CPU

# ✅ GOOD
batch_size = 32 if HAS_GPU else 16
```

3. **Don't ignore OOM errors**:
```python
# ❌ BAD
results = dataset.map_batches(...)  # May crash

# ✅ GOOD
try:
    results = dataset.map_batches(...)
except (RuntimeError, MemoryError):
    # Reduce batch_size and retry
```

---

## Quick Reference

### Minimal Adaptive Code Template

```python
import ray
import torch

# Initialize
ray.init()
HAS_GPU = torch.cuda.is_available()

# Define worker (adapts automatically)
class InferenceWorker:
    def __init__(self):
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.model = pipeline("image-classification", model="resnet-50", device=device)
    
    def __call__(self, batch):
        return [self.model(img) for img in batch["image"]]

# Run inference (adapts to hardware)
results = dataset.map_batches(
    InferenceWorker,
    num_gpus=1 if HAS_GPU else 0,
    num_cpus=1 if HAS_GPU else 2,
    batch_size=32 if HAS_GPU else 16,
    concurrency=torch.cuda.device_count() if HAS_GPU else 4
)
```

**This pattern works identically on CPU-only and GPU clusters!**

---

## Summary

**Key Takeaways:**

1. ✅ **All templates work on CPU and GPU** - just adapt resource parameters
2. ✅ **Same optimization patterns apply** - actor-based loading, batching, concurrency
3. ✅ **Hardware detection is simple** - use `torch.cuda.is_available()`
4. ✅ **Performance scales appropriately** - GPU faster, but CPU benefits too
5. ✅ **Development → Production path** - develop on CPU, deploy on GPU seamlessly

**The beauty of Ray Data**: Write once, run anywhere - CPU or GPU!

---

## Related Documentation

- [Part 1: Inference Fundamentals](01-inference-fundamentals.md) - See adaptive examples
- [Part 2: Advanced Optimization](02-advanced-optimization.md) - CPU+GPU optimization patterns
- [Production Checklist](../PRODUCTION_CHECKLIST.md) - Deployment guidance
- [Performance Tuning Guide](../PERFORMANCE_TUNING_GUIDE.md) - Optimization strategies

---

**Questions?** All template examples include CPU+GPU adaptive code - just run them on your available hardware!
