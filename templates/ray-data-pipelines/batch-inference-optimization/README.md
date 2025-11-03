# Batch Inference Optimization With Ray Data

**Time to complete**: 65 min (across 3 parts) | **Difficulty**: Intermediate | **Prerequisites**: ML inference experience, Python knowledge

Create an optimized ML batch inference pipeline that demonstrates the performance difference between naive and efficient approaches. Learn how Ray Data's actor-based patterns eliminate common bottlenecks in production ML inference.

## Template Parts

This template is split into three parts for comprehensive learning:

| Part | Description | Time | File |
|------|-------------|------|------|
| **Part 1** | Inference Fundamentals | 20 min | [01-inference-fundamentals.md](01-inference-fundamentals.md) |
| **Part 2** | Advanced Optimization | 20 min | [02-advanced-optimization.md](02-advanced-optimization.md) |
| **Part 3** | Ray Data Architecture | 25 min | [03-ray-data-architecture.md](03-ray-data-architecture.md) |

## What You'll Learn

### Part 1: Inference Fundamentals
Learn the core concepts of batch inference optimization by comparing inefficient and efficient approaches:
- **Setup**: Initialize Ray Data for accelerated inference (CPU or GPU)
- **The Wrong Way**: Understand anti-patterns that cause performance bottlenecks
- **Why It Fails**: Learn the technical reasons behind poor performance
- **The Right Way**: Implement optimized actor-based inference (works on CPU and GPU)

### Part 2: Ray Data Architecture
Understand how Ray Data's architecture enables optimization:
- **Streaming Execution**: How Ray Data processes unlimited datasets with constant memory
- **Blocks and Memory Model**: Understanding object store, heap memory, and zero-copy
- **Operators and Fusion**: How Ray Data combines operations for efficiency
- **Resource Management**: Automatic backpressure and dynamic allocation
- **Architecture-Informed Decisions**: Calculate optimal batch_size and concurrency from first principles

### Part 3: Advanced Optimization
Master systematic optimization techniques for production deployment:
- **Decision Framework**: Learn when to use each optimization parameter
- **Advanced Techniques**: Multi-model ensembles, systematic parameter tuning
- **Performance Monitoring**: Use Ray Dashboard for optimization decisions
- **Production Deployment**: Best practices for enterprise-scale inference

## Learning Objectives

**Why batch inference optimization matters**: Poor optimization wastes significant compute resources through repeated model loading and inefficient batching. Understanding these bottlenecks is crucial for production ML systems.

**Ray Data's inference capabilities**: Stateful per-worker model loading and distributed processing eliminate performance bottlenecks that plague traditional ML pipelines.

**Real-world optimization patterns**: Production ML systems at companies like Netflix, Tesla, and search engines process millions of inference requests using distributed techniques.

## Overview

**Challenge**: Batch inference bottlenecks waste compute resources and slow ML pipelines:
- **Repeated model loading**: Loading 500MB+ models for every batch wastes 97% of time
- **Poor resource utilization**: Small batches leave CPUs/GPUs idle 90% of the time
- **Memory inefficiency**: Materializing full datasets causes OOM errors
- **Sequential processing**: Single-threaded inference limits throughput

**Solution**: Ray Data's actor-based inference eliminates common bottlenecks:

| Inference Challenge | Naive Approach | Ray Data Solution | Performance Impact |
|---------------------|---------------|-------------------|-------------------|
| **Model Loading** | Load per batch (2-5 sec) | Load once per actor (one-time cost) | 10-100x throughput improvement |
| **Batch Sizing** | Small batches (4-16 samples) | Optimized batches (32-128 samples) | 5-10x resource utilization |
| **Resource Management** | Manual allocation | Automatic with `num_gpus` (GPU) or `num_cpus` (CPU) | Zero configuration |
| **Concurrency** | Sequential or over-subscribed | Optimal actor pool with `concurrency` param | Maximum cluster efficiency |

:::tip Ray Data for ML Inference
Batch inference showcases Ray Data's strengths for ML workloads:
- **Stateful actors**: Models load once in `__init__()`, reused across 1000s of batches
- **Resource allocation**: `num_gpus=1` (GPU) or `num_cpus=2` (CPU) for proper resource management
- **Batch optimization**: `batch_size` parameter controls memory vs throughput
- **Concurrency tuning**: `concurrency` parameter matches cluster resources (GPUs or CPU cores)
- **Built-in monitoring**: Ray Dashboard shows resource utilization and bottlenecks
- **Universal patterns**: Same optimization patterns work on CPU-only and GPU clusters
:::

**Impact**: Companies use these patterns at massive scale:
- **OpenAI**: Processes billions of ChatGPT requests using Ray Serve (built on Ray Data patterns)
- **Tesla**: Analyzes millions of autonomous driving images using distributed inference
- **Netflix**: Generates recommendations for 200M+ users using scalable ML pipelines
- **Shopify**: Processes millions of product images for search and categorization
- **Instacart**: Analyzes grocery images and generates recommendations at scale

**Performance results** you can expect:
- **10-100x throughput improvement** over naive implementations
- **90%+ resource utilization** vs 10-20% with poor patterns
- **Linear scaling** as you add more CPUs/GPUs to cluster
- **Sub-second latency** for batch sizes of 1000+ images

---

## Prerequisites

Before starting, ensure you have:
- [ ] Python 3.8+ with machine learning libraries installed
- [ ] Ray Data installed (`pip install "ray[data]>=2.7.0"` for full features)
- [ ] Transformers library (`pip install transformers torch`) for model examples
- [ ] Basic understanding of ML model inference concepts
- [ ] Familiarity with PyTorch or Transformers library (helpful but not required)
- [ ] Access to compute resources (CPU-only clusters work fine, GPU optional)

**Quick Installation:**
```bash
pip install "ray[data]>=2.7.0" transformers torch pillow
```

**Verify Installation:**


```python
import ray
import torch
from transformers import pipeline
print(f"Ray version: {ray.__version__}")
print(f"PyTorch version: {torch.__version__}")

# GPUs are optional for these templates
print(f"GPU available: {torch.cuda.is_available()}")
```


<div style="margin:1em 0; padding:12px 16px; border-left:4px solid #2e7d32; background:#f1f8e9; border-radius:4px;">
  <strong>All examples work on both CPU-only and GPU clusters</strong> 
  
- **GPU clusters**: Examples use `num_gpus=1` for optimal GPU acceleration
- **CPU clusters**: Simply set `num_gpus=0` or omit the parameter entirely

The optimization patterns and architecture concepts apply equally to both CPU and GPU workloads.
</div>

## Getting Started

**Recommended learning path** (65 minutes total):

1. **Start with Part 1** (20 min) - Understand fundamentals and common anti-patterns
   - See the problem: inefficient inference code
   - Learn the solution: actor-based patterns
   - Get immediate results: 10-50x throughput improvement
   
2. **Continue to Part 2** (25 min) - Deep dive into Ray Data architecture
   - How streaming execution works
   - Memory model and zero-copy optimizations
   - Calculate optimal parameters from first principles

3. **Finish with Part 3** (20 min) - Master advanced optimization techniques
   - Systematic decision frameworks
   - Multi-model ensemble patterns
   - Performance monitoring and tuning

Each part builds on the previous, so complete them in order for the best learning experience.

**Time-saving tips**:
- Each part includes a 3-5 minute "Quick Start" for immediate hands-on experience
- Code examples are copy-paste ready for rapid experimentation
- All examples work on both CPU-only and GPU clusters

**Alternative learning paths:**

**Quick path** (20 minutes - for immediate results):
- Part 1 only - Learn the basics and avoid common mistakes
- **Best for**: Quick wins, proof of concepts, immediate performance improvements
- **You'll learn**: Actor-based patterns, basic resource allocation
- **Result**: 10-50x throughput improvement in your inference code

**Architecture-focused path** (45 minutes - for deep understanding):
- Part 1 → Part 3 → Part 2
- **Best for**: Understanding how Ray Data works under the hood
- **You'll learn**: Streaming execution, memory model, operator fusion
- **Result**: Calculate optimal parameters and debug performance issues confidently

**Production path** (40 minutes - for deployment):
- Part 1 → Part 2
- **Best for**: Getting code production-ready quickly
- **You'll learn**: Practical optimization, monitoring, troubleshooting
- **Result**: Production-ready inference pipeline with monitoring
---

**Ready to begin?** → Start with [Part 1: Inference Fundamentals](01-inference-fundamentals.md)


