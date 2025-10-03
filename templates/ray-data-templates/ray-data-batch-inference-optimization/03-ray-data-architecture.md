# Part 3: Ray Data Architecture and Optimization

**⏱️ Time to complete**: 25 min

**[← Back to Part 2](02-advanced-optimization.md)** | **[Return to Overview](README.md)**

---

## What You'll Learn

Understanding Ray Data's architecture is critical for making informed optimization decisions. In this part, you'll learn:

1. How Ray Data's streaming execution model enables efficient batch inference
2. How blocks and the object store affect memory management
3. How operators and planning impact performance
4. How resource management affects optimization strategies
5. Why these architectural choices matter for your inference workloads

## Prerequisites

Complete [Part 1: Inference Fundamentals](01-inference-fundamentals.md) and [Part 2: Advanced Optimization](02-advanced-optimization.md) before starting this part.

## Table of Contents

1. [Streaming Execution Model](#streaming-execution-model)
2. [Datasets and Blocks](#datasets-and-blocks)
3. [Ray Memory Model](#ray-memory-model)
4. [Operators and Planning](#operators-and-planning)
5. [Resource Management](#resource-management-and-backpressure)
6. [Optimization Implications](#optimization-implications)

---

## Streaming Execution Model

### Why Streaming Execution Matters for Batch Inference

Traditional batch processing loads entire datasets into memory before processing. For batch inference with 1M+ images or documents, this approach fails:

**Traditional Batch Processing Problems:**
- **Memory explosion**: Loading 1M images × 3MB each = 3TB memory required
- **No pipeline parallelism**: Model loading, preprocessing, inference all sequential
- **Long time to first result**: Wait for all data to load before any inference
- **OOM errors**: Cluster runs out of memory frequently

**Ray Data Streaming Execution Solution:**
- **Constant memory**: Process 128MB blocks at a time, not full dataset
- **Pipeline parallelism**: Load, preprocess, and infer simultaneously
- **Fast time to first result**: Start inferring as soon as first block loads
- **Automatic backpressure**: Prevents memory overflow dynamically

### Visualizing the Difference

**Traditional Batch Processing:**

<img src="https://anyscale-materials.s3.us-west-2.amazonaws.com/cko-2025-q1/batch-processing.png" width="800" alt="Traditional Batch Processing">

**Problems with traditional approach:**
- x High memory - requires loading entire dataset
- x No parallelism - stages run sequentially  
- x Long latency - wait for complete load before processing
- x Wasted resources - GPUs idle during load/write stages

**Ray Data Streaming Execution:**

<img src="https://anyscale-materials.s3.us-west-2.amazonaws.com/cko-2025-q1/pipelining.png" width="800" alt="Ray Data Streaming Execution">

**Benefits of streaming execution:**
- - Low memory - constant 128MB blocks regardless of dataset size
- - Pipeline parallelism - all stages active simultaneously
- - Fast first result - inference starts immediately
- - Maximum throughput - all resources utilized continuously

### How This Affects Your Optimization Decisions

Understanding streaming execution helps you make better optimization choices:

**1. Batch Size Selection:**
- **Don't make batch_size too large**: Risk memory overflow
- **Don't make batch_size too small**: Waste GPU capacity
- **Sweet spot**: Match GPU memory and throughput needs

**2. Concurrency Configuration:**
- **Too many actors**: Backpressure kicks in, actors idle waiting for resources
- **Too few actors**: Underutilized cluster, low throughput
- **Optimal**: Match available GPUs and memory constraints

**3. Model Loading Strategy:**
- **Why actors work**: Model loads once, reused across many blocks
- **Why tasks fail**: Model reloads for every block, massive overhead
- **Architecture enables**: Stateful processing without memory bloat

### Practical Example

```python
import ray

# Example: How streaming execution enables large-scale inference
images = ray.data.read_images("s3://images/", num_cpus=0.05)  # 1M images, 3TB total

# This works even if cluster only has 64GB memory!
# Why? Streaming execution processes 128MB blocks at a time
results = images.map_batches(
    InferenceModel,  # Loads once per actor
    batch_size=32,   # Small batches prevent memory overflow
    num_gpus=1,      # One model per GPU
    concurrency=4    # 4 parallel actors
)

# As you iterate results, Ray Data:
# 1. Loads blocks from S3 (streaming)
# 2. Preprocesses in parallel (pipeline parallelism)
# 3. Runs inference on GPUs (distributed)
# 4. Writes results (continuous output)
# All while maintaining constant memory footprint!

for batch in results.iter_batches(batch_size=1000):
    # First results available immediately
    # Don't need to wait for all 1M images
    process_results(batch)
```

---

## Datasets and Blocks

### What Are Blocks?

A **block** is the fundamental unit of data in Ray Data. Understanding blocks is essential for optimization.

**Block characteristics:**
- **Size**: Typically 128MB (configurable via `target_max_block_size`)
- **Format**: Stored as PyArrow tables or pandas DataFrames
- **Location**: Ray object store (shared memory)
- **Processing unit**: One block = one task typically

### Why Block Size Matters for Batch Inference

Block size directly impacts inference performance:

**Block Size Too Small (e.g., 1MB):**
- x Too many tasks created (scheduling overhead)
- x Poor GPU utilization (small batches)
- x High network overhead (many small transfers)
- x Scheduler bottleneck (managing thousands of tasks)

**Block Size Too Large (e.g., 1GB):**
- x Memory pressure (blocks don't fit in object store)
- x Poor parallelism (few blocks = few parallel tasks)
- x Spilling to disk (performance degradation)
- x Uneven load balancing (some workers idle)

**Optimal Block Size (128MB default):**
- - Good parallelism (many blocks for distribution)
- - Low overhead (reasonable number of tasks)
- - Fits in memory (object store can hold multiple blocks)
- - Efficient transfer (network overhead manageable)

### Configuring Block Size for Inference

```python
import ray

# Configure block size for your inference workload
ctx = ray.data.DataContext.get_current()

# Default: 128MB blocks
print(f"Default max block size: {ctx.target_max_block_size / 1024**2:.0f}MB")

# For image inference with large images:
# Smaller blocks = more parallelism
ctx.target_max_block_size = 64 * 1024**2  # 64MB blocks

# For text inference with small documents:
# Larger blocks = less overhead
ctx.target_max_block_size = 256 * 1024**2  # 256MB blocks

# Load images with configured block size
images = ray.data.read_images("s3://images/", num_cpus=0.05)

print(f"Dataset blocks: {images.num_blocks()}")
print(f"Estimated blocks: {images.size_bytes() / ctx.target_max_block_size:.0f}")
```

### How Blocks Flow Through Inference Pipeline

Ray Data's block-based architecture enables parallelism at every stage:

<img src="https://docs.ray.io/en/latest/_images/dataset-arch.svg" width="700" alt="Ray Data Block Architecture">

**Key insights:**
- Each block contains a disjoint subset of rows
- Blocks are processed in parallel across the cluster
- Distributed object store enables efficient block transfer
- Tasks operate on individual blocks for maximum parallelism

---

## Ray Memory Model

### Object Store and Heap Memory

Ray manages two types of memory that affect batch inference:

<img src="https://docs.ray.io/en/latest/_images/memory.svg" width="600" alt="Ray Memory Model">

**1. Object Store Memory (30% of node memory by default):**
- **Purpose**: Shared memory for passing data between tasks
- **Contents**: Blocks (PyArrow tables), task outputs, intermediate results
- **Optimization impact**: Determines how many blocks can be in-flight
- **When full**: Triggers spilling to disk (major performance hit)

**2. Heap Memory (70% of node memory by default):**
- **Purpose**: Task execution, model loading, preprocessing
- **Contents**: Loaded models, batch data being processed, Python objects
- **Optimization impact**: Determines how many models fit in memory
- **When full**: Python out-of-memory errors, task failures

### How This Affects Batch Inference Optimization

```
Node Memory: 64GB
├── Object Store (30% = 19GB)
│   ├── Block 1 (128MB)
│   ├── Block 2 (128MB)
│   ├── ...
│   └── Block N (up to ~148 blocks fit)
│
└── Heap Memory (70% = 45GB)
    ├── Model weights (5GB per model)
    ├── Batch preprocessing (2GB per actor)
    ├── Python overhead (1GB)
    └── Available for actors: ~37GB
        → Can fit ~7 model actors at 5GB each
```

**Optimization implications:**

**Object Store Pressure:**
- **Symptom**: "Object store full" warnings in logs
- **Cause**: Too many blocks generated too fast
- **Solution**: Reduce `concurrency` or increase `batch_size`

**Heap Memory Pressure:**
- **Symptom**: Out-of-memory errors, task failures
- **Cause**: Too many models loaded or batch_size too large
- **Solution**: Reduce `concurrency` or `batch_size`

### Practical Memory Configuration

```python
import ray

# Configure Ray Data to respect memory limits
ctx = ray.data.DataContext.get_current()

# Set object store memory limit for inference
# Reserve 50% of object store (default) for Ray Data
ctx.execution_options.resource_limits.object_store_memory = 10e9  # 10GB limit

# This prevents Ray Data from overwhelming the object store
# Automatically triggers backpressure when limit reached

# Example: Conservative memory settings for large models
images = ray.data.read_images("s3://images/", num_cpus=0.05)

results = images.map_batches(
    LargeModelInference,  # 10GB model
    batch_size=16,        # Small batches (GPU memory constraint)
    num_gpus=1,
    concurrency=2         # Only 2 models (heap memory constraint)
)

# Ray Data will:
# - Backpressure image loading when object store fills
# - Limit concurrent tasks to respect memory limits
# - Automatically balance throughput vs memory usage
```

### Zero-Copy Optimization

Ray Data uses **zero-copy deserialization** for efficiency:

**What it means:**
- Blocks stored in object store are PyArrow tables
- Accessing a block doesn't copy data - just creates a pointer
- Multiple tasks can read same block without duplication

**Why it matters for inference:**
- **Memory efficiency**: 10 actors can share same preprocessed block
- **Performance**: No serialization overhead between stages
- **Scalability**: Enables high-throughput pipelines

**Practical impact:**
```python
# Without zero-copy (hypothetical):
# Block in object store: 128MB
# Actor 1 reads block: +128MB copy → 256MB total
# Actor 2 reads block: +128MB copy → 384MB total
# Result: 3x memory usage!

# With zero-copy (Ray Data actual):
# Block in object store: 128MB
# Actor 1 reads block: 0MB copy (pointer) → 128MB total
# Actor 2 reads block: 0MB copy (pointer) → 128MB total
# Result: Constant memory!
```

**Visual representation of object store usage:**

<img src="https://anyscale-materials.s3.us-west-2.amazonaws.com/ray-data-deep-dive/producer-consumer-object-store-v2.png" width="700" alt="Object Store Data Flow">

---

## Operators and Planning

### Logical vs Physical Plans

Ray Data transforms your code into an optimized execution plan:

**Your Code:**
```python
results = (
    ray.data.read_images("s3://images/")
    .map_batches(preprocess_images)
    .map_batches(InferenceModel, num_gpus=1)
)
```

**Logical Plan (What to do):**
```
ReadFiles → MapBatches[preprocess] → MapBatches[inference]
```

**Physical Plan (How to do it):**
```
TaskPoolMapOperator[ReadFiles→preprocess→inference]
```

**Note the fusion:** Ray Data combined all three operations into a single operator!

### Operator Fusion and Its Impact on Inference

**Operator fusion** combines multiple operations into single tasks to reduce overhead.

**Benefits for batch inference:**
- **Reduced data movement**: Preprocessed images go straight to model (no object store roundtrip)
- **Lower task overhead**: One task instead of three per block
- **Better GPU utilization**: Continuous processing without gaps
- **Memory efficiency**: Intermediate results stay in task memory

**Example showing fusion:**

```python
import ray
from ray.data._internal.logical.optimizers import get_execution_plan

# Create inference pipeline
ds = (
    ray.data.read_images("s3://images/")
    .map_batches(preprocess, batch_size=32)
    .map_batches(InferenceModel, batch_size=32, num_gpus=1)
)

# Inspect the execution plan
physical_plan = get_execution_plan(ds._plan._logical_plan)
print(physical_plan.dag)

# Output shows: TaskPoolMapOperator[ReadFiles->MapBatches->MapBatches]
# All three operations fused into single operator!
```

**When fusion happens:**
- Same compute configuration (both use tasks or both use actors)
- Compatible batch sizes
- Compatible resource requirements
- No shuffle/repartition operations between them

**When fusion doesn't happen:**
- Different compute strategies (task vs actor)
- Different resource requirements (CPU vs GPU)
- Shuffle operations (groupby, sort, repartition)

**Optimization strategy:**
Keep preprocessing and inference configs compatible to enable fusion:

```python
# Good: Fusion enabled
images.map_batches(preprocess, batch_size=32).map_batches(InferenceModel, batch_size=32, num_gpus=1)

# Suboptimal: Fusion disabled (different batch sizes)
images.map_batches(preprocess, batch_size=64).map_batches(InferenceModel, batch_size=32, num_gpus=1)
```

---

## Resource Management and Backpressure

### Dynamic Resource Allocation

Ray Data automatically manages resources across operators to maximize throughput. Understanding this helps you set optimal parameters.

**The Challenge:**
- **Too aggressive loading**: Object store fills up, spilling to disk
- **Too conservative loading**: GPUs idle waiting for data
- **Unbalanced pipeline**: Some stages bottleneck while others wait

**Ray Data's Solution:**
Dynamically allocates resources based on operator throughput and backpressure policies.

### Backpressure Mechanisms

**1. Submission-Based Backpressure:**
Prevents operators from submitting new tasks when resource budgets exceeded.

**Example scenario:**
```
GPU inference slower than data loading
↓
Object store filling with preprocessed images
↓
Ray Data backpressures data loading
↓
Loading slows down to match inference throughput
↓
Balanced pipeline - no memory overflow
```

**2. Output-Based Backpressure:**
Limits how many task outputs move to operator queues based on memory availability.

**Practical impact on inference:**
```python
# Scenario: Fast preprocessing, slow inference

images = ray.data.read_images("s3://images/", num_cpus=0.05)

results = images.map_batches(
    fast_preprocess,     # Processes 1000 images/sec
    batch_size=64,
    concurrency=16        # Many parallel preprocessors
).map_batches(
    SlowInferenceModel,  # Processes 100 images/sec
    batch_size=16,
    num_gpus=1,
    concurrency=2         # Only 2 GPUs available
)

# What Ray Data does automatically:
# 1. Preprocessing generates blocks faster than inference consumes
# 2. Object store starts filling with preprocessed blocks
# 3. Backpressure kicks in - preprocessing tasks not scheduled
# 4. Pipeline balances - preprocessing matches inference rate
# 5. Memory stays constant - no overflow despite throughput mismatch
```

### Resource Budgets and Limits

Ray Data allocates resources using reservation-based budgeting:

**Default behavior:**
- **50% reserved for outputs**: Ensures downstream operators have resources
- **50% shared across operators**: Enables flexible allocation
- **Dynamic adjustment**: Budgets change based on operator throughput

**Configure limits for inference workloads:**

```python
import ray

ctx = ray.data.DataContext.get_current()

# Set overall object store memory limit
# Useful when running inference alongside other workloads
ctx.execution_options.resource_limits.object_store_memory = 50e9  # 50GB

# Set CPU limits
# Useful to reserve CPUs for other processes
ctx.execution_options.resource_limits.cpu = 32  # Use only 32 CPUs

# Set GPU limits
# Useful when sharing cluster with training jobs
ctx.execution_options.resource_limits.gpu = 4  # Use only 4 GPUs

# Exclude specific resources
# Useful to reserve resources for head node or other services
from ray.data import ExecutionResources
ctx.execution_options.exclude_resources = ExecutionResources(cpu=4, gpu=1)

print("Resource limits configured for batch inference")
```

### Monitoring Resource Usage

```python
# Enable Ray Data resource manager debug logging
import os
os.environ['RAY_DATA_DEBUG_RESOURCE_MANAGER'] = '1'

# Run your inference pipeline
results = images.map_batches(InferenceModel, num_gpus=1, concurrency=4)

# You'll see output like:
# [ResourceManager] Operator budgets:
#   ReadImages: object_store_memory=5.0GB, cpu=8.0
#   MapBatches: object_store_memory=5.0GB, cpu=0.0, gpu=4.0
#   MapBatches: object_store_memory=10.0GB, cpu=0.0, gpu=0.0
#
# This shows:
# - How resources are allocated across operators
# - Where bottlenecks might occur
# - If backpressure is active
```

---

## Optimization Implications

### How Architecture Informs Optimization Decisions

Understanding Ray Data's architecture helps you make better optimization choices:

#### 1. Choosing Batch Size

**Architectural considerations:**
- **Block size**: Batch size should divide evenly into block size for efficiency
- **GPU memory**: Batch must fit in GPU memory during inference
- **Object store**: Preprocessed batches must fit in object store
- **Throughput**: Larger batches = better GPU utilization (up to a point)

**Decision framework:**
```python
# Calculate optimal batch size

# Factor 1: GPU memory constraint
gpu_memory_gb = 16  # Your GPU memory
model_size_gb = 5
batch_overhead_mb = 50  # Per sample
max_batch_from_gpu = int((gpu_memory_gb - model_size_gb) * 1024 / batch_overhead_mb)

# Factor 2: Block size alignment
block_size_mb = 128
samples_per_block = block_size_mb / 3  # 3MB per image
ideal_batch_for_blocks = int(samples_per_block / 4)  # 4 batches per block

# Factor 3: Throughput testing
# Test different sizes: 16, 32, 64, 128
# Choose largest that doesn't cause memory issues

# Final choice: Minimum of all constraints
optimal_batch_size = min(max_batch_from_gpu, ideal_batch_for_blocks, 128)
print(f"Optimal batch size: {optimal_batch_size}")
```

#### 2. Choosing Concurrency

**Architectural considerations:**
- **GPU count**: One actor per GPU maximum
- **Memory per actor**: Model size + batch size determines how many actors fit
- **Object store capacity**: More actors = more in-flight blocks
- **CPU availability**: Preprocessing may need CPUs

**Decision framework:**
```python
# Calculate optimal concurrency

# Factor 1: GPU constraint
num_gpus = 8  # Available GPUs
max_concurrency_gpu = num_gpus  # One model per GPU

# Factor 2: Memory constraint
node_heap_memory_gb = 64 * 0.7  # 70% of 64GB node
model_size_gb = 5
batch_memory_gb = 2
memory_per_actor = model_size_gb + batch_memory_gb
max_concurrency_memory = int(node_heap_memory_gb / memory_per_actor)

# Factor 3: Object store constraint
object_store_gb = 64 * 0.3  # 30% of 64GB node
block_size_gb = 0.128
blocks_in_flight_per_actor = 2  # Preprocessing + inference
required_object_store = concurrency * blocks_in_flight_per_actor * block_size_gb
max_concurrency_object_store = int(object_store_gb / (blocks_in_flight_per_actor * block_size_gb))

# Final choice: Minimum of all constraints
optimal_concurrency = min(
    max_concurrency_gpu,
    max_concurrency_memory,
    max_concurrency_object_store
)

print(f"Optimal concurrency: {optimal_concurrency}")
print(f"  GPU limit: {max_concurrency_gpu}")
print(f"  Memory limit: {max_concurrency_memory}")
print(f"  Object store limit: {max_concurrency_object_store}")
```

#### 3. Actor vs Task Decision

**Architectural insight:** Actors are stateful, tasks are stateless.

| Aspect | Tasks | Actors | Best For Inference |
|--------|-------|--------|-------------------|
| **Startup** | Launch per invocation | Launch once, reuse | - Actors (amortize model loading) |
| **State** | Stateless | Stateful | - Actors (keep model in memory) |
| **Resource overhead** | Low | Higher | Depends on model size |
| **Scheduling overhead** | Higher (many tasks) | Lower (few actors) | - Actors (fewer scheduling decisions) |
| **Memory** | Released after task | Held by actor | Tasks if memory-constrained |

**For batch inference:** Almost always use actors because:
- Model loading is expensive (2-5 seconds)
- Models are large (500MB - 10GB)
- Amortizing load cost across 1000s of batches is critical

#### 4. Understanding Performance Bottlenecks

**Use Ray Dashboard to identify architectural bottlenecks:**

**Symptom 1: Low GPU utilization**
- **Possible cause**: Object store full (loading backpressured)
- **Solution**: Increase object store limit or reduce block size
- **How to verify**: Check "Ray Data Metrics (Object Store Memory)"

**Symptom 2: Spilling to disk**
- **Possible cause**: Too many concurrent actors generating blocks
- **Solution**: Reduce concurrency or increase batch_size
- **How to verify**: Check "Spilled" metric in object store

**Symptom 3: High task overhead**
- **Possible cause**: Blocks too small, too many tasks
- **Solution**: Increase target_max_block_size
- **How to verify**: Check task count vs throughput

**Symptom 4: Actors idle**
- **Possible cause**: Upstream loading too slow
- **Solution**: Increase num_cpus for read operation
- **How to verify**: Check "Ray Data Metrics (Inputs)" throughput

---

## Key Takeaways: Architecture and Optimization

### Critical Architecture Concepts

**1. Streaming Execution:**
- Enables processing datasets larger than cluster memory
- Provides pipeline parallelism for maximum throughput
- Makes batch inference scalable from 1K to 1B samples

**2. Blocks:**
- 128MB default size balances parallelism and overhead
- More blocks = more parallelism (up to a point)
- Block size affects GPU batch size and task count

**3. Memory Model:**
- Object store (30%) holds blocks and transfers
- Heap memory (70%) runs tasks and loads models
- Both limits constrain concurrency and batch size

**4. Operator Fusion:**
- Combines operations to reduce overhead
- Keeps intermediate data in task memory
- Improves throughput and reduces latency

**5. Backpressure:**
- Automatically balances pipeline stages
- Prevents memory overflow
- Maximizes throughput within resource constraints

### Optimization Decision Framework

Use this framework informed by Ray Data architecture:

```python
# Step 1: Start with architectural constraints
gpu_count = 8
gpu_memory_gb = 16
node_memory_gb = 64
model_size_gb = 5

# Step 2: Calculate concurrency from memory
heap_memory = node_memory_gb * 0.7
actors_fit = int(heap_memory / (model_size_gb + 2))  # +2GB for batches
concurrency = min(gpu_count, actors_fit)

# Step 3: Calculate batch size from GPU memory
available_gpu_mem = gpu_memory_gb - model_size_gb
sample_size_mb = 3  # Per image
batch_size = int(available_gpu_mem * 1024 / sample_size_mb / 2)  # /2 for safety

# Step 4: Configure block size for efficiency
ctx = ray.data.DataContext.get_current()
# Make blocks contain ~4 batches worth of data
ctx.target_max_block_size = batch_size * sample_size_mb * 1024**2 * 4

# Step 5: Run with optimal settings
results = images.map_batches(
    InferenceModel,
    batch_size=batch_size,
    num_gpus=1,
    concurrency=concurrency
)

print(f"Optimized configuration:")
print(f"  Concurrency: {concurrency} (limited by {min(gpu_count, actors_fit)})")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {ctx.target_max_block_size / 1024**2:.0f}MB")
```

### Architecture-Aware Performance Tips

**Tip 1: Align batch_size with block_size**

Good configuration:
- Block size: 128MB, batch_size: 32 (4 batches per block - clean division)

Suboptimal configuration:
- Block size: 128MB, batch_size: 50 (2.56 batches per block - awkward division)

**Tip 2: Monitor object store, not just GPUs**
```python
# GPU utilization high but throughput low?
# Check object store - might be spilling to disk
# Ray Dashboard → Metrics → Object Store Memory
```

**Tip 3: Use fusion-friendly patterns**

```python
# Fusion-friendly: Same compute, compatible configs
images.map_batches(prep, batch_size=32, num_cpus=1).map_batches(model, batch_size=32, num_gpus=1)

# Fusion-incompatible: Different batch sizes  
images.map_batches(prep, batch_size=64).map_batches(model, batch_size=32)
```

**Tip 4: Respect memory limits**
```python
# Set limits based on architecture
ctx.execution_options.resource_limits.object_store_memory = node_memory * 0.3 * 0.5

# This leaves 50% object store for other workloads
# Prevents OOM when running multiple jobs
```

---

## Practical Architecture Examples

### Example 1: Small Model, High Throughput

**Scenario:** ResNet-50 (100MB model), process 1M images

**Architectural analysis:**
- **Model size**: Small (100MB) → Many actors fit in memory
- **GPU memory**: 16GB → Large batches possible (128 images)
- **Throughput goal**: Maximize images/second

**Optimal configuration:**
```python
results = images.map_batches(
    ResNet50Model,
    batch_size=128,    # Large batches for throughput
    num_gpus=1,
    concurrency=8      # Use all 8 GPUs
)

# Why this works:
# - Heap memory: 64GB * 0.7 = 45GB
# - Per actor: 0.1GB model + 1GB batch = 1.1GB
# - Can fit: 45GB / 1.1GB = 40 actors
# - Limited by: 8 GPUs → concurrency=8
# - Object store: Minimal pressure (small batches)
```

### Example 2: Large Model, Memory-Constrained

**Scenario:** LLaMA-70B (140GB model with quantization = 35GB), process 100K documents

**Architectural analysis:**
- **Model size**: Huge (35GB) → Very few actors fit
- **GPU memory**: 80GB A100 → Moderate batches (16 documents)
- **Memory goal**: Don't OOM

**Optimal configuration:**
```python
results = documents.map_batches(
    LLaMA70BModel,
    batch_size=16,     # Conservative for large model
    num_gpus=1,
    concurrency=2      # Only 2 models fit in cluster memory
)

# Why this works:
# - Heap memory: 256GB * 0.7 = 179GB (multi-node)
# - Per actor: 35GB model + 5GB batch = 40GB
# - Can fit: 179GB / 40GB = 4 actors theoretical
# - Use: 2 for safety margin (avoid OOM)
# - Object store: Backpressure prevents overflow
```

### Example 3: Balanced Pipeline

**Scenario:** BERT (500MB), preprocessing heavy (embedding generation)

**Architectural analysis:**
- **Preprocessing**: CPU-intensive (tokenization, embedding lookup)
- **Inference**: GPU-intensive (transformer forward pass)
- **Goal**: Balance both stages

**Optimal configuration:**
```python
results = (
    documents
    .map_batches(
        heavy_preprocessing,
        batch_size=64,      # CPU batches can be larger
        num_cpus=2,         # Allocate CPUs for preprocessing
        concurrency=16       # Many CPU workers
    )
    .map_batches(
        BERTInference,
        batch_size=32,      # GPU batch size
        num_gpus=1,
        concurrency=4        # 4 GPUs
    )
)

# Why this works:
# - Preprocessing: 16 workers × 2 CPUs = 32 CPUs used
# - Inference: 4 workers × 1 GPU = 4 GPUs used
# - Backpressure: Automatically balances if mismatch
# - Fusion: Disabled (different compute), but that's okay - different resources
```

---

## Cleanup

```python
# Clean up Ray resources
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```

---

## Summary: Architecture Drives Optimization

**Key Architectural Principles for Batch Inference:**

1. **Streaming execution** enables unlimited dataset sizes with constant memory
2. **Blocks** are the unit of parallelism - more blocks = more parallel tasks
3. **Object store** holds blocks and transfers - capacity limits in-flight data
4. **Heap memory** holds models and executions - limits concurrent actors
5. **Operator fusion** reduces overhead - keep configs compatible
6. **Backpressure** prevents overflow - trust Ray Data's automatic balancing

**Optimization Strategy:**
1. Start with memory constraints (heap and object store)
2. Calculate maximum concurrency from memory limits
3. Choose batch size for GPU utilization
4. Configure block size for parallelism
5. Monitor Ray Dashboard for bottlenecks
6. Adjust based on observed behavior

**[← Back to Part 2: Advanced Optimization](02-advanced-optimization.md)** | **[Return to Overview](README.md)**

---

*This architectural deep-dive completes the batch inference optimization series. You now understand not just how to optimize, but why specific optimizations work based on Ray Data's design.*

