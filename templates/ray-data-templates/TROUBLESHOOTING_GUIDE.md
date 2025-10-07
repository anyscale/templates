# Ray Data Troubleshooting Guide

**⏱ Time to complete**: 20 min | **Purpose**: Debug common Ray Data issues

---

## Table of Contents

1. [Performance Issues](#performance-issues)
2. [Memory Issues](#memory-issues)
3. [GPU Issues](#gpu-issues)
4. [Data Issues](#data-issues)
5. [Cluster Issues](#cluster-issues)

---

## Performance Issues

### Issue: Slow Data Loading

**Symptoms:**
- Data loading takes much longer than expected
- Progress bar shows slow file reading
- Low CPU utilization during loading

**Diagnosis:**
```python
# Check parallelism
dataset = ray.data.read_parquet(path)
print(f"Number of files: {dataset.num_blocks()}")
print(f"Parallelism: Check if num_cpus too high")
```

**Solutions:**

1. **Increase I/O parallelism**:
```python
# ✅ High parallelism for I/O
dataset = ray.data.read_parquet(
    path,
    num_cpus=0.025  # More parallel tasks
)
```

2. **Use faster storage**:
   - Use S3 instead of slower network filesystems
   - Use local SSD for frequently accessed data
   - Enable S3 request acceleration

3. **Optimize file format**:
   - Use Parquet instead of CSV (10x faster)
   - Enable compression
   - Partition large datasets

**Expected improvement:** 5-20x faster loading

---

### Issue: Slow Transformations

**Symptoms:**
- Transformations take much longer than data size suggests
- CPU utilization low during processing
- Progress bar moves slowly

**Diagnosis:**
```python
# Profile your transformation
import time

def timed_transform(batch):
    start = time.time()
    result = your_transform(batch)
    duration = time.time() - start
    print(f"Batch of {len(batch)} took {duration:.3f}s")
    return result

# Test with small sample
dataset.limit(100).map_batches(
    timed_transform,
    batch_size=10
).take_all()
```

**Solutions:**

1. **Use map_batches() instead of map()**:
```python
# ❌ SLOW: Row-by-row
dataset.map(transform_row)

# ✅ FAST: Batch processing
dataset.map_batches(transform_batch, batch_size=1000)
```

2. **Vectorize operations**:
```python
def vectorized_transform(batch):
    import pandas as pd
    df = pd.DataFrame(batch)
    # Use vectorized pandas operations
    df["result"] = df["value"] * 2  # Fast!
    return df.to_dict("records")
```

3. **Adjust num_cpus**:
```python
# Light work: low num_cpus
dataset.map_batches(light_work, num_cpus=0.25)

# Heavy work: high num_cpus
dataset.map_batches(heavy_work, num_cpus=1.0)
```

**Expected improvement:** 10-100x faster

---

### Issue: Low Throughput

**Symptoms:**
- Overall pipeline much slower than expected
- Some stages bottleneck the pipeline
- Resources underutilized

**Diagnosis:**
```python
# Enable detailed progress
ctx = ray.data.DataContext.get_current()
ctx.enable_operator_progress_bars = True

# Run and observe bottlenecks
result = dataset.map_batches(stage1).map_batches(stage2).take_all()

# Check Ray Dashboard for:
# - Which stage is slowest?
# - GPU/CPU utilization
# - Task execution times
```

**Solutions:**

1. **Balance num_cpus across stages**:
```python
# Balance resource allocation
dataset = ray.data.read_parquet(path, num_cpus=0.025)  # Fast load
preprocessed = dataset.map_batches(preprocess, num_cpus=0.1)  # Fast preprocess
results = preprocessed.map_batches(infer, num_gpus=1)  # GPU inference
```

2. **Increase parallelism of bottleneck stage**:
```python
# If stage2 is bottleneck, lower its num_cpus
dataset.map_batches(stage1, num_cpus=0.5).map_batches(
    stage2, 
    num_cpus=0.1  # More parallelism
)
```

3. **Add more resources**:
   - Scale cluster horizontally (more nodes)
   - Add GPUs for inference workloads
   - Increase memory if seeing pressure

**Expected improvement:** 2-10x faster

---

## Memory Issues

### Issue: Out of Memory (OOM) Errors

**Symptoms:**
```
OutOfMemoryError: Worker killed due to memory pressure
RuntimeError: CUDA out of memory
Ray object store full
```

**Diagnosis:**
```bash
# Monitor memory usage
ray status  # Check cluster memory
nvidia-smi  # Check GPU memory

# Check Ray Dashboard:
# - Object store usage
# - Heap memory per node
# - GPU memory per device
```

**Solutions:**

1. **Reduce batch size**:
```python
# Binary search for optimal batch size
for batch_size in [64, 32, 16, 8, 4]:
    try:
        result = dataset.limit(100).map_batches(
            transform,
            batch_size=batch_size
        ).take_all()
        print(f"✅ batch_size={batch_size} works!")
        break
    except (RuntimeError, OutOfMemoryError):
        print(f"❌ batch_size={batch_size} OOM")
```

2. **Enable eager memory freeing**:
```python
ctx = ray.data.DataContext.get_current()
ctx.eager_free = True  # Aggressive cleanup
```

3. **Reduce concurrency**:
```python
# Fewer parallel actors = less total memory
dataset.map_batches(
    transform,
    concurrency=2  # Reduced from 8
)
```

4. **Use streaming execution**:
```python
# Process in smaller chunks
for batch in dataset.iter_batches(batch_size=100):
    process(batch)  # Constant memory usage
```

**Expected improvement:** Eliminates OOM errors

---

### Issue: Object Store Full

**Symptoms:**
```
ObjectStoreFullError: Failed to put object in object store
Ray object store is at capacity
```

**Diagnosis:**
```python
# Check object store usage
import ray
print(ray.cluster_resources())
print(ray.available_resources())
```

**Solutions:**

1. **Enable spilling to disk**:
```python
# Ray automatically spills when object store full
# Check Ray Dashboard -> Memory -> Spilled bytes
```

2. **Reduce block size**:
```python
ctx = ray.data.DataContext.get_current()
ctx.target_max_block_size = 64 * 1024 * 1024  # 64MB blocks
```

3. **Process data incrementally**:
```python
# Don't materialize entire dataset
# Use .take() or .iter_batches() instead of .take_all()
for batch in dataset.iter_batches(batch_size=1000):
    process_and_write(batch)
```

4. **Increase object store size**:
```bash
# Start Ray with larger object store
ray start --object-store-memory=10000000000  # 10GB
```

**Expected improvement:** Eliminates spilling, faster execution

---

## GPU Issues

### Issue: GPU Utilization at 0%

**Symptoms:**
- GPU shows 0% utilization in nvidia-smi
- Inference running but GPU idle
- Expected GPU acceleration not happening

**Diagnosis:**
```bash
# Monitor GPU while running
watch -n 0.5 nvidia-smi

# Check Ray Dashboard:
# - Are tasks actually using GPUs?
# - Is num_gpus specified?
```

**Solutions:**

1. **Verify num_gpus specified**:
```python
# ❌ GPU not requested
dataset.map_batches(InferenceModel)

# ✅ GPU explicitly requested
dataset.map_batches(InferenceModel, num_gpus=1)
```

2. **Check model uses GPU**:
```python
class InferenceModel:
    def __init__(self):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model().to(device)  # Move to GPU!
        print(f"Model on device: {device}")
```

3. **Increase preprocessing parallelism**:
```python
# GPU starving - needs more data
dataset = ray.data.read_images(path, num_cpus=0.025)  # High parallelism
preprocessed = dataset.map_batches(preprocess, num_cpus=0.1)  # Fast prep
predictions = preprocessed.map_batches(infer, num_gpus=1)  # GPU fed
```

**Expected improvement:** GPU utilization > 80%

---

### Issue: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
torch.cuda.OutOfMemoryError
```

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Look for:
# - Memory usage near 100%
# - Multiple processes on same GPU
```

**Solutions:**

1. **Reduce batch size**:
```python
# Start large, reduce until stable
predictions = dataset.map_batches(
    InferenceModel,
    batch_size=16,  # Reduced from 64
    num_gpus=1
)
```

2. **Ensure only one model per GPU**:
```python
# Each actor gets dedicated GPU
predictions = dataset.map_batches(
    InferenceModel,
    num_gpus=1,  # Full GPU per actor
    concurrency=4  # 4 actors = 4 GPUs needed
)
```

3. **Use smaller model or mixed precision**:
```python
class InferenceModel:
    def __init__(self):
        import torch
        self.model = load_model()
        self.model.half()  # FP16 = half memory!
```

**Expected improvement:** Stable GPU inference

---

## Data Issues

### Issue: Data Not Found

**Symptoms:**
```
FileNotFoundError: Path does not exist
No files found matching pattern
```

**Diagnosis:**
```python
# Verify path exists
import os
print(os.path.exists(path))

# Check S3 permissions
import boto3
s3 = boto3.client("s3")
s3.list_objects_v2(Bucket="bucket", Prefix="prefix")
```

**Solutions:**

1. **Check path format**:
```python
# Local files
dataset = ray.data.read_parquet("./data/*.parquet")

# S3
dataset = ray.data.read_parquet("s3://bucket/path/*.parquet")

# HDFS
dataset = ray.data.read_parquet("hdfs://namenode:port/path")
```

2. **Verify permissions**:
```bash
# AWS
aws s3 ls s3://bucket/path/

# Check IAM roles and policies
```

3. **Use absolute paths**:
```python
from pathlib import Path
data_path = Path("/absolute/path/to/data").resolve()
dataset = ray.data.read_parquet(str(data_path))
```

---

### Issue: Schema Mismatches

**Symptoms:**
```
pyarrow.lib.ArrowInvalid: Schema mismatch
ValueError: Column types do not match
```

**Diagnosis:**
```python
# Check schema
dataset = ray.data.read_parquet(path)
print(dataset.schema())

# Compare schemas across files
for file in files:
    ds = ray.data.read_parquet(file)
    print(f"{file}: {ds.schema()}")
```

**Solutions:**

1. **Enforce schema**:
```python
import pyarrow as pa

schema = pa.schema([
    ("id", pa.int64()),
    ("value", pa.float64()),
    ("name", pa.string())
])

dataset = ray.data.read_parquet(path, schema=schema)
```

2. **Cast columns**:
```python
def cast_types(batch):
    import pandas as pd
    df = pd.DataFrame(batch)
    df["id"] = df["id"].astype("int64")
    df["value"] = df["value"].astype("float64")
    return df.to_dict("records")

dataset = dataset.map_batches(cast_types)
```

3. **Handle missing columns**:
```python
def add_missing_columns(batch):
    for row in batch:
        if "optional_column" not in row:
            row["optional_column"] = None
    return batch

dataset = dataset.map_batches(add_missing_columns)
```

---

## Cluster Issues

### Issue: Workers Not Connecting

**Symptoms:**
- Workers show as disconnected in Ray Dashboard
- Cluster has fewer resources than expected
- "Waiting for resources" messages

**Diagnosis:**
```bash
# Check cluster status
ray status

# Check worker logs
tail -f /tmp/ray/session_latest/logs/worker-*.log
```

**Solutions:**

1. **Verify network connectivity**:
```bash
# Workers must reach head node
ping head-node-ip

# Check firewall rules
# Ports 6379, 8265, 10001-10100 must be open
```

2. **Check Ray versions match**:
```bash
# On all nodes
ray --version

# Should be identical
```

3. **Restart cluster**:
```bash
# Stop all nodes
ray stop

# Start head
ray start --head

# Start workers
ray start --address="head-ip:6379"
```

---

## Quick Debugging Checklist

When things go wrong, check:

- [ ] **Logs**: Check `/tmp/ray/session_latest/logs/`
- [ ] **Dashboard**: Open Ray Dashboard at `localhost:8265`
- [ ] **Resources**: Run `ray status` to check available resources
- [ ] **GPU**: Run `nvidia-smi` to check GPU utilization
- [ ] **Memory**: Check object store and heap memory usage
- [ ] **Config**: Verify num_cpus, num_gpus, batch_size settings
- [ ] **Versions**: Ensure Ray version matches across cluster
- [ ] **Network**: Verify connectivity between nodes

---

## Getting Help

If you're still stuck:

1. **Check Ray documentation**: https://docs.ray.io/
2. **Search Ray Discuss**: https://discuss.ray.io/
3. **File GitHub issue**: https://github.com/ray-project/ray/issues
4. **Join Ray Slack**: https://forms.gle/9TSdDYUgxYs8SA9e8

**When asking for help, include:**
- Ray version (`ray --version`)
- Minimal reproducible example
- Full error message and stack trace
- Ray Dashboard screenshot
- System info (OS, Python version, GPU type)

