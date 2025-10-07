# Ray Data Quick Reference

**⏱ Time to complete**: 5 min | **Purpose**: Fast lookup for common patterns

---

## Essential Patterns

### Read Data

```python
# Parquet (recommended)
ds = ray.data.read_parquet("s3://bucket/path/*.parquet", num_cpus=0.025)

# CSV
ds = ray.data.read_csv("data/*.csv", num_cpus=0.025)

# JSON
ds = ray.data.read_json("data/*.json", num_cpus=0.025)

# Images
ds = ray.data.read_images("images/", mode="RGB", num_cpus=0.05)

# Binary files
ds = ray.data.read_binary_files("files/*", num_cpus=0.05)
```

### Transform Data

```python
# Batch processing (fast!)
ds.map_batches(transform_fn, batch_size=1000, num_cpus=0.5)

# Row-by-row (slow - avoid if possible)
ds.map(transform_row)

# Filter
ds.filter(lambda row: row["value"] > 0, num_cpus=0.1)

# Add column
ds.add_column("new_col", lambda row: row["old_col"] * 2)
```

### Aggregate Data

```python
from ray.data.aggregate import Count, Sum, Mean, Max, Min

# Group by and aggregate
result = ds.groupby("category").aggregate(
    Count(),
    Mean("value"),
    Sum("amount"),
    Max("score")
)
```

### Write Data

```python
# Parquet (recommended)
ds.write_parquet("output/", num_cpus=0.1)

# CSV
ds.write_csv("output/", num_cpus=0.1)

# JSON
ds.write_json("output/", num_cpus=0.1)
```

### GPU Inference

```python
class InferenceModel:
    def __init__(self):
        self.model = load_model()  # Load once
    
    def __call__(self, batch):
        return self.model(batch["data"])

predictions = ds.map_batches(
    InferenceModel,
    num_gpus=1,      # Reserve GPU
    batch_size=32,   # Batch for GPU
    concurrency=4    # Parallel actors
)
```

---

## Common Configurations

### ETL Pipeline

```python
# Extract
data = ray.data.read_parquet(
    "s3://input/",
    columns=["id", "value", "date"],  # Column pruning
    num_cpus=0.025  # High I/O parallelism
)

# Transform
transformed = data.map_batches(
    business_logic,
    num_cpus=0.5,  # Light CPU work
    batch_size=1000
)

# Load
transformed.write_parquet(
    "s3://output/",
    num_cpus=0.1  # Balanced writes
)
```

### Batch Inference

```python
class Model:
    def __init__(self):
        self.model = load_pretrained_model()
    
    def __call__(self, batch):
        predictions = self.model(batch["input"])
        batch["prediction"] = predictions
        return batch

results = dataset.map_batches(
    Model,
    num_gpus=1,  # GPU per actor
    batch_size=32,  # GPU-efficient batch
    concurrency=4  # = number of GPUs
)
```

### Document Processing

```python
# Load documents
docs = ray.data.read_binary_files(
    "documents/",
    num_cpus=0.05
)

# Extract text
text = docs.map_batches(
    extract_text_fn,
    num_cpus=0.25,
    batch_size=100
)

# Generate embeddings
embeddings = text.map_batches(
    EmbeddingModel,
    num_gpus=1,
    batch_size=64,
    concurrency=2
)
```

---

## num_cpus Cheat Sheet

| Operation Type | CPU Usage | num_cpus | Parallelism (8 CPUs) |
|----------------|-----------|----------|---------------------|
| **File I/O** | Very Low | 0.025-0.05 | 160-320 tasks |
| **Light CPU** | Low | 0.1-0.25 | 32-80 tasks |
| **Medium CPU** | Medium | 0.5-1.0 | 8-16 tasks |
| **Heavy CPU** | High | 1.0-2.0 | 4-8 tasks |
| **GPU Inference** | Low | 0.5-1.0 | - |

**Rule**: Lower num_cpus = More parallelism

---

## Batch Size Guide

| Workload | Batch Size | Reason |
|----------|-----------|--------|
| **CPU Transform** | 500-1000 | Balance overhead vs memory |
| **GPU Inference (Small Model)** | 32-64 | GPU utilization |
| **GPU Inference (Large Model)** | 8-16 | Memory constraints |
| **I/O Operations** | Default | Not critical |
| **Memory-Intensive** | 100-500 | Prevent OOM |

---

## Performance Optimization Order

1. ✅ **Use native operations** (`filter`, `groupby`, `aggregate`)
2. ✅ **Use map_batches() not map()**
3. ✅ **Tune num_cpus** for each operation
4. ✅ **Use Parquet not CSV**
5. ✅ **Column pruning** (read only needed columns)
6. ✅ **Optimize batch sizes**

---

## Common Mistakes to Avoid

❌ **Don't**: Use high num_cpus for I/O
```python
ds = ray.data.read_parquet(path, num_cpus=1.0)  # Only 8 parallel!
```

✅ **Do**: Use low num_cpus for I/O
```python
ds = ray.data.read_parquet(path, num_cpus=0.025)  # 320 parallel!
```

---

❌ **Don't**: Forget num_gpus for GPU work
```python
ds.map_batches(InferenceModel)  # Ray doesn't know GPU needed!
```

✅ **Do**: Always specify num_gpus
```python
ds.map_batches(InferenceModel, num_gpus=1)  # Proper GPU allocation
```

---

❌ **Don't**: Use map() for batch operations
```python
ds.map(lambda row: process(row))  # Slow row-by-row
```

✅ **Do**: Use map_batches()
```python
ds.map_batches(lambda batch: process_batch(batch))  # Fast batching
```

---

## Monitoring and Debugging

```python
# Enable progress bars
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

# Enable memory management
ctx.eager_free = True

# Run Ray Dashboard
# Open browser: http://localhost:8265
```

---

## Resource Commands

```bash
# Check cluster status
ray status

# Check GPU usage
watch -n 0.5 nvidia-smi

# View logs
tail -f /tmp/ray/session_latest/logs/worker-*.log

# Start Ray
ray start --head

# Stop Ray
ray stop
```

---

## Getting Help

- **Docs**: https://docs.ray.io/en/latest/data/data.html
- **Discuss**: https://discuss.ray.io/
- **Slack**: https://forms.gle/9TSdDYUgxYs8SA9e8
- **GitHub**: https://github.com/ray-project/ray

---

## Related Guides

- [Production Checklist](PRODUCTION_CHECKLIST.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING_GUIDE.md)
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md)

