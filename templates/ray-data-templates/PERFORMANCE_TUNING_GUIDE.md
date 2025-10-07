# Ray Data Performance Tuning Guide

**⏱ Time to complete**: 15 min | **Purpose**: Systematic performance optimization

---

## Performance Optimization Framework

### The 4-Level Optimization Hierarchy

**Always optimize in this order** for maximum impact with minimum effort:

```
Level 1: Use Native Operations (10x faster) ⭐⭐⭐⭐⭐
  ├─> filter(), groupby(), aggregate()
  ├─> Native readers (read_parquet, read_json)
  └─> Compiled implementations

Level 2: Batch Processing (5x faster) ⭐⭐⭐⭐
  ├─> map_batches() vs map()
  ├─> Vectorized operations
  └─> Efficient batch sizes

Level 3: Resource Tuning (3x faster) ⭐⭐⭐
  ├─> num_cpus specification
  ├─> num_gpus for GPU workloads
  └─> Concurrency settings

Level 4: Data Format (2x faster) ⭐⭐
  ├─> Parquet vs CSV
  ├─> Compression settings
  └─> Column pruning
```

:::tip Optimization ROI
**80% of performance gains** come from Level 1-2 optimizations.
**20% of performance gains** come from Level 3-4 optimizations.

Start with Level 1, move to Level 2, then consider 3-4 if needed!
:::

---

## Level 1: Use Native Operations

### ✅ Use Native Operations (Fast)

```python
# ✅ GOOD: Use native filter
valid_data = dataset.filter(lambda row: row["value"] > 0)

# ✅ GOOD: Use native groupby/aggregate
summary = dataset.groupby("category").aggregate(
    Count(),
    Mean("value")
)

# ✅ GOOD: Use native readers
data = ray.data.read_parquet(path)
```

### ❌ Avoid Custom Implementations (Slow)

```python
# ❌ BAD: Manual filtering with map
valid_data = dataset.map(
    lambda row: row if row["value"] > 0 else None
).filter(lambda row: row is not None)

# ❌ BAD: Manual aggregation
def manual_groupby(batch):
    # Complex groupby logic...
    return result
summary = dataset.map_batches(manual_groupby)

# ❌ BAD: Manual reading
def read_custom(path):
    # Custom read logic...
    return data
data = ray.data.from_items([read_custom(path)])
```

**Performance Impact:**
- Native operations: **10-100x faster**
- Compiled C++ implementations
- Optimized memory usage
- Better parallelization

---

## Level 2: Batch Processing

### ✅ Use map_batches() (Fast)

```python
# ✅ GOOD: Batch processing with vectorization
def process_batch(batch):
    import pandas as pd
    df = pd.DataFrame(batch)
    df["processed"] = df["value"] * 2  # Vectorized!
    return df.to_dict("records")

result = dataset.map_batches(
    process_batch,
    batch_size=1000,  # Process 1000 rows at once
    num_cpus=0.5
)
```

### ❌ Avoid Row-by-Row (Slow)

```python
# ❌ BAD: Row-by-row processing
def process_row(row):
    row["processed"] = row["value"] * 2  # One at a time!
    return row

result = dataset.map(process_row)  # No batching
```

**Performance Comparison:**

| Method | Rows/Second | Relative Speed |
|--------|-------------|----------------|
| `map()` row-by-row | 1,000 | 1x baseline |
| `map_batches()` batch=100 | 25,000 | 25x faster |
| `map_batches()` batch=1000 | 80,000 | 80x faster |

:::note Why Batching is Faster
- **Vectorization**: NumPy/Pandas operations on arrays
- **Amortized overhead**: Setup cost spread over many rows
- **Cache efficiency**: Better CPU cache utilization
- **Reduced task count**: Fewer Ray tasks to schedule
:::

---

## Level 3: Resource Tuning

### The num_cpus Parameter

**Golden Rule**: `num_cpus` should match **actual CPU usage**, not desired parallelism.

#### ✅ Correct num_cpus Usage

```python
# ✅ I/O bound (reading files)
dataset = ray.data.read_parquet(
    path,
    num_cpus=0.025  # Low CPU = High parallelism
)
# 8 CPUs = 320 parallel tasks!

# ✅ Light CPU work (simple transforms)
transformed = dataset.map_batches(
    light_transform,
    num_cpus=0.25  # Light CPU = Good parallelism
)
# 8 CPUs = 32 parallel tasks

# ✅ Heavy CPU work (complex computation)
processed = dataset.map_batches(
    heavy_computation,
    num_cpus=2.0  # Heavy CPU = Lower parallelism
)
# 8 CPUs = 4 parallel tasks
```

#### ❌ Common num_cpus Mistakes

```python
# ❌ BAD: High num_cpus for I/O work
dataset = ray.data.read_parquet(
    path,
    num_cpus=1.0  # Only 8 parallel tasks!
)
# Result: Slow loading, poor parallelism

# ❌ BAD: Low num_cpus for heavy computation
processed = dataset.map_batches(
    heavy_computation,
    num_cpus=0.1  # 80 parallel tasks!
)
# Result: CPU contention, worse performance
```

**Performance Impact Table:**

| Operation Type | CPU Usage | Recommended num_cpus | Parallelism (8 CPUs) |
|----------------|-----------|---------------------|---------------------|
| **File I/O** | < 5% | 0.025-0.05 | 160-320 tasks |
| **Light Transform** | 10-25% | 0.1-0.25 | 32-80 tasks |
| **Medium Transform** | 25-50% | 0.5-1.0 | 8-16 tasks |
| **Heavy Compute** | 75-100% | 1.0-2.0 | 4-8 tasks |

### GPU Resource Specification

#### ✅ Always Specify num_gpus for GPU Workloads

```python
# ✅ GOOD: Explicit GPU allocation
predictions = dataset.map_batches(
    InferenceModel,
    num_gpus=1,  # Reserve 1 GPU per actor
    batch_size=32,
    concurrency=4  # 4 actors = 4 GPUs used
)
```

#### ❌ Never Omit num_gpus

```python
# ❌ BAD: No GPU specification
predictions = dataset.map_batches(
    InferenceModel,
    batch_size=32
)
# Result: Ray doesn't know GPU needed, scheduling chaos!
```

---

## Level 4: Data Format Optimization

### File Format Comparison

| Format | Read Speed | Write Speed | Compression | Columnar | Recommended |
|--------|-----------|-------------|-------------|----------|-------------|
| **Parquet** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Excellent | ✅ Yes | **Production** |
| **CSV** | ⭐⭐ | ⭐⭐⭐ | ❌ Limited | ❌ No | Development only |
| **JSON** | ⭐ | ⭐⭐ | ❌ Limited | ❌ No | Small datasets |

**Performance Example:**

```
10GB Dataset Processing:

CSV:
  Read: 180 seconds
  Process: 45 seconds  
  Write: 120 seconds
  Total: 345 seconds

Parquet:
  Read: 12 seconds (15x faster!)
  Process: 45 seconds
  Write: 18 seconds (7x faster!)
  Total: 75 seconds (4.6x faster overall!)
```

### Column Pruning

**Only read columns you need** for massive speedups:

```python
# ❌ BAD: Read all columns
dataset = ray.data.read_parquet(path)  # Reads 50 columns

# ✅ GOOD: Read only needed columns
dataset = ray.data.read_parquet(
    path,
    columns=["id", "value", "timestamp"]  # Only 3 columns
)
# Result: 10-50x faster reads!
```

---

## Performance Tuning Workflow

### Step 1: Profile Your Pipeline

```python
# Enable performance monitoring
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True
ctx.enable_auto_log_stats = True

# Run your pipeline
result = dataset.map_batches(your_function).take_all()

# Review Ray Dashboard:
# - Which operations are slowest?
# - What's the GPU/CPU utilization?
# - Any memory pressure?
```

### Step 2: Identify Bottlenecks

| Observation | Likely Bottleneck | Optimization |
|-------------|------------------|--------------|
| GPU utilization < 80% | Preprocessing slow | Increase upstream parallelism (lower num_cpus) |
| CPU utilization low | Too few tasks | Lower num_cpus for more parallelism |
| Memory pressure | Batch size too large | Reduce batch_size |
| One stage much slower | Imbalanced resources | Tune num_cpus for that stage |

### Step 3: Apply Optimizations

**Use this systematic approach:**

1. **First**: Switch to native operations (if possible)
2. **Second**: Switch from `map()` to `map_batches()`
3. **Third**: Tune num_cpus based on actual usage
4. **Fourth**: Optimize batch sizes
5. **Fifth**: Use Parquet and column pruning
6. **Last**: Consider data format optimizations

### Step 4: Measure Improvements

```python
import time

# Baseline
start = time.time()
result_baseline = baseline_pipeline().take_all()
baseline_time = time.time() - start

# Optimized
start = time.time()
result_optimized = optimized_pipeline().take_all()
optimized_time = time.time() - start

# Compare
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x faster")
print(f"Time saved: {baseline_time - optimized_time:.1f} seconds")
```

---

## Common Performance Anti-Patterns

### ❌ Anti-Pattern 1: Using map() Instead of map_batches()

```python
# ❌ SLOW: Row-by-row
dataset.map(lambda row: transform(row))

# ✅ FAST: Batch processing
dataset.map_batches(lambda batch: transform_batch(batch))
```

**Impact**: 10-100x slower

### ❌ Anti-Pattern 2: Reading All Columns

```python
# ❌ SLOW: Read everything
dataset = ray.data.read_parquet(path)

# ✅ FAST: Column pruning
dataset = ray.data.read_parquet(path, columns=["id", "value"])
```

**Impact**: 5-50x slower

### ❌ Anti-Pattern 3: Using CSV in Production

```python
# ❌ SLOW: CSV
dataset = ray.data.read_csv(path)

# ✅ FAST: Parquet
dataset = ray.data.read_parquet(path)
```

**Impact**: 3-10x slower

### ❌ Anti-Pattern 4: Not Specifying num_gpus

```python
# ❌ SLOW: GPU not reserved
dataset.map_batches(InferenceModel, batch_size=32)

# ✅ FAST: GPU explicitly allocated
dataset.map_batches(InferenceModel, num_gpus=1, batch_size=32)
```

**Impact**: Unpredictable, often 5-10x slower

---

## Quick Reference: Optimization Checklist

**Before optimizing, check:**

- [ ] Using native operations where possible?
- [ ] Using `map_batches()` instead of `map()`?
- [ ] Specified `num_cpus` based on actual CPU usage?
- [ ] Specified `num_gpus` for GPU workloads?
- [ ] Using Parquet instead of CSV?
- [ ] Column pruning enabled?
- [ ] Batch sizes optimized?
- [ ] Monitoring enabled to verify improvements?

**Expected speedups if you fix all issues:**

- Native operations: 10-100x
- Batching: 10-80x  
- Resource tuning: 2-5x
- Data format: 2-10x

**Combined: Easily achieve 100-1000x speedups** with systematic optimization!

---

## Next Steps

1. **Profile** your current pipeline
2. **Identify** bottlenecks using Ray Dashboard
3. **Apply** optimizations in order (Level 1 → Level 4)
4. **Measure** improvements
5. **Iterate** until meeting performance targets

**Need more help?** See:
- [Production Checklist](PRODUCTION_CHECKLIST.md)
- [Ray Data Docs](https://docs.ray.io/en/latest/data/data.html)
- [Ray Community](https://discuss.ray.io/)

