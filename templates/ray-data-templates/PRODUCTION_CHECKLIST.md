# Production Deployment Checklist

**⏱ Time to complete**: 10 min | **Purpose**: Ensure production-ready Ray Data deployments

---

## Pre-Deployment Checklist

### 1. Performance Validation

- [ ] **Load tested** with production data volumes
- [ ] **GPU utilization** > 80% for inference workloads
- [ ] **CPU utilization** balanced across cluster
- [ ] **Memory usage** stable (no memory leaks)
- [ ] **Throughput** meets SLA requirements
- [ ] **Latency** within acceptable ranges

### 2. Resource Configuration

- [ ] **Cluster sizing** appropriate for workload
- [ ] **num_cpus** tuned for each operation type
- [ ] **num_gpus** specified for GPU workloads
- [ ] **batch_size** optimized through testing
- [ ] **concurrency** matches available resources
- [ ] **Block size** appropriate for data volumes

### 3. Error Handling

- [ ] **Retry logic** implemented for transient failures
- [ ] **Fallback strategies** for service degradation
- [ ] **Error monitoring** and alerting configured
- [ ] **Graceful degradation** for partial failures
- [ ] **Resource cleanup** on exceptions

### 4. Monitoring & Observability

- [ ] **Ray Dashboard** accessible for debugging
- [ ] **Custom metrics** exported to monitoring system
- [ ] **Log aggregation** configured
- [ ] **Alerting rules** defined for critical issues
- [ ] **Performance baselines** established

### 5. Data Quality

- [ ] **Input validation** implemented
- [ ] **Output verification** in place
- [ ] **Schema enforcement** for data types
- [ ] **Data quality metrics** tracked
- [ ] **Anomaly detection** for data issues

---

## Quick Reference: Common Configurations

### ETL Workloads

```python
# Optimized ETL configuration
dataset = ray.data.read_parquet(
    path,
    columns=essential_columns,  # Column pruning
    num_cpus=0.025  # High I/O concurrency
)

transformed = dataset.map_batches(
    transform_fn,
    num_cpus=0.5,  # Light CPU work
    batch_size=1000  # Efficient batching
)

transformed.write_parquet(
    output_path,
    num_cpus=0.1  # Balanced writes
)
```

### Batch Inference Workloads

```python
# Optimized inference configuration
class InferenceModel:
    def __init__(self):
        self.model = load_model()  # Load once
    
    def __call__(self, batch):
        return self.model(batch["data"])

predictions = dataset.map_batches(
    InferenceModel,
    num_gpus=1,  # Explicit GPU allocation
    batch_size=32,  # Optimize for GPU
    concurrency=4  # Match GPU count
)
```

### Document Processing Workloads

```python
# Optimized document processing
documents = ray.data.read_binary_files(
    path,
    num_cpus=0.05  # High parallelism
)

processed = documents.map_batches(
    extract_text,
    num_cpus=0.25,  # Light extraction
    batch_size=100
)

embeddings = processed.map_batches(
    generate_embeddings,
    num_gpus=1,  # GPU for embeddings
    batch_size=64,
    concurrency=2
)
```

---

## Production Best Practices

### Resource Allocation

| Workload Type | num_cpus | num_gpus | batch_size | concurrency |
|---------------|----------|----------|------------|-------------|
| **I/O Heavy** | 0.025-0.05 | 0 | Default | High |
| **Light CPU** | 0.25-0.5 | 0 | 500-1000 | Medium |
| **Heavy CPU** | 1.0-2.0 | 0 | 100-500 | Low |
| **GPU Inference** | 0.5-1.0 | 1 | 16-64 | = GPUs |

### Memory Management

- **Enable eager freeing**: `ctx.eager_free = True`
- **Set block size**: `ctx.target_max_block_size = 128 * 1024 * 1024`
- **Monitor object store**: Watch Ray Dashboard metrics
- **Tune batch sizes**: Reduce if seeing memory pressure

### Performance Monitoring

```python
# Enable performance monitoring
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True
ctx.enable_auto_log_stats = True
```

---

## Troubleshooting Guide

### Issue: Low Throughput

**Symptoms**: Processing slower than expected

**Diagnosis**:
1. Check GPU utilization (should be > 80%)
2. Check CPU utilization across nodes
3. Review Ray Dashboard for bottlenecks

**Solutions**:
- Increase parallelism (lower num_cpus)
- Increase batch size
- Add more workers/nodes

### Issue: OOM Errors

**Symptoms**: Workers crash with memory errors

**Diagnosis**:
1. Check batch sizes
2. Review object store usage
3. Monitor heap memory

**Solutions**:
- Reduce batch_size
- Reduce concurrency
- Enable eager_free
- Increase cluster memory

### Issue: GPU Idle

**Symptoms**: GPU utilization near 0%

**Diagnosis**:
1. Check preprocessing parallelism
2. Verify num_gpus specified
3. Review batch sizes

**Solutions**:
- Increase upstream parallelism (lower num_cpus)
- Verify GPU resources available
- Increase batch size

---

## Performance Optimization Checklist

### Level 1: Use Native Operations (10x faster)

- [ ] Use `filter()` instead of `map()` for filtering
- [ ] Use `groupby().aggregate()` for aggregations
- [ ] Use `read_parquet()` for Parquet files
- [ ] Use `write_parquet()` for outputs

### Level 2: Batch Processing (5x faster)

- [ ] Use `map_batches()` instead of `map()`
- [ ] Set appropriate batch sizes
- [ ] Vectorize operations in batch functions

### Level 3: Resource Tuning (3x faster)

- [ ] Tune num_cpus for each operation
- [ ] Set num_gpus for GPU workloads
- [ ] Adjust concurrency based on resources

### Level 4: Data Format (2x faster)

- [ ] Use Parquet instead of CSV
- [ ] Enable compression (snappy/gzip)
- [ ] Prune unnecessary columns early

---

## Success Metrics

Track these metrics for production health:

| Metric | Target | Action if Below Target |
|--------|--------|----------------------|
| **Throughput** | > Baseline | Optimize parallelism |
| **GPU Utilization** | > 80% | Increase batch size or parallelism |
| **CPU Utilization** | 60-80% | Balance num_cpus settings |
| **Memory Usage** | < 80% peak | Reduce batch sizes |
| **Error Rate** | < 0.1% | Improve error handling |
| **P99 Latency** | < 2x P50 | Investigate outliers |

---

## Next Steps

After deploying to production:

1. **Monitor continuously** using Ray Dashboard and custom metrics
2. **Iterate on configuration** based on observed performance
3. **Scale horizontally** by adding nodes as needed
4. **Update baselines** as workloads evolve
5. **Document learnings** for future optimizations

**Need help?** Join the Ray community:
- [Ray Slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
- [Ray Discourse](https://discuss.ray.io/)
- [GitHub Issues](https://github.com/ray-project/ray/issues)

