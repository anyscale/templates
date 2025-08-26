# Ray Data Batch Inference Optimization Template

## Overview

This template demonstrates how to optimize Ray Data pipelines for batch inference workloads. We'll start with a poorly performing pipeline that exhibits common mistakes, then systematically improve it using Ray Data optimization techniques.

**New in this version**: Support for both GPU and CPU-only clusters, and integration with real ImageNet data from S3!

## Learning Objectives

By the end of this template, you'll understand:
- Common performance pitfalls in Ray Data pipelines
- How to diagnose performance bottlenecks
- Optimization strategies for batch inference workloads
- Best practices for production-ready Ray Data pipelines
- **Device-agnostic optimization** for both GPU and CPU clusters
- **Real-world data integration** with cloud storage

## Template Structure

1. **Baseline Implementation** - A poorly performing pipeline with common mistakes
2. **Performance Analysis** - How to diagnose issues using Ray Dashboard and stats
3. **Optimization Techniques** - Step-by-step improvements
4. **Production-Ready Implementation** - Final optimized version
5. **Advanced Optimizations** - GPU utilization, checkpointing, and scaling
6. **Device-Agnostic Configuration** - Works on both GPU and CPU-only clusters

## Common Performance Mistakes We'll Address

### 1. **Inefficient Data Ingestion**
- Reading entire datasets without filtering
- Not using appropriate block sizes
- Missing column selection optimizations
- **No streaming for large datasets**

### 2. **Poor Transformation Patterns**
- Using `.map()` instead of `.map_batches()`
- Inefficient batch sizes
- Missing operator fusion opportunities
- **Processing one image at a time instead of batching**

### 3. **Resource Misallocation**
- Incorrect GPU/CPU allocation
- Memory pressure from oversized batches
- Poor parallelism configuration
- **No device-specific optimization**

### 4. **Output Inefficiencies**
- Too many small output files
- Missing repartitioning strategies
- Inefficient write operations

## Use Case: Image Classification Batch Inference

We'll optimize a pipeline that:
- **Processes real ImageNet data from S3** (`s3://anonymous@air-example-data-2/imagenette2/train/`)
- Runs inference through a pre-trained ResNet50 model
- Applies post-processing and filtering
- Outputs results to cloud storage
- **Works on both GPU and CPU-only clusters**

## Key Optimization Areas

### **Memory Management**
- Object store sizing and configuration
- Batch size tuning for device type (GPU vs CPU)
- Block size optimization
- Spill prevention strategies
- **Streaming data processing**

### **Device Optimization**
- **GPU clusters**: Proper GPU allocation, batch size optimization, CUDA memory management
- **CPU-only clusters**: Parallelism optimization, memory-efficient processing, vectorized operations
- **Adaptive configuration** based on available resources
- **Device-specific batch sizes** and worker scaling

### **Parallelism and Scaling**
- Block count optimization
- Worker distribution based on device type
- Resource allocation per task
- Autoscaling considerations
- **Device-aware parallelism**

### **I/O Optimization**
- **Cloud storage read optimization** (S3 ImageNet data)
- Column and partition filtering
- Output file size management
- Network bandwidth utilization
- **Streaming data ingestion**

## Performance Metrics We'll Monitor

- **Throughput**: Images processed per second
- **Device Utilization**: GPU percentage or CPU utilization
- **Memory Usage**: Object store and heap memory
- **I/O Performance**: Read/write speeds from S3
- **Resource Efficiency**: CPU/GPU utilization per worker
- **Cluster Resource Usage**: Available vs. utilized resources

## Expected Performance Improvements

After optimization, you should see:
- **2-5x faster inference** through proper batching and device optimization
- **50-80% reduction in memory pressure** through better resource allocation
- **Improved scalability** with linear performance gains as workers increase
- **Reduced costs** through better resource utilization
- **Device-agnostic performance** on both GPU and CPU clusters

## Cluster Compatibility

### **GPU Clusters**
- Automatic GPU detection and allocation
- Optimized batch sizes for GPU memory
- CUDA memory management and cleanup
- Multi-GPU parallelism

### **CPU-Only Clusters**
- Efficient CPU parallelism
- Memory-optimized batch processing
- Vectorized operations for CPU efficiency
- Resource-aware worker scaling

### **Hybrid Clusters**
- Automatic resource detection
- Optimal configuration selection
- Mixed GPU/CPU workload distribution

## Prerequisites

- Basic understanding of Ray Data concepts
- Familiarity with Python and machine learning workflows
- Access to Ray cluster (local or cloud)
- **GPU resources (optional)** - template works on both GPU and CPU-only clusters
- **Internet access** for S3 ImageNet data

## Next Steps

1. Review the baseline implementation to understand common mistakes
2. Run performance analysis to identify bottlenecks
3. Apply optimization techniques step-by-step
4. Compare performance metrics before and after
5. Understand when and how to apply each optimization
6. **Test on different cluster types** (GPU vs CPU-only)

## Template Files

- `baseline_implementation.py` - Poorly performing pipeline (26+ mistakes)
- `optimized_implementation.py` - Production-ready version (30+ optimizations)
- `performance_analysis.py` - Diagnostic tools and comparison
- `run_comparison.py` - Main script for running comparisons
- `requirements.txt` - All necessary dependencies
- `README.ipynb` - Interactive Jupyter notebook
- `configs/` - Cluster and pipeline configurations
- `models/` - Sample model and preprocessing code
- `data/` - Sample dataset and utilities

## Key Features

### **Real-World Data Integration**
- **S3 ImageNet dataset**: `s3://anonymous@air-example-data-2/imagenette2/train/`
- Streaming data processing for large datasets
- Cloud storage optimization
- Fallback to sample data if S3 unavailable

### **Device-Agnostic Optimization**
- **Automatic device detection** (GPU vs CPU)
- **Adaptive configuration** based on available resources
- **Device-specific batch sizes** and worker scaling
- **Optimal resource allocation** for each cluster type

### **Comprehensive Performance Analysis**
- Real-time resource monitoring
- Performance comparison tools
- Detailed optimization reports
- Cluster resource analysis

### **Production-Ready Patterns**
- Error handling and recovery
- Memory management and cleanup
- Progress monitoring and logging
- Output optimization and repartitioning

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run baseline pipeline** (see poor performance):
   ```bash
   python baseline_implementation.py
   ```

3. **Run optimized pipeline** (see dramatic improvements):
   ```bash
   python optimized_implementation.py
   ```

4. **Compare performance**:
   ```bash
   python run_comparison.py
   ```

5. **Interactive exploration**:
   ```bash
   jupyter notebook README.ipynb
   ```

## Cluster Setup Examples

### **Local Development**
```bash
ray start --head
python run_comparison.py
```

### **GPU Cluster**
```bash
ray start --head --num-gpus=4
python run_comparison.py
```

### **CPU-Only Cluster**
```bash
ray start --head --num-cpus=8
python run_comparison.py
```

### **Cloud Deployment**
```bash
# Works with any Ray cluster (AWS, GCP, Azure, etc.)
ray submit cluster.yaml run_comparison.py
```

---

*This template is designed to be run in sequence, with each step building on the previous optimizations. Follow along to see the dramatic performance improvements possible with Ray Data, whether you're running on GPU clusters, CPU-only clusters, or anywhere in between!*
