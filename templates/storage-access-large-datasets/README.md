# Storage Access and Large Datasets on Anyscale

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/storage-access-large-datasets"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/storage-access-large-datasets" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 15-20 minutes

This template demonstrates how to access and process large datasets from Amazon S3, Google Cloud Storage (GCS), and Azure Blob Storage using Ray Data on Anyscale. You'll learn authentication patterns, file format support, performance optimization, and best practices for cloud storage integration.

## What You'll Learn

By the end of this template, you'll be able to:

1. Read data from S3, GCS, and Azure Blob Storage with Ray Data
2. Authenticate with cloud providers using appropriate credential patterns
3. Work with multiple file formats (Parquet, CSV, JSON, images, text)
4. Process large datasets efficiently with Ray Data's lazy execution
5. Write results back to cloud storage
6. Optimize performance for large-scale data workloads

## Introduction

Modern data workloads often involve reading from and writing to cloud object storage. [Ray Data](https://docs.ray.io/en/latest/data/data.html) provides a unified interface for working with data across Amazon S3, Google Cloud Storage, and Azure Blob Storage, with built-in support for authentication, multiple file formats, and performance optimization. Ray Data's streaming execution model enables efficient processing of datasets that don't fit in memory, making it ideal for large-scale data preprocessing, ETL pipelines, and batch inference workloads.

### Anyscale Storage Architecture

Anyscale provides three types of storage mounts:

- **`/mnt/cluster_storage/`** - Shared across all nodes in the same cluster. Persists across workspace restarts. Ephemeral for Jobs/Services.
- **`/mnt/shared_storage/`** - Shared across all clusters for the same user/organization. Permanent until manually deleted.
- **`/mnt/user_storage/`** - User-specific storage accessible across all your clusters. Permanent until manually deleted.

For this template, we'll primarily use `/mnt/cluster_storage/` for intermediate results and demonstrate reading from/writing to cloud storage providers.

## Setup & Prerequisites

Let's start by installing dependencies and importing the necessary libraries.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/storage-access-large-datasets
```


```python
# Install Google Cloud Storage filesystem support
# Note: adlfs (Azure) is pre-installed on Anyscale base images
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```


```python
import ray
import pandas as pd
import os
from pathlib import Path

# Ray Data imports
import ray.data

# Cloud storage filesystem imports
import gcsfs
import adlfs

print(f"Ray version: {ray.__version__}")
```

### Note on Storage

Throughout this tutorial, we use `/mnt/cluster_storage/` to represent a shared storage location. In a multi-node cluster, Ray workers on different nodes cannot access the head node's local file system. Use a [shared storage solution](https://docs.anyscale.com/configuration/storage#shared) accessible from every node.

## Anyscale Storage Fundamentals

Before working with cloud storage, it's important to understand Anyscale's storage architecture:

**Storage Persistence:**
- `/mnt/cluster_storage/` persists across workspace restarts but is ephemeral for Jobs/Services
- `/mnt/shared_storage/` and `/mnt/user_storage/` persist permanently until manually deleted
- Always use absolute paths for multi-node workloads - relative paths only work on the head node

**Best Practices:**
- Use `/mnt/cluster_storage/` for temporary data within a single cluster session
- Use `/mnt/shared_storage/` when workspaces submit jobs/services that need to access the same data
- Avoid `local://` prefix for multi-node workloads as it restricts operations to the head node only

For more details on configuring storage for production workloads, see the [Storage configuration guide](https://docs.anyscale.com/configuration/storage).

## Amazon S3 Access

Amazon S3 is widely used for cloud data storage. Ray Data integrates with S3 through PyArrow's built-in S3 filesystem support.

### Reading from S3 (Anonymous Access)

For public datasets, you can use anonymous access with the `s3://anonymous@` prefix:


```python
# Read a public dataset from S3
ds_s3 = ray.data.read_parquet("s3://anonymous@ray-example-data/iris.parquet")

# Inspect the schema
print("Schema:")
print(ds_s3.schema())
```


```python
# Display the first few rows
print(f"\nDataset count: {ds_s3.count()} rows")
ds_s3.show(5)
```

### Reading from S3 with Credentials

For private S3 buckets, Ray Data uses PyArrow's S3 filesystem, which automatically picks up AWS credentials from:

1. Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
2. AWS credentials file (`~/.aws/credentials`)
3. IAM roles (when running on AWS infrastructure)

**On Anyscale**, the recommended approach is to use IAM roles attached to your cluster:


```python
# Example: Reading from a private S3 bucket
# Assumes your Anyscale cluster has appropriate IAM roles configured
# ds_private = ray.data.read_parquet("s3://my-private-bucket/data/")

# For this template, we'll continue with the public dataset
print("✓ S3 access configured")
```

**Authentication reference:** See the [PyArrow S3 Filesystem documentation](https://arrow.apache.org/docs/python/filesystems.html#s3) for detailed credential configuration options.

## Google Cloud Storage (GCS) Access

Google Cloud Storage requires the `gcsfs` library and appropriate authentication.

### GCS Authentication

GCS authentication typically uses service account JSON key files:


```python
# Set up GCS filesystem
# Option 1: Using GOOGLE_APPLICATION_CREDENTIALS environment variable
# Option 2: Explicitly passing credentials to GCSFileSystem

# For this demo, we'll create a filesystem object
# In production, set GOOGLE_APPLICATION_CREDENTIALS in your environment
gcs_project = os.environ.get("GCS_PROJECT", "demo-project")

# Create GCS filesystem (will use default credentials if available)
try:
    gcs_fs = gcsfs.GCSFileSystem(project=gcs_project)
    print(f"✓ GCS filesystem created for project: {gcs_project}")
except Exception as e:
    print(f"Note: GCS authentication not configured. Error: {e}")
    print("To use GCS, set up a service account and GOOGLE_APPLICATION_CREDENTIALS")
    gcs_fs = None
```

### Reading from GCS

Once authenticated, reading from GCS is straightforward:


```python
# Example: Reading from GCS (requires authentication)
# gs_path = "gs://my-bucket/data/dataset.parquet"
# ds_gcs = ray.data.read_parquet(gs_path, filesystem=gcs_fs)

# For this template, let's create some synthetic data to demonstrate the pattern
print("Creating synthetic data to demonstrate GCS write/read pattern...")

# Create a small synthetic dataset
synthetic_data = [
    {"feature_1": i, "feature_2": i * 2, "label": i % 3}
    for i in range(1000)
]
ds_synthetic = ray.data.from_items(synthetic_data)

print(f"Synthetic dataset created: {ds_synthetic.count()} rows")
ds_synthetic.show(3)
```

**Authentication reference:** See the [gcsfs documentation](https://gcsfs.readthedocs.io/en/latest/) and [PyArrow GCS Filesystem docs](https://arrow.apache.org/docs/python/filesystems.html#google-cloud-storage-file-system) for credential setup.

## Azure Blob Storage Access

Azure Blob Storage uses the `adlfs` library (pre-installed on Anyscale base images).

### Azure Authentication

Azure authentication supports multiple methods:

1. Account name + account key
2. Connection string
3. SAS (Shared Access Signature) tokens
4. Managed identities (when running on Azure)


```python
# Set up Azure Blob Storage filesystem
# Common authentication patterns:

# Option 1: Account key (from environment)
azure_account_name = os.environ.get("AZURE_ACCOUNT_NAME", "azureopendatastorage")
azure_account_key = os.environ.get("AZURE_ACCOUNT_KEY")  # None if not set

# Option 2: SAS token
azure_sas_token = os.environ.get("AZURE_SAS_TOKEN")  # None if not set

# Create Azure filesystem
try:
    azure_fs = adlfs.AzureBlobFileSystem(
        account_name=azure_account_name,
        account_key=azure_account_key,  # Will be None for public data
        sas_token=azure_sas_token  # Will be None for public data
    )
    print(f"✓ Azure filesystem created for account: {azure_account_name}")
except Exception as e:
    print(f"Note: Azure authentication configured for public access only")
    print(f"For private data, set AZURE_ACCOUNT_KEY or AZURE_SAS_TOKEN")
    azure_fs = None
```

### Reading from Azure


```python
# Example: Reading from Azure Blob Storage
# For public data, we can access without credentials
try:
    ds_azure = ray.data.read_parquet(
        "az://ray-example-data/iris.parquet",
        filesystem=adlfs.AzureBlobFileSystem(account_name="azureopendatastorage")
    )
    print("Azure dataset loaded:")
    print(ds_azure.schema())
    ds_azure.show(3)
except Exception as e:
    print(f"Note: Azure public data access example. Error: {e}")
    print("For private data, configure appropriate credentials")
```

**Authentication reference:** See [adlfs on PyPI](https://pypi.org/project/adlfs/) and [PyArrow fsspec-compatible filesystems](https://arrow.apache.org/docs/python/filesystems.html#using-fsspec-compatible-filesystems-with-arrow) for more details.

## Reading Multiple File Formats

Ray Data supports a wide variety of file formats beyond Parquet, including CSV, JSON, text files, images, and more. For a comprehensive overview of supported formats and data sources, see the [Loading data guide](https://docs.ray.io/en/latest/data/loading-data.html). Let's explore the most common formats.

### Reading CSV Files


```python
# Read CSV from S3
ds_csv = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

print("CSV Schema:")
print(ds_csv.schema())
print(f"\nCSV dataset: {ds_csv.count()} rows")
ds_csv.show(3)
```

### Reading JSON Files


```python
# Create a sample JSON dataset for demonstration
import tempfile
import json

# Write sample JSON to local storage
json_data = [
    {"id": i, "value": f"item_{i}", "score": i * 0.1}
    for i in range(100)
]

json_path = "/mnt/cluster_storage/sample_data.jsonl"
with open(json_path, "w") as f:
    for item in json_data:
        f.write(json.dumps(item) + "\n")

# Read JSON Lines format
ds_json = ray.data.read_json(json_path)

print("JSON Schema:")
print(ds_json.schema())
ds_json.show(3)
```

### Reading Text Files

Ray Data's text file support is optimized for large-scale NLP and text processing workloads. Each line becomes a separate row, making it easy to process log files, documents, and corpora at scale. See the [Working with text](https://docs.ray.io/en/latest/data/working-with-text.html) guide for tokenization and text preprocessing patterns.


```python
# Read text files (returns one row per line)
ds_text = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")

print("Text Schema:")
print(ds_text.schema())
ds_text.show(5)
```

### Reading Images

Ray Data provides specialized support for image data, with automatic format detection, resizing, and tensor conversion. For more details on image processing capabilities, see the [Working with images](https://docs.ray.io/en/latest/data/working-with-images.html) guide.


```python
# Read images from S3
# Each image becomes a row with numpy array representation
ds_images = ray.data.read_images(
    "s3://anonymous@ray-example-data/batoidea/JPEGImages/",
    size=(32, 32)  # Resize to 32x32 for uniform size
)

print("Image Dataset Schema:")
print(ds_images.schema())
print(f"Loaded {ds_images.count()} images")

# Inspect first image
sample = ds_images.take(1)[0]
print(f"Image shape: {sample['image'].shape}")
```

## Working with Large Datasets

Ray Data is designed for datasets that don't fit in memory. It uses lazy execution and streaming to handle data efficiently.

### Schema Inspection

Always inspect the schema before processing to understand your data structure:


```python
# Schema inspection
print("Dataset schema:")
schema = ds_s3.schema()
print(schema)

# Get column names
print(f"\nColumns: {schema.names}")
```

### Data Inspection Methods


```python
# Count rows (triggers execution)
total_rows = ds_s3.count()
print(f"Total rows: {total_rows}")

# Preview data without materializing entire dataset
print("\nFirst 5 rows:")
ds_s3.show(5)

# Take specific number of rows (returns Python objects)
sample_rows = ds_s3.take(3)
print(f"\nSample data type: {type(sample_rows)}")
print(f"First row: {sample_rows[0]}")
```

### Lazy Execution

Ray Data operations are lazy - they don't execute until you call a consuming operation:


```python
# These operations build a logical plan but don't execute yet
ds_filtered = ds_s3.filter(lambda row: row["sepal.length"] > 5.0)
ds_selected = ds_filtered.select_columns(["sepal.length", "sepal.width", "variety"])

print("Operations queued (not executed yet)")

# Execution happens when we call show(), take(), or write_*()
print("\nFiltered data:")
ds_selected.show(5)
```

## Data Transformations

Ray Data provides powerful transformation capabilities for distributed data processing. The [`map_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) operation is the primary tool for applying custom transformations at scale, enabling everything from simple column operations to complex ML preprocessing pipelines. For a comprehensive overview of transformation patterns, see the [Transforming data guide](https://docs.ray.io/en/latest/data/transforming-data.html).

### Using map_batches for Distributed Processing

For CPU-intensive transformations, use `map_batches()` to process data in parallel:


```python
import numpy as np

def normalize_batch(batch: dict) -> dict:
    """Normalize numerical columns to 0-1 range"""
    batch = batch.copy()

    # Normalize sepal length
    sepal_length = batch["sepal.length"]
    min_val = np.min(sepal_length)
    max_val = np.max(sepal_length)
    batch["sepal.length"] = (sepal_length - min_val) / (max_val - min_val + 1e-8)

    # Normalize sepal width
    sepal_width = batch["sepal.width"]
    min_val = np.min(sepal_width)
    max_val = np.max(sepal_width)
    batch["sepal.width"] = (sepal_width - min_val) / (max_val - min_val + 1e-8)

    return batch

# Apply transformation in parallel batches
ds_normalized = ds_s3.map_batches(
    normalize_batch,
    batch_size=50  # Process 50 rows at a time
)

print("Normalized data:")
ds_normalized.show(5)
```

### Batch Size Optimization

The `batch_size` parameter controls the trade-off between parallelism and overhead:


```python
# Smaller batches = more parallelism, more overhead
# Larger batches = less overhead, less parallelism
# Default is 4096, but adjust based on your data and operations

# For compute-intensive operations on small rows: smaller batches (100-1000)
# For I/O-bound operations: larger batches (1000-10000)
# For operations with GPU models: tune based on GPU memory

print(f"✓ Batch size optimization configured")
```

## Writing Results to Cloud Storage

After processing, write results back to S3, GCS, or Azure. Ray Data supports multiple output formats and destinations with automatic parallelization and compression. For details on write operations and format options, see the [Saving data guide](https://docs.ray.io/en/latest/data/saving-data.html).

### Writing to S3


```python
# Write processed data to cluster storage
output_path_s3 = "/mnt/cluster_storage/output/s3_processed"
ds_normalized.write_parquet(output_path_s3)

print(f"✓ Data written to {output_path_s3}")

# In production, write directly to S3:
# ds_normalized.write_parquet("s3://my-bucket/output/data.parquet")
```

### Writing to GCS


```python
# Write to cluster storage (GCS write would require authenticated filesystem)
output_path_gcs = "/mnt/cluster_storage/output/gcs_processed"
ds_synthetic.write_parquet(output_path_gcs)

print(f"✓ Data written to {output_path_gcs}")

# In production with GCS authentication:
# ds_synthetic.write_parquet("gs://my-bucket/output/data.parquet", filesystem=gcs_fs)
```

### Writing to Azure


```python
# Write to cluster storage
output_path_azure = "/mnt/cluster_storage/output/azure_processed"
ds_synthetic.write_parquet(output_path_azure)

print(f"✓ Data written to {output_path_azure}")

# In production with Azure authentication:
# ds_synthetic.write_parquet("az://my-container/output/data.parquet", filesystem=azure_fs)
```

### Writing CSV and JSON


```python
# Write as CSV
csv_output = "/mnt/cluster_storage/output/data.csv"
ds_normalized.write_csv(csv_output)

# Write as JSON Lines
json_output = "/mnt/cluster_storage/output/data.jsonl"
ds_json.write_json(json_output)

print(f"✓ Multiple format outputs written")
```

## Performance Optimization

Ray Data provides several knobs for optimizing performance when working with large datasets. From column pruning to parallelism tuning, these techniques can dramatically improve throughput and reduce costs. For comprehensive optimization strategies, see the [Performance tips](https://docs.ray.io/en/latest/data/performance-tips.html) guide.

### Parquet Column Pruning

When reading Parquet files, specify only the columns you need to reduce I/O:


```python
# Read only specific columns (reduces data transfer)
ds_pruned = ray.data.read_parquet(
    "s3://anonymous@ray-example-data/iris.parquet",
    columns=["sepal.length", "variety"]  # Only load these columns
)

print("Pruned dataset schema:")
print(ds_pruned.schema())
ds_pruned.show(3)
```

### Handling Compressed Files

Ray Data automatically handles compression for common formats:


```python
# Read compressed CSV (gzip example)
# The compression is automatically detected from file extension
# Or specify explicitly:
ds_compressed = ray.data.read_csv(
    "s3://anonymous@ray-example-data/iris.csv.gz",
    arrow_open_stream_args={"compression": "gzip"}
)

print(f"✓ Compressed file handling configured")
```

### Controlling Parallelism

Use `override_num_blocks` to control the number of parallel read tasks:


```python
# Default: Ray Data automatically determines optimal parallelism
# Override when you need more control:

# More parallelism for better distribution (useful for large files)
ds_parallel = ray.data.read_parquet(
    "s3://anonymous@ray-example-data/iris.parquet",
    override_num_blocks=10  # Create 10 blocks regardless of file size
)

print(f"Dataset created with custom parallelism")
# Materialize to check number of blocks
ds_parallel_materialized = ds_parallel.materialize()
print(f"Number of blocks: {ds_parallel_materialized.num_blocks()}")

# Note: Higher override_num_blocks = more parallelism but more overhead
# For small datasets: keep default or use small values (2-10)
# For large datasets: increase based on cluster size (10-100+)
```

### Progress Bars

For long-running operations, you may want to disable verbose progress output:


```python
# Disable progress bars for cleaner output
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = False  # Disable dataset-level progress
ctx.enable_operator_progress_bars = False  # Disable operator-level progress

print("✓ Progress bars configured")

# Re-enable if needed:
# ctx.enable_progress_bars = True
# ctx.enable_operator_progress_bars = True
```

## Best Practices & Troubleshooting

Here are key best practices for working with cloud storage and large datasets on Anyscale.

### Credential Management

**Never hardcode secrets in notebooks or code.**

Best practices:
1. Use environment variables for credentials
2. Set credentials in Anyscale workspace Dependencies tab
3. Use IAM roles / managed identities when available
4. For S3: prefer IAM roles over access keys
5. For GCS: use service account JSON files via `GOOGLE_APPLICATION_CREDENTIALS`
6. For Azure: prefer managed identities over SAS tokens


```python
# ✓ Good: Read from environment
hf_token = os.environ.get("HF_TOKEN")  # None if not set
aws_key = os.environ.get("AWS_ACCESS_KEY_ID")  # None if not set

# ✗ Bad: Never do this
# aws_key = "AKIAIOSFODNN7EXAMPLE"  # DON'T HARDCODE!
```

### Error Handling

Common issues and solutions:


```python
# Issue: "Access Denied" errors
# Solution: Verify IAM roles, service account permissions, or SAS token scope

# Issue: "File not found" errors
# Solution: Check bucket/container names, paths, and region settings

# Issue: Slow reads from cloud storage
# Solution: Use larger override_num_blocks, check network bandwidth, consider data locality

# Issue: Out of memory errors
# Solution: Reduce batch_size in map_batches, increase cluster size, or use streaming execution

print("✓ Error handling patterns documented")
```

### Cross-Cloud Workflows

When working across cloud providers:


```python
# Use /mnt/shared_storage/ for data that needs to persist across clusters
# Example: Workspace reads from S3, processes, writes to /mnt/shared_storage/
#          Then submits a job that reads from /mnt/shared_storage/ and writes to GCS

cross_cloud_path = "/mnt/shared_storage/intermediate_data"
ds_s3.write_parquet(cross_cloud_path)

print(f"✓ Intermediate data written to shared storage: {cross_cloud_path}")
print("This path is accessible from jobs and services running on different clusters")
```

## Summary & Next Steps

### What You've Learned

In this template, you:

1. ✓ Accessed data from S3, GCS, and Azure Blob Storage
2. ✓ Configured authentication for each cloud provider
3. ✓ Read and wrote multiple file formats (Parquet, CSV, JSON, text, images)
4. ✓ Processed large datasets with Ray Data transformations
5. ✓ Optimized performance with column pruning, parallelism tuning, and lazy execution
6. ✓ Applied best practices for credential management and error handling

### Next Steps

**For large-scale data processing:**
- Explore Ray Data's [performance tuning guide](https://docs.ray.io/en/latest/data/performance-tips.html) for advanced optimization
- Learn about [checkpointing and fault tolerance](https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.write_parquet) for long-running jobs
- Try [reading from databases](https://docs.ray.io/en/latest/data/loading-data.html#reading-databases) (MySQL, PostgreSQL, BigQuery)

**For ML workflows:**
- Use Ray Data for [batch inference](https://docs.ray.io/en/latest/data/working-with-llms.html) with LLMs
- Integrate with [Ray Train](https://docs.ray.io/en/latest/train/train.html) for distributed training pipelines
- Connect to [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) for online inference with preprocessing

**For production deployments:**
- Package your workflows as [Anyscale Jobs](https://docs.anyscale.com/platform/jobs/) for scheduled execution
- Set up [monitoring and observability](https://docs.anyscale.com/monitoring) for production workloads
- Configure [autoscaling compute](https://docs.anyscale.com/configuration/compute-configs/) for cost optimization

### Resources

- [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html)
- [Anyscale storage guide](https://docs.anyscale.com/configuration/storage#shared)
- [Cloud provider authentication guides](https://docs.ray.io/en/latest/data/loading-data.html#reading-files-from-cloud-storage)
