# ETL Processing and Optimization With Ray Data

**Time to complete**: 40 min | **Difficulty**: Intermediate | **Prerequisites**: ETL concepts, basic SQL knowledge, data processing experience

## What you'll build

Build comprehensive ETL pipelines using Ray Data's distributed processing capabilities, from foundational concepts with TPC-H benchmark to production-scale optimization techniques for enterprise data processing.

## Table of Contents

1. [ETL Fundamentals with TPC-H](#step-1-etl-fundamentals-with-tpc-h) (10 min)
2. [Data Transformations and Processing](#step-2-data-transformations-and-processing) (12 min)
3. [Performance Optimization Techniques](#step-3-performance-optimization-techniques) (10 min)
4. [Large-Scale ETL Patterns](#step-4-large-scale-etl-patterns) (8 min)

## Learning Objectives

**Why ETL optimization matters**: The difference between fast and slow data pipelines directly impacts business agility and operational costs. Understanding optimization techniques enables data teams to deliver insights faster while reducing infrastructure costs.

**Ray Data's ETL capabilities**: Native operations for distributed processing that automatically optimize memory, CPU, and I/O utilization. You'll learn how Ray Data's architecture enables efficient processing of large datasets.

**TPC-H benchmark patterns**: Learn ETL fundamentals using the TPC-H benchmark that simulates complex business environments with customers, orders, suppliers, and products.

**Production optimization strategies**: Memory management, parallel processing, and resource configuration patterns for production ETL workloads that scale from gigabytes to petabytes.

**Enterprise ETL patterns**: Techniques used by data engineering teams to process large datasets efficiently while maintaining data quality and performance.

## Prerequisites Checklist

Before starting, ensure you have:
- Understanding of ETL (Extract, Transform, Load) concepts
- Basic SQL knowledge for data transformations
- Python experience with data processing
- Familiarity with distributed computing concepts


## Quick start (3 minutes)

This section demonstrates ETL processing concepts using Ray Data:



```python
from typing import Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data.expressions import col, lit

from typing import Dict, Any, List
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data.aggregate import Count, Mean, Sum, Max
from ray.data.expressions import col, lit


# Configure Ray Data for optimal performance monitoring
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = False
ctx.enable_operator_progress_bars = False

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Load sample dataset for ETL demonstration
sample_data = ray.data.read_parquet(
    "s3://ray-benchmark-data/tpch/parquet/sf1/customer",
)

sample_data = sample_data.drop_columns(["column8"])
sample_data = sample_data.rename_columns([
    "c_custkey",
    "c_name",
    "c_address",
    "c_nationkey",
    "c_phone",
    "c_acctbal",
    "c_mktsegment",
    "c_comment",
    ])

print(f"Loaded ETL sample dataset: {sample_data.count()} records")
print(f"Schema: {sample_data.schema()}")
print("\nSample records:")
for i, record in enumerate(sample_data.take(3)):
    print(f"  {i+1}. Customer {record['c_custkey']}: {record['c_name']} from {record['c_mktsegment']}")

```

## Overview

**Challenge**: Traditional ETL tools struggle with modern data volumes and complexity. Processing large datasets can take significant time, creating bottlenecks in data-driven organizations.

**Solution**: Ray Data's distributed architecture and optimized operations enable efficient processing of large datasets through parallel computation and native operations.

**Impact**: Data engineering teams process terabytes of data daily using Ray Data's ETL capabilities. Companies transform raw data into analytics-ready datasets efficiently while maintaining data quality and performance.

### ETL pipeline architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ray Data ETL Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Extract              Transform              Load               │
│  ────────            ──────────            ──────              │
│                                                                 │
│  read_parquet()  →   map_batches()    →   write_parquet()     │
│  (TPC-H Data)        (Business Logic)     (Data Warehouse)     │
│                                                                 │
│  ↓ Column Pruning    ↓ Filter/Join       ↓ Partitioning       │
│  ↓ Parallel I/O      ↓ Aggregations      ↓ Compression        │
│  ↓ High Concurrency  ↓ Enrichment        ↓ Schema Optimization│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Data Flow:
  TPC-H Customer (150K) ─┐
  TPC-H Orders (1.5M)   ─┼→ Join → Enrich → Aggregate → Warehouse
  TPC-H LineItems (6M)  ─┘      ↓         ↓            ↓
                            Filter    Transform    Partition
```

### ETL performance comparison

| Approach | Data Loading | Transformations | Joins | Output | Use Case |
|-----------|--------------|------------------|--------|----------|-----------|
| **Traditional** | Sequential | Single-threaded | Memory-limited | Slow writes | Small datasets |
| **Ray Data** | Parallel I/O | Distributed | Scalable | Optimized writes | Production scale |

**Key advantages**:
- **Parallel processing**: Distribute transformations across cluster nodes
- **Memory efficiency**: Stream processing without materializing full datasets
- **Native operations**: Optimized filter, join, and aggregate functions
- **Scalability**: Handle datasets from gigabytes to petabytes


## Step 1: ETL Fundamentals with TPC-H

### Understanding TPC-H benchmark

**What is TPC-H?**

The TPC-H benchmark is used for testing database and data processing performance. It simulates a business environment with data relationships that represent business scenarios.

**TPC-H Business Context**: The benchmark models a wholesale supplier managing customer orders, inventory, and supplier relationships - representing business data systems.


### TPC-H schema overview

The TPC-H benchmark provides realistic business data for learning ETL patterns. Understanding the schema helps you apply these techniques to your own data.

| Table | Description | Typical Size (SF10) | Primary Use |
|-----------|------------------|--------------------------|------------------|
| **CUSTOMER** | Customer master data | 1.5M rows | Dimensional analysis |
| **ORDERS** | Order transactions | 15M rows | Fact table, time series |
| **LINEITEM** | Order line items | 60M rows | Largest fact table |
| **PART** | Product catalog | 2M rows | Product dimensions |
| **SUPPLIER** | Supplier information | 100K rows | Supplier analytics |
| **PARTSUPP** | Part-supplier links | 8M rows | Supply chain |
| **NATION** | Geographic data | 25 rows | Geographic grouping |
| **REGION** | Regional groups | 5 rows | High-level geography |

**Schema relationships**:

```
CUSTOMER ──one-to-many──→ ORDERS ──one-to-many──→ LINEITEM
                                                      ↓
NATION ──one-to-many──→ SUPPLIER                   PART
   ↓                        ↓                         ↓
REGION                  PARTSUPP ←────many-to-one────┘
```



```python
# TPC-H Schema Overview for ETL Processing
tpch_tables = {
    "customer": "Customer master data with demographics and market segments",
    "orders": "Order header information with dates, priorities, and status",
    "lineitem": "Detailed line items for each order (largest table)",
    "part": "Parts catalog with specifications and retail prices", 
    "supplier": "Supplier information including contact details",
    "partsupp": "Part-supplier relationships with costs",
    "nation": "Nation reference data with geographic regions",
    "region": "Regional groupings for geographic analysis"
}

print("TPC-H Schema (8 Tables):")
for table, description in tpch_tables.items():
    print(f"  {table.upper()}: {description}")

```

### Loading TPC-H data with Ray Data



```python
# TPC-H benchmark data location
TPCH_S3_PATH = "s3://ray-benchmark-data/tpch/parquet/sf10"

print("Loading TPC-H benchmark data for distributed processing...")
start_time = time.time()

try:
    # Read TPC-H Customer Master Data
    customers_ds = ray.data.read_parquet(
        f"{TPCH_S3_PATH}/customer",
        ray_remote_args={"num_cpus":0.25}
    )
    customers_ds = customers_ds.drop_columns(["column8"])
    customers_ds = customers_ds.rename_columns([
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
        ])
    
    # Read TPC-H Orders Data
    orders_ds = ray.data.read_parquet(
        f"{TPCH_S3_PATH}/orders", 
        ray_remote_args={"num_cpus":0.25}
    )
    orders_ds = (orders_ds
        .select_columns([f"column{i}" for i in range(9)])
        .rename_columns([
            "o_orderkey",
            "o_custkey",
            "o_orderstatus",
            "o_totalprice",
            "o_orderdate",
            "o_orderpriority",
            "o_clerk",
            "o_shippriority",
            "o_comment",
        ])
    )
    
    # Read TPC-H Line Items (largest table)
    lineitems_ds = ray.data.read_parquet(
        f"{TPCH_S3_PATH}/lineitem",
        ray_remote_args={"num_cpus":0.25}
    )
    lineitem_cols = [f"column{str(i).zfill(2)}" for i in range(16)]
    lineitems_ds = (lineitems_ds
        .select_columns(lineitem_cols)
        .rename_columns([
            "l_orderkey",
            "l_partkey",
            "l_suppkey",
            "l_linenumber",
            "l_quantity",
            "l_extendedprice",
            "l_discount",
            "l_tax",
            "l_returnflag",
            "l_linestatus",
            "l_shipdate",
            "l_commitdate",
            "l_receiptdate",
            "l_shipinstruct",
            "l_shipmode",
            "l_comment",
        ])
    )
    
    load_time = time.time() - start_time
    
    # Count records in parallel
    customer_count = customers_ds.count()
    orders_count = orders_ds.count()
    lineitems_count = lineitems_ds.count()
    
    print(f"TPC-H data loaded successfully in {load_time:.2f} seconds")
    print(f"   Customers: {customer_count:,}")
    print(f"   Orders: {orders_count:,}")
    print(f"   Line items: {lineitems_count:,}")
    print(f"   Total records: {customer_count + orders_count + lineitems_count:,}")
    
except Exception as e:
    print(f"ERROR: Failed to load TPC-H data: {e}")
    raise

```

### Basic ETL transformations



```python
# ETL Transform: Customer segmentation using Ray Data native operations
def segment_customers(batch: pd.DataFrame) -> pd.DataFrame:
    """Apply business rules for customer segmentation.
    
    This demonstrates common ETL pattern of adding derived business attributes
    based on rules and thresholds.
    
    Args:
        batch: Pandas DataFrame with customer records
        
    Returns:
        DataFrame with added customer_segment column
    """
    # Business logic for customer segmentation based on account balance
    batch['customer_segment'] = 'standard'
    batch.loc[batch['c_acctbal'] > 5000, 'customer_segment'] = 'premium'
    batch.loc[batch['c_acctbal'] > 10000, 'customer_segment'] = 'enterprise'
    
    return batch

# Apply customer segmentation transformation
print("Applying customer segmentation...")

try:
    segmented_customers = customers_ds.map_batches(
        segment_customers,
        num_cpus=0.5,  # Medium complexity transformation
        batch_format="pandas"
    )
    
    segment_count = segmented_customers.count()
    print(f"Customer segmentation completed: {segment_count:,} customers segmented")
    
except Exception as e:
    print(f"ERROR: Segmentation failed: {e}")
    raise

# ETL Filter: High-value customers using expressions API
print("Filtering high-value customers...")

try:
    high_value_customers = segmented_customers.filter(
        expr="c_acctbal > 1000",
        num_cpus=0.1
    )
    
    high_value_count = high_value_customers.count()
    total_count = segmented_customers.count()
    percentage = (high_value_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"High-value customers: {high_value_count:,} ({percentage:.1f}% of total)")
    
except Exception as e:
    print(f"Error during filtering: {e}")
    raise

# ETL Aggregation: Customer statistics by market segment
customer_stats = segmented_customers.groupby("c_mktsegment").aggregate(
    Count(),
    Mean("c_acctbal"),
    Sum("c_acctbal"),
    Max("c_acctbal")
)

print("Customer Statistics by Market Segment:")
print("=" * 70)
# Display customer statistics
stats_df = customer_stats.limit(10).to_pandas()
print(stats_df.to_string(index=False))
print("=" * 70)

```

## Step 2: Data Transformations and Processing

This section demonstrates how Ray Data handles common ETL transformation patterns including data enrichment, filtering, and complex business logic. You'll learn to build production-grade transformations that scale efficiently.

### Why transformations are critical

Data transformations convert raw data into business-valuable information. Common transformation patterns include:

- **Enrichment**: Adding calculated fields and derived metrics
- **Filtering**: Removing irrelevant or invalid records  
- **Joins**: Combining data from multiple sources
- **Aggregations**: Computing summary statistics and rollups
- **Type conversions**: Ensuring correct data types for analytics

### Transformation performance comparison

| Transformation Type | Traditional Approach | Ray Data Approach | Scalability |
|-------------------|---------------------|-------------------|--------------|
| **Column calculations** | Row-by-row processing | Vectorized batches | Linear scaling |
| **Date parsing** | Sequential parsing | Parallel batch parsing | High throughput |
| **Categorization** | Conditional logic loops | Pandas vectorization | Efficient |
| **Business rules** | Single-threaded | Distributed map_batches | Scales to cluster |


### Complex data transformations


<div style="margin:1em 0; padding:12px 16px; border-left:4px solid #2e7d32; background:#f1f8e9; border-radius:4px;">

  **GPU Acceleration for Pandas ETL Operations**: For complex pandas transformations in your ETL pipeline, you can use **NVIDIA RAPIDS cuDF** to accelerate DataFrame operations on GPUs. 
  
  Replace `import pandas as pd` with `import cudf as pd` in your `map_batches` functions to use GPU acceleration for operations like datetime parsing, groupby, joins, and aggregations.

**When to use cuDF**:
- Complex datetime operations (parsing, extracting components)
- Large aggregations and groupby operations
- String operations on millions of rows
- Join operations on large datasets
- Statistical calculations across many columns

**Performance benefit**: GPU-accelerated pandas operations can be 10-50x faster for large batches (1000+ rows) with complex transformations.

**Requirements**: Add `cudf` to your dependencies and ensure GPU-enabled cluster nodes.

**Before**

```python
def my_fnc(batch):
    # Process batch with pandas operations here
    res = ...
    return res

ds = ds.map_batches(my_fnc, format="pandas")
```

**After**

```python
def my_fnc(batch):
    batch = cudf.from_pandas(batch)
    res = ...
    return res

ds = ds.map_batches(my_fnc, format="pandas", num_gpus=1)
```

</div>



```python
# ETL Transform: Order enrichment with business metrics
def enrich_orders_with_metrics(batch):
    """Enrich orders with calculated business metrics.
    
    For GPU acceleration, replace 'import pandas as pd' with 'import cudf as pd'
    to speed up complex DataFrame operations like datetime parsing and categorization.
    """
    import pandas as pd  # or 'import cudf as pd' for GPU acceleration
    df = pd.DataFrame(batch)
    
    # Parse order date and create time dimensions
    # This datetime parsing is GPU-accelerated with cuDF
    df['o_orderdate'] = pd.to_datetime(df['o_orderdate'])
    df['order_year'] = df['o_orderdate'].dt.year
    df['order_quarter'] = df['o_orderdate'].dt.quarter
    df['order_month'] = df['o_orderdate'].dt.month
    
    # Business classifications
    # These conditional operations are GPU-accelerated with cuDF
    df['is_large_order'] = df['o_totalprice'] > 200000
    df['is_urgent'] = df['o_orderpriority'].isin(['1-URGENT', '2-HIGH'])
    df['revenue_tier'] = pd.cut(
        df['o_totalprice'],
        bins=[0, 50000, 150000, 300000, float('inf')],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    ).astype(str)  # Convert categorical to string for Ray Data compatibility
    
    return df
    
# Apply order enrichment
print("\nEnriching orders with business metrics...")

try:
    enriched_orders = orders_ds.map_batches(
        enrich_orders_with_metrics,
        num_cpus=0.5,  # Medium complexity transformation
        batch_format="pandas"
    )
    
    enriched_count = enriched_orders.count()
    print(f"Order enrichment completed: {enriched_count:,} orders processed")
    
    # Show sample enriched record
    sample = enriched_orders.take(1)[0]
    print(f"\nSample enriched order:")
    print(f"   Order ID: {sample.get('o_orderkey')}")
    print(f"   Year: {sample.get('order_year')}, Quarter: {sample.get('order_quarter')}")
    print(f"   Revenue Tier: {sample.get('revenue_tier')}")
    print(f"   Is Large Order: {sample.get('is_large_order')}")
    print(f"   Is Urgent: {sample.get('is_urgent')}")
    
except Exception as e:
    print(f"Error during enrichment: {e}")
    raise

```

### Advanced filtering and selection



```python
# Advanced filtering using Ray Data expressions API
print("Applying advanced filtering techniques...")

recent_high_value_orders = enriched_orders.filter(
expr="order_year >= 1995 and o_totalprice > 100000 and is_urgent",
num_cpus=0.1
)

enterprise_orders = enriched_orders.filter(
expr="revenue_tier == 'Enterprise'",
num_cpus=0.1
)

complex_filtered_orders = enriched_orders.filter(
expr="order_quarter == 4 and o_orderstatus == 'F' and o_totalprice > 50000",
num_cpus=0.1
)

print("Advanced filtering results:")
print(f"  Recent high-value orders: {recent_high_value_orders.count():,}")
print(f"  Enterprise orders: {enterprise_orders.count():,}")
print(f"  Complex filtered orders: {complex_filtered_orders.count():,}")

```


```python
recent_high_value_orders.limit(5).to_pandas()
```


```python
enterprise_orders.limit(5).to_pandas()
```


```python
complex_filtered_orders.limit(5).to_pandas()
```

### Data joins and relationships



```python
# ETL Join: Customer-Order analysis using Ray Data joins
print("\nPerforming distributed joins for customer-order analysis...")

try:
    # Join customers with their orders for comprehensive analysis
    # Ray Data optimizes join execution across distributed nodes
    customer_order_analysis = customers_ds.join(
        enriched_orders,
        on=("c_custkey",),
        right_on=("o_custkey",),
        join_type="inner",
        num_partitions=100
    )
    
    join_count = customer_order_analysis.count()
    print(f"Customer-order join completed: {join_count:,} records")
    
    # Calculate join statistics
    customer_count = customers_ds.count()
    orders_count = enriched_orders.count()
    join_ratio = (join_count / orders_count) * 100 if orders_count > 0 else 0
    
    print(f"   Input: {customer_count:,} customers, {orders_count:,} orders")
    print(f"   Join ratio: {join_ratio:.1f}% of orders matched")
    
except Exception as e:
    print(f"Error during join: {e}")
    raise

# Aggregate customer order metrics
customer_order_metrics = customer_order_analysis.groupby("c_mktsegment").aggregate(
    Count(),
    Mean("o_totalprice"),
    Sum("o_totalprice"),
    Count("o_orderkey")
)
```


```python
customer_order_metrics.limit(5).to_pandas()
```

## Step 3: Performance Optimization Techniques

This section covers advanced optimization techniques for production ETL workloads.



```python
# Configure Ray Data for optimal ETL performance
print("Configuring Ray Data for ETL optimization...")

# Memory optimization for large datasets
ctx.target_max_block_size = 128 * 1024 * 1024  # 128 MB blocks
ctx.eager_free = True  # Aggressive memory cleanup

# Enable performance monitoring
ctx.enable_auto_log_stats = True
ctx.memory_usage_poll_interval_s = 5.0

print("Ray Data configured for optimal ETL performance")

```

### Batch size and concurrency optimization



```python
import uuid
from datetime import datetime

# Demonstrate different batch size strategies for ETL operations
print("Testing ETL batch size optimization...")

# Small batch processing for memory-constrained operations
def memory_intensive_etl(batch):
    """Memory-intensive ETL transformation."""
    import pandas as pd
    import numpy as np
    df = pd.DataFrame(batch)
    
    # Simulate memory-intensive operations
    df['complex_metric'] = df['o_totalprice'] * np.log(df['o_totalprice'] + 1)
    df['percentile_rank'] = df['o_totalprice'].rank(pct=True)
    
    return df 

# Apply with optimized batch size for memory management
memory_optimized_orders = enriched_orders.map_batches(
    memory_intensive_etl,
    num_cpus=1.0,  # Fewer concurrent tasks for memory management
    batch_size=500,  # Smaller batches for memory efficiency
    batch_format="pandas"
)

print(f"Memory-optimized processing: {memory_optimized_orders.count():,} records")
```


```python
memory_optimized_orders.limit(5).to_pandas()
```


```python
import os

output_dir = "/mnt/cluster_storage/temp_etl_batches"
os.makedirs(output_dir, exist_ok=True)

def io_intensive_etl(batch):
    """I/O-intensive ETL transformation with actual disk writes."""
    import pandas as pd
    from datetime import datetime
    import uuid
    
    df = pd.DataFrame(batch)
    
    # Add processing metadata
    df['processing_timestamp'] = datetime.now().isoformat()
    batch_id = str(uuid.uuid4())[:8]
    df['batch_id'] = batch_id
    
    # Actual I/O operation: write batch to disk
    output_path = f"{output_dir}/batch_{batch_id}.parquet"
    df.to_parquet(output_path, index=False)
    
    return df

# Apply with optimized batch size for I/O efficiency
io_optimized_orders = enriched_orders.map_batches(
    io_intensive_etl,
    num_cpus=0.25,  # Higher concurrency for I/O operations
    batch_size=2000,  # Larger batches for I/O efficiency
    batch_format="pandas"
)

print(f"I/O-optimized processing: {io_optimized_orders.count():,} records")
print(f"Batch files written to: /mnt/cluster_storage/temp_etl_batches/")
```


```python
io_optimized_orders.limit(5).to_pandas()
```

### Column selection and schema optimization



```python
# ETL Optimization: Column pruning for performance
print("Applying column selection optimization...")

# Select only essential columns for downstream processing
essential_customer_columns = customers_ds.select_columns([
    "c_custkey", "c_name", "c_mktsegment", "c_acctbal", "c_nationkey"
])

essential_order_columns = enriched_orders.select_columns([
    "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate", 
    "order_year", "revenue_tier", "is_large_order"
])

print(f"Column optimization:")
print(f"  Customer columns: {len(essential_customer_columns.schema().names)}")
print(f"  Order columns: {len(essential_order_columns.schema().names)}")

# Optimized join with selected columns
optimized_join = essential_customer_columns.join(
    essential_order_columns,
    on=("c_custkey",),
    right_on=("o_custkey",),
    num_partitions=100,
    join_type="inner",
)

print(f"Optimized join completed: {optimized_join.count():,} records")

```


```python
optimized_join.limit(5).to_pandas()
```

## Step 4: Large-Scale ETL Patterns

Production ETL systems must handle billions of records efficiently. This section demonstrates Ray Data patterns for large-scale data processing including distributed aggregations, multi-dimensional analysis, and data warehouse integration.



```python
# Large-scale aggregations using Ray Data 
print("Performing large-scale distributed aggregations...")

# Multi-dimensional aggregations for business intelligence
comprehensive_metrics = optimized_join.groupby(["c_mktsegment", "order_year", "revenue_tier"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice"),
    Max("o_totalprice"),
    Mean("c_acctbal")
)

print("Comprehensive Business Metrics:")
print(comprehensive_metrics.limit(5).to_pandas())
```


```python
# Time-series aggregations for trend analysis
yearly_trends = optimized_join.groupby("order_year").aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice")
)
```


```python
yearly_trends.limit(5).to_pandas()
```


```python
# Customer segment performance analysis
segment_performance = optimized_join.groupby(["c_mktsegment", "revenue_tier"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("c_acctbal")
)
```


```python
segment_performance.limit(5).to_pandas()
```

### ETL output and data warehouse integration

Ray Data provides native write functions for various data warehouses and file formats, enabling you to export processed datasets directly to your target storage systems. You can write to Snowflake using `write_snowflake()`, which handles authentication and schema management automatically. 


For other data warehouses, Ray Data supports writing to BigQuery with `write_bigquery()`, SQL databases with `write_sql()`, and modern table formats like Delta Lake (`write_delta()` and `write_unity_catalog()`, *coming soon*) and Apache Iceberg (`write_iceberg()`). Additionally, you can write to file-based formats such as Parquet using `write_parquet(),` which offers efficient columnar storage with compression options. 


These native write functions integrate seamlessly with Ray Data's distributed processing, allowing you to scale data export operations across your cluster while maintaining data consistency and optimizing write performance.


```python
# Write ETL results to data warehouse formats
print("Writing ETL results to data warehouse...")

# Replace with S3 or other cloud storage in a real production use case
BASE_DIRECTORY = "/mnt/cluster_storage/"

# Write customer analytics with partitioning
enriched_customers = segmented_customers
enriched_customers.write_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/customers/",
    partition_cols=["customer_segment"],
    compression="snappy",
    ray_remote_args={"num_cpus": 0.1}
)

# Write order analytics with time-based partitioning
enriched_orders.write_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/orders/",
    partition_cols=["order_year"],
    compression="snappy",
    ray_remote_args={"num_cpus": 0.1}
)

# Write aggregated analytics for BI tools
final_analytics = optimized_join.groupby(["c_mktsegment", "revenue_tier", "order_year"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice"),
    Mean("c_acctbal")
)

final_analytics.write_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/analytics/",
    partition_cols=["order_year"],
    compression="snappy",
    ray_remote_args={"num_cpus": 0.1}
)
```

### Output Validation



```python
# Validate ETL pipeline performance
print("Validating ETL output...")

BASE_DIRECTORY = "/mnt/cluster_storage/"

# Read back and verify outputs
customer_verification = ray.data.read_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/customers/",
    ray_remote_args={"num_cpus":0.025}
)

order_verification = ray.data.read_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/orders/",
    ray_remote_args={"num_cpus":0.025}
)

analytics_verification = ray.data.read_parquet(
    f"{BASE_DIRECTORY}/etl_warehouse/analytics/",
    ray_remote_args={"num_cpus":0.025}
)

print(f"ETL Pipeline Verification:")
print(f"  Customer records: {customer_verification.count():,}")
print(f"  Order records: {order_verification.count():,}")
print(f"  Analytics records: {analytics_verification.count():,}")

# Display sample results
sample_analytics = analytics_verification.take(25)
print("\nSample ETL Analytics Results:")
for i, record in enumerate(sample_analytics):
    print(f"  {i+1}. Segment: {record['c_mktsegment']}, Tier: {record['revenue_tier']}, "
          f"Year: {record['order_year']}, Orders: {record['count()']}, Revenue: ${record['sum(o_totalprice)']:,.0f}")

```


