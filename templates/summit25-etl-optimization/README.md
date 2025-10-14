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

    2025-10-10 20:21:59,528	INFO worker.py:1771 -- Connecting to existing Ray cluster at address: 10.0.71.116:6379...
    2025-10-10 20:21:59,540	INFO worker.py:1942 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-77uweunq3awbhqefvry4lwcqq5.i.anyscaleuserdata.com [39m[22m
    2025-10-10 20:21:59,542	INFO packaging.py:380 -- Pushing file package 'gcs://_ray_pkg_bfd427b63b81c2f4449778ffaca41253837f9946.zip' (0.16MiB) to Ray cluster...
    2025-10-10 20:21:59,543	INFO packaging.py:393 -- Successfully pushed file package 'gcs://_ray_pkg_bfd427b63b81c2f4449778ffaca41253837f9946.zip'.
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/data/_internal/datasource/parquet_datasource.py:750: FutureWarning: The default `file_extensions` for `read_parquet` will change from `None` to ['parquet'] after Ray 2.43, and your dataset contains files that don't match the new `file_extensions`. To maintain backwards compatibility, set `file_extensions=None` explicitly.
      warnings.warn(
    2025-10-10 20:21:59,787	INFO logging.py:295 -- Registered dataset logger for dataset dataset_308_0
    2025-10-10 20:21:59,808	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_308_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:21:59,809	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_308_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)] -> LimitOperator[limit=1]
    2025-10-10 20:21:59,816	WARNING resource_manager.py:134 -- ⚠️  Ray's object store is configured to use only 27.9% of available memory (98.3GiB out of 352.0GiB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.
    2025-10-10 20:22:05,051	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_308_0 execution finished in 5.24 seconds
    2025-10-10 20:22:05,058	INFO logging.py:295 -- Registered dataset logger for dataset dataset_310_0
    2025-10-10 20:22:05,067	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_310_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:05,068	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_310_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> AggregateNumRows[AggregateNumRows]
    2025-10-10 20:22:10,287	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_310_0 execution finished in 5.22 seconds
    2025-10-10 20:22:10,295	INFO logging.py:295 -- Registered dataset logger for dataset dataset_311_0
    2025-10-10 20:22:10,302	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_311_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:10,303	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_311_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)] -> LimitOperator[limit=1] -> TaskPoolMapOperator[Project]


    Loaded ETL sample dataset: 150000 records


    2025-10-10 20:22:16,948	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_311_0 execution finished in 6.64 seconds
    2025-10-10 20:22:16,953	INFO dataset.py:3248 -- Tip: Use `take_batch()` instead of `take() / show()` to return records in pandas or numpy batch format.
    2025-10-10 20:22:16,955	INFO logging.py:295 -- Registered dataset logger for dataset dataset_312_0
    2025-10-10 20:22:16,964	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_312_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:16,965	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_312_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)] -> LimitOperator[limit=3] -> TaskPoolMapOperator[Project]


    Schema: Column        Type
    ------        ----
    c_custkey     int64
    c_name        string
    c_address     string
    c_nationkey   int64
    c_phone       string
    c_acctbal     double
    c_mktsegment  string
    c_comment     string
    
    Sample records:


    2025-10-10 20:22:17,536	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_312_0 execution finished in 0.57 seconds


      1. Customer 1: Customer#000000001 from BUILDING
      2. Customer 2: Customer#000000002 from AUTOMOBILE
      3. Customer 3: Customer#000000003 from AUTOMOBILE


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

    TPC-H Schema (8 Tables):
      CUSTOMER: Customer master data with demographics and market segments
      ORDERS: Order header information with dates, priorities, and status
      LINEITEM: Detailed line items for each order (largest table)
      PART: Parts catalog with specifications and retail prices
      SUPPLIER: Supplier information including contact details
      PARTSUPP: Part-supplier relationships with costs
      NATION: Nation reference data with geographic regions
      REGION: Regional groupings for geographic analysis


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

    2025-10-10 20:22:17,785	INFO logging.py:295 -- Registered dataset logger for dataset dataset_315_0
    2025-10-10 20:22:17,795	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_315_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:17,796	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_315_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)] -> LimitOperator[limit=1]


    Loading TPC-H benchmark data for distributed processing...


    2025-10-10 20:22:18,587	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_315_0 execution finished in 0.79 seconds
    2025-10-10 20:22:18,604	INFO logging.py:295 -- Registered dataset logger for dataset dataset_319_0
    2025-10-10 20:22:18,611	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_319_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:18,612	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_319_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> LimitOperator[limit=1] -> TaskPoolMapOperator[Project]
    2025-10-10 20:22:20,823	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_319_0 execution finished in 2.21 seconds
    2025-10-10 20:22:20,837	INFO logging.py:295 -- Registered dataset logger for dataset dataset_323_0
    2025-10-10 20:22:20,847	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_323_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:20,848	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_323_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> LimitOperator[limit=1] -> TaskPoolMapOperator[Project]
    2025-10-10 20:22:23,870	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_323_0 execution finished in 3.02 seconds
    2025-10-10 20:22:23,877	INFO logging.py:295 -- Registered dataset logger for dataset dataset_325_0
    2025-10-10 20:22:23,885	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_325_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:23,886	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_325_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> AggregateNumRows[AggregateNumRows]
    2025-10-10 20:22:24,730	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_325_0 execution finished in 0.84 seconds
    2025-10-10 20:22:24,738	INFO logging.py:295 -- Registered dataset logger for dataset dataset_326_0
    2025-10-10 20:22:24,744	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_326_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:24,744	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_326_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[MapBatches(count_rows)]
    2025-10-10 20:22:25,051	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_326_0 execution finished in 0.31 seconds
    2025-10-10 20:22:25,057	INFO logging.py:295 -- Registered dataset logger for dataset dataset_327_0
    2025-10-10 20:22:25,063	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_327_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:25,064	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_327_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[MapBatches(count_rows)]
    2025-10-10 20:22:25,469	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_327_0 execution finished in 0.41 seconds


    TPC-H data loaded successfully in 6.10 seconds
       Customers: 1,500,000
       Orders: 15,000,000
       Line items: 59,986,052
       Total records: 76,486,052


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

    2025-10-10 20:22:25,623	INFO logging.py:295 -- Registered dataset logger for dataset dataset_329_0
    2025-10-10 20:22:25,634	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_329_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:25,634	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_329_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> TaskPoolMapOperator[MapBatches(segment_customers)] -> AggregateNumRows[AggregateNumRows]


    Applying customer segmentation...


    2025-10-10 20:22:26,877	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_329_0 execution finished in 1.24 seconds
    2025-10-10 20:22:26,887	INFO logging.py:295 -- Registered dataset logger for dataset dataset_331_0
    2025-10-10 20:22:26,896	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_331_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:26,897	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_331_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> TaskPoolMapOperator[MapBatches(segment_customers)] -> TaskPoolMapOperator[Filter(<expression>)] -> AggregateNumRows[AggregateNumRows]


    Customer segmentation completed: 1,500,000 customers segmented
    Filtering high-value customers...


    2025-10-10 20:22:28,473	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_331_0 execution finished in 1.58 seconds
    2025-10-10 20:22:28,480	INFO logging.py:295 -- Registered dataset logger for dataset dataset_332_0
    2025-10-10 20:22:28,489	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_332_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:28,490	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_332_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> TaskPoolMapOperator[MapBatches(segment_customers)] -> AggregateNumRows[AggregateNumRows]
    2025-10-10 20:22:29,739	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_332_0 execution finished in 1.25 seconds
    2025-10-10 20:22:29,749	INFO logging.py:295 -- Registered dataset logger for dataset dataset_335_0
    2025-10-10 20:22:29,759	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_335_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:29,760	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_335_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> TaskPoolMapOperator[MapBatches(segment_customers)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=10]


    High-value customers: 1,227,529 (81.8% of total)
    Customer Statistics by Market Segment:
    ======================================================================


    2025-10-10 20:22:31,286	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: c_mktsegment: string
    count(): int64
    mean(c_acctbal): double
    sum(c_acctbal): double
    max(c_acctbal): double, new schema: None. This may lead to unexpected behavior.
    [36m(reduce pid=75571, ip=10.0.93.34)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:22:31,314	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_335_0 execution finished in 1.55 seconds


    c_mktsegment  count()  mean(c_acctbal)  sum(c_acctbal)  max(c_acctbal)
      AUTOMOBILE   300036      4496.230542    1.349031e+09         9999.96
        BUILDING   300276      4505.869852    1.353005e+09         9999.99
       FURNITURE   299496      4500.162798    1.347781e+09         9999.98
       HOUSEHOLD   299751      4499.862741    1.348838e+09         9999.99
       MACHINERY   300441      4492.427445    1.349709e+09         9999.96
    ======================================================================


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

    [36m(reduce pid=77073, ip=10.0.109.213)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:22:31,441	INFO logging.py:295 -- Registered dataset logger for dataset dataset_337_0
    2025-10-10 20:22:31,451	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_337_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:31,452	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_337_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> AggregateNumRows[AggregateNumRows]


    
    Enriching orders with business metrics...


    2025-10-10 20:22:36,198	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_337_0 execution finished in 4.75 seconds
    2025-10-10 20:22:36,205	INFO logging.py:295 -- Registered dataset logger for dataset dataset_338_0
    2025-10-10 20:22:36,216	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_338_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:36,216	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_338_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> LimitOperator[limit=1]


    Order enrichment completed: 15,000,000 orders processed


    2025-10-10 20:22:38,806	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_338_0 execution finished in 2.59 seconds


    
    Sample enriched order:
       Order ID: 4423681
       Year: 1996, Quarter: 4
       Revenue Tier: Medium
       Is Large Order: False
       Is Urgent: True


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

    2025-10-10 20:22:38,933	INFO logging.py:295 -- Registered dataset logger for dataset dataset_342_0
    2025-10-10 20:22:38,943	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_342_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:38,944	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_342_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> AggregateNumRows[AggregateNumRows]


    Applying advanced filtering techniques...
    Advanced filtering results:


    2025-10-10 20:22:44,212	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_342_0 execution finished in 5.27 seconds
    2025-10-10 20:22:44,221	INFO logging.py:295 -- Registered dataset logger for dataset dataset_343_0
    2025-10-10 20:22:44,229	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_343_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:44,230	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_343_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> AggregateNumRows[AggregateNumRows]


      Recent high-value orders: 2,176,683


    2025-10-10 20:22:49,489	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_343_0 execution finished in 5.26 seconds
    2025-10-10 20:22:49,497	INFO logging.py:295 -- Registered dataset logger for dataset dataset_344_0
    2025-10-10 20:22:49,505	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_344_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:49,506	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_344_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> AggregateNumRows[AggregateNumRows]


      Enterprise orders: 854,969


    2025-10-10 20:22:54,715	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_344_0 execution finished in 5.21 seconds


      Complex filtered orders: 1,479,415



```python
recent_high_value_orders.limit(5).to_pandas()
```

    2025-10-10 20:22:54,830	INFO logging.py:295 -- Registered dataset logger for dataset dataset_345_0
    2025-10-10 20:22:54,839	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_345_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:54,840	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_345_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> LimitOperator[limit=5]
    2025-10-10 20:22:57,743	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_345_0 execution finished in 2.90 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>o_orderkey</th>
      <th>o_custkey</th>
      <th>o_orderstatus</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>o_orderpriority</th>
      <th>o_clerk</th>
      <th>o_shippriority</th>
      <th>o_comment</th>
      <th>order_year</th>
      <th>order_quarter</th>
      <th>order_month</th>
      <th>is_large_order</th>
      <th>is_urgent</th>
      <th>revenue_tier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4423712</td>
      <td>1115005</td>
      <td>P</td>
      <td>309802.49</td>
      <td>1995-04-16</td>
      <td>1-URGENT</td>
      <td>Clerk#000009508</td>
      <td>0</td>
      <td>ounts. furiously bold accou</td>
      <td>1995</td>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>Enterprise</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4423719</td>
      <td>773638</td>
      <td>O</td>
      <td>140131.93</td>
      <td>1997-02-22</td>
      <td>2-HIGH</td>
      <td>Clerk#000000979</td>
      <td>0</td>
      <td>kages nag along the pending ideas. even, expre...</td>
      <td>1997</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>True</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4423745</td>
      <td>1477240</td>
      <td>O</td>
      <td>227181.93</td>
      <td>1995-09-30</td>
      <td>2-HIGH</td>
      <td>Clerk#000001033</td>
      <td>0</td>
      <td>ly whithout the final deposits;</td>
      <td>1995</td>
      <td>3</td>
      <td>9</td>
      <td>True</td>
      <td>True</td>
      <td>Large</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4423748</td>
      <td>528055</td>
      <td>O</td>
      <td>141626.74</td>
      <td>1996-06-11</td>
      <td>1-URGENT</td>
      <td>Clerk#000000618</td>
      <td>0</td>
      <td>ly regular sentiments integrate unusual reques...</td>
      <td>1996</td>
      <td>2</td>
      <td>6</td>
      <td>False</td>
      <td>True</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4423840</td>
      <td>325541</td>
      <td>O</td>
      <td>154640.69</td>
      <td>1996-04-28</td>
      <td>2-HIGH</td>
      <td>Clerk#000003311</td>
      <td>0</td>
      <td>de of the closely final pi</td>
      <td>1996</td>
      <td>2</td>
      <td>4</td>
      <td>False</td>
      <td>True</td>
      <td>Large</td>
    </tr>
  </tbody>
</table>
</div>




```python
enterprise_orders.limit(5).to_pandas()
```

    2025-10-10 20:22:57,873	INFO logging.py:295 -- Registered dataset logger for dataset dataset_346_0
    2025-10-10 20:22:57,886	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_346_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:22:57,887	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_346_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> LimitOperator[limit=5]
    2025-10-10 20:23:00,786	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_346_0 execution finished in 2.90 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>o_orderkey</th>
      <th>o_custkey</th>
      <th>o_orderstatus</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>o_orderpriority</th>
      <th>o_clerk</th>
      <th>o_shippriority</th>
      <th>o_comment</th>
      <th>order_year</th>
      <th>order_quarter</th>
      <th>order_month</th>
      <th>is_large_order</th>
      <th>is_urgent</th>
      <th>revenue_tier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58423751</td>
      <td>649027</td>
      <td>O</td>
      <td>328280.64</td>
      <td>1998-04-22</td>
      <td>2-HIGH</td>
      <td>Clerk#000008681</td>
      <td>0</td>
      <td>lly regular foxes. final, bold requests are da...</td>
      <td>1998</td>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>True</td>
      <td>Enterprise</td>
    </tr>
    <tr>
      <th>1</th>
      <td>58423810</td>
      <td>297619</td>
      <td>F</td>
      <td>330490.15</td>
      <td>1992-02-02</td>
      <td>4-NOT SPECIFIED</td>
      <td>Clerk#000006006</td>
      <td>0</td>
      <td>pending, unusual deposits haggle? carefully r...</td>
      <td>1992</td>
      <td>1</td>
      <td>2</td>
      <td>True</td>
      <td>False</td>
      <td>Enterprise</td>
    </tr>
    <tr>
      <th>2</th>
      <td>58423938</td>
      <td>968440</td>
      <td>F</td>
      <td>309202.74</td>
      <td>1992-05-10</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000006188</td>
      <td>0</td>
      <td>ly special accounts haggle? fluf</td>
      <td>1992</td>
      <td>2</td>
      <td>5</td>
      <td>True</td>
      <td>False</td>
      <td>Enterprise</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58424166</td>
      <td>1269979</td>
      <td>O</td>
      <td>318800.60</td>
      <td>1996-01-04</td>
      <td>5-LOW</td>
      <td>Clerk#000004535</td>
      <td>0</td>
      <td>. blithely final platelets wake blithely!</td>
      <td>1996</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>False</td>
      <td>Enterprise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58424423</td>
      <td>652471</td>
      <td>O</td>
      <td>312001.92</td>
      <td>1997-04-22</td>
      <td>5-LOW</td>
      <td>Clerk#000003204</td>
      <td>0</td>
      <td>ideas thrash never along the furiousl</td>
      <td>1997</td>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>Enterprise</td>
    </tr>
  </tbody>
</table>
</div>




```python
complex_filtered_orders.limit(5).to_pandas()
```

    2025-10-10 20:23:00,970	INFO logging.py:295 -- Registered dataset logger for dataset dataset_347_0
    2025-10-10 20:23:00,979	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_347_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:00,980	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_347_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Filter(<expression>)] -> LimitOperator[limit=5]
    2025-10-10 20:23:03,887	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_347_0 execution finished in 2.91 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>o_orderkey</th>
      <th>o_custkey</th>
      <th>o_orderstatus</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>o_orderpriority</th>
      <th>o_clerk</th>
      <th>o_shippriority</th>
      <th>o_comment</th>
      <th>order_year</th>
      <th>order_quarter</th>
      <th>order_month</th>
      <th>is_large_order</th>
      <th>is_urgent</th>
      <th>revenue_tier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4423715</td>
      <td>1391221</td>
      <td>F</td>
      <td>240515.93</td>
      <td>1992-12-24</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000002392</td>
      <td>0</td>
      <td>s; carefully bold packages solve slyly. specia...</td>
      <td>1992</td>
      <td>4</td>
      <td>12</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4423716</td>
      <td>236656</td>
      <td>F</td>
      <td>182232.10</td>
      <td>1992-11-02</td>
      <td>5-LOW</td>
      <td>Clerk#000006940</td>
      <td>0</td>
      <td>regular pinto beans. regula</td>
      <td>1992</td>
      <td>4</td>
      <td>11</td>
      <td>False</td>
      <td>False</td>
      <td>Large</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4423842</td>
      <td>624805</td>
      <td>F</td>
      <td>212789.42</td>
      <td>1994-12-26</td>
      <td>5-LOW</td>
      <td>Clerk#000009544</td>
      <td>0</td>
      <td>breach furiously. carefully regular patterns a...</td>
      <td>1994</td>
      <td>4</td>
      <td>12</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4423873</td>
      <td>493880</td>
      <td>F</td>
      <td>184870.28</td>
      <td>1992-10-28</td>
      <td>1-URGENT</td>
      <td>Clerk#000001600</td>
      <td>0</td>
      <td>sly unusual accounts play furiously across the...</td>
      <td>1992</td>
      <td>4</td>
      <td>10</td>
      <td>False</td>
      <td>True</td>
      <td>Large</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4423878</td>
      <td>174965</td>
      <td>F</td>
      <td>64043.56</td>
      <td>1994-11-18</td>
      <td>1-URGENT</td>
      <td>Clerk#000008651</td>
      <td>0</td>
      <td>ackages. fluffily ironic r</td>
      <td>1994</td>
      <td>4</td>
      <td>11</td>
      <td>False</td>
      <td>True</td>
      <td>Medium</td>
    </tr>
  </tbody>
</table>
</div>



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

    2025-10-10 20:23:04,007	INFO logging.py:295 -- Registered dataset logger for dataset dataset_349_0
    2025-10-10 20:23:04,031	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_349_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:04,032	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_349_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AggregateNumRows[AggregateNumRows]


    
    Performing distributed joins for customer-order analysis...


    [36m(HashShuffleAggregator pid=78838, ip=10.0.116.84)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:23:15,154	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_349_0 execution finished in 11.12 seconds
    2025-10-10 20:23:15,209	INFO logging.py:295 -- Registered dataset logger for dataset dataset_350_0
    2025-10-10 20:23:15,223	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_350_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:15,224	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_350_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> AggregateNumRows[AggregateNumRows]


    Customer-order join completed: 15,000,000 records


    2025-10-10 20:23:16,205	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_350_0 execution finished in 0.98 seconds
    2025-10-10 20:23:16,214	INFO logging.py:295 -- Registered dataset logger for dataset dataset_351_0
    2025-10-10 20:23:16,223	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_351_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:16,224	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_351_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> AggregateNumRows[AggregateNumRows]
    2025-10-10 20:23:21,066	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_351_0 execution finished in 4.84 seconds


       Input: 1,500,000 customers, 15,000,000 orders
       Join ratio: 100.0% of orders matched



```python
customer_order_metrics.limit(5).to_pandas()
```

    2025-10-10 20:23:21,275	INFO logging.py:295 -- Registered dataset logger for dataset dataset_354_0
    2025-10-10 20:23:21,295	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_354_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:21,296	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_354_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=5]
    [36m(HashShuffleAggregator pid=82352, ip=10.0.99.160)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:23:36,024	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: c_mktsegment: string
    count(): int64
    mean(o_totalprice): double
    sum(o_totalprice): double
    count(o_orderkey): int64, new schema: None. This may lead to unexpected behavior.
    2025-10-10 20:23:36,179	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_354_0 execution finished in 14.88 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c_mktsegment</th>
      <th>count()</th>
      <th>mean(o_totalprice)</th>
      <th>sum(o_totalprice)</th>
      <th>count(o_orderkey)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUTOMOBILE</td>
      <td>3000540</td>
      <td>151096.214697</td>
      <td>4.533702e+11</td>
      <td>3000540</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BUILDING</td>
      <td>3004382</td>
      <td>151053.909702</td>
      <td>4.538236e+11</td>
      <td>3004382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FURNITURE</td>
      <td>3001268</td>
      <td>151022.834817</td>
      <td>4.532600e+11</td>
      <td>3001268</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOUSEHOLD</td>
      <td>2990828</td>
      <td>151207.419625</td>
      <td>4.522354e+11</td>
      <td>2990828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MACHINERY</td>
      <td>3002982</td>
      <td>151052.827336</td>
      <td>4.536089e+11</td>
      <td>3002982</td>
    </tr>
  </tbody>
</table>
</div>



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

    Configuring Ray Data for ETL optimization...
    Ray Data configured for optimal ETL performance


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

    2025-10-10 20:23:36,440	INFO logging.py:295 -- Registered dataset logger for dataset dataset_356_0
    2025-10-10 20:23:36,450	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_356_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:36,451	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_356_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[MapBatches(memory_intensive_etl)] -> AggregateNumRows[AggregateNumRows]


    Testing ETL batch size optimization...


    2025-10-10 20:23:48,070	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_356_0 execution finished in 11.62 seconds


    Memory-optimized processing: 15,000,000 records



```python
memory_optimized_orders.limit(5).to_pandas()
```

    2025-10-10 20:23:48,185	INFO logging.py:295 -- Registered dataset logger for dataset dataset_357_0
    2025-10-10 20:23:48,196	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_357_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:48,197	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_357_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[MapBatches(memory_intensive_etl)] -> LimitOperator[limit=5]
    2025-10-10 20:23:56,723	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_357_0 execution finished in 8.53 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>o_orderkey</th>
      <th>o_custkey</th>
      <th>o_orderstatus</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>o_orderpriority</th>
      <th>o_clerk</th>
      <th>o_shippriority</th>
      <th>o_comment</th>
      <th>order_year</th>
      <th>order_quarter</th>
      <th>order_month</th>
      <th>is_large_order</th>
      <th>is_urgent</th>
      <th>revenue_tier</th>
      <th>complex_metric</th>
      <th>percentile_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28423681</td>
      <td>1067063</td>
      <td>F</td>
      <td>163555.68</td>
      <td>1992-12-14</td>
      <td>2-HIGH</td>
      <td>Clerk#000001449</td>
      <td>0</td>
      <td>es since the quickly final requests haggle</td>
      <td>1992</td>
      <td>4</td>
      <td>12</td>
      <td>False</td>
      <td>True</td>
      <td>Large</td>
      <td>1.963472e+06</td>
      <td>0.570</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28423682</td>
      <td>122650</td>
      <td>F</td>
      <td>208484.81</td>
      <td>1994-06-11</td>
      <td>4-NOT SPECIFIED</td>
      <td>Clerk#000002316</td>
      <td>0</td>
      <td>al instructions. even deposits detect carefull...</td>
      <td>1994</td>
      <td>2</td>
      <td>6</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
      <td>2.553444e+06</td>
      <td>0.712</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28423683</td>
      <td>372205</td>
      <td>O</td>
      <td>113902.82</td>
      <td>1998-07-19</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000001667</td>
      <td>0</td>
      <td>g furiously even de</td>
      <td>1998</td>
      <td>3</td>
      <td>7</td>
      <td>False</td>
      <td>False</td>
      <td>Medium</td>
      <td>1.326183e+06</td>
      <td>0.390</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28423684</td>
      <td>638413</td>
      <td>O</td>
      <td>121829.88</td>
      <td>1996-06-17</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000006284</td>
      <td>0</td>
      <td>ve the blithely ironic deposi</td>
      <td>1996</td>
      <td>2</td>
      <td>6</td>
      <td>False</td>
      <td>False</td>
      <td>Medium</td>
      <td>1.426675e+06</td>
      <td>0.412</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28423685</td>
      <td>305218</td>
      <td>O</td>
      <td>148385.59</td>
      <td>1996-11-09</td>
      <td>4-NOT SPECIFIED</td>
      <td>Clerk#000008442</td>
      <td>0</td>
      <td>ns serve slyly against the b</td>
      <td>1996</td>
      <td>4</td>
      <td>11</td>
      <td>False</td>
      <td>False</td>
      <td>Medium</td>
      <td>1.766913e+06</td>
      <td>0.518</td>
    </tr>
  </tbody>
</table>
</div>




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

    2025-10-10 20:23:56,873	INFO logging.py:295 -- Registered dataset logger for dataset dataset_359_0
    2025-10-10 20:23:56,882	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_359_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:23:56,883	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_359_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[MapBatches(io_intensive_etl)] -> AggregateNumRows[AggregateNumRows]
    2025-10-10 20:24:29,266	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_359_0 execution finished in 32.38 seconds


    I/O-optimized processing: 15,000,000 records
    Batch files written to: /mnt/cluster_storage/temp_etl_batches/



```python
io_optimized_orders.limit(5).to_pandas()
```

    2025-10-10 20:24:29,908	INFO logging.py:295 -- Registered dataset logger for dataset dataset_360_0
    2025-10-10 20:24:29,918	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_360_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:24:29,918	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_360_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[MapBatches(io_intensive_etl)] -> LimitOperator[limit=5]


    [36m(autoscaler +2m36s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.


    2025-10-10 20:24:47,969	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_360_0 execution finished in 18.05 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>o_orderkey</th>
      <th>o_custkey</th>
      <th>o_orderstatus</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>o_orderpriority</th>
      <th>o_clerk</th>
      <th>o_shippriority</th>
      <th>o_comment</th>
      <th>order_year</th>
      <th>order_quarter</th>
      <th>order_month</th>
      <th>is_large_order</th>
      <th>is_urgent</th>
      <th>revenue_tier</th>
      <th>processing_timestamp</th>
      <th>batch_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40423681</td>
      <td>1202450</td>
      <td>O</td>
      <td>24797.87</td>
      <td>1996-10-07</td>
      <td>2-HIGH</td>
      <td>Clerk#000002862</td>
      <td>0</td>
      <td>ly fluffy orbits. unusual, unusual ideas thras...</td>
      <td>1996</td>
      <td>4</td>
      <td>10</td>
      <td>False</td>
      <td>True</td>
      <td>Small</td>
      <td>2025-10-10T20:24:32.461085</td>
      <td>0bf79647</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40423682</td>
      <td>65267</td>
      <td>O</td>
      <td>290473.34</td>
      <td>1998-01-31</td>
      <td>1-URGENT</td>
      <td>Clerk#000007249</td>
      <td>0</td>
      <td>ide of the platelets; slyly silent requests af...</td>
      <td>1998</td>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>True</td>
      <td>Large</td>
      <td>2025-10-10T20:24:32.461085</td>
      <td>0bf79647</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40423683</td>
      <td>441604</td>
      <td>P</td>
      <td>226618.10</td>
      <td>1995-04-17</td>
      <td>4-NOT SPECIFIED</td>
      <td>Clerk#000009686</td>
      <td>0</td>
      <td>cuses about the furiously even account</td>
      <td>1995</td>
      <td>2</td>
      <td>4</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
      <td>2025-10-10T20:24:32.461085</td>
      <td>0bf79647</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40423684</td>
      <td>16730</td>
      <td>F</td>
      <td>245172.80</td>
      <td>1993-06-24</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000009880</td>
      <td>0</td>
      <td>heaves. even requests sleep b</td>
      <td>1993</td>
      <td>2</td>
      <td>6</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
      <td>2025-10-10T20:24:32.461085</td>
      <td>0bf79647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40423685</td>
      <td>677485</td>
      <td>O</td>
      <td>239125.96</td>
      <td>1996-08-04</td>
      <td>3-MEDIUM</td>
      <td>Clerk#000009403</td>
      <td>0</td>
      <td>elieve finally above the regular, final reques...</td>
      <td>1996</td>
      <td>3</td>
      <td>8</td>
      <td>True</td>
      <td>False</td>
      <td>Large</td>
      <td>2025-10-10T20:24:32.461085</td>
      <td>0bf79647</td>
    </tr>
  </tbody>
</table>
</div>



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

    2025-10-10 20:24:48,115	INFO logging.py:295 -- Registered dataset logger for dataset dataset_363_0
    2025-10-10 20:24:48,124	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_363_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:24:48,125	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_363_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)] -> LimitOperator[limit=1] -> TaskPoolMapOperator[Project]


    Applying column selection optimization...
    Column optimization:


    2025-10-10 20:24:49,111	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_363_0 execution finished in 0.99 seconds
    2025-10-10 20:24:49,117	INFO logging.py:295 -- Registered dataset logger for dataset dataset_364_0
    2025-10-10 20:24:49,126	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_364_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:24:49,127	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_364_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> LimitOperator[limit=1] -> TaskPoolMapOperator[Project]


      Customer columns: 5


    2025-10-10 20:24:51,828	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_364_0 execution finished in 2.70 seconds
    2025-10-10 20:24:51,836	INFO logging.py:295 -- Registered dataset logger for dataset dataset_366_0
    2025-10-10 20:24:51,856	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_366_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:24:51,857	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_366_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AggregateNumRows[AggregateNumRows]


      Order columns: 7


    [36m(HashShuffleAggregator pid=91607, ip=10.0.127.67)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:25:01,325	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_366_0 execution finished in 9.47 seconds


    Optimized join completed: 15,000,000 records



```python
optimized_join.limit(5).to_pandas()
```

    2025-10-10 20:25:01,468	INFO logging.py:295 -- Registered dataset logger for dataset dataset_367_0
    2025-10-10 20:25:01,487	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_367_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:25:01,488	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_367_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> LimitOperator[limit=5]
    [36m(HashShuffleAggregator pid=89233, ip=10.0.83.124)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:25:10,851	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_367_0 execution finished in 9.36 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c_custkey</th>
      <th>c_name</th>
      <th>c_mktsegment</th>
      <th>c_acctbal</th>
      <th>c_nationkey</th>
      <th>o_orderkey</th>
      <th>o_totalprice</th>
      <th>o_orderdate</th>
      <th>order_year</th>
      <th>revenue_tier</th>
      <th>is_large_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>841561</td>
      <td>Customer#000841561</td>
      <td>MACHINERY</td>
      <td>1961.63</td>
      <td>16</td>
      <td>10423846</td>
      <td>168237.54</td>
      <td>1994-07-20</td>
      <td>1994</td>
      <td>Large</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>701276</td>
      <td>Customer#000701276</td>
      <td>BUILDING</td>
      <td>5727.18</td>
      <td>13</td>
      <td>10423936</td>
      <td>228113.77</td>
      <td>1995-05-19</td>
      <td>1995</td>
      <td>Large</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>266596</td>
      <td>Customer#000266596</td>
      <td>HOUSEHOLD</td>
      <td>7294.46</td>
      <td>1</td>
      <td>10424358</td>
      <td>193268.14</td>
      <td>1992-05-07</td>
      <td>1992</td>
      <td>Large</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>562309</td>
      <td>Customer#000562309</td>
      <td>AUTOMOBILE</td>
      <td>7688.35</td>
      <td>18</td>
      <td>10424513</td>
      <td>248754.64</td>
      <td>1994-10-05</td>
      <td>1994</td>
      <td>Large</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>887143</td>
      <td>Customer#000887143</td>
      <td>MACHINERY</td>
      <td>7866.95</td>
      <td>5</td>
      <td>10424577</td>
      <td>64566.07</td>
      <td>1993-11-20</td>
      <td>1993</td>
      <td>Medium</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



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

    2025-10-10 20:25:11,092	INFO logging.py:295 -- Registered dataset logger for dataset dataset_370_0
    2025-10-10 20:25:11,112	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_370_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:25:11,113	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_370_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=5]


    Performing large-scale distributed aggregations...
    Comprehensive Business Metrics:
    [33m(raylet)[0m A worker died or was killed while executing a task by an unexpected system error. To troubleshoot the problem, check the logs for the dead worker. RayTask ID: ffffffffffffffff628638057c2f0b7254ad1dc90a000000 Worker ID: ab015acd9569881de1592e4b363f7dfec9124d774b02ca1717232f4b Node ID: 45eb4f9cbe98aac441206a556e4505c3153634633056153efe93760e Worker IP address: 10.0.83.247 Worker port: 10525 Worker PID: 92026 Worker exit type: SYSTEM_ERROR Worker exit detail: Worker exits unexpectedly by a signal. SystemExit is raised (sys.exit is called). Exit code: 1. The process receives a SIGTERM.


    {"asctime":"2025-10-10 20:25:23,000","levelname":"E","message":":info_message: Attempting to recover 1 lost objects by resubmitting their tasks or setting a new primary location from existing copies. To disable object reconstruction, set @ray.remote(max_retries=0).","filename":"core_worker.cc","lineno":445}
    {"asctime":"2025-10-10 20:25:23,800","levelname":"E","message":":info_message: Attempting to recover 1 lost objects by resubmitting their tasks or setting a new primary location from existing copies. To disable object reconstruction, set @ray.remote(max_retries=0).","filename":"core_worker.cc","lineno":445}
    {"asctime":"2025-10-10 20:25:25,601","levelname":"E","message":":info_message: Attempting to recover 1 lost objects by resubmitting their tasks or setting a new primary location from existing copies. To disable object reconstruction, set @ray.remote(max_retries=0).","filename":"core_worker.cc","lineno":445}
    {"asctime":"2025-10-10 20:25:26,001","levelname":"E","message":":info_message: Attempting to recover 1 lost objects by resubmitting their tasks or setting a new primary location from existing copies. To disable object reconstruction, set @ray.remote(max_retries=0).","filename":"core_worker.cc","lineno":445}



    Cannot execute code, session has been disposed. Please try restarting the Kernel.



    The Kernel crashed while executing code in the the current cell or a previous cell. Please review the code in the cell(s) to identify a possible cause of the failure. Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.



```python
# Time-series aggregations for trend analysis
yearly_trends = optimized_join.groupby("order_year").aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice")
)
```

    2025-10-10 18:56:27,516	INFO logging.py:295 -- Registered dataset logger for dataset dataset_207_0
    2025-10-10 18:56:27,534	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_207_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:56:27,535	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_207_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=5]


    
    Yearly Trends Analysis:


    [36m(HashShuffleAggregator pid=43668, ip=10.0.127.67)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 18:56:41,444	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: order_year: int64
    count(): int64
    sum(o_totalprice): double
    mean(o_totalprice): double, new schema: None. This may lead to unexpected behavior.
    2025-10-10 18:56:41,596	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_207_0 execution finished in 14.06 seconds


       order_year  count()  sum(o_totalprice)  mean(o_totalprice)
    0        1992  2281205       3.444725e+11       151004.621657
    1        1993  2276638       3.440619e+11       151127.204812
    2        1994  2275919       3.440890e+11       151186.860472
    3        1995  2275575       3.437713e+11       151070.064800
    4        1996  2281938       3.447880e+11       151094.386277



```python
yearly_trends.limit(5).to_pandas()
```

    2025-10-10 20:18:00,460	INFO logging.py:295 -- Registered dataset logger for dataset dataset_305_0
    2025-10-10 20:18:00,479	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_305_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 20:18:00,480	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_305_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=5]
    [36m(HashShuffleAggregator pid=75410, ip=10.0.83.124)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 20:18:16,923	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: order_year: int64
    count(): int64
    sum(o_totalprice): double
    mean(o_totalprice): double, new schema: None. This may lead to unexpected behavior.
    2025-10-10 20:18:17,078	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_305_0 execution finished in 16.60 seconds





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_year</th>
      <th>count()</th>
      <th>sum(o_totalprice)</th>
      <th>mean(o_totalprice)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1992</td>
      <td>2281205</td>
      <td>3.444725e+11</td>
      <td>151004.621657</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1993</td>
      <td>2276638</td>
      <td>3.440619e+11</td>
      <td>151127.204812</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1994</td>
      <td>2275919</td>
      <td>3.440890e+11</td>
      <td>151186.860472</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1995</td>
      <td>2275575</td>
      <td>3.437713e+11</td>
      <td>151070.064800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1996</td>
      <td>2281938</td>
      <td>3.447880e+11</td>
      <td>151094.386277</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Customer segment performance analysis
segment_performance = optimized_join.groupby(["c_mktsegment", "revenue_tier"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("c_acctbal")
)
```

    2025-10-10 18:56:41,750	INFO logging.py:295 -- Registered dataset logger for dataset dataset_210_0
    2025-10-10 18:56:41,766	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_210_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:56:41,767	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_210_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> LimitOperator[limit=5]


    
    Customer Segment Performance:


    [36m(HashShuffleAggregator pid=45378, ip=10.0.127.67)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 18:56:55,131	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: c_mktsegment: string
    revenue_tier: string
    count(): int64
    sum(o_totalprice): double
    mean(c_acctbal): double, new schema: None. This may lead to unexpected behavior.
    2025-10-10 18:56:55,322	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_210_0 execution finished in 13.55 seconds


      c_mktsegment revenue_tier  count()  sum(o_totalprice)  mean(c_acctbal)
    0   AUTOMOBILE   Enterprise   171395       5.749540e+10      4496.630832
    1   AUTOMOBILE        Large  1264302       2.703461e+11      4497.482494
    2   AUTOMOBILE       Medium  1141136       1.136020e+11      4496.741620
    3   AUTOMOBILE        Small   423707       1.192671e+10      4501.300247
    4     BUILDING   Enterprise   170942       5.733719e+10      4514.232680



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

    2025-10-10 18:56:55,476	INFO logging.py:295 -- Registered dataset logger for dataset dataset_212_0
    2025-10-10 18:56:55,486	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_212_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:56:55,489	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_212_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project] -> TaskPoolMapOperator[MapBatches(segment_customers)] -> TaskPoolMapOperator[Write]


    Writing ETL results to data warehouse...


    2025-10-10 18:56:57,295	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_212_0 execution finished in 1.81 seconds
    2025-10-10 18:56:57,346	INFO dataset.py:4871 -- Data sink Parquet finished. 1500000 rows and 665.0MB data written.
    2025-10-10 18:56:57,352	INFO logging.py:295 -- Registered dataset logger for dataset dataset_215_0
    2025-10-10 18:56:57,363	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_215_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:56:57,364	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_215_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Write]
    2025-10-10 18:57:04,817	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_215_0 execution finished in 7.45 seconds
    2025-10-10 18:57:04,850	INFO dataset.py:4871 -- Data sink Parquet finished. 15000000 rows and 5.3GB data written.
    2025-10-10 18:57:04,858	INFO logging.py:295 -- Registered dataset logger for dataset dataset_220_0
    2025-10-10 18:57:04,874	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_220_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:57:04,875	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_220_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(drop_columns)->Project], InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[MapBatches(enrich_orders_with_metrics)] -> TaskPoolMapOperator[Project] -> JoinOperatorWithPolars[Join(num_partitions=100)] -> AllToAllOperator[Aggregate] -> TaskPoolMapOperator[Write]
    [36m(HashShuffleAggregator pid=44991, ip=10.0.99.160)[0m Failed to hash the schemas (for deduplication): unhashable type: 'dict'
    2025-10-10 18:57:32,014	WARNING streaming_executor_state.py:793 -- Operator produced a RefBundle with a different schema than the previous one. Previous schema: c_mktsegment: string
    revenue_tier: string
    order_year: int64
    count(): int64
    sum(o_totalprice): double
    mean(o_totalprice): double
    mean(c_acctbal): double, new schema: None. This may lead to unexpected behavior.
    2025-10-10 18:57:32,336	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_220_0 execution finished in 27.46 seconds
    2025-10-10 18:57:32,425	INFO dataset.py:4871 -- Data sink Parquet finished. 140 rows and 8.7KB data written.


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

    2025-10-10 18:57:32,543	INFO logging.py:295 -- Registered dataset logger for dataset dataset_225_0
    2025-10-10 18:57:32,546	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_225_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:57:32,547	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_225_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[MapBatches(count_rows)]


    Validating ETL output...
    ETL Pipeline Verification:


    2025-10-10 18:57:32,751	INFO streaming_executor.py:264 -- Operator 2 MapBatches(count_rows): 1 tasks executed, 1 blocks produced in 0.14s
    * Remote wall time: 138.37ms min, 138.37ms max, 138.37ms mean, 138.37ms total
    * Remote cpu time: 72.1ms min, 72.1ms max, 72.1ms mean, 72.1ms total
    * UDF time: 136.67ms min, 136.67ms max, 136.67ms mean, 136.67ms total
    * Peak heap memory usage (MiB): 848.18 min, 848.18 max, 848 mean
    * Output num rows per block: 5 min, 5 max, 5 mean, 5 total
    * Output size bytes per block: 40 min, 40 max, 40 mean, 40 total
    * Output rows per task: 5 min, 5 max, 5 mean, 1 tasks used
    * Tasks per node: 1 min, 1 max, 1 mean; 1 nodes used
    * Operator throughput:
    	* Ray Data throughput: 36.13621943996574 rows/s
    	* Estimated single node throughput: 36.13621943996574 rows/s
    
    Dataset throughput:
    	* Ray Data throughput: 5.706518275526538 rows/s
    	* Estimated single node throughput: 31.402439216023936 rows/s
    
    2025-10-10 18:57:32,752	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_225_0 execution finished in 0.20 seconds
    2025-10-10 18:57:32,759	INFO logging.py:295 -- Registered dataset logger for dataset dataset_226_0
    2025-10-10 18:57:32,763	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_226_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:57:32,763	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_226_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[MapBatches(count_rows)]


      Customer records: 6,000,000


    2025-10-10 18:57:34,786	INFO streaming_executor.py:264 -- Operator 2 MapBatches(count_rows): 2 tasks executed, 2 blocks produced in 1.73s
    * Remote wall time: 226.21ms min, 1.73s max, 978.67ms mean, 1.96s total
    * Remote cpu time: 135.91ms min, 1.1s max, 618.5ms mean, 1.24s total
    * UDF time: 224.13ms min, 1.72s max, 970.72ms mean, 1.94s total
    * Peak heap memory usage (MiB): 538.77 min, 1204.13 max, 871 mean
    * Output num rows per block: 8 min, 63 max, 35 mean, 71 total
    * Output size bytes per block: 64 min, 504 max, 284 mean, 568 total
    * Output rows per task: 8 min, 63 max, 35 mean, 2 tasks used
    * Tasks per node: 1 min, 1 max, 1 mean; 2 nodes used
    * Operator throughput:
    	* Ray Data throughput: 41.013836292613426 rows/s
    	* Estimated single node throughput: 36.27380445871822 rows/s
    
    Dataset throughput:
    	* Ray Data throughput: 41.013836292613426 rows/s
    	* Estimated single node throughput: 32.23611590996947 rows/s
    
    2025-10-10 18:57:34,787	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_226_0 execution finished in 2.02 seconds
    2025-10-10 18:57:34,795	INFO logging.py:295 -- Registered dataset logger for dataset dataset_227_0
    2025-10-10 18:57:34,799	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_227_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:57:34,799	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_227_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[MapBatches(count_rows)]


      Order records: 60,000,000


    2025-10-10 18:57:35,316	INFO streaming_executor.py:264 -- Operator 2 MapBatches(count_rows): 1 tasks executed, 1 blocks produced in 0.39s
    * Remote wall time: 391.92ms min, 391.92ms max, 391.92ms mean, 391.92ms total
    * Remote cpu time: 240.02ms min, 240.02ms max, 240.02ms mean, 240.02ms total
    * UDF time: 386.1ms min, 386.1ms max, 386.1ms mean, 386.1ms total
    * Peak heap memory usage (MiB): 1116.09 min, 1116.09 max, 1116 mean
    * Output num rows per block: 27 min, 27 max, 27 mean, 27 total
    * Output size bytes per block: 216 min, 216 max, 216 mean, 216 total
    * Output rows per task: 27 min, 27 max, 27 mean, 1 tasks used
    * Tasks per node: 1 min, 1 max, 1 mean; 1 nodes used
    * Operator throughput:
    	* Ray Data throughput: 68.89138940244673 rows/s
    	* Estimated single node throughput: 68.89138940244673 rows/s
    
    Dataset throughput:
    	* Ray Data throughput: 42.26306318732737 rows/s
    	* Estimated single node throughput: 57.932190036850805 rows/s
    
    2025-10-10 18:57:35,317	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_227_0 execution finished in 0.52 seconds
    2025-10-10 18:57:35,324	INFO logging.py:295 -- Registered dataset logger for dataset dataset_228_0
    2025-10-10 18:57:35,330	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_228_0. Full logs are in /tmp/ray/session_2025-10-10_16-23-49_015346_2333/logs/ray-data
    2025-10-10 18:57:35,330	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_228_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[ReadFiles] -> LimitOperator[limit=25]


      Analytics records: 420


    2025-10-10 18:57:37,106	INFO streaming_executor.py:264 -- Operator 3 limit=25: 1 tasks executed, 1 blocks produced in 0.09s
    * Remote wall time: 89.4ms min, 89.4ms max, 89.4ms mean, 89.4ms total
    * Remote cpu time: 186.83ms min, 186.83ms max, 186.83ms mean, 186.83ms total
    * UDF time: 0us min, 0us max, 0.0us mean, 0us total
    * Peak heap memory usage (MiB): 856.52 min, 856.52 max, 856 mean
    * Output num rows per block: 25 min, 25 max, 25 mean, 25 total
    * Output size bytes per block: 1594 min, 1594 max, 1594 mean, 1594 total
    * Output rows per task: 25 min, 25 max, 25 mean, 1 tasks used
    * Tasks per node: 1 min, 1 max, 1 mean; 1 nodes used
    * Operator throughput:
    	* Ray Data throughput: 279.62736677939245 rows/s
    	* Estimated single node throughput: 279.62736677939245 rows/s
    
    Dataset throughput:
    	* Ray Data throughput: 13.58305853827714 rows/s
    	* Estimated single node throughput: 46.42515826415853 rows/s
    
    2025-10-10 18:57:37,107	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_228_0 execution finished in 1.78 seconds


    
    Sample ETL Analytics Results:
      1. Segment: AUTOMOBILE, Tier: Enterprise, Year: 1992, Orders: 25919, Revenue: $8,691,666,789
      2. Segment: AUTOMOBILE, Tier: Large, Year: 1992, Orders: 191421, Revenue: $40,936,308,218
      3. Segment: AUTOMOBILE, Tier: Medium, Year: 1992, Orders: 173366, Revenue: $17,240,817,851
      4. Segment: AUTOMOBILE, Tier: Small, Year: 1992, Orders: 64755, Revenue: $1,819,743,455
      5. Segment: BUILDING, Tier: Enterprise, Year: 1992, Orders: 25978, Revenue: $8,716,921,785
      6. Segment: BUILDING, Tier: Large, Year: 1992, Orders: 192720, Revenue: $41,234,190,203
      7. Segment: BUILDING, Tier: Medium, Year: 1992, Orders: 174342, Revenue: $17,355,772,628
      8. Segment: BUILDING, Tier: Small, Year: 1992, Orders: 64775, Revenue: $1,828,037,736
      9. Segment: FURNITURE, Tier: Enterprise, Year: 1992, Orders: 25807, Revenue: $8,650,677,599
      10. Segment: FURNITURE, Tier: Large, Year: 1992, Orders: 192641, Revenue: $41,209,906,769
      11. Segment: FURNITURE, Tier: Medium, Year: 1992, Orders: 173672, Revenue: $17,294,243,789
      12. Segment: FURNITURE, Tier: Small, Year: 1992, Orders: 64385, Revenue: $1,806,817,337
      13. Segment: HOUSEHOLD, Tier: Enterprise, Year: 1992, Orders: 26105, Revenue: $8,753,166,132
      14. Segment: HOUSEHOLD, Tier: Large, Year: 1992, Orders: 191579, Revenue: $40,959,940,953
      15. Segment: HOUSEHOLD, Tier: Medium, Year: 1992, Orders: 173111, Revenue: $17,211,742,048
      16. Segment: HOUSEHOLD, Tier: Small, Year: 1992, Orders: 63916, Revenue: $1,800,192,999
      17. Segment: MACHINERY, Tier: Enterprise, Year: 1992, Orders: 26091, Revenue: $8,751,905,563
      18. Segment: MACHINERY, Tier: Large, Year: 1992, Orders: 192131, Revenue: $41,070,411,989
      19. Segment: MACHINERY, Tier: Medium, Year: 1992, Orders: 173966, Revenue: $17,325,248,289
      20. Segment: MACHINERY, Tier: Small, Year: 1992, Orders: 64525, Revenue: $1,814,785,814
      21. Segment: AUTOMOBILE, Tier: Enterprise, Year: 1992, Orders: 25919, Revenue: $8,691,666,789
      22. Segment: AUTOMOBILE, Tier: Large, Year: 1992, Orders: 191421, Revenue: $40,936,308,218
      23. Segment: AUTOMOBILE, Tier: Medium, Year: 1992, Orders: 173366, Revenue: $17,240,817,851
      24. Segment: AUTOMOBILE, Tier: Small, Year: 1992, Orders: 64755, Revenue: $1,819,743,455
      25. Segment: BUILDING, Tier: Enterprise, Year: 1992, Orders: 25978, Revenue: $8,716,921,785


    [33m(raylet)[0m WARNING: 32 PYTHON worker processes have been started on node: 649a75ab4f03d0f79d9397c9d249bc0977d766b0eb9085dfc81526c2 with address: 10.0.127.67. This could be a result of using a large number of actors, or due to tasks blocked in ray.get() calls (see https://github.com/ray-project/ray/issues/3644 for some discussion of workarounds).



