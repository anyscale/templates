# ETL Processing and Optimization with Ray Data

**⏱️ Time to complete**: 40 min | **Difficulty**: Intermediate | **Prerequisites**: Understanding of ETL concepts, basic SQL knowledge, data processing experience

## What You'll Build

Build comprehensive ETL pipelines using Ray Data's distributed processing capabilities, from foundational concepts with TPC-H benchmark to production-scale optimization techniques for enterprise data processing.

## Table of Contents

1. [ETL Fundamentals with TPC-H](#step-1-etl-fundamentals-with-tpc-h) (10 min)
2. [Data Transformations and Processing](#step-2-data-transformations-and-processing) (12 min)
3. [Performance Optimization Techniques](#step-3-performance-optimization-techniques) (10 min)
4. [Large-Scale ETL Patterns](#step-4-large-scale-etl-patterns) (8 min)

## Learning objectives

**Why ETL optimization matters**: The difference between fast and slow data pipelines directly impacts business agility and operational costs. Understanding optimization techniques enables data teams to deliver insights faster while reducing infrastructure costs.

**Ray Data's ETL capabilities**: Native operations for distributed processing that automatically optimize memory, CPU, and I/O utilization. You'll learn how Ray Data's architecture enables efficient processing of large datasets.

**TPC-H benchmark patterns**: Learn ETL fundamentals using the TPC-H benchmark that simulates complex business environments with customers, orders, suppliers, and products.

**Production optimization strategies**: Memory management, parallel processing, and resource configuration patterns for production ETL workloads that scale from gigabytes to petabytes.

**Enterprise ETL patterns**: Techniques used by data engineering teams to process large datasets efficiently while maintaining data quality and performance.

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of ETL (Extract, Transform, Load) concepts
- [ ] Basic SQL knowledge for data transformations
- [ ] Python experience with data processing
- [ ] Familiarity with distributed computing concepts

## Quick start (3 minutes)

This section demonstrates ETL processing concepts using Ray Data:

```python
from typing import Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data.expressions import col, lit

# Initialize Ray for distributed ETL processing
ray.init()

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

# Load sample dataset for ETL demonstrationsample_data = ray.data.read_parquet(
    "s3://ray-benchmark-data/tpch/parquet/sf1/customer",
    num_cpus=0.025  # High I/O concurrency
)

print(f"Loaded ETL sample dataset: {sample_data.count()} records")
print(f"Schema: {sample_data.schema()}")
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
|----------|-------------|-----------------|-------|---------|----------|
| **Traditional** | Sequential | Single-threaded | Memory-limited | Slow writes | Small datasets |
| **Ray Data** | Parallel I/O | Distributed | Scalable | Optimized writes | Production scale |

**Key advantages**:
- **Parallel processing**: Distribute transformations across cluster nodes
- **Memory efficiency**: Stream processing without materializing full datasets
- **Native operations**: Optimized filter, join, and aggregate functions
- **Scalability**: Handle datasets from gigabytes to petabytes

---

## Step 1: ETL Fundamentals with TPC-H

### Understanding TPC-H Benchmark

```python
from typing import Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from ray.data.expressions import col, lit
from ray.data.aggregate import Count, Mean, Sum, Max

# Initialize Ray for ETL processingray.init(ignore_reinit_error=True)

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

print(f"Ray version: {ray.__version__}")
print(f"Ray cluster resources: {ray.cluster_resources()}")
```

**What is TPC-H?**

The TPC-H benchmark is used for testing database and data processing performance. It simulates a business environment with data relationships that represent business scenarios.

**TPC-H Business Context**: The benchmark models a wholesale supplier managing customer orders, inventory, and supplier relationships - representing business data systems.

### TPC-H Schema Overview

The TPC-H benchmark provides realistic business data for learning ETL patterns. Understanding the schema helps you apply these techniques to your own data.

| Table | Description | Typical Size (SF10) | Primary Use |
|-------|-------------|-------------------|-------------|
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

### Loading TPC-H Data with Ray Data

```python
# Tpc-H benchmark data locationTPCH_S3_PATH = "s3://ray-benchmark-data/tpch/parquet/sf10"

print("Loading TPC-H benchmark data for distributed processing...")
start_time = time.time()

# Read TPC-H Customer Master Datacustomers_ds = ray.data.read_parquet(
    f"{TPCH_S3_PATH}/customer",
    num_cpus=0.025  # High I/O concurrency for reading
)

# Read TPC-H Orders Dataorders_ds = ray.data.read_parquet(
    f"{TPCH_S3_PATH}/orders", 
    num_cpus=0.025
)

# Read TPC-H Line Items (largest table)lineitems_ds = ray.data.read_parquet(
    f"{TPCH_S3_PATH}/lineitem",
    num_cpus=0.025
)

load_time = time.time() - start_time
print(f"TPC-H data loaded in {load_time:.2f} seconds")
print(f"  Customers: {customers_ds.count():,}")
print(f"  Orders: {orders_ds.count():,}")
print(f"  Line items: {lineitems_ds.count():,}")
```

### Basic ETL Transformations

```python
# Etl Transform: Customer segmentation using Ray Data native operationsdef segment_customers(batch):
    """Apply business rules for customer segmentation."""
    import pandas as pd
    df = pd.DataFrame(batch)
    
    # Business logic for customer segmentation

    # Data transformation    df['customer_segment'] = 'standard'
    df.loc[df['c_acctbal'] > 5000, 'customer_segment'] = 'premium'
    df.loc[df['c_acctbal'] > 10000, 'customer_segment'] = 'enterprise'
    
    return df.to_dict('records')

# Apply customer segmentation transformationsegmented_customers = customers_ds.map_batches(
    segment_customers,
    num_cpus=0.5,  # Medium complexity transformation
    batch_format="pandas"
)

print(f"Customer segmentation completed: {segmented_customers.count():,} customers segmented")

# Etl Filter: High-value customers using expressions APIhigh_value_customers = segmented_customers.filter(
    col("c_acctbal") > lit(1000),
    num_cpus=0.1
)

print(f"High-value customers: {high_value_customers.count():,}")

# Etl Aggregation: Customer statistics by market segmentcustomer_stats = segmented_customers.groupby("c_mktsegment").aggregate(
    Count(),
    Mean("c_acctbal"),
    Sum("c_acctbal"),
    Max("c_acctbal")
)

print("Customer Statistics by Market Segment:")
# Display customer statistics
print("Customer Statistics by Market Segment:")
print(customer_stats.limit(10).to_pandas())
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
|-------------------|---------------------|-------------------|-------------|
| **Column calculations** | Row-by-row processing | Vectorized batches | Linear scaling |
| **Date parsing** | Sequential parsing | Parallel batch parsing | High throughput |
| **Categorization** | Conditional logic loops | Pandas vectorization | Efficient |
| **Business rules** | Single-threaded | Distributed map_batches | Scales to cluster |

### Complex Data Transformations

:::tip GPU Acceleration for Pandas ETL Operations
For complex pandas transformations in your ETL pipeline, you can use **NVIDIA RAPIDS cuDF** to accelerate DataFrame operations on GPUs. Simply replace `import pandas as pd` with `import cudf as pd` in your `map_batches` functions to leverage GPU acceleration for operations like datetime parsing, groupby, joins, and aggregations.

**When to use cuDF**:
- Complex datetime operations (parsing, extracting components)
- Large aggregations and groupby operations
- String operations on millions of rows
- Join operations on large datasets
- Statistical calculations across many columns

**Performance benefit**: GPU-accelerated pandas operations can be 10-50x faster for large batches (1000+ rows) with complex transformations.

**Requirements**: Add `cudf` to your dependencies and ensure GPU-enabled cluster nodes.
:::

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

    # Data transformation    df['o_orderdate'] = pd.to_datetime(df['o_orderdate'])

    # Data transformation    df['order_year'] = df['o_orderdate'].dt.year

    # Data transformation    df['order_quarter'] = df['o_orderdate'].dt.quarter

    # Data transformation    df['order_month'] = df['o_orderdate'].dt.month
    
    # Business classifications
    # These conditional operations are GPU-accelerated with cuDF

    # Data transformation    df['is_large_order'] = df['o_totalprice'] > 200000

    # Data transformation    df['is_urgent'] = df['o_orderpriority'].isin(['1-URGENT', '2-HIGH'])

    # Data transformation    df['revenue_tier'] = pd.cut(
        df['o_totalprice'],
        bins=[0, 50000, 150000, 300000, float('inf')],
        labels=['Small', 'Medium', 'Large', 'Enterprise']
    )
    
    return df.to_dict('records')

# Apply order enrichmentenriched_orders = orders_ds.map_batches(
    enrich_orders_with_metrics,
    num_cpus=0.5,  # Medium complexity transformation
    batch_format="pandas"
)

print(f"Order enrichment completed: {enriched_orders.count():,} orders processed")
```

### Advanced Filtering and Selection

```python
# Advanced filtering using Ray Data expressions APIprint("Applying advanced filtering techniques...")

# Filter recent high-value ordersrecent_high_value_orders = enriched_orders.filter(
    (col("order_year") >= lit(1995)) & 
    (col("o_totalprice") > lit(100000)) &
    (col("is_urgent") == lit(True)),
    num_cpus=0.1
)

# Filter by revenue tier using expressionsenterprise_orders = enriched_orders.filter(
    col("revenue_tier") == lit("Enterprise"),
    num_cpus=0.1
)

# Complex filtering with multiple conditionscomplex_filtered_orders = enriched_orders.filter(
    (col("order_quarter") == lit(4)) &
    (col("o_orderstatus") == lit("F")) &
    (col("o_totalprice") > lit(50000)),
    num_cpus=0.1
)

print(f"Advanced filtering results:")
print(f"  Recent high-value orders: {recent_high_value_orders.count():,}")
print(f"  Enterprise orders: {enterprise_orders.count():,}")
print(f"  Complex filtered orders: {complex_filtered_orders.count():,}")
```

### Data Joins and Relationships

```python
# Etl Join: Customer-Order analysis using Ray Data joinsprint("Performing distributed joins for customer-order analysis...")

# Join customers with their orders for comprehensive analysiscustomer_order_analysis = customers_ds.join(
    enriched_orders,
    left_key="c_custkey",
    right_key="o_custkey",
    join_type="inner"
)

print(f"Customer-order join completed: {customer_order_analysis.count():,} records")

# Aggregate customer order metricscustomer_order_metrics = customer_order_analysis.groupby("c_mktsegment").aggregate(
    Count(),
    Mean("o_totalprice"),
    Sum("o_totalprice"),
    Count("o_orderkey")
)

print("Customer Order Metrics by Market Segment:")
# Display customer order metrics
print("Customer Order Metrics by Market Segment:")
print(customer_order_metrics.limit(10).to_pandas())
```

## Step 3: Performance Optimization Techniques

This section covers advanced optimization techniques for production ETL workloads. You'll learn how to tune memory usage, batch sizes, and resource allocation for optimal performance.

### ETL optimization decision framework

Understanding when and how to optimize is crucial for production ETL systems. Follow this systematic approach:

```
ETL Performance Issue
├── Is data loading slow?
│   └── Solution: Increase I/O concurrency (num_cpus=0.025)
│       Column pruning, file format optimization
│
├── Are transformations slow?
│   └── Solution: Balance num_cpus based on complexity
│       Light transforms: num_cpus=0.25-0.5
│       Heavy transforms: num_cpus=1.0-2.0
│
├── Are joins slow?
│   └── Solution: Column selection before joins
│       Filter data early, use expressions API
│
├── Is output slow?
│   └── Solution: Optimize write concurrency (num_cpus=0.1)
│       Compression, partitioning strategy
│
└── Memory issues?
    └── Solution: Adjust batch_size and block_size
        Monitor object store usage
```

### Resource allocation guidelines for ETL

| ETL Stage | Operation Type | num_cpus Recommendation | Batch Size | Reasoning |
|-----------|---------------|------------------------|------------|-----------|
| **Extract** | I/O-bound | 0.025 | Default | Maximum parallel reads |
| **Light Transform** | CPU-light | 0.25-0.5 | Default | Balanced parallelism |
| **Heavy Transform** | CPU-intensive | 1.0-2.0 | 500-1000 | Reduce task overhead |
| **Filter** | Simple logic | 0.1 | Default | Fast filtering |
| **Join** | Shuffle operation | Default | Default | Let Ray optimize |
| **Aggregate** | Reduction | Default | Default | Ray Data optimized |
| **Load** | I/O-bound | 0.1 | Default | Balanced writes |

### Memory and Resource Optimization

```python
# Configure Ray Data for optimal ETL performanceprint("Configuring Ray Data for ETL optimization...")

# Memory optimization for large datasetsctx.target_max_block_size = 128 * 1024 * 1024  # 128 MB blocks
ctx.eager_free = True  # Aggressive memory cleanup

# Enable performance monitoringctx.enable_auto_log_stats = True
ctx.memory_usage_poll_interval_s = 5.0

print("Ray Data configured for optimal ETL performance")
```

### Batch Size and Concurrency Optimization

```python
# Demonstrate different batch size strategies for ETL operationsprint("Testing ETL batch size optimization...")

# Small batch processing for memory-constrained operationsdef memory_intensive_etl(batch):
    """Memory-intensive ETL transformation."""
    import pandas as pd
    df = pd.DataFrame(batch)
    
    # Simulate memory-intensive operations

    # Data transformation    df['complex_metric'] = df['o_totalprice'] * np.log(df['o_totalprice'] + 1)

    # Data transformation    df['percentile_rank'] = df['o_totalprice'].rank(pct=True)
    
    return df.to_dict('records')

# Apply with optimized batch size for memory managementmemory_optimized_orders = enriched_orders.map_batches(
    memory_intensive_etl,
    num_cpus=1.0,  # Fewer concurrent tasks for memory management
    batch_size=500,  # Smaller batches for memory efficiency
    batch_format="pandas"
)

print(f"Memory-optimized processing: {memory_optimized_orders.count():,} records")

# Large batch processing for I/O-intensive operationsdef io_intensive_etl(batch):
    """I/O-intensive ETL transformation."""
    # Simulate I/O operations
    processed_records = []
    for record in batch:
        processed_record = {
            **record,
            'processing_timestamp': datetime.now().isoformat(),
            'batch_id': str(uuid.uuid4())[:8]
        }
        processed_records.append(processed_record)
    
    return processed_records

# Apply with optimized batch size for I/O efficiencyio_optimized_orders = enriched_orders.map_batches(
    io_intensive_etl,
    num_cpus=0.25,  # Higher concurrency for I/O operations
    batch_size=2000  # Larger batches for I/O efficiency
, batch_format="pandas")

print(f"I/O-optimized processing: {io_optimized_orders.count():,} records")
```

### Column Selection and Schema Optimization

```python
# Etl Optimization: Column pruning for performanceprint("Applying column selection optimization...")

# Select only essential columns for downstream processingessential_customer_columns = customers_ds.select_columns([
    "c_custkey", "c_name", "c_mktsegment", "c_acctbal", "c_nationkey"
])

essential_order_columns = enriched_orders.select_columns([
    "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate", 
    "order_year", "revenue_tier", "is_large_order"
])

print(f"Column optimization:")
print(f"  Customer columns: {len(essential_customer_columns.schema().names)}")
print(f"  Order columns: {len(essential_order_columns.schema().names)}")

# Optimized join with selected columnsoptimized_join = essential_customer_columns.join(
    essential_order_columns,
    left_key="c_custkey",
    right_key="o_custkey"
)

print(f"Optimized join completed: {optimized_join.count():,} records")
```

## Step 4: Large-Scale ETL Patterns

Production ETL systems must handle billions of records efficiently. This section demonstrates Ray Data patterns for large-scale data processing including distributed aggregations, multi-dimensional analysis, and data warehouse integration.

### Why scale matters in ETL

As data volumes grow, ETL approaches must evolve:

| Data Scale | Traditional ETL Challenge | Ray Data Solution |
|------------|--------------------------|-------------------|
| **< 100 GB** | Single machine sufficient | Ray Data still faster with parallelism |
| **100 GB - 1 TB** | Memory constraints appear | Streaming execution prevents OOMs |
| **1 TB - 10 TB** | Processing takes hours/days | Distributed processing reduces time |
| **> 10 TB** | May not complete | Scales horizontally across cluster |

**Scaling dimensions**:
- **Data volume**: From gigabytes to petabytes
- **Cluster size**: From single node to hundreds of nodes
- **Complexity**: From simple transforms to complex business logic
- **Concurrency**: From sequential to massively parallel

### Distributed Aggregations

```python
# Large-scale aggregations using Ray Data native operationsprint("Performing large-scale distributed aggregations...")

# Multi-dimensional aggregations for business intelligencecomprehensive_metrics = optimized_join.groupby(["c_mktsegment", "order_year", "revenue_tier"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice"),
    Max("o_totalprice"),
    Mean("c_acctbal")
)

print("Comprehensive Business Metrics:")
# Display comprehensive business metrics
print("Comprehensive Business Metrics:")
print(comprehensive_metrics.limit(20).to_pandas())

# Time-series aggregations for trend analysisyearly_trends = optimized_join.groupby("order_year").aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice")
)

print("Yearly Trends Analysis:")
# Display yearly trends
print("Yearly Trends Analysis:")
print(yearly_trends.limit(10).to_pandas())

# Customer segment performance analysissegment_performance = optimized_join.groupby(["c_mktsegment", "revenue_tier"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("c_acctbal")
)

print("Customer Segment Performance:")
# Display segment performance
print("Customer Segment Performance:")
print(segment_performance.limit(10).to_pandas())
```

### ETL Pipeline Optimization

```python
# Demonstrate optimized ETL pipeline patternsprint("Building optimized ETL pipeline...")

def create_optimized_etl_pipeline():
    """Create optimized ETL pipeline with Ray Data best practices."""
    
    # Extract: Optimized data loading with column selection
    customers = ray.data.read_parquet(
        f"{TPCH_S3_PATH}/customer",
        columns=["c_custkey", "c_name", "c_mktsegment", "c_acctbal", "c_nationkey"],
        num_cpus=0.025  # High I/O concurrency
    )
    
    orders = ray.data.read_parquet(
        f"{TPCH_S3_PATH}/orders",
        columns=["o_orderkey", "o_custkey", "o_totalprice", "o_orderdate", "o_orderstatus"],
        num_cpus=0.025
    )
    
    # Transform: Apply business logic with optimized batch processing
    enriched_customers = customers.map_batches(
        lambda batch: [
            {
                **record,
                "customer_tier": "premium" if record["c_acctbal"] > 5000 else "standard",
                "market_priority": "high" if record["c_mktsegment"] in ["BUILDING", "AUTOMOBILE"] else "medium"
            }
            for record in batch
        ],
        num_cpus=0.5,  # Medium complexity transformation
        batch_size=1000
    , batch_format="pandas")
    
    # Transform: Order processing with time dimensions
    processed_orders = orders.map_batches(
        lambda batch: [
            {
                **record,
                "order_year": int(record["o_orderdate"][:4], batch_format="pandas"),
                "order_size": "large" if record["o_totalprice"] > 200000 else "small",
                "processing_timestamp": datetime.now().isoformat()
            }
            for record in batch
        ],
        num_cpus=0.5,
        batch_size=1000
    )
    
    # Load: Join and aggregate for analytics output
    final_analytics = enriched_customers.join(
        processed_orders,
        left_key="c_custkey",
        right_key="o_custkey"
    ).groupby(["customer_tier", "order_year"]).aggregate(
        Count(),
        Sum("o_totalprice"),
        Mean("c_acctbal")
    )
    
    return final_analytics

# Execute optimized ETL pipelineoptimized_results = create_optimized_etl_pipeline()
print("Optimized ETL Pipeline Results:")
# Display optimized pipeline results
print("Optimized ETL Pipeline Results:")
print(optimized_results.limit(10).to_pandas())
```

### Large-Scale Data Processing

```python
# Process large datasets with optimization techniquesprint("Demonstrating large-scale data processing...")

# Load larger TPC-H scale factor for performance testinglarge_orders = ray.data.read_parquet(
    f"{TPCH_S3_PATH}/lineitem",  # Largest TPC-H table
    columns=["l_orderkey", "l_partkey", "l_quantity", "l_extendedprice", "l_discount"],
    num_cpus=0.025  # High concurrency for large dataset
)

print(f"Large dataset loaded: {large_orders.count():,} line items")

# Apply distributed transformations for large-scale processingdef calculate_line_metrics(batch):
    """Calculate line item business metrics."""
    import pandas as pd
    df = pd.DataFrame(batch)
    
    # Business calculations
    df['discounted_price'] = df['l_extendedprice'] * (1 - df['l_discount'])
    df['revenue_impact'] = df['discounted_price'] * df['l_quantity']
    df['volume_category'] = pd.cut(
        df['l_quantity'],
        bins=[0, 10, 50, 100, float('inf')],
        labels=['Low', 'Medium', 'High', 'Bulk']
    )
    
    return df.to_dict('records')

# Process large dataset with optimized settingsprocessed_lineitems = large_orders.map_batches(
    calculate_line_metrics,
    num_cpus=0.5,  # Balanced processing
    batch_size=1000,
    batch_format="pandas"
)

# Large-scale aggregationsrevenue_analysis = processed_lineitems.groupby("volume_category").aggregate(
    Count(),
    Sum("revenue_impact"),
    Mean("discounted_price")
)

print("Large-Scale Revenue Analysis:")
# Display revenue analysis results
print("Large-Scale Revenue Analysis:")
print(revenue_analysis.limit(10).to_pandas())
```

### ETL Output and Data Warehouse Integration

```python
# Write ETL results to data warehouse formatsprint("Writing ETL results to data warehouse...")

# Write customer analytics with partitioningenriched_customers.write_parquet(
    "/tmp/etl_warehouse/customers/",
    partition_cols=["customer_tier"],
    compression="snappy",
    num_cpus=0.1
)

# Write order analytics with time-based partitioningprocessed_orders.write_parquet(
    "/tmp/etl_warehouse/orders/",
    partition_cols=["order_year"],
    compression="snappy",
    num_cpus=0.1
)

# Write aggregated analytics for BI toolsfinal_analytics = optimized_join.groupby(["c_mktsegment", "revenue_tier", "order_year"]).aggregate(
    Count(),
    Sum("o_totalprice"),
    Mean("o_totalprice"),
    Mean("c_acctbal")
)

final_analytics.write_parquet(
    "/tmp/etl_warehouse/analytics/",
    partition_cols=["order_year"],
    compression="snappy",
    num_cpus=0.1
)

print("ETL warehouse output completed")
```

### Performance Monitoring and Validation

```python
# Validate ETL pipeline performanceprint("Validating ETL pipeline performance...")

# Read back and verify outputscustomer_verification = ray.data.read_parquet(
    "/tmp/etl_warehouse/customers/",
    num_cpus=0.025
)

order_verification = ray.data.read_parquet(
    "/tmp/etl_warehouse/orders/",
    num_cpus=0.025
)

analytics_verification = ray.data.read_parquet(
    "/tmp/etl_warehouse/analytics/",
    num_cpus=0.025
)

print(f"ETL Pipeline Verification:")
print(f"  Customer records: {customer_verification.count():,}")
print(f"  Order records: {order_verification.count():,}")
print(f"  Analytics records: {analytics_verification.count():,}")

# Display sample resultssample_analytics = analytics_verification.take(5)
print("Sample ETL Analytics Results:")
for i, record in enumerate(sample_analytics):
    print(f"  {i+1}. Segment: {record['c_mktsegment']}, Tier: {record['revenue_tier']}, "
          f"Year: {record['order_year']}, Orders: {record['count()']}, Revenue: ${record['sum(o_totalprice)']:,.0f}")
```

## ETL pipeline execution results

### Processing metrics

This ETL pipeline processed TPC-H benchmark data demonstrating production-scale capabilities:

| Dataset | Records Processed | Ray Data Operation | Purpose |
|---------|------------------|-------------------|---------|
| **Customers** | 1.5M rows | `read_parquet()` | Customer master data |
| **Orders** | 15M rows | `read_parquet()` | Order transactions |
| **LineItems** | 60M rows | `read_parquet()` | Detailed line items |
| **Final Analytics** | Aggregated | `groupby().aggregate()` | Business intelligence |

### Ray Data operations demonstrated

**Data loading and extraction**:
- `read_parquet()` with column pruning - Optimized data loading
- High I/O concurrency with `num_cpus=0.025` - Maximum parallelism

**Data transformations**:
- `map_batches()` - Distributed transformations with business logic
- `filter()` with expressions API - Advanced filtering
- `select_columns()` - Schema optimization for performance

**Data integration**:
- `join()` - Distributed joins across datasets
- `groupby().aggregate()` - Large-scale aggregations
- Native aggregations: `Count()`, `Sum()`, `Mean()`, `Max()`

**Data output**:
- `write_parquet()` - Data warehouse output with partitioning
- Compression and optimization for query performance

### ETL optimization techniques applied

| Technique | Implementation | Benefit |
|-----------|---------------|---------|
| **Column Pruning** | Specify columns in `read_parquet()` | 60-95% I/O reduction |
| **Early Filtering** | Use `filter()` after read | Reduce data volume early |
| **Resource Allocation** | Stage-specific `num_cpus` values | Balanced parallelism |
| **Batch Sizing** | Appropriate batch sizes | Memory management |
| **Partitioned Output** | `partition_cols` parameter | Query optimization |
| **Compression** | Snappy compression | Storage efficiency |

### Production ETL patterns demonstrated

This template showcases enterprise-ready patterns:
-  TPC-H benchmark for standardized testing
-  Business logic transformations with real data
-  Data quality and validation checks
-  Analytics-ready output formats (partitioned Parquet)
-  Performance monitoring with progress bars
-  Proper resource cleanup

### ETL pipeline performance visualization

```python
# Visualize ETL pipeline performance using utility functions
from util.viz_utils import (
    visualize_etl_performance,
    create_interactive_etl_pipeline,
    create_data_lineage_diagram,
    create_tpch_schema_diagram
)

# Generate ETL performance visualization
fig = visualize_etl_performance()
print("ETL performance visualization created")

# Create interactive pipeline dashboard
pipeline_results = {
    'stages': ['Extract', 'Transform', 'Load'],
    'records': [1500000, 1200000, 1200000],
    'processing_time': [12, 45, 10]
}
interactive_dashboard = create_interactive_etl_pipeline(pipeline_results)
interactive_dashboard.write_html('etl_pipeline_dashboard.html')
print("Interactive ETL pipeline dashboard saved")

# Create data lineage visualization
lineage_diagram = create_data_lineage_diagram()
lineage_diagram.write_html('data_lineage.html')
print("Data lineage diagram saved")

# Create TPC-H schema diagram
schema_diagram = create_tpch_schema_diagram()
schema_diagram.write_html('tpch_schema.html')
print("TPC-H schema diagram saved")
```

**Interactive visualizations created:**
- **ETL pipeline dashboard**: Shows data volumes, processing times, and resource utilization
- **Data lineage diagram**: Sankey chart showing data flow through ETL stages
- **TPC-H schema**: Visual representation of table relationships
- **Performance metrics**: Interactive charts for throughput and optimization impact

### Ray Data's streaming execution model

Ray Data uniquely combines the best aspects of structured streaming and batch processing:

#### How streaming execution works

```
Traditional Batch Processing:
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│ Read │ → │ Proc │ → │ Agg  │ → │Write │
│ ALL  │   │ ALL  │   │ ALL  │   │ ALL  │
└──────┘   └──────┘   └──────┘   └──────┘
  Wait       Wait       Wait       Wait
  
 High memory (load everything)
 No parallelism across stages
 Long time to first result

Ray Data Streaming Execution:
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Block1│ → │Block1│ → │Block1│ → │Block1│
│Block2│ → │Block2│ → │Block2│ → │Block2│ (Parallel!)
│Block3│ → │Block3│ → │Block3│ → │Block3│
└──────┘   └──────┘   └──────┘   └──────┘
  Read      Process    Aggregate   Write

 Low memory (process in blocks)
 Pipeline parallelism (all stages active)
 Fast time to first result
```

#### Best of structured streaming

From structured streaming systems (Spark Structured Streaming, Flink), Ray Data inherits:

| Streaming Benefit | How Ray Data Implements It | ETL Advantage |
|------------------|---------------------------|---------------|
| **Low memory footprint** | Process 128 MB blocks, not full dataset | Handle TB datasets with GB clusters |
| **Pipeline parallelism** | All stages run simultaneously | Better resource utilization |
| **Backpressure control** | Automatic flow management | Prevents memory overflow |
| **Incremental results** | Output available immediately | Faster feedback |

#### Best of batch processing

From batch processing systems (Spark SQL, pandas), Ray Data inherits:

| Batch Benefit | How Ray Data Implements It | ETL Advantage |
|--------------|---------------------------|---------------|
| **Simple API** | Familiar operations (filter, join, groupby) | Easy to learn and use |
| **High throughput** | Optimized block sizes (128 MB) | Efficient processing |
| **Rich transformations** | Full SQL-like operations | Complex business logic |
| **No complexity** | No windowing or watermarking | Simpler code |

#### Unique Ray Data advantages

What makes Ray Data's approach special:

**Automatic optimization**:
- No manual micro-batch tuning required
- Intelligent backpressure between stages
- Dynamic resource allocation

**Unified API**:
- Same code for batch and streaming
- No separate APIs to learn
- Consistent behavior

**Example: Streaming execution in ETL**
```python
# This pipeline runs all stages simultaneously:
etl_results = (
    # Stage 1: Read (starts immediately)
    ray.data.read_parquet("s3://data/",
    num_cpus=0.025
)
    
    # Stage 2: Filter (processes blocks as they arrive)  
    .filter(lambda x: x["valid"] == True, num_cpus=0.1)
    
    # Stage 3: Transform (runs in parallel with read/filter)
    .map_batches(enrich_data, num_cpus=0.5, batch_format="pandas")
    
    # Stage 4: Write (starts as soon as first blocks ready)
    .write_parquet("s3://output/",
    num_cpus=0.1
)
)

# All stages active simultaneously - pipeline parallelism!
# Block 1 can be writing while Block 10 is being read
# Memory stays constant regardless of total dataset size
```

**Why this matters**:
- Process 100 TB with 64 GB cluster memory
- Results available while pipeline still running
- All CPUs active across all stages
- No manual tuning of batch sizes or windows

## Cleanup

```python
# Cleanup Ray resources following best practicesif ray.is_initialized():
    ray.shutdown()
print("Ray shutdown completed")
```

## Ray Data ETL operations summary

### Native operations demonstrated

This template showcases the full range of Ray Data's native ETL operations:

| Operation Category | Functions Used | Purpose | Performance Benefit |
|-------------------|---------------|---------|-------------------|
| **Data Loading** | `read_parquet()` | Extract data from sources | Parallel I/O with column pruning |
| **Filtering** | `filter()` with expressions | Remove unwanted data | Push-down optimization |
| **Transformations** | `map_batches()` | Apply business logic | Distributed processing |
| **Joins** | `join()` | Combine datasets | Scalable distributed joins |
| **Aggregations** | `groupby().aggregate()` | Calculate metrics | Parallel aggregation |
| **Column Operations** | `select_columns()` | Schema optimization | Reduce memory usage |
| **Output** | `write_parquet()` | Load to warehouse | Partitioned, compressed |

### ETL pipeline optimization techniques applied

| Technique | Implementation | Performance Impact | When to Use |
|-----------|---------------|-------------------|-------------|
| **Column Pruning** | `columns=["col1", "col2"]` in read operations | 60-95% I/O reduction | Always for large datasets |
| **num_cpus Tuning** | Stage-specific values (0.025-2.0) | Balanced resource utilization | When CPU <80% |
| **Batch Sizing** | `batch_size=500-2000` | Memory management | For memory-intensive ops |
| **Early Filtering** | `filter()` immediately after read | Data volume reduction | When you can filter early |
| **Expression API** | `col()` and `lit()` for filters | Query optimization | For complex filtering |
| **Partitioned Output** | `partition_cols=["date"]` | Query performance | For time-series data |
| **Compression** | `compression="snappy"` | Storage efficiency | All warehouse outputs |

### Key takeaways

**Ray Data ETL advantages**:
- **Parallel processing**: Distribute operations across cluster automatically
- **Memory efficiency**: Stream processing without full dataset materialization
- **Native operations**: Optimized implementations for common ETL patterns
- **Scalability**: Handle datasets from gigabytes to petabytes with same code
- **Performance**: Proper resource allocation delivers optimal throughput

**Best practices demonstrated**:
-  Use `read_parquet()` with column pruning for efficient extraction
-  Apply filters early using expressions API for data reduction
-  Specify `num_cpus` based on operation complexity
-  Use `map_batches()` with batch_format="pandas" for transformations
-  Leverage native `join()` and `groupby()` operations
-  Write with partitioning and compression for warehouse optimization
-  Monitor progress with Ray Dashboard and progress bars

**Production patterns**:
- **TPC-H benchmark**: Industry-standard for testing and learning
- **Column selection**: Only read/process what you need
- **Resource allocation**: Stage-specific `num_cpus` values
- **Memory management**: Batch sizes and block sizes for stability
- **Data warehouse**: Partitioned, compressed Parquet output
- **Validation**: Verify outputs and data quality

This template covers complete ETL workflows from fundamental concepts to production optimization techniques using real benchmark data.
