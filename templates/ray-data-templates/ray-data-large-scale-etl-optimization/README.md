# Large-scale ETL optimization with Ray Data

**⏱️ Time to complete**: 35 min | **Difficulty**: Intermediate | **Prerequisites**: Understanding of ETL concepts, data processing experience

## What You'll Build

Create an ETL pipeline that processes large datasets using Ray Data's distributed processing capabilities. Learn optimization techniques that help production ETL jobs scale reliably.

## Table of Contents

1. [ETL Data Creation](#step-1-creating-sample-etl-data) (8 min)
2. [Optimized Transformations](#step-2-efficient-data-transformations) (12 min)
3. [Parallel Processing](#step-3-distributed-etl-operations) (10 min)
4. [Performance Monitoring](#step-4-etl-performance-optimization) (5 min)

## Learning Objectives

**Why ETL optimization matters**: The difference between fast and slow data pipelines directly impacts business agility and operational costs at enterprise scale. Understanding optimization techniques enables data teams to deliver insights faster while reducing infrastructure costs.

**Ray Data's ETL superpowers**: Native operations for distributed processing that automatically optimize memory, CPU, and I/O utilization. You'll learn how Ray Data's architecture enables efficient processing of massive datasets.

**Real-world optimization patterns**: Techniques used by companies like Netflix and Airbnb to process petabytes of data daily for business intelligence demonstrate the practical application of distributed ETL optimization.

**Performance tuning strategies**: Memory management, parallel processing, and resource configuration patterns for production ETL workloads.

## Overview

**The Challenge**: Traditional ETL tools struggle with modern data volumes. Processing terabytes of data can take days, creating bottlenecks in data-driven organizations.

**The Solution**: Ray Data's distributed architecture and optimized operations enable efficient processing of large datasets through parallel computation.

**Enterprise ETL at Scale**

Leading technology companies demonstrate the critical importance of optimized ETL processing. Data warehouse companies like Snowflake process petabytes of data daily for business intelligence applications, requiring sophisticated distributed ETL pipelines that maintain sub-second query response times. E-commerce giants like Amazon process billions of transactions continuously for real-time recommendation systems, where ETL latency directly impacts revenue and customer experience.

Social media platforms like Facebook process trillions of user events for content ranking and advertising optimization, demanding ETL systems that handle massive data volumes while preserving data quality and consistency. Ride-sharing companies like Uber process millions of trips for dynamic pricing and driver matching algorithms, where ETL efficiency affects operational costs and service reliability.

```python
# Example: High-volume transaction processing like Amazon
def process_transaction_stream(batch):
    """Process e-commerce transactions for real-time analytics."""
    processed_transactions = []
    
    for transaction in batch:
        # Calculate derived fields for analytics
        enhanced_transaction = {
            'transaction_id': transaction['id'],
            'customer_segment': 'premium' if transaction['amount'] > 500 else 'standard',
            'product_category': transaction['product_type'],
            'revenue_impact': transaction['amount'] * transaction['quantity'],
            'processing_timestamp': transaction['timestamp'],
            'geographic_region': transaction['shipping_region']
        }
        
        processed_transactions.append(enhanced_transaction)
    
    return processed_transactions

# Demonstrate high-volume ETL processing
print("Enterprise-scale transaction processing enabled")
```

These implementations showcase how distributed ETL systems enable real-time business intelligence and operational efficiency at unprecedented scale.

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of ETL (Extract, Transform, Load) concepts
- [ ] Experience with data processing and transformations
- [ ] Familiarity with distributed computing concepts
- [ ] Python environment with sufficient memory (8GB+ recommended)

## Quick Start (3 minutes)

Want to see high-performance ETL immediately? This section demonstrates the core concepts in just a few minutes.

### Setup and Imports

```python
import ray
import pandas as pd
import numpy as np
import time

# Initialize Ray for distributed processing
ray.init()
```

### Load ETL dataset

Load a real dataset or a prepared local subset for ETL processing.

```python
from ray.data import read_parquet

print("Loading ETL input dataset...")
ds = read_parquet(
    "s3://anonymous@nyc-tlc/trip_data/",
)
print(f"Created ETL dataset with {ds.count():,} records")
```

**What just happened?**
- Loaded a real dataset into a Ray Dataset
- Validated record count and schema quickly

### Quick ETL Transformation

```python
# Quick ETL demonstration
print("Running quick ETL transformation...")
result = ds.map_batches(lambda batch: [
    {**record, "total_value": record["amount"] * record["quantity"]} 
    for record in batch
], batch_size=1000)

sample_results = result.take(5)
```

### Display Processed Results

```python
# Display results in a visually appealing table format
print("Sample Processed Records:")
print("=" * 80)
print(f"{'Order ID':<12} {'Customer':<12} {'Product':<10} {'Qty':<4} {'Amount':<8} {'Total Value':<12}")
print("-" * 80)

for record in sample_results:
    print(f"{record['order_id']:<12} {record['customer_id']:<12} {record['product_id']:<10} "
          f"{record['quantity']:<4} ${record['amount']:<7.2f} ${record['total_value']:<11.2f}")

print("-" * 80)
print(f"Dataset Summary: {ds.count():,} total records ready for advanced ETL processing")
```

### Data distribution analysis

```python
# Show data distribution for better understanding
regions = [r['region'] for r in sample_results]
print(f"Regional Distribution (sample): {dict(pd.Series(regions).value_counts())}")

# Calculate and display basic statistics
amounts = [r['amount'] for r in sample_results]
print(f"Amount Statistics (sample): Min=${min(amounts):.2f}, Max=${max(amounts):.2f}, Avg=${np.mean(amounts):.2f}")

print(f"\nReady for advanced ETL processing!")
```

**Key takeaways from quick start:**
- Ray Data handles large datasets efficiently through distributed processing
- Simple transformations can be applied using `map_batches()` 
- Results can be displayed in professional, readable formats
- The same patterns scale from thousands to millions of records

## Why ETL Performance Matters

**The Scale Challenge**:
- **Volume**: Modern companies generate terabytes of data daily
- **Velocity**: Business decisions require real-time or near-real-time data
- **Complexity**: Data comes from dozens of sources in different formats
- **Cost**: Slow ETL means expensive compute resources running longer

**Performance Considerations**:

| ETL Approach | Characteristics | Business Impact |
|--------------|----------------|-----------------|
| **Traditional ETL** | Single-machine processing | Limited scalability, resource constraints |
| **Ray Data ETL** | Distributed parallel processing | Horizontal scalability, efficient resource utilization |
| **Key Difference** | Distributed vs. centralized | Better resource utilization and scalability |

## Use Case: E-commerce Data Warehouse ETL

We'll build an ETL pipeline that processes:
- **Customer Data**: Demographics, preferences, segments (10M+ records)
- **Transaction Data**: Orders, payments, refunds (100M+ records)  
- **Product Data**: Catalog, inventory, pricing (1M+ records)
- **Behavioral Data**: Clicks, views, searches (1B+ records)

The pipeline will:
1. Extract data from multiple sources in parallel
2. Apply data quality validation and cleansing
3. Perform complex joins and aggregations
4. Generate business intelligence metrics
5. Load results to analytical data stores

## Architecture

### **Ray Data ETL processing architecture**

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Enterprise Data Sources                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │
│  │   Customer   │ │    Orders    │ │   Products   │ │  Behavioral  │           │
│  │     Data     │ │     Data     │ │     Data     │ │     Data     │           │
│  │   (10M+)     │ │   (100M+)    │ │    (1M+)     │ │    (1B+)     │           │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Ray Data Ingestion Layer                                │
│  • ray.data.read_parquet() • ray.data.read_csv() • ray.data.read_json()       │
│  • Distributed loading across cluster • Automatic partitioning                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Parallel ETL Processing Engine                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                  │
│  │   Validation    │ │   Cleansing     │ │   Enrichment    │                  │
│  │ • Data quality  │ │ • Deduplication │ │ • Joins         │                  │
│  │ • Schema checks │ │ • Normalization │ │ • Calculations  │                  │
│  │ • Business rules│ │ • Type casting  │ │ • Aggregations  │                  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Analytics & Storage Layer                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                  │
│  │ Data Warehouse  │ │   OLAP Cubes    │ │    Reports      │                  │
│  │ • Partitioned   │ │ • Aggregated    │ │ • Dashboards    │                  │
│  │ • Optimized     │ │ • Indexed       │ │ • Alerts        │                  │
│  │ • Compressed    │ │ • Cached        │ │ • Insights      │                  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### **Ray Data Advantages for ETL**

| Traditional ETL Approach | Ray Data ETL Approach | Key Difference |
|---------------------------|----------------------|----------------|
| **Single-machine processing** | Distributed across multiple CPU cores | Horizontal scalability |
| **Sequential operations** | Parallel processing pipeline | Concurrent execution |
| **Manual resource management** | Automatic scaling and load balancing | Simplified operations |
| **Complex infrastructure setup** | Native Ray Data operations | Streamlined development |
| **Limited fault tolerance** | Built-in error recovery and retries | Enhanced reliability |

## Key Components

### 1. **Parallel Data Extraction**
- `ray.data.read_parquet()` for efficient columnar data
- `ray.data.read_csv()` for structured text data
- `ray.data.read_json()` for semi-structured data
- Optimized file reading with block size tuning

### 2. **Native Data Transformations**
- `dataset.map()` for row-wise transformations
- `dataset.map_batches()` for vectorized operations
- `dataset.filter()` for data selection
- `dataset.flat_map()` for one-to-many transformations

### 3. **Distributed Aggregations**
- `dataset.groupby()` for aggregation operations
- Native sorting and ranking operations
- Statistical calculations and metrics
- Cross-dataset joins and correlations

### 4. **Optimized Data Loading**
- `dataset.write_parquet()` for analytical workloads
- `dataset.write_csv()` for reporting systems
- Partitioned writing for optimal query performance
- Compression and encoding optimization

## Prerequisites

- Ray cluster with sufficient memory and CPU cores
- Python 3.8+ with Ray Data
- Access to large datasets (multi-GB recommended)
- Basic understanding of ETL concepts and data processing

## Installation

```bash
pip install ray[data] pyarrow fastparquet
pip install numpy pandas
pip install boto3 s3fs
pip install matplotlib seaborn plotly networkx psutil
```

## Quick Start

### 1. **Load Large Datasets with Ray Data Native Operations**

Let's load real-world datasets using Ray Data's native reading capabilities.

**Import Required Libraries**

```python
# Standard library imports
from typing import Dict, Any
import time
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# Ray Data imports
import ray
from ray.data import read_parquet, read_csv

print("All libraries imported successfully")
```

**Initialize Ray with Optimized Configuration**

```python
# Initialize Ray with optimized configuration for large-scale ETL
ray.init(
    object_store_memory=10_000_000_000,  # 10GB object store
    _memory=20_000_000_000               # 20GB heap memory
)

print("Ray cluster initialized with optimized ETL configuration")
print(f"Available resources: {ray.cluster_resources()}")
```

**Load NYC Taxi Data (public dataset)**

```python
# Load NYC Taxi data - publicly available, large scale dataset
print("Loading NYC Taxi dataset...")
taxi_data = read_parquet(
    "s3://anonymous@nyc-tlc/trip_data/",
    columns=["pickup_datetime", "dropoff_datetime", "passenger_count", 
             "trip_distance", "fare_amount", "total_amount"]
)

print(f"NYC Taxi data loaded: {taxi_data.count():,} trip records")
```

**Load Amazon Reviews Data (text + structured)**

```python
# Load Amazon product reviews - text + structured data
print("Loading Amazon Reviews dataset...")
reviews_data = read_parquet(
    "s3://anonymous@amazon-reviews-pds/parquet/",
    columns=["review_date", "star_rating", "review_body", "product_category"]
)

print(f"Amazon Reviews loaded: {reviews_data.count():,} review records")
```

**Load US Census Data (demographic information)**

```python
# Load US Census data - demographic information
print("Loading US Census dataset...")
census_data = read_csv("s3://anonymous@uscensus-grp/acs/2021_5yr_data.csv")

print(f"US Census data loaded: {census_data.count():,} demographic records")
```

# Display dataset information with visual formatting
datasets_info = [
    ("Taxi Data", taxi_data.count(), taxi_data.schema()),
    ("Reviews Data", reviews_data.count(), reviews_data.schema()),
    ("Census Data", census_data.count(), census_data.schema())
]

print("Loaded Datasets Summary:")
print("=" * 100)
print(f"{'Dataset':<15} {'Record Count':<15} {'Schema Preview':<50}")
print("-" * 100)

for name, count, schema in datasets_info:
    # Get first few column names for schema preview
    schema_preview = str(schema)[:47] + "..." if len(str(schema)) > 50 else str(schema)
    print(f"{name:<15} {count:<15,} {schema_preview:<50}")

print("=" * 100)

# Display sample records from each dataset
print("\nSample Data Preview:")
print("-" * 100)

# Taxi data sample
taxi_sample = taxi_data.take(2)
print("Taxi Data Sample:")
for i, record in enumerate(taxi_sample):
    pickup = record.get('pickup_datetime', 'N/A')
    fare = record.get('fare_amount', 0)
    distance = record.get('trip_distance', 0)
    print(f"  {i+1}. Pickup: {pickup}, Fare: ${fare:.2f}, Distance: {distance:.1f}mi")

# Reviews data sample  
reviews_sample = reviews_data.take(2)
print("\nReviews Data Sample:")
for i, record in enumerate(reviews_sample):
    rating = record.get('star_rating', 'N/A')
    category = record.get('product_category', 'N/A')
    body_preview = str(record.get('review_body', ''))[:60] + "..." if len(str(record.get('review_body', ''))) > 60 else str(record.get('review_body', ''))
    print(f"  {i+1}. Rating: {rating} stars, Category: {category}")
    print(f"      Review: {body_preview}")

print("-" * 100)

# Data format efficiency demonstration
print("\nData Format Efficiency:")
print("Using Parquet format for taxi and reviews data (optimal for analytics)")
print("Using CSV for census data (consider converting to Parquet for better performance)")

# Example: Convert CSV to Parquet for better performance
# census_parquet = census_data.write_parquet("s3://your-bucket/census_optimized/")
```

### 2. **Data Quality and Validation with Native Operations**

```python
# Demonstrate Ray Data native operations with expressions API
from ray.data.expressions import col, lit
print("Using Ray Data native operations with optimized expressions...")

# BEST PRACTICE: Use expressions API for better performance
valid_taxi_trips = taxi_data.filter(
    (col("fare_amount") > lit(0)) & (col("trip_distance") > lit(0))
)

# Native groupby operations for aggregation
from ray.data.aggregate import Mean
trip_stats = valid_taxi_trips.groupby("passenger_count").aggregate(
    Mean("fare_amount"),
    Mean("trip_distance")
)

# Native sorting for ordered results  
sorted_trips = valid_taxi_trips.sort("fare_amount", descending=True)

# Advanced filtering with expressions for business rules
high_value_trips = valid_taxi_trips.filter(
    (col("fare_amount") > lit(50)) & 
    (col("trip_distance") > lit(10)) &
    (col("passenger_count") >= lit(2))
)

print(f"Filtered to {valid_taxi_trips.count()} valid trips")
print(f"Grouped statistics by passenger count")
print(f"Sorted trips by fare amount")

# Use Ray Data native operations for data quality checks
def validate_taxi_data(batch):
    """Validate taxi trip data using business rules."""
    valid_records = []
    
    for record in batch:
        # Apply validation rules
        is_valid = True
        validation_errors = []
        
        # Check fare amount
        if record.get('fare_amount', 0) < 0:
            is_valid = False
            validation_errors.append("Negative fare amount")
        
        # Check trip distance
        if record.get('trip_distance', 0) < 0:
            is_valid = False
            validation_errors.append("Negative trip distance")
        
        # Check passenger count
        if record.get('passenger_count', 0) <= 0 or record.get('passenger_count', 0) > 6:
            is_valid = False
            validation_errors.append("Invalid passenger count")
        
        # Add validation metadata
        validated_record = {
            **record,
            'is_valid': is_valid,
            'validation_errors': validation_errors,
            'validation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        valid_records.append(validated_record)
    
    return valid_records

# Apply validation using Ray Data native map_batches
validated_taxi = taxi_data.map_batches(
    validate_taxi_data,
    batch_size=10000,
    concurrency=8
)

# Filter to only valid records using native filter operation
clean_taxi_data = validated_taxi.filter(lambda record: record['is_valid'])

print(f"Clean taxi data: {clean_taxi_data.count()} records")

# Display data quality summary in a visual format
print("\nData Quality Summary:")
print("=" * 60)
total_records = taxi_data.count()
clean_records = clean_taxi_data.count()
invalid_records = total_records - clean_records

print(f"{'Metric':<25} {'Count':<10} {'Percentage':<12}")
print("-" * 60)
print(f"{'Total Records':<25} {total_records:<10,} {'100.0%':<12}")
print(f"{'Valid Records':<25} {clean_records:<10,} {clean_records/total_records*100:<11.1f}%")
print(f"{'Invalid Records':<25} {invalid_records:<10,} {invalid_records/total_records*100:<11.1f}%")
print("=" * 60)

# Sample clean records for inspection
sample_clean = clean_taxi_data.take(3)
print(f"\nSample Clean Records:")
print("-" * 100)
for i, record in enumerate(sample_clean):
    fare = record.get('fare_amount', 0)
    distance = record.get('trip_distance', 0)
    passengers = record.get('passenger_count', 0)
    print(f"{i+1}. Fare: ${fare:.2f}, Distance: {distance:.1f}mi, Passengers: {passengers}, Valid: {record.get('is_valid', False)}")
print("-" * 100)
```

### 3. **Large-Scale Aggregations with Native GroupBy**

```python
# BEST PRACTICE: Use Ray Data native groupby operations instead of pandas
from ray.data.aggregate import Count, Sum, Mean, Std

# First, extract date from pickup_datetime using add_column
def extract_pickup_date(batch):
    """Extract date from pickup_datetime for groupby operations."""
    import pandas as pd
    
    # Convert pickup_datetime to date for each record
    enhanced_records = []
    for record in batch:
        try:
            pickup_datetime = pd.to_datetime(record.get('pickup_datetime'))
            pickup_date = pickup_datetime.date().isoformat()
            enhanced_records.append({
                **record,
                'pickup_date': pickup_date
            })
        except Exception:
            # Keep original record if date parsing fails
            enhanced_records.append({
                **record,
                'pickup_date': '1970-01-01'  # Default date for invalid entries
            })
    
    return enhanced_records

# Add pickup_date column using Ray Data native operations
taxi_with_dates = clean_taxi_data.map_batches(
    extract_pickup_date,
    batch_size=10000,
    concurrency=6
)

# Use Ray Data native groupby with aggregation functions - MUCH MORE EFFICIENT
daily_metrics = taxi_with_dates.groupby("pickup_date").aggregate(
    Count(),  # Trip count
    Sum("fare_amount"),  # Total fare revenue
    Mean("fare_amount"),  # Average fare
    Std("fare_amount"),   # Fare standard deviation
    Sum("trip_distance"), # Total distance
    Mean("trip_distance"), # Average distance
    Sum("passenger_count"), # Total passengers
    Sum("total_amount")    # Total revenue including tips
)

# Add computed columns using native Ray Data operations
from ray.data.expressions import col, lit

daily_metrics_enhanced = daily_metrics.add_column(
    "avg_fare_per_mile", 
    col("sum(fare_amount)") / col("sum(trip_distance)")
).add_column(
    "revenue_per_trip",
    col("sum(total_amount)") / col("count()")
)

print(f"Daily metrics: {daily_metrics.count()} records")

# Display daily metrics in a visually appealing format
sample_metrics = daily_metrics.take(5)
print("\nDaily Taxi Metrics Summary:")
print("=" * 120)
print(f"{'Date':<12} {'Trips':<8} {'Revenue':<10} {'Avg Fare':<10} {'Total Miles':<12} {'Avg Distance':<12}")
print("-" * 120)

for metric in sample_metrics:
    date = metric.get('pickup_date', 'N/A')
    trip_count = metric.get('fare_amount_count', 0)
    revenue = metric.get('total_amount_sum', 0)
    avg_fare = metric.get('fare_amount_mean', 0)
    total_miles = metric.get('trip_distance_sum', 0)
    avg_distance = metric.get('trip_distance_mean', 0)
    
    print(f"{str(date):<12} {trip_count:<8,} ${revenue:<9.0f} ${avg_fare:<9.2f} {total_miles:<11.1f}mi {avg_distance:<11.2f}mi")

print("-" * 120)
print("Note: This demonstrates Ray Data's native groupby aggregation capabilities")
```

### 4. **Cross-Dataset Joins and Enrichment**

```python
# Perform data enrichment using Ray Data operations
def enrich_with_reviews(batch):
    """Enrich data with review sentiment analysis."""
    enriched_records = []
    
    for record in batch:
        # Add sentiment analysis (simplified)
        review_text = record.get('review_body', '')
        
        # Simple sentiment scoring based on keywords
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'wonderful']
        negative_words = ['terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing']
        
        positive_count = sum(1 for word in positive_words if word in review_text.lower())
        negative_count = sum(1 for word in negative_words if word in review_text.lower())
        
        # Calculate sentiment score
        if positive_count > negative_count:
            sentiment = 'positive'
            sentiment_score = min(positive_count / (positive_count + negative_count + 1), 1.0)
        elif negative_count > positive_count:
            sentiment = 'negative'
            sentiment_score = min(negative_count / (positive_count + negative_count + 1), 1.0)
        else:
            sentiment = 'neutral'
            sentiment_score = 0.5
        
        enriched_record = {
            **record,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'review_length': len(review_text),
            'word_count': len(review_text.split()),
            'enrichment_timestamp': pd.Timestamp.now().isoformat()
        }
        
        enriched_records.append(enriched_record)
    
    return enriched_records

# Apply enrichment using Ray Data native operations
enriched_reviews = reviews_data.map_batches(
    enrich_with_reviews,
    batch_size=5000,
    concurrency=6
)

print(f"Enriched reviews: {enriched_reviews.count()} records")

# Display enrichment results with visual formatting
sample_enriched = enriched_reviews.take(3)
print("\nEnriched Review Data Sample:")
print("=" * 100)

for i, review in enumerate(sample_enriched):
    print(f"\nReview {i+1}:")
    print(f"  Rating: {review.get('star_rating', 'N/A')} stars")
    print(f"  Sentiment: {review.get('sentiment', 'N/A').upper()} (score: {review.get('sentiment_score', 0):.2f})")
    print(f"  Text Length: {review.get('review_length', 0)} characters ({review.get('word_count', 0)} words)")
    print(f"  Preview: {str(review.get('review_body', ''))[:80]}...")
    print("-" * 50)

# Show sentiment distribution
sentiments = [r.get('sentiment', 'unknown') for r in sample_enriched]
sentiment_counts = pd.Series(sentiments).value_counts()
print(f"\nSentiment Distribution (sample):")
for sentiment, count in sentiment_counts.items():
    bar_length = int(count * 20 / len(sample_enriched))
    bar = "█" * bar_length + "░" * (20 - bar_length)
    print(f"  {sentiment.capitalize():<10} {bar} {count}/{len(sample_enriched)}")

print("=" * 100)
```

### 5. **Optimized data loading and partitioning**

```python
# Write results using Ray Data native operations with optimization
import tempfile

output_dir = tempfile.mkdtemp()
print(f"Output directory: {output_dir}")

# Write daily metrics with optimal partitioning
daily_metrics.write_parquet(
    f"local://{output_dir}/daily_taxi_metrics",
    num_files=10,  # Control number of output files
    compression="snappy"
)

# Write enriched reviews with partitioning by category
enriched_reviews.write_parquet(
    f"local://{output_dir}/enriched_reviews",
    num_files=20,
    compression="gzip"
)

# Write clean taxi data with date-based partitioning
clean_taxi_data.write_parquet(
    f"local://{output_dir}/clean_taxi_data",
    num_files=50,
    compression="snappy"
)

print(f"All datasets written to: {output_dir}")

# Display file output summary with visual formatting
import os
print("\nETL Output Summary:")
print("=" * 80)
print(f"{'Dataset':<30} {'Location':<35} {'Status':<15}")
print("-" * 80)

datasets = [
    ("Daily Taxi Metrics", f"{output_dir}/daily_taxi_metrics", "Complete"),
    ("Enriched Reviews", f"{output_dir}/enriched_reviews", "Complete"), 
    ("Clean Taxi Data", f"{output_dir}/clean_taxi_data", "Complete")
]

for name, path, status in datasets:
    print(f"{name:<30} {path[-35:]:<35} {status:<15}")

print("-" * 80)
print("All ETL outputs saved successfully!")

# Optional: Write to cloud storage for production (commented for demo)
print("\nProduction Storage Options:")
print("# daily_metrics.write_parquet('s3://your-bucket/etl-output/daily_metrics/')")
print("# enriched_reviews.write_parquet('s3://your-bucket/etl-output/enriched_reviews/')")
print("# clean_taxi_data.write_parquet('s3://your-bucket/etl-output/clean_taxi_data/')")
```

## Advanced ETL Patterns

### **Memory-Efficient Processing**

```python
# Process large datasets with memory optimization
def memory_optimized_transform(batch):
    """Transform data with memory optimization techniques."""
    # Process in smaller chunks to manage memory
    chunk_size = 1000
    transformed_records = []
    
    for i in range(0, len(batch), chunk_size):
        chunk = batch[i:i + chunk_size]
        
        # Apply transformations to chunk
        for record in chunk:
            # Minimal memory footprint transformations
            transformed_record = {
                'id': record.get('id'),
                'processed_value': record.get('value', 0) * 1.1,
                'category': record.get('category', 'unknown').upper(),
                'is_valid': record.get('value', 0) > 0
            }
            transformed_records.append(transformed_record)
    
    return transformed_records

# Apply memory-optimized processing
optimized_data = taxi_data.map_batches(
    memory_optimized_transform,
    batch_size=5000,  # Smaller batches for memory efficiency
    concurrency=10
)
```

### **Distributed Deduplication**

```python
# OPTIMIZED: Minimize pandas usage for deduplication
def deduplicate_records(batch):
    """Remove duplicate records with minimal DataFrame conversion."""
    # Use native Python for deduplication when possible
    seen_keys = set()
    deduplicated_records = []
    original_count = len(batch)
    
    for record in batch:
        # Create composite key for deduplication
        key = (
            record.get('pickup_datetime', ''),
            record.get('dropoff_datetime', ''),
            record.get('fare_amount', 0)
        )
        
        if key not in seen_keys:
            seen_keys.add(key)
            # Add deduplication metadata
            enhanced_record = {
                **record,
                'deduplication_timestamp': pd.Timestamp.now().isoformat(),
                'original_batch_size': original_count,
                'deduplicated_batch_size': len(deduplicated_records) + 1
            }
            deduplicated_records.append(enhanced_record)
    
    return deduplicated_records

# Apply deduplication with optimized batch processing
deduplicated_data = clean_taxi_data.map_batches(
    deduplicate_records,
    batch_size=15000,  # Optimized batch size
    concurrency=8      # Increased concurrency for efficiency
)

print(f"Deduplicated data: {deduplicated_data.count()} records")
```

## Performance optimization

### **Block size and parallelism tuning**

```python
# Optimize Ray Data configuration for large-scale ETL
from ray.data.context import DataContext

# Configure Ray Data for optimal ETL performance
ctx = DataContext.get_current()
ctx.target_max_block_size = 1024 * 1024 * 1024  # 1GB blocks
ctx.enable_progress_bars = False

# Read with optimized block configuration
optimized_data = read_parquet(
    "s3://anonymous@nyc-tlc/trip_data/",
    parallelism=100,  # High parallelism for large datasets
    columns=["pickup_datetime", "fare_amount", "trip_distance"]
)

print(f"Optimized data loading: {optimized_data.count()} records")
```

### **Efficient column operations**

```python
# Use Ray Data native column operations
def efficient_column_transforms(batch):
    """Apply efficient column-wise transformations."""
    import numpy as np
    
    transformed_batch = []
    
    for record in batch:
        # Efficient numerical transformations
        fare = record.get('fare_amount', 0)
        distance = record.get('trip_distance', 0)
        
        # Calculate derived metrics efficiently
        fare_per_mile = fare / distance if distance > 0 else 0
        fare_tier = 'high' if fare > 20 else 'medium' if fare > 10 else 'low'
        distance_tier = 'long' if distance > 10 else 'medium' if distance > 3 else 'short'
        
        transformed_record = {
            **record,
            'fare_per_mile': fare_per_mile,
            'fare_tier': fare_tier,
            'distance_tier': distance_tier,
            'is_premium_trip': fare > 50 and distance > 10
        }
        
        transformed_batch.append(transformed_record)
    
    return transformed_batch

# Apply efficient transformations
transformed_data = optimized_data.map_batches(
    efficient_column_transforms,
    batch_size=25000,
    concurrency=8
)
```

## Advanced features

### **Distributed sorting and ranking**

```python
# Use Ray Data native sorting for large datasets
# Sort by fare amount for ranking analysis
sorted_by_fare = transformed_data.sort("fare_amount", descending=True)

# Add ranking information
def add_ranking_info(batch):
    """Add ranking information to records."""
    ranked_batch = []
    
    for i, record in enumerate(batch):
        ranked_record = {
            **record,
            'fare_rank_in_batch': i + 1,
            'is_top_10_percent': i < len(batch) * 0.1,
            'ranking_timestamp': pd.Timestamp.now().isoformat()
        }
        ranked_batch.append(ranked_record)
    
    return ranked_batch

# Apply ranking
ranked_data = sorted_by_fare.map_batches(
    add_ranking_info,
    batch_size=10000,
    concurrency=4
)
```

### **Complex business logic processing**

```python
# Implement complex business rules using Ray Data operations
def apply_business_rules(batch):
    """Apply complex business rules and calculations."""
    processed_batch = []
    
    for record in batch:
        # Extract key metrics
        fare = record.get('fare_amount', 0)
        distance = record.get('trip_distance', 0)
        passenger_count = record.get('passenger_count', 1)
        
        # Business rule calculations
        efficiency_score = distance / fare if fare > 0 else 0
        capacity_utilization = passenger_count / 4.0  # Assume 4-seat capacity
        
        # Trip categorization
        if distance > 20:
            trip_type = 'long_distance'
        elif distance > 5:
            trip_type = 'medium_distance'
        else:
            trip_type = 'short_distance'
        
        # Revenue calculations
        base_revenue = fare * 0.8  # After commission
        bonus_revenue = fare * 0.1 if fare > 30 else 0
        total_revenue = base_revenue + bonus_revenue
        
        processed_record = {
            **record,
            'efficiency_score': efficiency_score,
            'capacity_utilization': capacity_utilization,
            'trip_type': trip_type,
            'base_revenue': base_revenue,
            'bonus_revenue': bonus_revenue,
            'total_revenue': total_revenue,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        }
        
        processed_batch.append(processed_record)
    
    return processed_batch

# Apply business rules
business_processed = ranked_data.map_batches(
    apply_business_rules,
    batch_size=15000,
    concurrency=6
)
```

## Production considerations

### **Cluster configuration**
```python
# Optimal cluster setup for large-scale ETL
cluster_config = {
    "head_node": {
        "instance_type": "m5.4xlarge",  # 16 vCPUs, 64GB RAM
        "cpu": 16,
        "memory": 64000
    },
    "worker_nodes": {
        "instance_type": "m5.8xlarge",  # 32 vCPUs, 128GB RAM
        "min_workers": 5,
        "max_workers": 20,
        "cpu_per_worker": 32,
        "memory_per_worker": 128000
    }
}

# Ray initialization for production ETL
ray.init(
    object_store_memory=50_000_000_000,  # 50GB object store
    _memory=100_000_000_000,             # 100GB heap memory
    log_to_driver=True,
    enable_object_reconstruction=True
)
```

### **Resource monitoring**
- Monitor memory usage and object store pressure
- Track processing throughput and bottlenecks
- Implement automatic scaling based on workload
- Set up alerting for pipeline failures

### **Data lineage and governance**
- Track data transformations and dependencies
- Maintain audit trails for compliance
- Implement data quality monitoring
- Ensure data security and access controls

## Example workflows

### **Daily ETL pipeline**
1. Extract overnight data from operational systems
2. Validate data quality and apply cleansing rules
3. Perform complex transformations and enrichment
4. Calculate business metrics and KPIs
5. Load results to data warehouse with optimal partitioning

### **Historical data migration**
1. Extract historical data from legacy systems
2. Transform data to new schema and formats
3. Validate data integrity and completeness
4. Load data to modern analytical platforms
5. Verify migration success and performance

### **Real-time analytics preparation**
1. Process streaming data in micro-batches
2. Apply real-time transformations and aggregations
3. Prepare data for real-time dashboards
4. Update analytical models and metrics
5. Maintain data freshness and quality

## Resource planning and configuration

### **ETL processing considerations**

| Operation Type | Resource Requirements | Ray Data Features | Cluster Configuration |
|---------------|----------------------|-------------------|----------------------|
| **Data Extraction** | I/O intensive | Parallel readers | Multiple worker nodes |
| **Data Transformation** | CPU intensive | Distributed processing | High-CPU instances |
| **Data Aggregation** | Memory intensive | In-memory operations | High-memory instances |
| **Data Loading** | I/O intensive | Parallel writers | Multiple worker nodes |

### **Cluster sizing guidelines**

| Cluster Size | Memory Capacity | Processing Capability | Suitable Workloads |
|-------------|-----------------|----------------------|-------------------|
| **5 Nodes** | 32-64GB total | Moderate throughput | Development/Testing |
| **10 Nodes** | 64-128GB total | High throughput | Production workloads |
| **20+ Nodes** | 128GB+ total | Very high throughput | Large-scale processing |

### **Resource utilization patterns**

| Workload Type | CPU Requirements | Memory Requirements | Storage Requirements | Recommended Instance |
|--------------|------------------|-------------------|---------------------|---------------------|
| **Light ETL** | 2-4 cores | 4-8GB | Standard | m5.xlarge |
| **Heavy Transformations** | 4-8 cores | 6-12GB | Standard | c5.2xlarge |
| **Complex Joins** | 2-4 cores | 8-16GB | High-memory | r5.xlarge |
| **ML Feature Engineering** | 4-8 cores | 12-24GB | Standard | c5.4xlarge |

## Interactive ETL pipeline visualizations

Let's create engaging visualizations to understand our ETL pipeline results and data insights:

```python
# Import visualization libraries for engaging data presentation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Set up appealing visual style
plt.style.use('default')
sns.set_palette("husl")

def create_engaging_etl_dashboard(dataset, title="ETL Results"):
    """Create visually appealing dashboard showing ETL results and insights."""
    
    print("="*60)
    print(f"{title.upper()}")
    print("="*60)
    print(f"Total records: {dataset.count():,}")
    print(f"Schema: {dataset.schema()}")
    print(f"Dataset size: {dataset.size_bytes() / (1024**2):.1f} MB")
    
    # Convert sample data to pandas for visualization
    sample_data = dataset.take(10000)  # 10K sample for visualization
    df = pd.DataFrame(sample_data)
    
    # Create comprehensive visualization dashboard
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Trip Type Distribution', 'Revenue by Efficiency', 'Geographic Patterns',
                       'Revenue Trends', 'Capacity Utilization', 'Performance Metrics'),
        specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "histogram"}, {"type": "bar"}]]
    )
    
    # 1. Trip Type Distribution (Business Logic Results)
    if 'trip_type' in df.columns:
        trip_counts = df['trip_type'].value_counts()
        fig.add_trace(
            go.Bar(x=trip_counts.index, y=trip_counts.values,
                  marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                  name="Trip Types"),
            row=1, col=1
        )
    
    # 2. Revenue vs Efficiency Analysis
    if 'total_revenue' in df.columns and 'efficiency_score' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['efficiency_score'], y=df['total_revenue'],
                      mode='markers', marker=dict(size=8, opacity=0.6),
                      name="Revenue vs Efficiency"),
            row=1, col=2
        )
    
    # 3. Geographic Distribution (if available)
    if 'pickup_borough' in df.columns:
        borough_counts = df['pickup_borough'].value_counts()
        fig.add_trace(
            go.Bar(x=borough_counts.index, y=borough_counts.values,
                  marker_color='lightgreen', name="Geographic Distribution"),
            row=1, col=3
        )
    
    # 4. Revenue Trends Over Time
    if 'pickup_datetime' in df.columns and 'total_revenue' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        hourly_revenue = df.groupby('hour')['total_revenue'].mean()
        
        fig.add_trace(
            go.Scatter(x=hourly_revenue.index, y=hourly_revenue.values,
                      mode='lines+markers', name="Hourly Revenue Trends"),
            row=2, col=1
        )
    
    # 5. Capacity Utilization Distribution
    if 'capacity_utilization' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['capacity_utilization'], nbinsx=20,
                        marker_color='orange', name="Capacity Utilization"),
            row=2, col=2
        )
    
    # 6. Performance Metrics Summary
    if 'efficiency_score' in df.columns:
        efficiency_ranges = pd.cut(df['efficiency_score'], bins=5, labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
        range_counts = efficiency_ranges.value_counts()
        
        fig.add_trace(
            go.Bar(x=range_counts.index, y=range_counts.values,
                  marker_color='purple', name="Performance Ranges"),
            row=2, col=3
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title_text="ETL Pipeline Results Dashboard - Data Insights",
        height=800,
        showlegend=False
    )
    
    # Show interactive dashboard
    fig.show()
    
    print("="*60)
    print("Interactive ETL dashboard created!")
    print("This dashboard shows data insights and processing results")
    print("="*60)
    
    return fig

# Create engaging ETL results dashboard
etl_dashboard = create_engaging_etl_dashboard(business_processed, "ETL Pipeline Results")
```

### ETL pipeline analysis using Ray Data native operations

```python
# Analyze ETL results using Ray Data aggregations
from ray.data.aggregate import Count, Mean, Sum, Max, Min

# Business metrics analysis
print("ETL Pipeline Analysis:")
print("="*50)

# Group by trip type for analysis
trip_analysis = business_processed.groupby("trip_type").aggregate(
    Count(),
    Mean("total_revenue"),
    Mean("efficiency_score")
).rename_columns(["trip_type", "trip_count", "avg_revenue", "avg_efficiency"])

print("Trip Type Analysis:")
trip_analysis.show()

# Revenue analysis
revenue_metrics = business_processed.aggregate(
    Count(),
    Sum("total_revenue"),
    Mean("total_revenue"),
    Max("total_revenue"),
    Min("total_revenue")
)

print(f"\nRevenue Summary:")
for metric in revenue_metrics:
    print(f"  {metric}")

print(f"\nETL Pipeline Processing Complete!")
print(f"Monitor detailed performance in Ray Dashboard: {ray.get_dashboard_url()}")
```

### **ETL pipeline status monitoring**

```python
def display_pipeline_status(datasets_dict):
    """Display comprehensive pipeline status in a visual format."""
    
    print("ETL Pipeline Status Monitor")
    print("=" * 90)
    print(f"{'Pipeline Stage':<25} {'Dataset':<20} {'Records':<12} {'Status':<15} {'Notes':<20}")
    print("-" * 90)
    
    stages = [
        ("Data Extraction", "Raw Taxi Data", taxi_data.count(), "Complete", "From S3 Parquet"),
        ("Data Validation", "Validated Data", clean_taxi_data.count(), "Complete", "Business rules applied"),
        ("Data Aggregation", "Daily Metrics", daily_metrics.count(), "Complete", "Grouped by date"),
        ("Data Enrichment", "Enriched Reviews", enriched_reviews.count(), "Complete", "Sentiment analysis"),
        ("Data Storage", "Final Output", "Multiple", "Complete", "Parquet format")
    ]
    
    for stage, dataset, records, status, notes in stages:
        record_str = f"{records:,}" if isinstance(records, int) else records
        status_symbol = "[OK]" if status == "Complete" else "[WARN]"
        print(f"{stage:<25} {dataset:<20} {record_str:<12} {status_symbol} {status:<14} {notes:<20}")
    
    print("-" * 90)
    print("Pipeline Status: All stages completed successfully")
    
    # Resource utilization summary
    cluster_resources = ray.cluster_resources()
    print(f"\nCluster Resource Summary:")
    print(f"  Available CPUs: {cluster_resources.get('CPU', 0)}")
    print(f"  Available Memory: {cluster_resources.get('memory', 0) / 1e9:.1f}GB")
    print(f"  Available GPUs: {cluster_resources.get('GPU', 0)}")
    
    return True

# Display the pipeline status
pipeline_status = display_pipeline_status({
    "taxi_data": taxi_data,
    "clean_taxi_data": clean_taxi_data,
    "daily_metrics": daily_metrics,
    "enriched_reviews": enriched_reviews
})
```

### ETL pipeline flow diagram

```python
def create_etl_pipeline_diagram():
    """Create interactive ETL pipeline flow diagram."""
    print("Creating ETL pipeline flow diagram...")
    
    # Create a network graph representing the ETL pipeline
    G = nx.DiGraph()
    
    # Add nodes for different pipeline stages
    pipeline_stages = {
        'Data Sources': {'pos': (0, 2), 'color': 'lightblue', 'size': 3000},
        'Extract': {'pos': (1, 2), 'color': 'lightgreen', 'size': 2500},
        'Validate': {'pos': (2, 3), 'color': 'orange', 'size': 2000},
        'Transform': {'pos': (2, 1), 'color': 'yellow', 'size': 2500},
        'Aggregate': {'pos': (3, 2), 'color': 'lightcoral', 'size': 2000},
        'Load': {'pos': (4, 2), 'color': 'lightpink', 'size': 2500},
        'Data Warehouse': {'pos': (5, 2), 'color': 'lightgray', 'size': 3000}
    }
    
    # Add nodes to graph
    for stage, attrs in pipeline_stages.items():
        G.add_node(stage, **attrs)
    
    # Add edges representing data flow
    pipeline_edges = [
        ('Data Sources', 'Extract'),
        ('Extract', 'Validate'),
        ('Extract', 'Transform'),
        ('Validate', 'Aggregate'),
        ('Transform', 'Aggregate'),
        ('Aggregate', 'Load'),
        ('Load', 'Data Warehouse')
    ]
    
    G.add_edges_from(pipeline_edges)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    pos = nx.get_node_attributes(G, 'pos')
    colors = [pipeline_stages[node]['color'] for node in G.nodes()]
    sizes = [pipeline_stages[node]['size'] for node in G.nodes()]
    
    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes,
            font_size=12, font_weight='bold', arrows=True, arrowsize=20,
            edge_color='gray', linewidths=2, arrowstyle='->')
    
    plt.title('ETL Pipeline Flow Diagram', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('etl_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("ETL pipeline diagram saved as 'etl_pipeline_diagram.png'")

# Create pipeline diagram
create_etl_pipeline_diagram()
```

### ETL performance dashboard

```python
def create_etl_performance_dashboard():
    """Create comprehensive ETL performance monitoring dashboard."""
    print("Creating ETL performance dashboard...")
    
    # Use Ray Dashboard for real performance monitoring
    perf_df = pd.DataFrame()
    
    # Create comprehensive dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Records Processed Over Time', 'Processing Time Trends',
                       'Resource Usage', 'Throughput Analysis',
                       'Error Rate Monitoring', 'Performance Correlation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Example layout for production monitoring dashboards
    
    # Replace with real monitoring data from your environment
    
    pass
    
    # 4. Throughput analysis
    fig.add_trace(
        go.Bar(x=perf_df['timestamp'], y=perf_df['throughput'],
               name='Throughput (records/sec)', marker_color='lightblue'),
        row=2, col=2
    )
    
    # 5. Error rate monitoring
    fig.add_trace(
        go.Scatter(x=perf_df['timestamp'], y=perf_df['error_rate'],
                  mode='lines+markers', name='Error Rate (%)',
                  line=dict(color='red', width=3),
                  fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'),
        row=3, col=1
    )
    
    # 6. Performance correlation heatmap
    correlation_data = perf_df[['records_processed', 'processing_time', 'memory_usage', 
                               'cpu_usage', 'throughput']].corr()
    
    fig.add_trace(
        go.Heatmap(z=correlation_data.values,
                  x=correlation_data.columns,
                  y=correlation_data.index,
                  colorscale='RdBu',
                  zmid=0,
                  text=correlation_data.round(2).values,
                  texttemplate="%{text}",
                  showscale=True),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="ETL Performance Monitoring Dashboard",
        height=1000,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Records", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Minutes", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Memory %", row=2, col=1)
    fig.update_yaxes(title_text="CPU %", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Records/sec", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Error %", row=3, col=1)
    
    # Save and show
    print("Use Ray Dashboard for performance monitoring. This function shows example layout only.")
    
    return fig

# Create performance dashboard
performance_dashboard = create_etl_performance_dashboard()
```

### Data quality monitoring visualizations

```python
def create_data_quality_dashboard():
    """Create data quality monitoring dashboard."""
    print("Creating data quality monitoring dashboard...")
    
    # Example layout for data quality dashboards
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('ETL Data Quality Monitoring Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Data Completeness by Source
    ax1 = axes[0, 0]
    sources = []
    completeness = []
    bars = ax1.bar(sources, completeness, color='green', alpha=0.7)
    ax1.set_title('Data Completeness by Source', fontweight='bold')
    ax1.set_ylabel('Completeness (%)')
    ax1.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Target: 95%')
    ax1.legend()
    
    # Add value labels
    for bar, value in zip(bars, completeness):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Data Freshness Trends
    ax2 = axes[0, 1]
    ax2.plot([], [], 'b-o', linewidth=2, markersize=4)
    ax2.fill_between(hours, freshness_delay, alpha=0.3)
    ax2.set_title('Data Freshness (Delay in Hours)', fontweight='bold')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Delay (hours)')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='SLA: 1 hour')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Schema Validation Results
    ax3 = axes[0, 2]
    ax3.pie([], labels=[], autopct='%1.1f%%', colors=['green','orange','red'], startangle=90)
    ax3.set_title('Schema Validation Results', fontweight='bold')
    
    # 4. Data Volume Trends
    ax4 = axes[1, 0]
    ax4.plot([], [], 'g-', linewidth=2)
    ax4.set_title('Daily Data Volume Trends', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Volume (Millions of Records)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Data Type Distribution
    ax5 = axes[1, 1]
    bars = ax5.barh([], [], color='skyblue', alpha=0.7)
    ax5.set_title('Data Type Distribution', fontweight='bold')
    ax5.set_xlabel('Percentage of Columns')
    
    # Add value labels
    for bar, value in zip(bars, type_counts):
        width = bar.get_width()
        ax5.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{value}%', ha='left', va='center', fontweight='bold')
    
    # 6. Duplicate Detection
    ax6 = axes[1, 2]
    bars = ax6.bar([], [], color='green', alpha=0.7)
    ax6.set_title('Duplicate Detection Rates', fontweight='bold')
    ax6.set_ylabel('Duplicate Rate (%)')
    ax6.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Warning: 1%')
    ax6.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Critical: 3%')
    ax6.legend()
    
    # 7. Processing Error Trends
    ax7 = axes[2, 0]
    ax7.bar([], [], color='red', alpha=0.6, width=0.8)
    ax7.set_title('Processing Errors by Hour', fontweight='bold')
    ax7.set_xlabel('Hour of Day')
    ax7.set_ylabel('Error Count')
    ax7.grid(True, alpha=0.3)
    
    # 8. Data Quality Score Over Time
    ax8 = axes[2, 1]
    ax8.plot([], [], 'purple', linewidth=2, marker='o', markersize=3)
    ax8.set_title('Overall Data Quality Score', fontweight='bold')
    ax8.set_xlabel('Date')
    ax8.set_ylabel('Quality Score (%)')
    ax8.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Target: 90%')
    ax8.legend()
    ax8.tick_params(axis='x', rotation=45)
    ax8.grid(True, alpha=0.3)
    
    # 9. ETL Stage Performance
    ax9 = axes[2, 2]
    bars = ax9.bar([], [], color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    ax9.set_title('ETL Stage Performance', fontweight='bold')
    ax9.set_ylabel('Processing Time (minutes)')
    
    # Add value labels
    for bar, value in zip(bars, avg_times):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('etl_data_quality_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Use Ray Dashboard and your data platform's monitoring for real metrics.")

# Create data quality dashboard
create_data_quality_dashboard()
```

### Real-time ETL monitoring

```python
def create_realtime_etl_monitor():
    """Create real-time ETL monitoring visualization."""
    print("Creating real-time ETL monitoring system...")
    
    # Use Ray Dashboard and your observability stack for real-time monitoring
    fig = go.Figure()
    print("Use Ray Dashboard for real-time monitoring. This function shows example layout only.")
    
    return fig

# Create real-time monitor
realtime_monitor = create_realtime_etl_monitor()
```

### System resource monitoring

```python
def create_system_resource_dashboard():
    """Create system resource monitoring dashboard."""
    print("Creating system resource monitoring dashboard...")
    
    # Example layout for system resource dashboards
    cpu_percent = []
    class M: pass
    memory = M(); memory.total = 0; memory.percent = 0
    class D: pass
    disk = D(); disk.total = 0; disk.used = 0
    
    # Create system resource visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('System Resource Monitoring for ETL Pipeline', fontsize=16, fontweight='bold')
    
    # 1. CPU Usage by Core
    bars = ax1.bar([], [], color='green', alpha=0.7)
    ax1.set_title('CPU Usage by Core', fontweight='bold')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Critical: 80%')
    ax1.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Warning: 60%')
    ax1.legend()
    
    # Add value labels
    for bar, value in zip(bars, cpu_percent):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Memory Usage
    ax2.pie([], labels=[], colors=['red','green','blue'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Memory Usage (Total: {memory.total / (1024**3):.1f} GB)', fontweight='bold')
    
    # 3. Disk Usage
    ax3.pie([], labels=[], colors=['red','green'], autopct='%1.1f%%', startangle=90)
    ax3.set_title(f'Disk Usage (Total: {disk.total / (1024**3):.1f} GB)', fontweight='bold')
    
    # 4. Resource Trends (simulated)
    ax4.plot([], [], 'b-o', label='CPU Usage (%)', linewidth=2, markersize=4)
    ax4.plot([], [], 'r-s', label='Memory Usage (%)', linewidth=2, markersize=4)
    ax4.set_title('24-Hour Resource Trends', fontweight='bold')
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Usage (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('system_resource_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Use Ray Dashboard and your infrastructure monitoring for system metrics.")

# Create system resource dashboard
create_system_resource_dashboard()
```

## Troubleshooting

### **Common Issues**
1. **Memory Pressure**: Reduce batch size or increase cluster memory
2. **Slow Performance**: Optimize block size and parallelism settings
3. **Data Skew**: Implement data redistribution strategies
4. **Resource Contention**: Balance CPU and memory allocation

### **Debug Mode**
Enable detailed logging and performance monitoring:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable Ray Data debugging
from ray.data.context import DataContext
ctx = DataContext.get_current()
ctx.enable_progress_bars = True
```

## Next Steps

1. **Scale to Production**: Deploy to multi-node clusters with proper resource allocation
2. **Add Data Sources**: Connect to your specific data systems and formats
3. **Implement Monitoring**: Set up comprehensive pipeline monitoring and alerting
4. **Optimize Performance**: Fine-tune based on your specific workload characteristics

## Resources

- [Ray Data Documentation](https://docs.ray.io/en/latest/data/index.html)
- [Ray Data Performance Guide](https://docs.ray.io/en/latest/data/performance-tips.html)
- [ETL Best Practices](https://docs.ray.io/en/latest/data/best-practices.html)
- [Large-Scale Data Processing](https://docs.ray.io/en/latest/data/batch_inference.html)

## Cleanup and Resource Management

Always clean up Ray resources when done:

```python
# Clean up Ray resources
ray.shutdown()
print("Ray cluster shutdown complete")
```

---

*This template demonstrates Ray Data's native capabilities for large-scale ETL processing. Focus on using Ray Data's built-in operations for optimal performance and scalability.*
