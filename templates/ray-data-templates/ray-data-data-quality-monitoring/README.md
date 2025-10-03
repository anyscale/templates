# Data Quality Monitoring and Validation with Ray Data

**⏱️ Time to complete**: 25 min | **Difficulty**: Intermediate | **Prerequisites**: Data engineering experience, understanding of data quality concepts

## What You'll Build

Build an automated data quality monitoring system that validates data, detects anomalies, and helps ensure your data pipelines produce reliable results for data-driven organizations.

## Table of Contents

1. [Data Quality Setup](#step-1-data-quality-setup) (6 min)
2. [Quality Validation](#step-2-automated-quality-checks) (8 min)
3. [Anomaly Detection](#step-3-data-drift-monitoring) (7 min)
4. [Quality Dashboard](#step-4-quality-reporting) (4 min)

## Learning objectives

**Why data quality matters**: Poor data quality affects organizations through incorrect insights and operational problems. Understanding data quality monitoring helps enable reliable data-driven decision making.

**Ray Data's quality capabilities**: Automate quality checks across large datasets using distributed processing. You'll learn how to scale data validation from sample-based to full-dataset monitoring.

**Real-world applications**: Streaming services monitor data quality across viewing events to support content recommendations. Booking platforms validate booking records for pricing accuracy and fraud prevention. Transportation companies track data quality across trip data for safety and operational efficiency. E-commerce platforms monitor product catalog data quality to maintain customer experience and search accuracy.

## Overview

**The Challenge**: Poor data quality significantly impacts business decisions and organizational efficiency. Data quality issues can lead to incorrect insights and operational problems.

**The Solution**: Ray Data enables continuous, automated data quality monitoring with large datasets, catching issues before they impact business decisions.

**Real-world Impact**:

| Industry | Use Case | Quality Impact |
|----------|----------|----------------|
| **Financial Services** | Transaction monitoring | Prevent fraud through real-time validation |
| **E-commerce** | Product catalogs | Ensure catalog accuracy for better customer experience |
| **Healthcare** | Patient records | Validate data quality for accurate diagnosis |
| **Analytics** | Data pipelines | Ensure reliable insights through quality monitoring |

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of data quality concepts (completeness, accuracy, consistency)
- [ ] Experience with data validation and testing
- [ ] Familiarity with statistical concepts for anomaly detection
- [ ] Python environment with data processing libraries

## Quick start (3 minutes)

### Setup and Dependencies

```python
import numpy as np
import pandas as pd
import ray

# Initialize Ray for distributed processing
ray.init()

# Configure Ray Data for optimal performance monitoring
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

print("Ray initialized for data quality monitoring")
```

### Load Sample Dataset

```python
# Load pre-built customer dataset with realistic quality issues
customer_dataset = ray.data.read_parquet(
    "ecommerce_customers_with_quality_issues.parquet",
    num_cpus=0.025
)

print(f"Loaded customer dataset with quality issues:")
print(f"  Records: {customer_dataset.count():,}")
print(f"  Schema: {customer_dataset.schema()}")

quality_dataset = customer_dataset
```

**What we have:**
- 100,000+ customer records with realistic data patterns
- Intentional quality issues: missing values, invalid data, outliers, duplicates
- Pre-built Parquet dataset for optimal Ray Data performance

## Step 1: Data Quality Setup

### Schema Validation

```python
# Check data schema and types
print("Data Schema Validation:")
print(f"Dataset schema: {quality_dataset.schema()}")
print(f"Record count: {quality_dataset.count():,}")

# Sample record structure
sample_record = quality_dataset.take(1)[0]
print(f"Sample record keys: {list(sample_record.keys())}")
```

### Basic Quality Overview

```python
# Simple data quality analysis
def analyze_basic_quality(dataset):
    """Quick quality overview using Ray Data operations."""
    total_records = dataset.count()
    sample_records = dataset.take(1000)
    
    print("Basic Quality Metrics:")
    print(f"  Total records: {total_records:,}")
    print(f"  Sample analyzed: {len(sample_records):,}")
    
    return sample_records

# Analyze our dataset
sample_data = analyze_basic_quality(quality_dataset)
```

## Step 2: Automated Quality Checks

### Missing Data Analysis

```python
# Use Ray Data native operations for missing data analysis
from ray.data.expressions import col

def check_missing_values(dataset):
    """Analyze missing values using Ray Data operations."""
    sample_records = dataset.take(1000)  # Efficient sampling
    
    if not sample_records:
        return {}
    
    missing_stats = {}
    for key in sample_records[0].keys():
        missing_count = sum(1 for record in sample_records 
                          if record.get(key) is None or record.get(key) == '')
        missing_stats[key] = {
            'missing_count': missing_count,
            'missing_rate': missing_count / len(sample_records) * 100
        }
    
    return missing_stats

# Analyze missing data
missing_analysis = check_missing_values(dataset)
```

#### Missing Data Summary

| Field | Missing Count | Missing Rate | Status |
|-------|---------------|--------------|---------|
| **Email** | Sample analysis | Calculated % | High / Medium / Good |
| **Age** | Sample analysis | Calculated % | High / Medium / Good |
| **Income** | Sample analysis | Calculated % | High / Medium / Good |

### Data Quality Dashboard

```python
# Create data quality dashboard using utility function
from util.viz_utils import create_quality_dashboard, create_interactive_quality_dashboard

# Generate quality dashboard
fig = create_quality_dashboard(missing_analysis, email_validation)
print("Quality dashboard created")

# Create interactive plotly dashboard
interactive_fig = create_interactive_quality_dashboard(quality_dataset)
interactive_fig.write_html('interactive_quality_dashboard.html')
print("Interactive quality dashboard saved to 'interactive_quality_dashboard.html'")
```

**What the dashboard shows:**
- **Missing data heatmap**: Visualize missing patterns across fields and records
- **Quality scores**: Overall data quality across multiple dimensions
- **Outlier detection**: Box plots showing data distribution and outliers
- **Data freshness**: Timeline of record ingestion and processing

### Accuracy Validation

```python
# Email validation using Ray Data filtering
def validate_email_format(dataset):
    """Validate email formats using Ray Data operations."""
    
    # Use simple lambda filtering for email validation
    valid_emails = dataset.filter(
        lambda record: '@' in str(record.get('email', ''))
    )
    total_records = dataset.count()
    valid_count = valid_emails.count()
    
    return {
        'total_records': total_records,
        'valid_emails': valid_count,
        'validity_rate': (valid_count / total_records * 100) if total_records > 0 else 0
    }

# Run email validation
email_validation = validate_email_format(dataset)
print(f"Email validation: {email_validation['validity_rate']:.1f}% valid formats")
```

## Step 3: Data Drift Monitoring

### Statistical Analysis with Native Ray Data

```python
# Use Ray Data native aggregations for statistical analysis
from ray.data.aggregate import Count, Mean, Std, Min, Max

# Calculate statistics for numeric columns
try:
    age_stats = dataset.aggregate(
        Count(),
        Mean('age'),
        Std('age'),
        Min('age'),
        Max('age')
    )
    print("Age Statistics:")
    print(f"  Count: {age_stats['count()']:,}")
    print(f"  Mean: {age_stats['mean(age)']:.1f}")
    print(f"  Std Dev: {age_stats['std(age)']:.1f}")
except Exception as e:
    print(f"Age statistics calculation: {e}")
```

### Anomaly Detection Using Statistical Methods

```python
def detect_statistical_anomalies(dataset):
    """Detect statistical anomalies using Ray Data operations."""
    
    # Calculate statistical thresholds
    sample_records = dataset.take(1000)
    
    if not sample_records:
        return {}
    
    # Analyze numeric fields
    numeric_fields = ['age', 'income', 'score']
    anomaly_results = {}
    
    for field in numeric_fields:
        values = [r.get(field) for r in sample_records if r.get(field) is not None]
        
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Calculate Z-scores for outlier detection
            outliers = sum(1 for v in values if abs(v - mean_val) > 3 * std_val)
            
            anomaly_results[field] = {
                'mean': mean_val,
                'std': std_val,
                'outliers': outliers,
                'outlier_rate': (outliers / len(values) * 100) if values else 0
            }
    
    return anomaly_results

# Detect anomalies
anomalies = detect_statistical_anomalies(quality_dataset)

print("Statistical Anomaly Detection Results:")
for field, stats in anomalies.items():
    print(f"  {field}: {stats['outliers']} outliers ({stats['outlier_rate']:.2f}%)")
```

### Data Drift Detection Over Time

```python
def detect_data_drift(current_dataset, historical_stats):
    """Detect data drift by comparing current vs historical distributions."""
    
    current_sample = current_dataset.take(1000)
    
    drift_results = {}
    
    # Compare distributions
    for field in ['age', 'income']:
        current_values = [r.get(field) for r in current_sample if r.get(field) is not None]
        
        if current_values:
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)
            
            # Compare with historical baselines
            historical_mean = historical_stats.get(f'{field}_mean', current_mean)
            historical_std = historical_stats.get(f'{field}_std', current_std)
            
            mean_drift = abs(current_mean - historical_mean) / historical_mean * 100
            std_drift = abs(current_std - historical_std) / historical_std * 100
            
            drift_status = "High" if mean_drift > 15 else "Medium" if mean_drift > 5 else "Low"
            
            drift_results[field] = {
                'mean_drift_pct': mean_drift,
                'std_drift_pct': std_drift,
                'status': drift_status
            }
    
    return drift_results

# Example historical statistics (in production, load from storage)
historical_stats = {
    'age_mean': 45.0,
    'age_std': 15.0,
    'income_mean': 65000,
    'income_std': 25000
}

# Detect drift
drift_analysis = detect_data_drift(quality_dataset, historical_stats)

print("Data Drift Analysis:")
for field, drift in drift_analysis.items():
    print(f"  {field}: {drift['mean_drift_pct']:.2f}% drift - Status: {drift['status']}")
```

### Business Rule Validation

```python
def validate_business_rules(dataset):
    """Validate business logic rules using Ray Data filtering."""
    from ray.data.expressions import col, lit
    
    total_records = dataset.count()
    validation_results = {}
    
    # Rule 1: Age should be between 0 and 120
    valid_age = dataset.filter(
        (col("age") >= lit(0)) & (col("age") <= lit(120))
    ).count()
    validation_results['valid_age_range'] = {
        'valid_count': valid_age,
        'valid_rate': (valid_age / total_records * 100) if total_records > 0 else 0,
        'rule': '0 <= age <= 120'
    }
    
    # Rule 2: Income should be positive
    valid_income = dataset.filter(
        col("income") > lit(0)
    ).count()
    validation_results['valid_income'] = {
        'valid_count': valid_income,
        'valid_rate': (valid_income / total_records * 100) if total_records > 0 else 0,
        'rule': 'income > 0'
    }
    
    return validation_results

# Run business rule validation
business_validation = validate_business_rules(quality_dataset)

print("Business Rule Validation Results:")
for rule_name, result in business_validation.items():
    print(f"  {rule_name}: {result['valid_rate']:.1f}% valid - {result['rule']}")
```

#### Quality Statistics Summary

| Metric | Age | Income | Score |
|--------|-----|--------|-------|
| **Count** | Native Ray Data aggregation | Native calculation | Native calculation |
| **Mean** | Distributed processing | Distributed processing | Distributed processing |
| **Std Dev** | Scalable statistics | Scalable statistics | Scalable statistics |

## Step 4: Quality Reporting

### Generate Quality Report

```python
# Create comprehensive quality report
def generate_quality_report(dataset, missing_stats, email_validation):
    """Generate a comprehensive quality report."""
    total_records = dataset.count()
    
    print("="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    print(f"Dataset size: {total_records:,} records")
    print(f"Email validity: {email_validation['validity_rate']:.1f}%")
    
    # Missing data summary
    print("\nMissing Data Summary:")
    for field, stats in missing_stats.items():
        status = "High" if stats['missing_rate'] > 10 else "Medium" if stats['missing_rate'] > 5 else "Good"
        print(f"  {field}: {stats['missing_rate']:.1f}% missing {status}")
    
    print("="*60)

# Generate final report
generate_quality_report(dataset, missing_analysis, email_validation)
```

### Quality Score Calculation

```python
# Calculate overall quality score using native operations
def calculate_quality_score_native(dataset):
    """Calculate quality score using Ray Data operations."""
    total_records = dataset.count()
    
    # Find records with complete critical fields using lambda filtering
    complete_records = dataset.filter(
        lambda record: (
            record.get('email') is not None and 
            record.get('age') is not None and
            str(record.get('email', '')) != '' and
            str(record.get('email', '')) != 'None'
        )
    ).count()
    
    completeness_score = complete_records / total_records if total_records > 0 else 0
    
    quality_result = {
        "total_records": total_records,
        "complete_records": complete_records,
        "completeness_score": completeness_score,
        "quality_grade": "A" if completeness_score > 0.9 else "B" if completeness_score > 0.7 else "C"
    }
    
    return quality_result

# Calculate quality score
overall_quality = calculate_quality_score_native(dataset)
print(f"Overall Quality: {overall_quality['completeness_score']:.1%} (Grade: {overall_quality['quality_grade']})")
```

## Key Takeaways

### Ray Data Quality Advantages

| Traditional Approach | Ray Data Approach | Key Benefit |
|---------------------|-------------------|-------------|
| **Batch validation** | Continuous monitoring | Real-time insights |
| **Single-machine** | Distributed processing | Horizontal scaling |
| **Manual rules** | Automated detection | Streamlined development |
| **Point-in-time** | Historical tracking | Comprehensive management |

### Quality Framework

:::tip Data quality pillars
The template implements six key quality dimensions:
- **Completeness** (25%) - Missing value detection
- **Accuracy** (25%) - Format and range validation  
- **Consistency** (20%) - Schema compliance
- **Timeliness** (15%) - Freshness monitoring
- **Validity** (10%) - Business rule validation
- **Uniqueness** (5%) - Duplicate detection
:::

## Action Items

### Immediate Implementation
- [ ] Set up automated quality monitoring for your datasets
- [ ] Define business-specific validation rules
- [ ] Implement quality score calculations
- [ ] Create quality dashboards and alerts

### Advanced Features
- [ ] Add statistical anomaly detection
- [ ] Implement data drift monitoring
- [ ] Build quality trend analysis
- [ ] Create automated quality improvement recommendations

## Related Templates

### Recommended Next Steps
- **[enterprise-data-catalog](../ray-data-enterprise-data-catalog/)**: Extend quality monitoring with automated data discovery
- **[etl-optimization](../ray-data-etl-optimization/)**: Apply quality checks within ETL pipelines
- **[log-ingestion](../ray-data-log-ingestion/)**: Monitor data pipeline logs for quality issues

### Advanced Applications
- **[financial-forecasting](../ray-data-financial-forecasting/)**: Apply quality monitoring to financial time series data
- **[medical-connectors](../ray-data-medical-connectors/)**: Implement HIPAA-compliant data quality validation

## Resources

- [Ray Data Documentation](https://docs.ray.io/en/latest/data/index.html)
- [Data Quality Best Practices](https://docs.ray.io/en/latest/data/best-practices.html)
- [Ray Security Documentation](https://docs.ray.io/en/latest/ray-security.html)

## Cleanup

```python
# Clean up Ray resources
if ray.is_initialized():
    ray.shutdown()
    print("Ray cluster shutdown complete")
```

---

*This template provides a foundation for building production-ready data quality monitoring pipelines with Ray Data. Start with basic validation and gradually add complexity based on your specific requirements.*
