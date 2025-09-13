# Data Quality Monitoring with Ray Data

**Time to complete**: 25 min | **Difficulty**: Intermediate | **Prerequisites**: Data analysis experience, understanding of data quality concepts

## What You'll Build

Create an automated data quality monitoring system that continuously validates data, detects anomalies, and ensures your data pipelines produce reliable, trustworthy results - essential for any data-driven organization.

## Table of Contents

1. [Data Quality Setup](#step-1-creating-test-data) (6 min)
2. [Quality Validation](#step-2-automated-quality-checks) (8 min)
3. [Anomaly Detection](#step-3-data-drift-monitoring) (7 min)
4. [Quality Dashboard](#step-4-quality-reporting) (4 min)

## Learning Objectives

By completing this tutorial, you'll understand:

- **Why data quality matters**: How poor data quality costs organizations millions annually
- **Ray Data's quality capabilities**: Automate quality checks across large datasets using distributed processing
- **Real-world applications**: How companies like Netflix and Airbnb ensure data reliability at scale
- **Quality frameworks**: Implement comprehensive data validation and monitoring systems

## Overview

**The Challenge**: Poor data quality significantly impacts business decisions and organizational efficiency. Data quality issues can lead to incorrect insights and operational problems.

**The Solution**: Ray Data enables continuous, automated data quality monitoring at scale, catching issues before they impact business decisions.

**Real-world Impact**:
- **Financial Services**: Banks prevent fraud by monitoring transaction data quality in real-time
- **E-commerce**: Retailers ensure product catalog accuracy for better customer experience
- **Healthcare**: Hospitals validate patient data quality for accurate diagnosis and treatment
- **Analytics**: Data teams ensure reliable insights by monitoring data pipeline quality

---

## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of data quality concepts (completeness, accuracy, consistency)
- [ ] Experience with data validation and testing
- [ ] Familiarity with statistical concepts for anomaly detection
- [ ] Python environment with data processing libraries

## Quick Start (3 minutes)

Want to see data quality monitoring immediately? This section demonstrates core data quality concepts in just a few minutes.

### Setup and Dependencies

```python
import ray
import numpy as np
import pandas as pd
import time

# Initialize Ray for distributed processing
ray.init()
```

### Create Sample Dataset with Quality Issues

We'll create a realistic dataset that contains common data quality issues found in real-world data.

```python
# Set up data generation parameters
print("Creating sample dataset with quality issues...")
np.random.seed(42)  # For reproducible results

# Define data quality issue rates (realistic percentages)
MISSING_AGE_RATE = 0.10      # 10% missing ages
INVALID_INCOME_RATE = 0.05   # 5% invalid income values
INVALID_EMAIL_RATE = 0.08    # 8% invalid email formats
MISSING_CATEGORY_RATE = 0.12 # 12% missing categories
OUTLIER_SCORE_RATE = 0.03    # 3% score outliers
```

```python
# Generate realistic customer data with quality issues
data = []
for i in range(10000):  # 10K records for meaningful analysis
    record = {
        "customer_id": f"CUST_{i:05d}",
        "age": np.random.randint(18, 80) if np.random.random() > MISSING_AGE_RATE else None,
        "income": np.random.normal(50000, 20000) if np.random.random() > INVALID_INCOME_RATE else -1,
        "email": f"user{i}@example.com" if np.random.random() > INVALID_EMAIL_RATE else "invalid_email",
        "category": np.random.choice(["A", "B", "C"]) if np.random.random() > MISSING_CATEGORY_RATE else None,
        "score": np.random.uniform(0, 100) if np.random.random() > OUTLIER_SCORE_RATE else 999,
        "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
    }
    data.append(record)

print(f"Generated {len(data):,} customer records with intentional quality issues")
```

```python
# Create Ray Dataset using native from_items operation
ds = ray.data.from_items(data)
print(f"Created Ray Dataset with {ds.count():,} records for quality monitoring")
```

**What we created:**
- 10,000 customer records with realistic data patterns
- Intentional quality issues: missing values, invalid data, outliers  
- Ray Dataset ready for distributed quality analysis

### Quick Quality Assessment

```python
# Quick data quality assessment
sample_data = ds.take(5)

print("Sample Data Quality Preview:")
print("=" * 100)
print(f"{'Customer ID':<12} {'Age':<5} {'Income':<10} {'Email':<20} {'Category':<10} {'Score':<8}")
print("-" * 100)

for record in sample_data:
    age = str(record.get('age', 'NULL'))
    income = f"${record.get('income', 0):,.0f}" if record.get('income', 0) > 0 else "INVALID"
    email = record.get('email', 'NULL')[:18] + "..." if len(record.get('email', '')) > 20 else record.get('email', 'NULL')
    category = record.get('category', 'NULL')
    score = f"{record.get('score', 0):.1f}"
    
    print(f"{record['customer_id']:<12} {age:<5} {income:<10} {email:<20} {category:<10} {score:<8}")

print("-" * 100)
print("Notice: Some records have missing (NULL) or invalid values - this is intentional for demonstration")
```

### Import Visualization Libraries

```python
# Import visualization libraries for quality dashboards
import matplotlib.pyplot as plt
import seaborn as sns

# Set up plotting style for professional visualizations
plt.style.use('default')
sns.set_palette("husl")

print("Visualization libraries loaded - ready for quality analysis!")
```

## Why Data Quality Monitoring is Critical

**The Cost of Poor Data Quality**:
- **Business Impact**: 1 in 3 business leaders don't trust their data for decision-making
- **Financial Loss**: Organizations lose $12.9M annually on average due to bad data
- **Operational Issues**: 40% of business decisions use stale or outdated data
- **Customer Impact**: Poor data quality leads to bad recommendations and customer churn

**The Scale of the Problem:**
- **Data Volume Growth**: Enterprise data grows 40-60% annually
- **Source Proliferation**: Average enterprise has 400+ data sources
- **Quality Degradation**: Data quality degrades 2% monthly without active monitoring
- **Business Impact**: 1 in 3 business leaders don't trust their data for decision-making

**Common Data Quality Issues:**
- **Completeness**: 15-25% of enterprise data has missing values
- **Accuracy**: 10-20% of data contains errors or inconsistencies
- **Consistency**: Schema changes break 30% of downstream applications
- **Timeliness**: 40% of business decisions use stale or outdated data

### **Ray Data's Data Quality Advantages**

Ray Data revolutionizes data quality monitoring by providing:

| Traditional Approach | Ray Data Approach | Key Difference |
|---------------------|-------------------|----------------|
| **Batch quality checks** | Continuous monitoring | Real-time quality insights |
| **Single-machine validation** | Distributed validation | Horizontal scalability |
| **Manual rule creation** | Automated pattern detection | Streamlined rule development |
| **Point-in-time analysis** | Historical trend tracking | Comprehensive quality management |
| **Siloed quality tools** | Integrated data pipeline | Unified data operations |

### **Enterprise Data Quality Framework**

This template implements a comprehensive data quality framework based on industry best practices:

**The Six Pillars of Data Quality:**

1. **Completeness** (25% of quality score)
   - Missing value detection and analysis
   - Coverage assessment across data sources
   - Null pattern identification and trends

2. **Accuracy** (25% of quality score)
   - Business rule validation and enforcement
   - Format and range validation
   - Cross-reference verification

3. **Consistency** (20% of quality score)
   - Schema compliance monitoring
   - Data type validation
   - Referential integrity checks

4. **Timeliness** (15% of quality score)
   - Data freshness monitoring
   - Update frequency analysis
   - Staleness detection and alerting

5. **Validity** (10% of quality score)
   - Domain-specific validation rules
   - Constraint checking
   - Business logic compliance

6. **Uniqueness** (5% of quality score)
   - Duplicate detection and analysis
   - Primary key validation
   - Record deduplication

### **Business Impact and ROI**

Organizations implementing comprehensive data quality monitoring see:

| Quality Aspect | Traditional Approach | Ray Data Approach | Key Benefit |
|---------------|---------------------|-------------------|-------------|
| **Monitoring Scope** | Limited sample checking | Comprehensive full-dataset analysis | Complete visibility |
| **Detection Method** | Manual rule checking | Automated pattern detection | Systematic identification |
| **Processing Scale** | Single-machine analysis | Distributed processing | Horizontal scalability |
| **Issue Response** | Reactive detection | Proactive monitoring | Earlier intervention |
| **Resource Focus** | Manual quality tasks | Automated workflows | Strategic optimization |

## Learning Objectives

By the end of this template, you'll understand:
- How to implement automated data quality checks
- Schema validation and business rule enforcement
- Statistical anomaly detection and monitoring
- Building scalable data quality pipelines
- Integration with monitoring and alerting systems

## Use Case: Enterprise Data Quality Monitoring

We'll build a pipeline that monitors:
- **Data Completeness**: Missing values, null rates, data coverage
- **Data Accuracy**: Value validation, range checks, format compliance
- **Data Consistency**: Schema compliance, data type validation
- **Data Freshness**: Timeliness, update frequency, staleness detection
- **Data Integrity**: Referential integrity, constraint validation

## Architecture

```
Data Sources → Ray Data → Quality Checks → Validation Engine → Monitoring → Alerts
     ↓           ↓           ↓              ↓                ↓          ↓
  Databases   Parallel    Schema Check    Business Rules   Metrics    Notifications
  Files       Processing  Completeness    Anomaly Detection  Scoring   Dashboards
  APIs        GPU Workers  Accuracy       Drift Detection   Reports   APIs
  Streams     Validation   Consistency    Integrity Check   Trends    Actions
```

## Key Components

### 1. **Data Quality Checks**
- Schema validation and type checking
- Completeness and coverage analysis
- Accuracy and range validation
- Consistency and integrity verification

### 2. **Statistical Monitoring**
- Distribution analysis and drift detection
- Outlier detection and anomaly identification
- Trend analysis and pattern recognition
- Statistical significance testing

### 3. **Business Rule Engine**
- Custom validation rules and constraints
- Domain-specific quality requirements
- Regulatory compliance checking
- Automated rule generation and testing

### 4. **Quality Scoring and Reporting**
- Comprehensive quality metrics
- Trend analysis and historical tracking
- Automated alerting and notifications
- Quality improvement recommendations

## Prerequisites

- Ray cluster with data processing capabilities
- Python 3.8+ with data quality libraries
- Access to data sources for monitoring
- Basic understanding of data quality concepts

## Installation

```bash
pip install ray[data] pandas numpy great-expectations
pip install scikit-learn scipy statsmodels
pip install plotly dash streamlit
pip install pyarrow boto3
```

## Quick Start

### 1. **Load Real Enterprise Data for Quality Monitoring**

```python
import ray
from ray.data import read_parquet, read_csv
import pandas as pd

# Initialize Ray
ray.init()

# Load real enterprise datasets from public sources
# NYC Taxi data - publicly available
taxi_data = read_parquet("s3://anonymous@nyc-tlc/trip_data/yellow_tripdata_2023-01.parquet")

# US Government spending data - publicly available
spending_data = read_csv("s3://anonymous@usaspending-gov/download_center/Custom_Account_Data.csv")

# Public company financial data - publicly available
financial_data = read_parquet("s3://anonymous@sec-edgar/financial_statements/2023/")

# Healthcare provider data - publicly available (CMS)
healthcare_data = read_csv("s3://anonymous@cms-gov/provider-data/Physician_Compare_National_Downloadable_File.csv")

print(f"Taxi data: {taxi_data.count()}")
print(f"Spending data: {spending_data.count()}")
print(f"Financial data: {financial_data.count()}")
print(f"Healthcare data: {healthcare_data.count()}")
```

### 2. **Schema Validation**

```python
from typing import Dict, Any, List
import json

class SchemaValidator:
    """Validate data schema and structure."""
    
    def __init__(self, expected_schema: Dict[str, Any]):
        self.expected_schema = expected_schema
    
    def __call__(self, batch):
        """Validate schema for a batch of data."""
        validation_results = []
        
        for item in batch:
            try:
                # Check required columns
                missing_columns = []
                extra_columns = []
                
                for expected_col in self.expected_schema["required_columns"]:
                    if expected_col not in item:
                        missing_columns.append(expected_col)
                
                for actual_col in item.keys():
                    if actual_col not in self.expected_schema["allowed_columns"]:
                        extra_columns.append(actual_col)
                
                # Check data types
                type_violations = []
                for col, expected_type in self.expected_schema["column_types"].items():
                    if col in item:
                        actual_value = item[col]
                        if not self._check_type(actual_value, expected_type):
                            type_violations.append({
                                "column": col,
                                "expected_type": expected_type,
                                "actual_value": str(actual_value)[:100]
                            })
                
                # Create validation result
                validation_result = {
                    "record_id": item.get("id", "unknown"),
                    "schema_valid": len(missing_columns) == 0 and len(type_violations) == 0,
                    "missing_columns": missing_columns,
                    "extra_columns": extra_columns,
                    "type_violations": type_violations,
                    "validation_timestamp": pd.Timestamp.now().isoformat()
                }
                
                validation_results.append(validation_result)
                
            except Exception as e:
                validation_results.append({
                    "record_id": item.get("id", "unknown"),
                    "schema_valid": False,
                    "error": str(e),
                    "validation_timestamp": pd.Timestamp.now().isoformat()
                })
        
        return {"schema_validation": validation_results}
    
    def _check_type(self, value, expected_type):
        """Check if value matches expected type."""
        try:
            if expected_type == "string":
                return isinstance(value, str)
            elif expected_type == "integer":
                return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            elif expected_type == "float":
                return isinstance(value, (int, float))
            elif expected_type == "boolean":
                return isinstance(value, bool)
            elif expected_type == "datetime":
                return pd.api.types.is_datetime64_any_dtype(value) or isinstance(value, pd.Timestamp)
            else:
                return True  # Unknown type, assume valid
        except:
            return False

# Define expected schema
customer_schema = {
    "required_columns": ["customer_id", "name", "email", "registration_date"],
    "allowed_columns": ["customer_id", "name", "email", "registration_date", "phone", "address"],
    "column_types": {
        "customer_id": "string",
        "name": "string",
        "email": "string",
        "registration_date": "datetime",
        "phone": "string",
        "address": "string"
    }
}

# Apply schema validation
schema_validation = customer_data.map_batches(
    SchemaValidator(customer_schema),
    batch_size=1000,
    concurrency=4
)
```

### 3. **Data Completeness Analysis**

```python
class CompletenessAnalyzer:
    """Analyze data completeness and missing value patterns."""
    
    def __init__(self):
        self.completeness_metrics = {}
    
    def __call__(self, batch):
        """Analyze completeness for a batch of data."""
        if not batch:
            return {"completeness_analysis": {}}
        
        # Convert batch to DataFrame for easier analysis
        df = pd.DataFrame(batch)
        
        # Calculate completeness metrics
        total_records = len(df)
        completeness_metrics = {}
        
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            null_count = df[column].isna().sum()
            
            completeness_metrics[column] = {
                "total_records": total_records,
                "non_null_count": int(non_null_count),
                "null_count": int(null_count),
                "completeness_rate": float(non_null_count / total_records),
                "missing_rate": float(null_count / total_records)
            }
        
        # Calculate overall completeness
        overall_completeness = np.mean([metrics["completeness_rate"] for metrics in completeness_metrics.values()])
        
        return {
            "completeness_analysis": {
                "overall_completeness": overall_completeness,
                "column_metrics": completeness_metrics,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
        }

# Apply completeness analysis
completeness_analysis = customer_data.map_batches(
    CompletenessAnalyzer(),
    batch_size=1000,
    concurrency=2
)
```

### 4. **Data Accuracy Validation**

```python
import re
from datetime import datetime

class AccuracyValidator:
    """Validate data accuracy and business rules."""
    
    def __init__(self):
        self.validation_rules = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?\d{9,15}$",
            "customer_id": r"^CUST\d{6}$"
        }
    
    def __call__(self, batch):
        """Validate data accuracy for a batch."""
        accuracy_results = []
        
        for item in batch:
            try:
                validation_errors = []
                
                # Email validation
                if "email" in item and item["email"]:
                    if not re.match(self.validation_rules["email"], str(item["email"])):
                        validation_errors.append("Invalid email format")
                
                # Phone validation
                if "phone" in item and item["phone"]:
                    if not re.match(self.validation_rules["phone"], str(item["phone"])):
                        validation_errors.append("Invalid phone format")
                
                # Customer ID validation
                if "customer_id" in item and item["customer_id"]:
                    if not re.match(self.validation_rules["customer_id"], str(item["customer_id"])):
                        validation_errors.append("Invalid customer ID format")
                
                # Date validation
                if "registration_date" in item and item["registration_date"]:
                    try:
                        registration_date = pd.to_datetime(item["registration_date"])
                        if registration_date > pd.Timestamp.now():
                            validation_errors.append("Registration date cannot be in the future")
                    except:
                        validation_errors.append("Invalid date format")
                
                # Business rule: Customer ID must be unique (simplified check)
                # In production, you'd check against a reference dataset
                
                accuracy_result = {
                    "record_id": item.get("id", "unknown"),
                    "is_accurate": len(validation_errors) == 0,
                    "validation_errors": validation_errors,
                    "validation_timestamp": pd.Timestamp.now().isoformat()
                }
                
                accuracy_results.append(accuracy_result)
                
            except Exception as e:
                accuracy_results.append({
                    "record_id": item.get("id", "unknown"),
                    "is_accurate": False,
                    "error": str(e),
                    "validation_timestamp": pd.Timestamp.now().isoformat()
                })
        
        return {"accuracy_validation": accuracy_results}

# Apply accuracy validation
accuracy_validation = customer_data.map_batches(
    AccuracyValidator(),
    batch_size=1000,
    concurrency=4
)
```

### 5. **Statistical Anomaly Detection**

```python
from scipy import stats
import numpy as np

class AnomalyDetector:
    """Detect statistical anomalies in data."""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.statistical_metrics = {}
    
    def __call__(self, batch):
        """Detect anomalies in a batch of data."""
        if not batch:
            return {"anomaly_detection": {}}
        
        # Convert batch to DataFrame
        df = pd.DataFrame(batch)
        
        anomaly_results = {}
        
        for column in df.select_dtypes(include=[np.number]).columns:
            values = df[column].dropna()
            
            if len(values) > 0:
                # Calculate statistical metrics
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Detect outliers using Z-score
                z_scores = np.abs(stats.zscore(values))
                outliers = values[z_scores > self.threshold]
                
                # Calculate outlier percentage
                outlier_percentage = len(outliers) / len(values) * 100
                
                anomaly_results[column] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "outlier_count": int(len(outliers)),
                    "outlier_percentage": float(outlier_percentage),
                    "is_anomalous": outlier_percentage > 5.0  # More than 5% outliers
                }
        
        return {
            "anomaly_detection": {
                "statistical_metrics": anomaly_results,
                "detection_timestamp": pd.Timestamp.now().isoformat()
            }
        }

# Apply anomaly detection
anomaly_detection = customer_data.map_batches(
    AnomalyDetector(threshold=2.5),
    batch_size=1000,
    concurrency=2
)
```

### 6. **Data Quality Scoring**

```python
class QualityScorer:
    """Calculate comprehensive data quality scores."""
    
    def __init__(self):
        self.quality_weights = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "consistency": 0.2,
            "timeliness": 0.2
        }
    
    def __call__(self, batch):
        """Calculate quality scores for a batch."""
        # This would typically combine results from multiple validation steps
        # For demonstration, we'll create sample quality scores
        
        quality_scores = {
            "overall_quality_score": 0.85,
            "completeness_score": 0.92,
            "accuracy_score": 0.78,
            "consistency_score": 0.88,
            "timeliness_score": 0.82,
            "quality_grade": "B",
            "recommendations": [
                "Improve email format validation",
                "Reduce missing phone numbers",
                "Standardize customer ID format"
            ],
            "scoring_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return {"quality_scoring": quality_scores}

# Apply quality scoring
quality_scoring = customer_data.map_batches(
    QualityScorer(),
    batch_size=1000,
    concurrency=2
)
```

## Advanced Features

### **Ray Data Fault Tolerance**
- **Automatic Task Retries**: Failed validation tasks are automatically retried
- **Worker Failure Recovery**: Ray Data reschedules tasks when workers fail
- **Data Block Replication**: Critical data blocks are replicated for reliability
- **RayTurbo Checkpointing**: Job-level checkpointing on Anyscale for long-running quality jobs

```python
# Example: Fault-tolerant data quality pipeline
from ray.data.context import DataContext

# Configure fault tolerance
ctx = DataContext.get_current()
ctx.enable_auto_log_stats = True

# Ray Data automatically handles:
# - Task failures and retries
# - Worker node failures
# - Memory pressure recovery
# - Network interruptions

# On Anyscale with RayTurbo:
# - Job-level checkpointing
# - Automatic job recovery
# - State preservation across failures
```

### **Data Drift Detection**
- Statistical drift detection methods
- Distribution comparison techniques
- Automated drift monitoring
- Alert generation for significant changes

### **Custom Validation Rules**
- Domain-specific business rules
- Regulatory compliance checking
- Automated rule generation
- Rule performance monitoring

### **Quality Trend Analysis**
- Historical quality tracking
- Trend identification and forecasting
- Quality improvement recommendations
- Automated reporting and dashboards

## Production Considerations

### **Performance Optimization**
- Efficient validation algorithms
- Parallel processing strategies
- Caching and incremental updates
- Resource optimization

### **Scalability**
- Horizontal scaling across nodes
- Load balancing for validation workloads
- Distributed rule processing
- Efficient data partitioning

### **Monitoring and Alerting**
- Real-time quality monitoring
- Automated alert generation
- Escalation procedures
- Performance tracking

## Example Workflows

### **Customer Data Quality Monitoring**
1. Load customer data from multiple sources
2. Validate schema and data types
3. Check completeness and accuracy
4. Detect anomalies and outliers
5. Generate quality reports and alerts

### **Financial Data Validation**
1. Process transaction and financial data
2. Validate business rules and constraints
3. Check for fraud indicators
4. Monitor data consistency
5. Generate compliance reports

### **Product Data Quality**
1. Validate product catalog data
2. Check pricing and inventory accuracy
3. Monitor data freshness
4. Detect duplicate and invalid entries
5. Generate quality improvement recommendations

## Performance Analysis

### **Data Quality Assessment Framework**

| Quality Dimension | Validation Method | Measurement Output | Visualization |
|------------------|-------------------|-------------------|---------------|
| **Completeness** | Missing value analysis | Completeness scores | Heatmaps |
| **Accuracy** | Business rule validation | Error rates | Error distribution |
| **Consistency** | Schema compliance | Violation counts | Compliance charts |
| **Timeliness** | Freshness analysis | Age metrics | Freshness trends |

### **Quality Scoring Methodology**

```
Data Quality Score Calculation:
┌─────────────────┐
│ Raw Data Input  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Schema          │    │ Completeness    │    │ Accuracy        │
│ Validation      │    │ Analysis        │    │ Validation      │
│ (30% weight)    │    │ (25% weight)    │    │ (25% weight)    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
         ┌─────────────────┐    │    ┌─────────────────┐
         │ Consistency     │    │    │ Statistical     │
         │ Checks          │    ▼    │ Anomaly         │
         │ (10% weight)    │    │    │ Detection       │
         └────────┬────────┘    │    │ (10% weight)    │
                  │             │    └────────┬────────┘
                  └─────────────┼─────────────┘
                                ▼
                      ┌─────────────────┐
                      │ Overall Quality │
                      │ Score (0-100)   │
                      └─────────────────┘
```

### **Quality Monitoring Dashboard**

The template generates comprehensive quality monitoring visualizations:

| Dashboard Component | Chart Type | File Output |
|-------------------|------------|-------------|
| **Quality Scores** | Gauge charts | `quality_dashboard.html` |
| **Trend Analysis** | Time series | `quality_trends.html` |
| **Issue Distribution** | Bar charts | `quality_issues.html` |
| **Data Profiling** | Statistical summaries | `data_profile.html` |

### **Expected Quality Metrics Output**

```
Data Quality Report:
┌─────────────────────────────────────────┐
│ Overall Quality Score: [Calculated]     │
├─────────────────────────────────────────┤
│ Completeness Score: [%]                 │
│ ├─ Missing Values: [Count]              │
│ ├─ Null Percentage: [%]                 │
│ └─ Coverage Rate: [%]                   │
├─────────────────────────────────────────┤
│ Accuracy Score: [%]                     │
│ ├─ Format Violations: [Count]           │
│ ├─ Range Violations: [Count]            │
│ └─ Business Rule Failures: [Count]      │
├─────────────────────────────────────────┤
│ Consistency Score: [%]                  │
│ ├─ Schema Violations: [Count]           │
│ ├─ Type Mismatches: [Count]             │
│ └─ Constraint Failures: [Count]         │
└─────────────────────────────────────────┘
```

## Troubleshooting

### **Common Issues**
1. **Performance Issues**: Optimize validation rules and batch sizes
2. **Memory Issues**: Reduce batch size or optimize data structures
3. **False Positives**: Adjust validation thresholds and rules
4. **Scalability**: Optimize data partitioning and resource allocation

### **Debug Mode**
Enable detailed logging and validation debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable validation debugging
import warnings
warnings.filterwarnings("ignore")
```

## Next Steps

1. **Customize Rules**: Implement domain-specific validation rules
2. **Define Metrics**: Create quality metrics relevant to your use case
3. **Build Dashboards**: Create monitoring and alerting systems
4. **Scale Production**: Deploy to multi-node clusters

### **Security Considerations** (rule #656)

**Data Privacy and Security**:
- Use encrypted connections when accessing external data sources
- Implement proper authentication for data access
- Consider data anonymization for sensitive datasets
- Follow data retention policies and compliance requirements

**Ray Cluster Security**:
```python
# Example: Initialize Ray with security considerations
ray.init(
    dashboard_port=None,  # Disable dashboard for security
    include_dashboard=False,
    _temp_dir="/secure/temp/path"  # Use secure temporary directory
)
```

## Resources

- [Ray Data Documentation](https://docs.ray.io/en/latest/data/index.html)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [pandas Data Validation](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Data Quality Best Practices](https://docs.ray.io/en/latest/data/best-practices.html)
- [Ray Security Documentation](https://docs.ray.io/en/latest/ray-security.html)

---

*This template provides a foundation for building production-ready data quality monitoring pipelines with Ray Data. Start with the basic examples and gradually add complexity based on your specific data quality requirements.*
