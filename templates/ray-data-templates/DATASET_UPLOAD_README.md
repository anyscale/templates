# Ray Data Template Dataset Upload Guide

This directory contains scripts and data files needed to populate the `s3://ray-benchmark-data` bucket with all datasets required by the Ray Data templates.

## Overview

The Ray Data templates reference datasets stored in `s3://ray-benchmark-data/` across various paths. This upload script handles:

1. **Existing local files**: Converts and uploads parquet files to required formats
2. **External datasets**: Downloads and processes public datasets (Titanic, S&P 500, etc.)
3. **Synthetic datasets**: Creates realistic synthetic data where needed

## Quick Start

### Prerequisites

```bash
# Configure AWS credentials for the ray-benchmark-data bucket
aws configure
# OR set environment variables:
# export AWS_ACCESS_KEY_ID=your_key
# export AWS_SECRET_ACCESS_KEY=your_secret
```

### Run Upload

**Option 1: Real Kaggle Datasets (Recommended)**
```bash
# Install Kaggle dependencies
pip install -r requirements_kaggle.txt

# Configure Kaggle API
kaggle configure  # Follow prompts to add API token

# Download real datasets from Kaggle and public sources
python download_real_datasets.py
```

**Option 2: Simplified Upload (if Kaggle not available)**
```bash
# Install minimal dependencies
pip install -r requirements_simple.txt

# Upload core datasets (avoids pandas/numpy issues)
python upload_benchmark_data_simple.py
```

**Option 3: Full Upload (if pandas/numpy work)**
```bash
# Install all dependencies
pip install -r requirements_upload.txt

# Upload all datasets including financial and TPC-H
python upload_benchmark_data.py
```

**For Kaggle dataset download:**
```bash
# After running download_real_datasets.py, use the generated scripts:
bash downloads/download_all_real_datasets.sh
```

## Real Dataset Sources

### Kaggle Datasets (Recommended)
| Dataset | Kaggle Source | Template Usage | S3 Path |
|---------|---------------|----------------|---------|
| **Titanic** | `titanic` competition | ML Feature Engineering | `ml-datasets/titanic.csv` |
| **Amazon Reviews** | `snap/amazon-fine-food-reviews` | NLP Text Analytics | `nlp/amazon-reviews.csv` |
| **Brazilian E-Commerce** | `olistbr/brazilian-ecommerce` | Data Quality, Catalog | `catalog/customer_data.csv` |
| **Credit Card Fraud** | `mlg-ulb/creditcardfraud` | Financial Analytics | `financial/credit-card-fraud.csv` |
| **Twitter Sentiment** | `kazanova/sentiment140` | NLP Text Analytics | `nlp/twitter-sentiment.csv` |

### Public API Sources
| Dataset | Source | Template Usage | S3 Path |
|---------|--------|----------------|---------|
| **S&P 500 Companies** | GitHub datasets | Financial Forecasting | `financial/sp500_companies.csv` |
| **S&P 500 Prices** | GitHub datasets | Financial Forecasting | `financial/sp500_daily_2years.csv` |
| **NYC Taxi Data** | NYC.gov | Geospatial Analysis | `nyc-taxi/yellow_tripdata_2023-01.parquet` |
| **Economic Data** | FRED API | Financial Forecasting | `financial/economic_indicators.csv` |

## Dataset Mapping

### Financial Data
- **Source**: Real S&P 500 data from GitHub datasets + Yahoo Finance API
- **S3 Path**: `s3://ray-benchmark-data/financial/sp500_daily_2years.csv` (parquet preferred)
- **Used by**: `ray-data-financial-forecasting`
- **Format**: CSV/Parquet with columns: symbol, date, open, high, low, close, volume
- **Real Data**: ✅ 5 years of actual market data from public sources

### ML Datasets
- **Source**: Real Titanic dataset from Kaggle competition
- **S3 Path**: `s3://ray-benchmark-data/ml-datasets/titanic.csv`
- **Used by**: `ray-data-ml-feature-engineering`
- **Format**: CSV with standard Titanic dataset columns
- **Real Data**: ✅ Actual passenger data from Kaggle's Titanic competition

### Geospatial Data
- **Source**: Synthetic NYC taxi trip data
- **S3 Path**: `s3://ray-benchmark-data/nyc-taxi/yellow_tripdata_2023-01.parquet`
- **Used by**: `ray-data-geospatial-analysis`
- **Format**: Parquet with lat/lon coordinates and trip metadata

### TPC-H Benchmark Data
- **Source**: Synthetic TPC-H compliant data
- **S3 Paths**: 
  - `s3://ray-benchmark-data/tpch/parquet/sf10/customer.parquet`
  - `s3://ray-benchmark-data/tpch/parquet/sf10/nation.parquet`
  - `s3://ray-benchmark-data/tpch/parquet/sf10/orders.parquet`
  - `s3://ray-benchmark-data/tpch/parquet/sf10/lineitem.parquet`
- **Used by**: `ray-data-etl-tpch`
- **Format**: Parquet files following TPC-H schema

### Log Data
- **Source**: Local parquet files converted to text/JSON formats
- **S3 Paths**:
  - `s3://ray-benchmark-data/logs/apache-access.log`
  - `s3://ray-benchmark-data/logs/application.json`
  - `s3://ray-benchmark-data/logs/security.log`
- **Used by**: `ray-data-log-ingestion`
- **Format**: Text logs and JSON structured logs

### Medical Data
- **Source**: Local parquet files converted to appropriate formats
- **S3 Paths**:
  - `s3://ray-benchmark-data/medical/dicom-metadata.json`
  - `s3://ray-benchmark-data/medical/hl7-messages/messages.hl7`
  - `s3://ray-benchmark-data/medical/patient-records.csv`
  - `s3://ray-benchmark-data/medical/laboratory_results.parquet`
- **Used by**: `ray-data-medical-connectors`
- **Format**: JSON, HL7 text, CSV, and Parquet formats

### Text Data
- **Source**: Generated text captions
- **S3 Path**: `s3://ray-benchmark-data/text/captions.txt`
- **Used by**: `ray-data-multimodal-ai-pipeline`
- **Format**: Plain text, one caption per line

### Support Data
- **Source**: Synthetic support ticket data
- **S3 Path**: `s3://ray-benchmark-data/support/tickets/tickets.json`
- **Used by**: `ray-data-nlp-text-analytics`
- **Format**: JSON Lines format

### Image Data
- **Source**: ImageNette dataset (requires manual download)
- **S3 Path**: `s3://ray-benchmark-data/imagenette2/train/`
- **Used by**: `ray-data-batch-inference-optimization`, `ray-data-multimodal-ai-pipeline`
- **Format**: Image files organized by class
- **Note**: Due to size, ImageNette must be downloaded separately from https://github.com/fastai/imagenette

### Data Quality Data
- **Source**: Local parquet file
- **S3 Path**: `s3://ray-benchmark-data/catalog/customer_data.parquet`
- **Used by**: `ray-data-data-quality-monitoring`, `ray-data-enterprise-data-catalog`
- **Format**: Parquet with customer records including quality issues

## Local Files

The following files exist locally in template directories and will be uploaded:

```
ray-data-medical-connectors/
├── laboratory_results.parquet
├── patient_medical_records.parquet  
├── dicom_imaging_metadata.parquet
└── hl7_medical_messages.parquet

ray-data-log-ingestion/
├── apache_access_logs.parquet
├── application_logs.parquet
└── security_logs.parquet

ray-data-data-quality-monitoring/
└── ecommerce_customers_with_quality_issues.parquet
```

## Manual ImageNette Setup

For the image datasets, you'll need to manually download ImageNette:

```bash
# Download ImageNette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz

# Upload to S3 (requires appropriate permissions)
aws s3 sync imagenette2/train/ s3://ray-benchmark-data/imagenette2/train/
```

## Verification

After upload, verify the datasets are accessible:

```python
import ray

# Test financial data
financial_data = ray.data.read_parquet("s3://ray-benchmark-data/financial/sp500_daily_2years.parquet")
print(f"Financial data: {financial_data.count()} records")

# Test Titanic data  
titanic_data = ray.data.read_csv("s3://ray-benchmark-data/ml-datasets/titanic.csv")
print(f"Titanic data: {titanic_data.count()} records")

# Test TPC-H data
customer_data = ray.data.read_parquet("s3://ray-benchmark-data/tpch/parquet/sf10/customer.parquet")
print(f"TPC-H customers: {customer_data.count()} records")
```

## Data Sizes

Expected dataset sizes after upload:

- Financial data: ~50MB (10 symbols × 2 years daily)
- Titanic data: ~100KB (891 passengers)
- NYC Taxi data: ~50MB (50K trips)
- TPC-H data: ~2GB total (SF10 scale)
- Log data: ~100MB (converted text logs)
- Medical data: ~200MB (various formats)
- Text captions: ~500KB (5K captions)
- Support tickets: ~50MB (10K tickets)

**Total estimated size**: ~3GB (excluding ImageNette images)

## Security Considerations

- Ensure AWS credentials have appropriate S3 permissions
- All generated data is synthetic and contains no real PII
- Medical data follows HIPAA-compliant synthetic patterns
- Financial data uses public market data or synthetic equivalents

## Troubleshooting

### Pandas/Numpy Compatibility Issues
```bash
# Error: "numpy.dtype size changed, may indicate binary incompatibility"
# Solution 1: Use simplified uploader
python upload_benchmark_data_simple.py

# Solution 2: Fix the environment
./fix_pandas_numpy.sh

# Solution 3: Create fresh environment
conda create -n ray_upload python=3.9
conda activate ray_upload
pip install boto3 pandas pyarrow requests yfinance
python upload_benchmark_data.py
```

### Permission Errors
```bash
# Verify S3 access
aws s3 ls s3://ray-benchmark-data/

# Check IAM permissions for s3:PutObject on ray-benchmark-data bucket
```

### Download Failures
- yfinance failures will fallback to synthetic financial data
- Titanic download failures will fallback to synthetic passenger data
- All critical datasets have synthetic fallbacks

### Large Dataset Handling
- Script uses temporary directory for processing
- Files are cleaned up after upload
- Monitor disk space during TPC-H generation

### Import Errors
```bash
# Missing dependencies
pip install -r requirements_simple.txt  # For simplified version
pip install -r requirements_upload.txt  # For full version

# AWS SDK issues
pip install --upgrade boto3

# Network/SSL issues
pip install --upgrade requests urllib3
```

## Contributing

When adding new templates that require datasets:

1. Add dataset requirements to this documentation
2. Update `upload_benchmark_data.py` with new dataset generation
3. Test upload and template integration
4. Document expected S3 paths in template README

## Support

For issues with dataset upload or access:

1. Check AWS credentials and S3 permissions
2. Verify network connectivity to external data sources
3. Monitor CloudWatch logs for S3 upload issues
4. Use synthetic data fallbacks for development
