# Ray Data Reader Usage Guide

This guide documents the proper Ray Data native readers to use for different file formats across all templates.

## Native Ray Data Readers

### Text-Based Data

#### `ray.data.read_text()`
**Use for**:
- Apache/nginx access logs (raw text format)
- System logs (syslog format)
- Security logs (text-based)
- HL7 v2 messages (pipe-delimited text)
- Plain text documents
- Custom text-based formats

**Examples**:
```python
# Apache access logs
apache_logs = ray.data.read_text("s3://logs/apache-access/*.log")

# HL7 medical messages
hl7_messages = ray.data.read_text("s3://medical/hl7-messages/*.hl7")

# Security logs
security_logs = ray.data.read_text("s3://security/logs/*.log")

# Text documents
documents = ray.data.read_text("s3://documents/*.txt")
```

#### `ray.data.read_json()`
**Use for**:
- Application logs (JSON-formatted)
- API response data
- DICOM metadata (JSON format)
- FHIR resources (healthcare standard)
- NoSQL database exports
- Structured application events

**Examples**:
```python
# Application logs
app_logs = ray.data.read_json("s3://logs/application/*.json", num_cpus=0.05)

# DICOM metadata
dicom_metadata = ray.data.read_json("s3://medical/dicom-metadata/*.json", num_cpus=0.05)

# API events
api_events = ray.data.read_json("s3://events/api/*.json", num_cpus=0.05)

# Audit trails
audit_logs = ray.data.read_json("s3://security/audit/*.json", num_cpus=0.05)
```

#### `ray.data.read_csv()`
**Use for**:
- Patient demographics (EHR data)
- Laboratory results
- Financial transaction data
- Sensor data exports
- Traditional database exports
- Spreadsheet exports

**Examples**:
```python
# Patient demographics
patient_data = ray.data.read_csv("s3://medical/patient-demographics.csv", num_cpus=0.05)

# Lab results
lab_results = ray.data.read_csv("s3://medical/lab-results.csv", num_cpus=0.05)

# Financial data
transactions = ray.data.read_csv("s3://finance/transactions.csv", num_cpus=0.05)
```

### Binary and Structured Data

#### `ray.data.read_parquet()`
**Use for**:
- Pre-processed/cleaned data
- Data warehouse tables
- Analytics-ready datasets
- ETL pipeline outputs
- ML feature stores
- High-performance analytics

**Examples**:
```python
# Pre-processed data
processed_data = ray.data.read_parquet("s3://warehouse/processed/", num_cpus=0.025)

# Analytics tables
analytics_data = ray.data.read_parquet("s3://warehouse/analytics/", num_cpus=0.025)

# Feature store
features = ray.data.read_parquet("s3://ml/features/", num_cpus=0.025)
```

#### `ray.data.read_images()`
**Use for**:
- Medical images (DICOM pixel data)
- Satellite imagery
- Product images
- Computer vision datasets
- Photo collections

**Examples**:
```python
# Medical images
medical_images = ray.data.read_images("s3://medical/images/", mode="RGB")

# Product catalog
product_images = ray.data.read_images("s3://ecommerce/products/", mode="RGB")

# Computer vision dataset
imagenet = ray.data.read_images("s3://datasets/imagenet/", mode="RGB")
```

#### `ray.data.read_binary_files()`
**Use for**:
- PDF documents
- Word documents (.docx)
- PowerPoint presentations (.pptx)
- Audio files
- Video files
- Arbitrary binary formats

**Examples**:
```python
# Document ingestion
documents = ray.data.read_binary_files(
    "s3://documents/",
    include_paths=True,
    num_cpus=0.025
)

# Audio files
audio_data = ray.data.read_binary_files("s3://audio/*.wav")
```

### Database and Specialized Sources

#### `ray.data.read_datasource()`
**Use for**:
- Custom data formats (HL7, DICOM, FHIR)
- Proprietary formats
- Complex parsing logic
- Integration with external systems

**Examples**:
```python
# HL7 message parsing
hl7_dataset = ray.data.read_datasource(
    HL7Datasource(),
    paths=["s3://medical/hl7/"]
)

# DICOM image parsing
dicom_dataset = ray.data.read_datasource(
    DICOMDatasource(),
    paths=["s3://medical/dicom/"]
)
```

## Template-Specific Format Guidelines

### ray-data-log-ingestion
**Correct formats**:
- Apache/nginx logs: `read_text()` for raw logs
- Application logs: `read_json()` for JSON-formatted logs
- Security logs: `read_text()` for syslog format
- Audit trails: `read_json()` for structured audit logs
- Pre-processed logs: `read_parquet()` after parsing

### ray-data-medical-connectors
**Correct formats**:
- HL7 v2 messages: `read_text()` for pipe-delimited text
- DICOM metadata: `read_json()` for metadata files
- Patient records: `read_csv()` for demographic data
- Lab results: `read_csv()` or custom datasource
- Pre-processed medical data: `read_parquet()` after enrichment

### ray-data-financial-forecasting
**Correct formats**:
- Stock prices: `read_csv()` or `read_parquet()`
- Market data: `read_csv()` for time series
- Financial reports: `read_json()` for structured data
- Trading data: `read_parquet()` for high-frequency data

### ray-data-nlp-text-analytics
**Correct formats**:
- Reviews/comments: `read_json()` or `read_text()`
- Documents: `read_text()` for plain text
- Structured text data: `read_parquet()` after preprocessing
- Web scraping results: `read_json()`

### ray-data-geospatial-analysis
**Correct formats**:
- GPS coordinates: `read_csv()` or `read_parquet()`
- GeoJSON: `read_json()`
- Shapefiles: Custom datasource
- Processed spatial data: `read_parquet()`

### ray-data-unstructured-ingestion
**Correct formats**:
- PDF/Word/PPT: `read_binary_files()` with custom parsing
- HTML: `read_text()` or `read_binary_files()`
- Images in documents: `read_binary_files()` + extraction
- Processed documents: `read_parquet()` after extraction

## File Format Decision Tree

```
What Ray Data reader should I use?
├── Is it a text-based log file?
│   ├── Apache/nginx/syslog → read_text()
│   ├── JSON-formatted logs → read_json()
│   └── Already processed logs → read_parquet()
│
├── Is it healthcare data?
│   ├── HL7 v2 messages → read_text() or read_datasource(HL7Datasource)
│   ├── FHIR resources → read_json()
│   ├── DICOM images → read_datasource(DICOMDatasource)
│   ├── DICOM metadata → read_json()
│   ├── Patient demographics → read_csv()
│   └── Processed medical data → read_parquet()
│
├── Is it image/media data?
│   ├── Images (JPG/PNG) → read_images()
│   ├── Audio/Video → read_binary_files()
│   └── Medical images → read_datasource() for DICOM
│
├── Is it structured tabular data?
│   ├── Analytics-ready → read_parquet() (BEST)
│   ├── Database exports → read_csv() then convert to parquet
│   ├── Real-time data → read_json()
│   └── Time series → read_csv() or read_parquet()
│
├── Is it document data?
│   ├── PDF/Word/PowerPoint → read_binary_files()
│   ├── Plain text → read_text()
│   ├── HTML → read_text() or read_binary_files()
│   └── Processed documents → read_parquet()
│
└── Is it custom/proprietary format?
    └── Implement custom datasource → read_datasource()
```

## Best Practices

### 1. Use Native Format When Possible
- Read data in its native format first
- Parse and structure using Ray Data operations
- Save processed data as Parquet for analytics

### 2. Parquet for Performance
- Use Parquet for pre-processed, analytics-ready data
- Convert raw data to Parquet after cleaning/parsing
- Use column pruning with Parquet readers

### 3. Custom Datasources for Complex Formats
- Implement custom datasources for HL7, DICOM, FHIR
- Use `read_datasource()` for specialized medical/healthcare formats
- Properly handle format-specific parsing requirements

### 4. Optimization by Format

| Format | Reader | num_cpus Guideline | Reasoning |
|--------|--------|-------------------|-----------|
| **Text logs** | `read_text()` | Not specified (default) | Simple reading |
| **JSON logs** | `read_json()` | 0.05 | Moderate parsing overhead |
| **CSV files** | `read_csv()` | 0.05 | Moderate parsing overhead |
| **Parquet files** | `read_parquet()` | 0.025 | High I/O concurrency |
| **Images** | `read_images()` | 0.05 (default) | Image decoding overhead |
| **Binary files** | `read_binary_files()` | 0.025 | High I/O concurrency |

## Common Mistakes to Avoid

### x Don't Read Logs as Parquet
```python
# WRONG: Raw logs should not be parquet
apache_logs = ray.data.read_parquet("apache-access-logs.parquet")
```

```python
# CORRECT: Read raw logs as text
apache_logs = ray.data.read_text("s3://logs/apache-access/*.log")

# Parse logs to structured format
parsed_logs = apache_logs.map_batches(parse_apache_logs, num_cpus=0.25)

# THEN save as parquet for future analytics
parsed_logs.write_parquet("s3://processed-logs/apache/", num_cpus=0.1)
```

### x Don't Read HL7 as Parquet
```python
# WRONG: HL7 messages are text-based
hl7_data = ray.data.read_parquet("hl7_messages.parquet")
```

```python
# CORRECT: Read HL7 as text
hl7_data = ray.data.read_text("s3://medical/hl7/*.hl7")

# Or use custom datasource for proper parsing
hl7_dataset = ray.data.read_datasource(
    HL7Datasource(),
    paths=["s3://medical/hl7/"]
)
```

### x Don't Use Custom Readers for Standard Formats
```python
# WRONG: Don't implement custom readers for standard formats
def custom_csv_reader(path):
    # Custom CSV reading logic
    pass

dataset = ray.data.from_items(custom_csv_reader(path))
```

```python
# CORRECT: Use native Ray Data readers
dataset = ray.data.read_csv(
    path,
    columns=["col1", "col2"],  # Column pruning
    num_cpus=0.05
)
```

## Validation Checklist

Before finalizing a template, verify:
- [ ] **Native formats**: Raw data read in native format (text, JSON, CSV)
- [ ] **Proper reader**: Using appropriate Ray Data native reader
- [ ] **Parquet for processed**: Parquet used only for processed/analytics data
- [ ] **Custom datasources**: Used for specialized formats (HL7, DICOM)
- [ ] **num_cpus optimization**: Appropriate values for each reader type
- [ ] **Format documentation**: Comments explain format choices

## Summary

All Ray Data templates now use proper file formats and native readers:
- - **Logs**: Text/JSON formats with appropriate readers
- - **Medical data**: Native formats (HL7 text, DICOM datasource, JSON metadata)
- - **Images**: Native `read_images()` reader
- - **Documents**: Binary file reader with custom parsing
- - **Analytics data**: Parquet for performance
- - **Custom formats**: Datasource implementations for specialized data

Templates demonstrate the full workflow from raw data ingestion through processing to analytics-ready Parquet storage.
