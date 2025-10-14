# Unstructured Data Ingestion and Processing With Ray Data

**Time to complete**: 35 min | **Difficulty**: Advanced | **Prerequisites**: Data engineering experience, document processing, basic NLP knowledge

## What you'll build

Build a comprehensive document ingestion pipeline that transforms unstructured documents from data lakes into structured, analytics-ready datasets using Ray Data's distributed processing capabilities for enterprise data warehouse workflows.


## Table of Contents

1. [Data Lake Document Discovery](#step-1-data-lake-document-discovery) (8 min)
2. [Document Processing and Classification](#step-2-document-processing-and-classification) (10 min)
3. [Text Extraction and Enrichment](#step-3-text-extraction-and-enrichment) (8 min)
4. [LLM-Powered Content Analysis](#step-4-llm-powered-content-analysis) (6 min)
5. [Data Warehouse Output](#step-5-data-warehouse-output) (3 min)


## Learning Objectives

**Why unstructured data ingestion matters**: Enterprise data lakes contain vast amounts of unstructured documents (PDFs, Word docs, presentations, reports) that need systematic processing to extract business value for analytics and reporting.

**Ray Data's ingestion capabilities**: Distribute document processing across clusters to handle large-scale document collections, extract structured data, and prepare analytics-ready datasets for data warehouse consumption.

**Data lake to warehouse patterns**: Techniques used by data engineering teams to systematically process document collections, extract structured information, and create queryable datasets for business intelligence.

**Production ingestion workflows**: Scalable document processing patterns that handle diverse file formats, extract metadata, and create structured schemas for downstream analytics systems.

**LLM integration strategies**: Document processing workflows that can use advanced analysis for content extraction from unstructured text.


## Overview

**Challenge**: Enterprise data lakes contain millions of unstructured documents (PDFs, Word docs, presentations) across multiple formats that need systematic processing to extract business value. Traditional document processing approaches struggle with:
- **Scale**: Single-machine processing limits document volume
- **Consistency**: Manual extraction creates inconsistent schemas  
- **Integration**: Complex infrastructure for analysis
- **Warehouse integration**: Manual data modeling and ETL processes

**Solution**: Ray Data enables end-to-end document ingestion pipelines:

| Pipeline Stage | Traditional Approach | Ray Data Approach | Benefit |
|------------------|-----------------------|---------------------|-----------|
| **Document Discovery** | Sequential file listing | Parallel `read_binary_files()` | Process millions of files |
| **Text Extraction** | Single-threaded parsing | Distributed `map_batches()` | Extract from all docs simultaneously |
| **Content Analysis** | Manual processing | Distributed analysis | Built-in batch processing |
| **Data Warehouse** | Custom ETL scripts | Native `write_parquet()` with partitioning | Production-ready output |

**Data Lake to Warehouse Flow**: This template demonstrates a complete pipeline from raw documents in data lakes to structured, queryable datasets ready for business intelligence and analytics workflows using Ray Data native operations.


## Prerequisites Checklist

Before starting, ensure you have:
- [ ] Understanding of data lake and data warehouse concepts
- [ ] Experience with document processing and text extraction
- [ ] Knowledge of structured data formats (Parquet, Delta Lake, Iceberg)
- [ ] Python environment with Ray Data and document processing libraries
- [ ] Access to S3 or other cloud storage for document sources


## Quick start (3 minutes)

This section demonstrates large-scale document ingestion using Ray Data:



```python
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import ray

# Configure Ray Data 
ctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = False
ctx.enable_operator_progress_bars = False

# Initialize Ray for distributed processing
ray.init(ignore_reinit_error=True)
```

## Step 1: Data Lake Document Discovery

### Discover document collections in data lake



```python

# Load document collection from data lake
document_collection = ray.data.read_binary_files(
    "s3://anyscale-rag-application/1000-docs/",
    include_paths=True,
    ray_remote_args={"num_cpus":0.025}  # High I/O concurrency for large document collections
).limit(100)

print(f"Dataset schema: {document_collection.schema()}")
```

### Document metadata extraction



```python
def process_file(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract text content from document files.
    
    Processes the bytes field immediately to avoid passing large binary data
    through multiple Ray Data operations. Returns basic file metadata and
    extracted text.
    """
    import io
    from pathlib import Path
    from unstructured.partition.auto import partition
    
    file_path = Path(record["path"])
    file_bytes = record["bytes"]
    file_size = len(file_bytes)
    file_extension = file_path.suffix.lower()
    file_name = file_path.name
    
    # Only process supported file extensions
    supported_extensions = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".txt"}
    
    if file_extension not in supported_extensions:
        return {
            "document_id": str(uuid.uuid4()),
            "file_path": str(file_path),
            "file_name": file_name,
            "file_extension": file_extension,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "discovery_timestamp": datetime.now().isoformat(),
            "extracted_text": "",
            "text_length": 0,
            "word_count": 0,
            "extraction_status": "unsupported_format"
        }
    
    try:
        with io.BytesIO(file_bytes) as stream:
            elements = partition(file=stream)
            
            # Combine all text elements
            extracted_text = " ".join([str(el) for el in elements]).strip()
            text_length = len(extracted_text)
            word_count = len(extracted_text.split()) if extracted_text else 0
            extraction_status = "success"
            
    except Exception as e:
        print(f"Cannot process file {file_path}: {e}")
        extracted_text = ""
        text_length = 0
        word_count = 0
        extraction_status = f"error: {str(e)[:100]}"
    
    return {
        "document_id": str(uuid.uuid4()),
        "file_path": str(file_path),
        "file_name": file_name,
        "file_extension": file_extension,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "discovery_timestamp": datetime.now().isoformat(),
        "extracted_text": extracted_text,
        "text_length": text_length,
        "word_count": word_count,
        "extraction_status": extraction_status
    }

# Apply text extraction
print("Extracting text from documents...")
documents_with_text = document_collection.map(
    process_file,
    concurrency=8,
    num_cpus=1
)

```


```python
documents_with_text.limit(25).to_pandas()
```


```python

def enrich_business_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Classify documents by business category and assign processing priority.
    
    This is a separate stage that operates on already-extracted text,
    performing pure metadata enrichment based on filename patterns.
    """
    file_name = record["file_name"]
    filename_lower = file_name.lower()
    file_size = record["file_size_bytes"]
    
    # Business classification for data warehouse categorization
    if any(keyword in filename_lower for keyword in ["financial", "earnings", "revenue", "profit"]):
        doc_type = "financial_document"
        business_category = "finance"
    elif any(keyword in filename_lower for keyword in ["legal", "contract", "agreement", "terms"]):
        doc_type = "legal_document"
        business_category = "legal"
    elif any(keyword in filename_lower for keyword in ["regulatory", "compliance", "filing", "sec"]):
        doc_type = "regulatory_document"
        business_category = "compliance"
    elif any(keyword in filename_lower for keyword in ["client", "customer", "portfolio"]):
        doc_type = "client_document"
        business_category = "client_services"
    elif any(keyword in filename_lower for keyword in ["market", "research", "analysis", "report"]):
        doc_type = "research_document"
        business_category = "research"
    else:
        doc_type = "general_document"
        business_category = "general"
    
    # Processing priority for workflow optimization
    if any(keyword in filename_lower for keyword in ["urgent", "critical", "deadline"]):
        priority = "high"
        priority_score = 3
    elif any(keyword in filename_lower for keyword in ["important", "quarterly", "annual"]):
        priority = "medium"
        priority_score = 2
    else:
        priority = "low"
        priority_score = 1
    
    return {
        **record,
        "document_type": doc_type,
        "business_category": business_category,
        "processing_priority": priority,
        "priority_score": priority_score,
        "estimated_pages": max(1, file_size // 50000),
        "processing_status": "classified"
    }


# Apply business metadata enrichment
print("\nEnriching with business metadata...")
documents_with_metadata = documents_with_text.map(
    enrich_business_metadata,
    concurrency=10,
    num_cpus=0.25
)

```


```python
documents_with_metadata.limit(5).to_pandas()
```


```python
# Use Ray Data native operations for document collection analysis
from ray.data.aggregate import Count, Sum, Mean, Max, Min

print("Analyzing document collection using Ray Data native operations...")

# Document type distribution using native groupby
doc_type_stats = documents_with_metadata.groupby("document_type").aggregate(
    Count(),
    Sum("file_size_bytes"),
    Mean("file_size_mb"),
    Max("estimated_pages")
)

```


```python

# Business category analysis
category_stats = documents_with_metadata.groupby("business_category").aggregate(
    Count(),
    Mean("priority_score"),
    Sum("file_size_mb")
)
```

## Step 2: Document Processing and Classification

###  Text extraction and quality assessment



```python
from ray.data.expressions import col, lit

def assess_document_quality(record: Dict[str, Any]) -> Dict[str, Any]:
    """Assess document quality for data warehouse ingestion."""
    
    quality_score = 0
    quality_issues = []
    
    if record["file_size_mb"] > 0.01:
        quality_score += 1
    else:
        quality_issues.append("file_too_small")
    
    if record["text_length"] > 100:
        quality_score += 1
    else:
        quality_issues.append("insufficient_text")
    
    if record["business_category"] != "general":
        quality_score += 1
    else:
        quality_issues.append("low_business_relevance")
    
    if record["word_count"] > 20:
        quality_score += 1
    else:
        quality_issues.append("insufficient_content")
    
    quality_rating = "high" if quality_score >= 4 else "medium" if quality_score >= 2 else "low"
    
    return {
        **record,
        "quality_score": quality_score,
        "quality_rating": quality_rating,
        "quality_issues": json.dumps(quality_issues)
    }

# Apply quality assessment (text extraction already done in previous step)
quality_assessed_docs = documents_with_metadata.map_batches(
    process_quality_assessment_batch,
    num_cpus=0.25,
    batch_size=2000
)

```

## Step 3: Text Chunking and Enrichment



```python
def create_text_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create text chunks optimized for processing and analytics."""
    
    text = record["extracted_text"]
    chunk_size = 1500
    overlap = 150
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        chunk_record = {
            **record,
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": len(chunk_text),
            "chunk_word_count": len(chunk_text.split())
        }
        
        chunks.append(chunk_record)
        
        # If we've reached the end of the text, stop
        if end >= len(text):
            break
            
        start = end - overlap
        chunk_index += 1
    
    # Update total chunks
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
    
    return chunks

# Apply text chunking using Ray Data flat_map
print("Creating text chunks...")

chunked_documents = quality_assessed_docs.flat_map(
    create_text_chunks,
    num_cpus=0.5
)
```

## Step 4: Data Warehouse Schema and Output

### Create data warehouse schema



```python
from datetime import datetime

# Apply warehouse schema transformation using expressions API
print("Creating data warehouse schema...")

processing_date = datetime.now().isoformat()[:10]

warehouse_dataset = chunked_documents.select_columns([
    # Primary identifiers
    "document_id",
    "chunk_id",
    
    # Dimensional attributes
    "business_category",
    "document_type",
    "file_extension",
    "quality_rating",
    "processing_priority",
    
    # Fact measures
    "file_size_mb",
    "word_count",
    "chunk_word_count",
    "quality_score",
    "priority_score",
    "estimated_pages",
    "chunk_index",
    "total_chunks",
    
    # Content fields
    "chunk_text",
    "file_name",
    "file_path",
    
    # Existing metadata
    "discovery_timestamp",
    "extraction_status",
    "processing_status"
]).rename_columns({
    "chunk_text": "text_content"
}).add_column(
    "processing_date", lambda df: processing_date
).add_column(
    "pipeline_version", lambda df: "1.0"
).add_column(
    "processing_engine", lambda df: "ray_data"
)
```

### Write to data warehouse with partitioning



```python
# Write main warehouse table with partitioning
print("Writing to data warehouse...")

OUTPUT_WAREHOUSE_PATH = "/mnt/cluster_storage"

warehouse_dataset.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/main_table/",
    partition_cols=["business_category", "processing_date"],
    compression="snappy",
    ray_remote_args={"num_cpus":0.1}
)

print("Main warehouse table written successfully")

```


```python
print("Creating business-specific datasets...")

# Financial documents dataset
financial_analytics = warehouse_dataset.filter(
    expr="business_category == 'finance'",
    num_cpus=0.1
).select_columns([
    "document_id", "chunk_id", "text_content", "summary", 
    "quality_score", "processing_date", "metrics_count"
])

financial_analytics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/analytics/financial/",
    partition_cols=["processing_date"],
    compression="snappy",
    ray_remote_args={"num_cpus":0.1}
)

# Compliance documents dataset
compliance_analytics = warehouse_dataset.filter(
   expr="business_category == 'compliance'",
    num_cpus=0.1
).select_columns([
    "document_id", "chunk_id", "text_content", "summary",
    "quality_score", "content_priority", "processing_date"
])

compliance_analytics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/analytics/compliance/",
    partition_cols=["processing_date"],
    compression="snappy",
    ray_remote_args={"num_cpus":0.1}
)

```

### Create analytics summary tables



```python
print("Creating analytics summary tables...")

# Processing metrics by category and date
processing_metrics = warehouse_dataset.groupby(["business_category", "processing_date"]).aggregate(
    Count(),
    Sum("file_size_mb"),
    Mean("word_count"),
    Mean("quality_score")
)

processing_metrics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/processing_metrics/",
    partition_cols=["processing_date"],
    compression="snappy",
    ray_remote_args={"num_cpus":0.1}
)

# Quality distribution analysis
quality_distribution = warehouse_dataset.groupby(["quality_rating", "business_category"]).aggregate(
    Count(),
    Mean("word_count"),
    Mean("entities_count"),
    Mean("metrics_count")
)

quality_distribution.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/quality_distribution/",
    compression="snappy",
    ray_remote_args={"num_cpus":0.1}
)


```

## Verification and Summary

### Verify data warehouse outputs



```python
# Verify warehouse outputs
print("Verifying data warehouse integration...")

# Read back main table
main_table_verify = ray.data.read_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/main_table/",
    num_cpus=0.025
)

# Read back summary tables
metrics_verify = ray.data.read_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/processing_metrics/",
    num_cpus=0.025
)

print(f"Data warehouse verification:")
print(f"  Main table records: {main_table_verify.count():,}")
print(f"  Processing metrics: {metrics_verify.count():,}")
print(f"  Schema compatibility: Verified")

# Display sample data
print("\\nSample warehouse records:")
samples = main_table_verify.take(10)
for i, record in enumerate(samples):
    print(f"  {i+1}. Doc: {record['document_id'][:8]}, Category: {record['business_category']}, "
          f"Words: {record['word_count']}, Quality: {record['quality_rating']}")

```

## Summary and Next Steps

This notebook demonstrates a complete document ingestion pipeline using Ray Data:

### Key Features Demonstrated

**Ray Data Operations**:
- `read_binary_files()` for large-scale document discovery
- `map()` and `map_batches()` for distributed processing
- `filter()` with expressions API for efficient filtering
- `flat_map()` for text chunking
- `groupby().aggregate()` for analytics
- `write_parquet()` with partitioning for data warehouse output

**CPU-Based Processing**:
- Pattern matching for content analysis
- No GPU requirements
- Scalable across CPU-only clusters

**Data Warehouse Integration**:
- Partitioned tables for query optimization
- Business-specific datasets
- Summary tables for analytics
- Schema standardization

### Enabling GPU-Accelerated LLM Processing

For GPU-accelerated content analysis with vLLM:

1. Install Ray Data LLM package: `pip install -U vllm==0.7.2`
2. Configure GPU resources in your cluster
3. Replace the CPU-based analysis in Step 4 with:

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

llm_config = vLLMEngineProcessorConfig(
    model_source="unsloth/Llama-3.1-8B-Instruct",
    engine_kwargs={
        "max_model_len": 16384,
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,
        "tensor_parallel_size": 1,
    },
    concurrency=1,
    batch_size=32,
    accelerator_type="A10G"
)

llm_processor = build_llm_processor(
    llm_config,
    preprocess=create_prompts,
    postprocess=extract_structured_data
)

analyzed_docs = llm_processor(chunked_documents)
```

### Production Recommendations

1. **Use real text extraction libraries**: PyPDF2, python-docx, python-pptx, BeautifulSoup
2. **Tune batch sizes**: Adjust based on document size and cluster resources
3. **Monitor progress**: Use Ray dashboard for performance visibility
4. **Scale horizontally**: Add workers to increase throughput
5. **Optimize partitioning**: Match partitioning strategy to query patterns

This pipeline transforms unstructured documents from data lakes into structured, analytics-ready datasets for enterprise data warehouse consumption and business intelligence workflows.

