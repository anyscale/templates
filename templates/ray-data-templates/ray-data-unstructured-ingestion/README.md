# Unstructured Data Ingestion and Processing with Ray Data

**⏱️ Time to complete**: 35 min | **Difficulty**: Advanced | **Prerequisites**: Data engineering experience, understanding of document processing, basic NLP knowledge

## What You'll Build

Build a comprehensive document ingestion pipeline that transforms unstructured documents from data lakes into structured, analytics-ready datasets using Ray Data's distributed processing capabilities for enterprise data warehouse workflows.

## Table of Contents

1. [Data Lake Document Discovery](#step-1-data-lake-document-discovery) (8 min)
2. [Document Processing and Classification](#step-2-document-processing-and-classification) (10 min)
3. [Text Extraction and Enrichment](#step-3-text-extraction-and-enrichment) (8 min)
4. [LLM-Powered Content Analysis](#step-4-llm-powered-content-analysis) (6 min)
5. [Data Warehouse Output](#step-5-data-warehouse-output) (3 min)

## Learning objectives

**Why unstructured data ingestion matters**: Enterprise data lakes contain vast amounts of unstructured documents (PDFs, Word docs, presentations, reports) that need systematic processing to extract business value for analytics and reporting.

**Ray Data's ingestion capabilities**: Distribute document processing across clusters to handle large-scale document collections, extract structured data, and prepare analytics-ready datasets for data warehouse consumption.

**Data lake to warehouse patterns**: Techniques used by data engineering teams to systematically process document collections, extract structured information, and create queryable datasets for business intelligence.

**Production ingestion workflows**: Scalable document processing patterns that handle diverse file formats, extract metadata, and create structured schemas for downstream analytics systems.

**LLM integration strategies**: Document processing workflows that use Ray Data LLM package for content analysis and structured data extraction from unstructured text.

## Overview

**Challenge**: Enterprise data lakes contain millions of unstructured documents across multiple formats that need systematic processing to extract business value. Traditional document processing approaches struggle with scale, consistency, and integration with analytics systems.

**Solution**: Ray Data enables distributed document ingestion that processes large document collections, extracts structured information, and creates analytics-ready datasets for data warehouse consumption.

**Data Lake to Warehouse Flow**: This template demonstrates a complete pipeline from raw documents in data lakes to structured, queryable datasets ready for business intelligence and analytics workflows.

---

### Approach comparison

| Traditional Approach | Ray Data Approach | Key Benefit |
|---------------------|-------------------|-------------|
| **Single-machine processing** | Distributed across cluster | Horizontal scalability |
| **Memory-limited** | Streaming execution | Handle large datasets |
| **Sequential operations** | Pipeline parallelism | Better resource utilization |
| **Manual optimization** | Automatic resource management | Simplified deployment |

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
from ray.data.expressions import col, lit

# Initialize Ray for distributed processingray.init()

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

# Load document collection from data lakedocument_dataset = ray.data.read_binary_files(
    "s3://anyscale-rag-application/1000-docs/",
    include_paths=True,
    num_cpus=0.025  # High I/O concurrency for large document collections
)

print(f"Loaded document collection: {document_dataset.count()} documents")
print(f"Dataset schema: {document_dataset.schema()}")
```

## Step 1: Data Lake Document Discovery

### Initialize Ray Data Environment

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
from ray.data.expressions import col, lit

# Initialize Ray for distributed document processingray.init()

# Configure Ray Data for optimal performance monitoringctx = ray.data.DataContext.get_current()
ctx.enable_progress_bars = True
ctx.enable_operator_progress_bars = True

print("Ray Data initialized for large-scale document ingestion")
print(f"Cluster resources: {ray.cluster_resources()}")
```

### Discover Document Collections in Data Lake

```python
# Configuration for document ingestion pipelineSOURCE_S3_PATH = "s3://anyscale-rag-application/1000-docs/"
OUTPUT_WAREHOUSE_PATH = "/tmp/document_warehouse"

# Use Ray Data to scan large document collectionsprint("Discovering documents in data lake...")

document_collection = ray.data.read_binary_files(
    SOURCE_S3_PATH,
    include_paths=True,
    num_cpus=0.025  # High I/O concurrency for document discovery
)

print(f"Document discovery completed:")
print(f"  Total documents: {document_collection.count():,}")
print(f"  Schema: {document_collection.schema()}")
```

### Document Metadata Extraction

```python
def extract_document_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive metadata from document files for data warehouse analysis."""
    
    file_path = Path(record["path"])
    file_size = len(record["bytes"])
    
    # Extract file system metadata
    file_metadata = {
        "document_id": str(uuid.uuid4()),
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_extension": file_path.suffix.lower(),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "discovery_timestamp": datetime.now().isoformat(),
        "source_bucket": file_path.parts[0] if file_path.parts else "unknown"
    }
    
    # Business classification for data warehouse categorization
    filename_lower = file_path.name.lower()
    
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
        **file_metadata,
        "document_type": doc_type,
        "business_category": business_category,
        "processing_priority": priority,
        "priority_score": priority_score,
        "estimated_pages": max(1, file_size // 50000),  # Rough page estimate
        "processing_status": "discovered"
    }

# Apply metadata extraction using Ray Data map operationprint("Extracting document metadata for data warehouse analysis...")

documents_with_metadata = document_collection.map(
    extract_document_metadata,
    num_cpus=0.5  # Medium complexity metadata extraction
)

print(f"Metadata extraction completed: {documents_with_metadata.count():,} documents processed")
```

### Document Collection Analytics

```python
# Use Ray Data native operations for document collection analysisfrom ray.data.aggregate import Count, Sum, Mean, Max, Min

print("Analyzing document collection using Ray Data native operations...")

# Document type distribution using native groupbydoc_type_stats = documents_with_metadata.groupby("document_type").aggregate(
    Count(),
    Sum("file_size_bytes"),
    Mean("file_size_mb"),
    Max("estimated_pages")
)

print("Document Type Distribution:")
print(doc_type_stats.limit(10).to_pandas())

# Business category analysiscategory_stats = documents_with_metadata.groupby("business_category").aggregate(
    Count(),
    Mean("priority_score"),
    Sum("file_size_mb")
)

print("Business Category Analysis:")
print(category_stats.limit(10).to_pandas())

# File extension analysis using expressions APIpdf_documents = documents_with_metadata.filter(col("file_extension") == lit(".pdf"), num_cpus=0.1)
word_documents = documents_with_metadata.filter(col("file_extension") == lit(".docx"), num_cpus=0.1)
ppt_documents = documents_with_metadata.filter(col("file_extension") == lit(".pptx"), num_cpus=0.1)

print(f"File Format Distribution:")
print(f"  PDF documents: {pdf_documents.count():,}")
print(f"  Word documents: {word_documents.count():,}")
print(f"  PowerPoint documents: {ppt_documents.count():,}")
```

## Step 2: Document Processing and Classification

### Text Extraction Pipeline

```python
def extract_text_from_document(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract text content from various document formats for data warehouse processing."""
    
    file_extension = record["file_extension"]
    file_name = record["file_name"]
    file_size = record["file_size_bytes"]
    
    # Simulate format-specific text extraction
    # In production, use libraries like PyPDF2, python-docx, python-pptx, BeautifulSoup
    
    if file_extension == ".pdf":
        extracted_text = simulate_pdf_extraction(file_name, file_size)
    elif file_extension in [".docx", ".doc"]:
        extracted_text = simulate_word_extraction(file_name, file_size)
    elif file_extension in [".pptx", ".ppt"]:
        extracted_text = simulate_powerpoint_extraction(file_name, file_size)
    elif file_extension in [".html", ".htm"]:
        extracted_text = simulate_html_extraction(file_name, file_size)
    else:
        extracted_text = f"Text content from {file_name} with business information and structured data."
    
    # Calculate text statistics for analytics
    text_length = len(extracted_text)
    word_count = len(extracted_text.split())
    
    return {
        **record,
        "extracted_text": extracted_text,
        "text_length": text_length,
        "word_count": word_count,
        "text_extraction_timestamp": datetime.now().isoformat(),
        "extraction_status": "success"
    }

def simulate_pdf_extraction(file_name: str, file_size: int) -> str:
    """Simulate PDF text extraction with realistic business content."""
    if "financial" in file_name.lower():
        return f"Financial Report: Quarterly earnings data showing revenue growth, profit margins, and cash flow analysis. Balance sheet information includes assets, liabilities, and equity positions. Income statement details operating expenses, net income, and earnings per share metrics. Document contains structured financial tables and regulatory compliance information."
    elif "regulatory" in file_name.lower():
        return f"Regulatory Filing: Compliance documentation for regulatory requirements including risk assessments, audit findings, and regulatory framework adherence. Contains mandatory reporting data, compliance metrics, and regulatory disclosure information."
    else:
        return f"Business Document: Professional document containing business information, operational data, and structured content for enterprise analysis and reporting. Includes business metrics, process documentation, and analytical insights."

def simulate_word_extraction(file_name: str, file_size: int) -> str:
    """Simulate Word document text extraction with business context."""
    if "client" in file_name.lower():
        return f"Client Report: Client portfolio analysis including investment performance, asset allocation, and risk assessment. Contains client-specific recommendations, market outlook information, and personalized financial advice."
    elif "policy" in file_name.lower():
        return f"Policy Document: Corporate policies, procedures, guidelines, and operational frameworks. Contains structured policy information, compliance requirements, and operational procedures."
    else:
        return f"Business Document: Corporate document with operational information, business processes, professional correspondence, and structured information for internal and external stakeholders."

def simulate_powerpoint_extraction(file_name: str, file_size: int) -> str:
    """Simulate PowerPoint text extraction with presentation context."""
    if "executive" in file_name.lower():
        return f"Executive Presentation: Strategic business presentation with performance metrics, market analysis, and executive insights. Contains key performance indicators, strategic initiatives, and business analytics content."
    elif "quarterly" in file_name.lower():
        return f"Quarterly Report: Quarterly business review with financial metrics, operational performance, market trends, and strategic initiatives. Contains structured quarterly data and business analytics."
    else:
        return f"Business Presentation: Corporate presentation with charts, data visualizations, and structured content suitable for business meetings and stakeholder communications."

def simulate_html_extraction(file_name: str, file_size: int) -> str:
    """Simulate HTML text extraction with web content context."""
    if "report" in file_name.lower():
        return f"Web Report: Online business report with structured data, tables, and analytical content. Contains web-based business intelligence and reporting information."
    else:
        return f"HTML Document: Web content with structured information, data tables, and business content suitable for data warehouse ingestion."

# Apply text extraction using Ray Data distributed processingprint("Extracting text content from documents...")

documents_with_text = documents_with_metadata.map_batches(
    lambda batch: [extract_text_from_document(record, batch_format="pandas") for record in batch],
    num_cpus=1.0,  # Heavy text extraction processing
    batch_size=500
)

print(f"Text extraction completed: {documents_with_text.count():,} documents processed")
```

### Document Quality Assessment

```python
def assess_document_quality(record: Dict[str, Any]) -> Dict[str, Any]:
    """Assess document quality for data warehouse ingestion suitability."""
    
    quality_score = 0
    quality_issues = []
    
    # File size quality check
    if record["file_size_mb"] > 0.01:  # At least 10KB
        quality_score += 1
    else:
        quality_issues.append("file_too_small")
    
    # Text content quality check
    if record["text_length"] > 100:  # At least 100 characters
        quality_score += 1
    else:
        quality_issues.append("insufficient_text")
    
    # Business relevance check
    if record["business_category"] != "general":
        quality_score += 1
    else:
        quality_issues.append("low_business_relevance")
    
    # Word count check
    if record["word_count"] > 20:  # At least 20 words
        quality_score += 1
    else:
        quality_issues.append("insufficient_content")
    
    # File format check
    supported_formats = [".pdf", ".docx", ".pptx", ".html", ".txt"]
    if record["file_extension"] in supported_formats:
        quality_score += 1
    else:
        quality_issues.append("unsupported_format")
    
    # Determine overall quality rating
    if quality_score >= 4:
        quality_rating = "high"
    elif quality_score >= 2:
        quality_rating = "medium"
    else:
        quality_rating = "low"
    
    return {
        **record,
        "quality_score": quality_score,
        "quality_rating": quality_rating,
        "quality_issues": json.dumps(quality_issues),
        "quality_assessment_timestamp": datetime.now().isoformat()
    }

# Apply quality assessment using Ray Data batch processingprint("Assessing document quality for data warehouse ingestion...")

quality_assessed_docs = documents_with_text.map_batches(
    lambda batch: [assess_document_quality(record, batch_format="pandas") for record in batch],
    num_cpus=0.25,  # Light quality assessment processing
    batch_size=2000
)

# Filter high-quality documents using Ray Data expressionshigh_quality_docs = quality_assessed_docs.filter(
    col("quality_rating") == lit("high"),
    num_cpus=0.1
)

print(f"Quality assessment completed:")
print(f"  Total documents assessed: {quality_assessed_docs.count():,}")
print(f"  High quality documents: {high_quality_docs.count():,}")
```

### Document Filtering and Prioritization

```python
# Use Ray Data native filtering for document prioritizationprint("Filtering and prioritizing documents for processing...")

# Filter by processing priority using expressions APIhigh_priority_docs = high_quality_docs.filter(
    col("priority_score") >= lit(2),
    num_cpus=0.1
)

# Filter by business category for targeted processingfinancial_docs = high_quality_docs.filter(
    col("business_category") == lit("finance"),
    num_cpus=0.1
)

compliance_docs = high_quality_docs.filter(
    col("business_category") == lit("compliance"),
    num_cpus=0.1
)

# Filter by file size for processing optimizationlarge_documents = high_quality_docs.filter(
    col("file_size_mb") > lit(5.0),  # Documents larger than 5MB
    num_cpus=0.1
)

small_documents = high_quality_docs.filter(
    col("file_size_mb") <= lit(5.0),  # Documents 5MB or smaller
    num_cpus=0.1
)

print(f"Document filtering results:")
print(f"  High priority documents: {high_priority_docs.count():,}")
print(f"  Financial documents: {financial_docs.count():,}")
print(f"  Compliance documents: {compliance_docs.count():,}")
print(f"  Large documents (>5MB): {large_documents.count():,}")
print(f"  Small documents (5MB): {small_documents.count():,}")
```

## Step 3: Text Extraction and Enrichment

### Text Chunking for LLM Processing

```python
def create_text_chunks_for_analytics(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create text chunks optimized for LLM processing and analytics."""
    
    text = record["extracted_text"]
    chunk_size = 1500  # Optimal size for LLM context window
    overlap = 150      # 10% overlap for context preservation
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        # Create analytics-optimized chunk record
        chunk_record = {
            **record,
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": len(chunk_text),
            "chunk_word_count": len(chunk_text.split()),
            "chunk_start_position": start,
            "chunk_end_position": end,
            "chunking_timestamp": datetime.now().isoformat()
        }
        
        chunks.append(chunk_record)
        
        # Move to next chunk with overlap
        start = end - overlap
        chunk_index += 1
        
        if start >= len(text):
            break
    
    # Update total chunks for all records
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)
    
    return chunks

# Apply text chunking using Ray Data flat_mapprint("Creating text chunks for LLM processing...")

chunked_documents = high_quality_docs.flat_map(
    create_text_chunks_for_analytics,
    num_cpus=0.5  # Medium complexity chunking operation
)

print(f"Text chunking completed: {chunked_documents.count():,} text chunks created")
```

### Content Preprocessing for Analytics

```python
def preprocess_content_for_analytics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess text content for analytics and LLM processing."""
    
    text = record["chunk_text"]
    
    # Basic text cleaning and normalization
    cleaned_text = text.strip()
    cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
    
    # Extract content indicators for analytics
    content_indicators = {
        "contains_financial_terms": any(term in text.lower() for term in ["revenue", "profit", "earnings", "financial"]),
        "contains_compliance_terms": any(term in text.lower() for term in ["compliance", "regulatory", "audit", "risk"]),
        "contains_dates": any(term in text for term in ["2023", "2024", "January", "February", "March"]),
        "contains_numbers": any(char.isdigit() for char in text),
        "contains_entities": any(word[0].isupper() for word in text.split() if len(word) > 2)
    }
    
    # Calculate content metrics
    sentences = text.split('. ')
    paragraphs = text.split('\n')
    
    content_metrics = {
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
        "text_density": record["chunk_word_count"] / record["chunk_length"] if record["chunk_length"] > 0 else 0
    }
    
    return {
        **record,
        "cleaned_text": cleaned_text,
        "content_indicators": json.dumps(content_indicators),
        "content_metrics": json.dumps(content_metrics),
        "preprocessing_timestamp": datetime.now().isoformat(),
        "llm_ready": len(cleaned_text) > 50  # Minimum text for LLM processing
    }

# Apply content preprocessing using Ray Dataprint("Preprocessing content for analytics...")

preprocessed_chunks = chunked_documents.map_batches(
    lambda batch: [preprocess_content_for_analytics(record, batch_format="pandas") for record in batch],
    num_cpus=0.25,  # Light preprocessing
    batch_size=2000
)

# Filter chunks ready for LLM processingllm_ready_chunks = preprocessed_chunks.filter(
    col("llm_ready") == lit(True),
    num_cpus=0.1
)

print(f"Content preprocessing completed: {llm_ready_chunks.count():,} chunks ready for LLM")
```

## Step 4: LLM-Powered Content Analysis

### Configure Ray Data LLM Processing

```python
# Configure LLM processing using Ray Data LLM packageprint("Configuring Ray Data LLM processing...")

# Install required packages for LLM processing# Pip install -U vllm==0.7.2
try:
    from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
    
    # Configure vLLM engine for document analysis
    llm_config = vLLMEngineProcessorConfig(
        model_source="unsloth/Llama-3.1-8B-Instruct",
        engine_kwargs={
            "max_model_len": 16384,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        concurrency=1,
        batch_size=32,
        accelerator_type="A10G"
    )
    
    print("vLLM configuration created successfully")
    llm_available = True
    
except ImportError:
    print("Ray Data LLM package not available - will use fallback processing")
    llm_config = None
    llm_available = False
```

### Business Intelligence Prompt Engineering

```python
def create_business_analysis_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    """Create specialized prompts for business document analysis using Ray Data LLM."""
    
    business_context = f"Document Type: {row['document_type']}, Category: {row['business_category']}"
    text_content = row["cleaned_text"]
    
    # Create system prompt for structured business analysis
    system_prompt = """You are a business analyst specializing in document intelligence for enterprise data warehouses. 
    Analyze documents and extract structured business information for analytics and reporting.
    
    For each document chunk, provide analysis in this exact format:
    SUMMARY: [2-3 sentence business summary]
    KEY_METRICS: [Extract numerical data, percentages, financial figures]
    ENTITIES: [Companies, people, locations, products mentioned]
    CATEGORY: [financial/operational/strategic/compliance/research]
    PRIORITY: [high/medium/low based on business importance]
    
    Keep responses structured and factual."""
    
    # Create user prompt with business context
    user_prompt = f"""Analyze this business document chunk for data warehouse ingestion:
    
    Context: {business_context}
    
    Content: {text_content}
    
    Provide structured analysis following the format above."""
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "sampling_params": {
            "temperature": 0.2,  # Low temperature for consistent business analysis
            "max_tokens": 300,   # Sufficient for structured output
            "top_p": 0.9
        }
    }

def extract_structured_analysis(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured analysis from LLM response for data warehouse storage."""
    
    llm_response = row.get("generated_text", "")
    
    # Parse LLM response for structured data warehouse fields
    summary = extract_summary_from_response(llm_response)
    metrics = extract_metrics_from_response(llm_response)
    entities = extract_entities_from_response(llm_response)
    category = extract_category_from_response(llm_response)
    priority = extract_priority_from_response(llm_response)
    
    return {
        **row,
        "llm_summary": summary,
        "llm_metrics": json.dumps(metrics),
        "llm_entities": json.dumps(entities),
        "llm_category": category,
        "llm_priority": priority,
        "llm_analysis_timestamp": datetime.now().isoformat()
    }

def extract_summary_from_response(response: str) -> str:
    """Extract summary from LLM response."""
    if "SUMMARY:" in response:
        start = response.find("SUMMARY:") + len("SUMMARY:")
        end = response.find("KEY_METRICS:") if "KEY_METRICS:" in response else len(response)
        return response[start:end].strip()
    return "Summary not available"

def extract_metrics_from_response(response: str) -> List[str]:
    """Extract metrics from LLM response."""
    if "KEY_METRICS:" in response:
        start = response.find("KEY_METRICS:") + len("KEY_METRICS:")
        end = response.find("ENTITIES:") if "ENTITIES:" in response else len(response)
        metrics_text = response[start:end].strip()
        # Extract numerical values and percentages
        metrics = []
        words = metrics_text.split()
        for word in words:
            if any(char.isdigit() for char in word) or '%' in word:
                metrics.append(word.strip(".,!?;:()[]{}"))
        return metrics[:10]  # Limit to top 10 metrics
    return []

def extract_entities_from_response(response: str) -> List[str]:
    """Extract business entities from LLM response."""
    if "ENTITIES:" in response:
        start = response.find("ENTITIES:") + len("ENTITIES:")
        end = response.find("CATEGORY:") if "CATEGORY:" in response else len(response)
        entity_text = response[start:end].strip()
        # Extract entities (companies, people, locations)
        entities = []
        words = entity_text.split()
        for word in words:
            clean_word = word.strip(".,!?;:()[]{}\"'")
            if len(clean_word) > 2 and clean_word[0].isupper():
                entities.append(clean_word)
        return list(set(entities))[:15]  # Unique entities, max 15
    return []

def extract_category_from_response(response: str) -> str:
    """Extract category classification from LLM response."""
    if "CATEGORY:" in response:
        start = response.find("CATEGORY:") + len("CATEGORY:")
        end = response.find("PRIORITY:") if "PRIORITY:" in response else len(response)
        category_text = response[start:end].strip().lower()
        
        if "financial" in category_text:
            return "financial"
        elif "operational" in category_text:
            return "operational"
        elif "strategic" in category_text:
            return "strategic"
        elif "compliance" in category_text:
            return "compliance"
        elif "research" in category_text:
            return "research"
    return "general"

def extract_priority_from_response(response: str) -> str:
    """Extract priority assessment from LLM response."""
    if "PRIORITY:" in response:
        start = response.find("PRIORITY:") + len("PRIORITY:")
        priority_text = response[start:].strip().lower()
        
        if "high" in priority_text:
            return "high"
        elif "medium" in priority_text:
            return "medium"
        else:
            return "low"
    return "medium"

# Build and apply LLM processor using Ray Data LLM packageif llm_available:
    print("Building LLM processor for business analysis...")
    
    llm_processor = build_llm_processor(
        llm_config,
        preprocess=create_business_analysis_prompt,
        postprocess=extract_structured_analysis
    )
    
    # Apply LLM processing to document chunks
    print("Applying LLM analysis to document chunks...")
    
    llm_analyzed_docs = llm_processor(llm_ready_chunks)
    
    print(f"LLM analysis completed: {llm_analyzed_docs.count():,} chunks analyzed")
    
else:
    print("Using fallback analysis without LLM...")
    
    def fallback_analysis(record: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when Ray Data LLM package is not available."""
        return {
            **record,
            "llm_summary": f"Document analysis for {record['document_type']} in {record['business_category']} category",
            "llm_metrics": json.dumps([]),
            "llm_entities": json.dumps([]),
            "llm_category": record["business_category"],
            "llm_priority": record["processing_priority"],
            "llm_analysis_timestamp": datetime.now().isoformat()
        }
    
    llm_analyzed_docs = llm_ready_chunks.map_batches(
        lambda batch: [fallback_analysis(record, batch_format="pandas") for record in batch],
        num_cpus=0.25,  # Light fallback processing
        batch_size=2000
    )
    
    print(f"Fallback analysis completed: {llm_analyzed_docs.count():,} chunks processed")
```

### Content Enrichment and Entity Extraction

```python
def enrich_content_with_business_intelligence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich content with additional business intelligence for analytics."""
    
    # Parse LLM-extracted data
    try:
        llm_metrics = json.loads(record.get("llm_metrics", "[]"))
        llm_entities = json.loads(record.get("llm_entities", "[]"))
        content_indicators = json.loads(record.get("content_indicators", "{}"))
    except:
        llm_metrics = []
        llm_entities = []
        content_indicators = {}
    
    # Calculate enrichment metrics
    enrichment_metrics = {
        "metrics_extracted_count": len(llm_metrics),
        "entities_extracted_count": len(llm_entities),
        "has_financial_indicators": content_indicators.get("contains_financial_terms", False),
        "has_compliance_indicators": content_indicators.get("contains_compliance_terms", False),
        "has_date_references": content_indicators.get("contains_dates", False),
        "has_numerical_data": content_indicators.get("contains_numbers", False),
        "content_richness_score": len(llm_metrics) + len(llm_entities)
    }
    
    # Determine analytics value for data warehouse
    analytics_value = "high" if enrichment_metrics["content_richness_score"] > 5 else "medium" if enrichment_metrics["content_richness_score"] > 2 else "low"
    
    return {
        **record,
        "enrichment_metrics": json.dumps(enrichment_metrics),
        "analytics_value": analytics_value,
        "enrichment_timestamp": datetime.now().isoformat()
    }

# Apply content enrichment using Ray Dataprint("Enriching content with business intelligence...")

enriched_documents = llm_analyzed_docs.map_batches(
    lambda batch: [enrich_content_with_business_intelligence(record, batch_format="pandas") for record in batch],
    num_cpus=0.25,  # Light enrichment processing
    batch_size=2000
)

print(f"Content enrichment completed: {enriched_documents.count():,} documents enriched")
```

## Step 4: Structured Data Transformation

### Data Warehouse Schema Creation

```python
def create_data_warehouse_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """Create final data warehouse schema optimized for analytics and BI tools."""
    
    # Parse JSON fields for structured storage
    try:
        llm_metrics = json.loads(record.get("llm_metrics", "[]"))
        llm_entities = json.loads(record.get("llm_entities", "[]"))
        enrichment_metrics = json.loads(record.get("enrichment_metrics", "{}"))
    except:
        llm_metrics = []
        llm_entities = []
        enrichment_metrics = {}
    
    # Create analytics-optimized record for data warehouse
    warehouse_record = {
        # Primary identifiers for data warehouse
        "document_id": record["document_id"],
        "chunk_id": record["chunk_id"],
        
        # Dimensional attributes for analytics
        "business_category": record["business_category"],
        "document_type": record["document_type"],
        "department": record.get("department", "unknown"),
        "file_extension": record["file_extension"],
        "quality_rating": record["quality_rating"],
        "llm_category": record["llm_category"],
        "llm_priority": record["llm_priority"],
        "analytics_value": record["analytics_value"],
        
        # Fact measures for analytics
        "file_size_mb": record["file_size_mb"],
        "text_length": record["text_length"],
        "word_count": record["word_count"],
        "chunk_length": record["chunk_length"],
        "chunk_word_count": record["chunk_word_count"],
        "quality_score": record["quality_score"],
        "priority_score": record["priority_score"],
        "chunk_index": record["chunk_index"],
        "total_chunks": record["total_chunks"],
        "estimated_pages": record["estimated_pages"],
        "metrics_count": len(llm_metrics),
        "entities_count": len(llm_entities),
        "content_richness_score": enrichment_metrics.get("content_richness_score", 0),
        
        # Content fields
        "text_content": record["chunk_text"],
        "llm_summary": record["llm_summary"],
        "source_path": record["file_path"],
        "file_name": record["file_name"],
        
        # Analytics flags
        "has_financial_content": enrichment_metrics.get("has_financial_indicators", False),
        "has_compliance_content": enrichment_metrics.get("has_compliance_indicators", False),
        "has_numerical_data": enrichment_metrics.get("has_numerical_data", False),
        "has_date_references": enrichment_metrics.get("has_date_references", False),
        
        # Processing timestamps for data lineage
        "discovery_date": record["discovery_timestamp"][:10],  # Extract date only
        "processing_date": datetime.now().isoformat()[:10],
        "discovery_timestamp": record["discovery_timestamp"],
        "extraction_timestamp": record["text_extraction_timestamp"],
        "llm_analysis_timestamp": record["llm_analysis_timestamp"],
        "enrichment_timestamp": record["enrichment_timestamp"],
        
        # Data warehouse metadata
        "pipeline_version": "1.0",
        "processing_engine": "ray_data",
        "llm_engine": "ray_data_llm_vllm" if llm_available else "fallback",
        "source_system": "enterprise_data_lake"
    }
    
    return warehouse_record

# Apply data warehouse schema transformationprint("Creating data warehouse schema...")

warehouse_dataset = enriched_documents.map_batches(
    lambda batch: [create_data_warehouse_schema(record, batch_format="pandas") for record in batch],
    num_cpus=0.25,  # Light schema transformation
    batch_size=2000
)

print(f"Data warehouse schema created: {warehouse_dataset.count():,} records")
```

### Data Validation for Warehouse Integration

```python
def validate_warehouse_data(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data quality for data warehouse integration."""
    
    validation_results = {
        "has_required_ids": all(field in record and record[field] for field in ["document_id", "chunk_id"]),
        "has_content": len(record.get("text_content", "")) > 10,
        "has_business_classification": record.get("business_category") != "general",
        "has_quality_metrics": record.get("quality_score", 0) > 0,
        "has_processing_timestamps": all(field in record for field in ["discovery_timestamp", "llm_analysis_timestamp"]),
        "has_analytics_metadata": record.get("analytics_value") in ["high", "medium", "low"]
    }
    
    # Calculate validation score
    validation_score = sum(validation_results.values())
    validation_passed = validation_score >= 5  # Require 5/6 validations to pass
    
    return {
        **record,
        "validation_results": json.dumps(validation_results),
        "validation_score": validation_score,
        "validation_passed": validation_passed,
        "validation_timestamp": datetime.now().isoformat(),
        "warehouse_ready": validation_passed
    }

# Apply data validation using Ray Dataprint("Validating data for warehouse integration...")

validated_documents = warehouse_dataset.map_batches(
    lambda batch: [validate_warehouse_data(record, batch_format="pandas") for record in batch],
    num_cpus=0.25,  # Light validation processing
    batch_size=2000
)

# Filter documents ready for warehouse storagewarehouse_ready_docs = validated_documents.filter(
    col("warehouse_ready") == lit(True),
    num_cpus=0.1
)

print(f"Data validation completed:")
print(f"  Total documents validated: {validated_documents.count():,}")
print(f"  Warehouse-ready documents: {warehouse_ready_docs.count():,}")
```

## Step 5: Data Warehouse Output

### Document Processing Analytics

```python
# Visualize document processing pipeline resultsimport matplotlib.pyplot as plt
import numpy as np

# Generate document processing analytics using utility function
from util.viz_utils import visualize_document_processing, create_processing_funnel

fig = visualize_document_processing()
print("Document processing visualization created")

# Create interactive processing funnel
funnel_fig = create_processing_funnel(quality_assessed_docs.to_pandas())
funnel_fig.write_html('document_processing_funnel.html')
print("Interactive processing funnel saved")
```

### Write to Data Warehouse Formats

```python
# Write main warehouse table with partitioning for query optimizationprint("Writing to data warehouse formats...")

# Main warehouse table partitioned by business category and processing datewarehouse_ready_docs.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/main_table/",
    partition_cols=["business_category", "processing_date"],
    compression="snappy",
    num_cpus=0.1  # Moderate write concurrency
)

print("Main warehouse table written with partitioning")

# Create business-specific datasets for targeted analyticsprint("Creating business-specific analytics datasets...")

# Financial documents for financial analyticsfinancial_analytics = warehouse_ready_docs.filter(
    col("business_category") == lit("finance"),
    num_cpus=0.1
).select_columns([
    "document_id", "chunk_id", "text_content", "llm_summary", 
    "file_size_mb", "word_count", "quality_score", "processing_date",
    "has_financial_content", "metrics_count"
])

financial_analytics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/analytics/financial/",
    partition_cols=["processing_date"],
    compression="snappy",
    num_cpus=0.1
)

# Compliance documents for regulatory reportingcompliance_analytics = warehouse_ready_docs.filter(
    col("business_category") == lit("compliance"),
    num_cpus=0.1
).select_columns([
    "document_id", "chunk_id", "text_content", "llm_summary",
    "quality_score", "llm_priority", "processing_date",
    "has_compliance_content", "entities_count"
])

compliance_analytics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/analytics/compliance/",
    partition_cols=["processing_date"],
    compression="snappy",
    num_cpus=0.1
)

# Research documents for business intelligenceresearch_analytics = warehouse_ready_docs.filter(
    col("business_category") == lit("research"),
    num_cpus=0.1
).select_columns([
    "document_id", "chunk_id", "text_content", "llm_summary",
    "analytics_value", "content_richness_score", "processing_date"
])

research_analytics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/analytics/research/",
    partition_cols=["processing_date"],
    compression="snappy",
    num_cpus=0.1
)

print(f"Business-specific datasets created:")
print(f"  Financial analytics: {financial_analytics.count():,} records")
print(f"  Compliance analytics: {compliance_analytics.count():,} records")
print(f"  Research analytics: {research_analytics.count():,} records")
```

### Create Analytics Summary Tables

```python
# Create comprehensive analytics summaries using Ray Data native operationsprint("Creating analytics summary tables for data warehouse...")

from ray.data.aggregate import Count, Sum, Mean, Max, Min

# Document processing metrics by category and dateprocessing_metrics = warehouse_ready_docs.groupby(["business_category", "processing_date"]).aggregate(
    Count(),
    Sum("file_size_mb"),
    Mean("text_length"),
    Mean("word_count"),
    Mean("quality_score"),
    Sum("chunk_word_count"),
    Mean("content_richness_score")
)

processing_metrics.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/processing_metrics/",
    partition_cols=["processing_date"],
    compression="snappy",
    num_cpus=0.1
)

# Document quality distribution analysisquality_distribution = warehouse_ready_docs.groupby(["quality_rating", "business_category"]).aggregate(
    Count(),
    Mean("text_length"),
    Mean("entities_count"),
    Mean("metrics_count"),
    Sum("file_size_mb")
)

quality_distribution.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/quality_distribution/",
    compression="snappy",
    num_cpus=0.1
)

# Llm analysis effectiveness summaryllm_analysis_summary = warehouse_ready_docs.groupby(["llm_category", "llm_priority"]).aggregate(
    Count(),
    Mean("chunk_length"),
    Sum("word_count"),
    Mean("content_richness_score")
)

llm_analysis_summary.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/llm_analysis/",
    compression="snappy",
    num_cpus=0.1
)

# Document type and format analysisformat_analysis = warehouse_ready_docs.groupby(["document_type", "file_extension"]).aggregate(
    Count(),
    Mean("file_size_mb"),
    Sum("estimated_pages"),
    Mean("analytics_value")
)

format_analysis.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/format_analysis/",
    compression="snappy",
    num_cpus=0.1
)

print("Analytics summary tables created:")
print("  - Processing metrics by category and date")
print("  - Quality distribution analysis")
print("  - LLM analysis effectiveness summary")
print("  - Document format analysis")
```

### Data Catalog and Lineage

```python
# Create comprehensive data catalog for warehouse integrationdef create_data_warehouse_catalog():
    """Create data catalog metadata for all warehouse tables."""
    
    catalog_metadata = {
        "catalog_version": "1.0",
        "created_timestamp": datetime.now().isoformat(),
        "pipeline_name": "ray_data_document_ingestion",
        "total_input_documents": document_collection.count(),
        "total_warehouse_records": warehouse_ready_docs.count(),
        "processing_engine": "ray_data",
        "llm_integration": "ray_data_llm_package" if llm_available else "fallback",
        
        "warehouse_tables": {
            "main_table": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/main_table/",
                "description": "Main document warehouse table with full content and metadata",
                "partitioning": ["business_category", "processing_date"],
                "record_count": warehouse_ready_docs.count(),
                "schema_version": "1.0",
                "update_frequency": "daily",
                "retention_policy": "7_years"
            },
            "financial_analytics": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/analytics/financial/",
                "description": "Financial documents optimized for financial analytics",
                "partitioning": ["processing_date"],
                "record_count": financial_analytics.count(),
                "specialized_for": "financial_reporting_and_analysis"
            },
            "compliance_analytics": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/analytics/compliance/",
                "description": "Compliance documents for regulatory reporting",
                "partitioning": ["processing_date"],
                "record_count": compliance_analytics.count(),
                "specialized_for": "regulatory_compliance_monitoring"
            },
            "research_analytics": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/analytics/research/",
                "description": "Research documents for business intelligence",
                "partitioning": ["processing_date"],
                "record_count": research_analytics.count(),
                "specialized_for": "business_intelligence_research"
            }
        },
        
        "summary_tables": {
            "processing_metrics": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/summaries/processing_metrics/",
                "description": "Daily processing metrics by business category",
                "aggregation_level": "daily_by_category"
            },
            "quality_distribution": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/summaries/quality_distribution/",
                "description": "Document quality distribution analysis",
                "aggregation_level": "quality_rating_by_category"
            },
            "llm_analysis": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/summaries/llm_analysis/",
                "description": "LLM analysis effectiveness metrics",
                "aggregation_level": "llm_category_by_priority"
            },
            "format_analysis": {
                "path": f"{OUTPUT_WAREHOUSE_PATH}/summaries/format_analysis/",
                "description": "Document format and type analysis",
                "aggregation_level": "document_type_by_format"
            }
        },
        
        "data_lineage": {
            "source_system": "enterprise_data_lake",
            "processing_pipeline": "ray_data_document_ingestion",
            "llm_processing": "ray_data_llm_package",
            "output_format": "parquet_partitioned",
            "compression": "snappy",
            "schema_version": "1.0"
        },
        
        "ray_data_operations_used": [
            "read_binary_files() - Large-scale document discovery",
            "map() - Metadata extraction and preprocessing",
            "map_batches() - Text extraction and content analysis",
            "filter() - Quality assessment and document filtering",
            "flat_map() - Text chunking for LLM processing",
            "groupby().aggregate() - Business analytics and summaries",
            "select_columns() - Schema optimization for analytics",
            "write_parquet() - Data warehouse output with partitioning",
            "ray.data.llm - Integrated LLM processing for content analysis"
        ]
    }
    
    return catalog_metadata

# Create and save data catalog using Ray Datacatalog_data = create_data_warehouse_catalog()

# Save catalog metadata as JSONcatalog_dataset = ray.data.from_items([catalog_data])
catalog_dataset.write_json(
    f"{OUTPUT_WAREHOUSE_PATH}/catalog/",
    compression="gzip",
    num_cpus=0.1
)

print("Data warehouse catalog created and saved")
```

## Verification and Analytics Validation

### Comprehensive Output Verification

```python
# Verify all data warehouse outputsprint("Verifying data warehouse integration...")

# Verify main warehouse tablemain_table_verification = ray.data.read_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/main_table/",
    num_cpus=0.025
)

# Verify summary tablesprocessing_metrics_verification = ray.data.read_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/processing_metrics/",
    num_cpus=0.025
)

quality_verification = ray.data.read_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/summaries/quality_distribution/",
    num_cpus=0.025
)

print(f"Data warehouse verification:")
print(f"  Main table records: {main_table_verification.count():,}")
print(f"  Processing metrics: {processing_metrics_verification.count():,}")
print(f"  Quality analysis: {quality_verification.count():,}")
print(f"  Schema compatibility:  Verified")

# Display sample analytics dataprint("Sample analytics data:")
sample_analytics = main_table_verification.take(3)
for i, record in enumerate(sample_analytics):
    print(f"  {i+1}. Doc: {record['document_id'][:8]}, Category: {record['business_category']}, "
          f"Type: {record['document_type']}, Words: {record['word_count']}, Quality: {record['quality_rating']}")
```

### Business Intelligence Integration

```python
# Create BI-ready views using Ray Data operationsprint("Creating business intelligence views...")

# Executive dashboard viewexecutive_view = warehouse_ready_docs.select_columns([
    "document_id", "business_category", "document_type", "llm_summary",
    "quality_score", "analytics_value", "processing_date"
]).filter(
    col("analytics_value") == lit("high"),
    num_cpus=0.1
)

executive_view.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/bi_views/executive_dashboard/",
    partition_cols=["business_category"],
    compression="snappy",
    num_cpus=0.1
)

# Operational metrics viewoperational_view = warehouse_ready_docs.select_columns([
    "document_id", "file_size_mb", "text_length", "word_count",
    "quality_score", "processing_date", "llm_category"
]).filter(
    col("quality_score") >= lit(3),
    num_cpus=0.1
)

operational_view.write_parquet(
    f"{OUTPUT_WAREHOUSE_PATH}/bi_views/operational_metrics/",
    partition_cols=["processing_date"],
    compression="snappy",
    num_cpus=0.1
)

print(f"Business intelligence views created:")
print(f"  Executive dashboard: {executive_view.count():,} records")
print(f"  Operational metrics: {operational_view.count():,} records")
```

## Pipeline Performance Summary

```python
# Calculate comprehensive pipeline processing metricsprint("=" * 80)
print("DOCUMENT INGESTION PIPELINE SUMMARY")
print("=" * 80)

# Processing metricstotal_input_docs = document_collection.count()
total_chunks_created = enriched_documents.count()
total_warehouse_records = warehouse_ready_docs.count()
total_financial_records = financial_analytics.count()
total_compliance_records = compliance_analytics.count()

print(f" PROCESSING METRICS:")
print(f"    Input documents discovered: {total_input_docs:,}")
print(f"    Text chunks created: {total_chunks_created:,}")
print(f"    Warehouse records generated: {total_warehouse_records:,}")
print(f"    Financial records: {total_financial_records:,}")
print(f"    Compliance records: {total_compliance_records:,}")
print(f"    Average chunks per document: {total_chunks_created / total_input_docs:.1f}")

# Data warehouse statisticsbusiness_categories = warehouse_ready_docs.select_columns(["business_category"]).distinct(num_cpus=0.1).take_all()
document_types = warehouse_ready_docs.select_columns(["document_type"]).distinct(num_cpus=0.1).take_all()

print(f" DATA WAREHOUSE STATISTICS:")
print(f"    Business categories: {len(business_categories)}")
print(f"    Document types: {len(document_types)}")
print(f"    Output formats: Parquet with Snappy compression")
print(f"    Partitioning strategy: business_category, processing_date")
print(f"    Summary tables: 4 analytics tables created")
print(f"    BI views: 2 specialized views for business intelligence")

print(f" RAY DATA OPERATIONS DEMONSTRATED:")
print(f"    read_binary_files() - Large-scale document discovery")
print(f"    map() - Metadata extraction and preprocessing")
print(f"    map_batches() - Text extraction and content analysis")
print(f"    filter() - Quality assessment and document filtering")
print(f"    flat_map() - Text chunking for LLM processing")
print(f"    groupby().aggregate() - Business analytics and summaries")
print(f"    select_columns() - Schema optimization for analytics")
print(f"    distinct() - Data deduplication and analysis")
print(f"    write_parquet() - Data warehouse output with partitioning")
print(f"    write_json() - Metadata and catalog management")
print(f"    ray.data.llm - Integrated LLM processing for content analysis")

print(f" DATA WAREHOUSE INTEGRATION:")
print(f"    Partitioned tables for query optimization")
print(f"    Business-specific analytics datasets")
print(f"    Summary tables for operational monitoring")
print(f"    BI views for executive dashboards")
print(f"    Data catalog for metadata management")
print(f"    Schema standardization for BI tool integration")
print(f"    Data lineage tracking for governance")

print("=" * 80)
```

## Advanced Ray Data Features Demonstrated

### Native Operations Usage

This template showcases comprehensive Ray Data native operations:

**Data Discovery and Loading:**
- `ray.data.read_binary_files()` - Efficient binary file reading from S3
- `include_paths=True` - Path tracking for data lineage
- `num_cpus=0.025` - High I/O concurrency for large collections

**Data Transformation:**
- `map()` - Row-wise metadata extraction and preprocessing
- `map_batches()` - Vectorized text extraction and content analysis
- `flat_map()` - One-to-many text chunking operations

**Data Filtering and Selection:**
- `filter()` with expressions API - Optimized filtering using `col()` and `lit()`
- `select_columns()` - Schema optimization for analytics
- `distinct()` - Data deduplication and unique value analysis

**Analytics and Aggregation:**
- `groupby().aggregate()` - Distributed business analytics
- Native aggregation functions: `Count()`, `Sum()`, `Mean()`, `Max()`, `Min()`
- Multi-dimensional grouping for comprehensive analytics

**Data Output and Storage:**
- `write_parquet()` - Efficient columnar storage with compression
- `write_json()` - Metadata and catalog management
- `partition_cols` - Query optimization through partitioning
- `compression="snappy"` - Storage optimization

**LLM Integration:**
- `ray.data.llm.vLLMEngineProcessorConfig` - LLM engine configuration
- `build_llm_processor()` - Integrated LLM processing pipeline
- Custom preprocessing and postprocessing functions
- Batch inference optimization for document analysis

### Performance Optimization Patterns

**Resource Allocation Following Ray Data Best Practices:**
- **I/O Operations**: `num_cpus=0.025-0.05` for maximum parallelism
- **Light Processing**: `num_cpus=0.25` for quality assessment and validation
- **Medium Processing**: `num_cpus=0.5` for text chunking and content analysis
- **Heavy Processing**: `num_cpus=1.0` for text extraction and complex transformations
- **Write Operations**: `num_cpus=0.1` for balanced output concurrency

**Batch Size Optimization:**
- **Small batches (500)** for heavy text extraction
- **Medium batches (1000-2000)** for content analysis and validation
- **Large batches (2000)** for light schema transformations
- **LLM batches (32)** for optimal GPU utilization

## Cleanup

```python
# Cleanup Ray resources following best practicesif ray.is_initialized():
    ray.shutdown()
print("Ray shutdown completed")
```

## Ray Data Performance Summary

This template demonstrates comprehensive Ray Data best practices for enterprise document ingestion:

- **Native operations**: Extensive use of `read_binary_files()`, `filter()`, `groupby()`, `aggregate()`, `select_columns()`, `distinct()`
- **Proper resource allocation**: All operations specify `num_cpus` following optimization guidelines
- **Expressions API**: Using `col()` and `lit()` for query optimization and performance
- **LLM integration**: Using `ray.data.llm` package for content analysis and structured extraction
- **Monitoring setup**: Progress bars enabled for performance visibility and bottleneck identification
- **Data warehouse patterns**: Partitioned output, summary tables, BI views, and data catalog
- **Analytics integration**: Business-specific datasets optimized for different analytics use cases
- **Large-scale processing**: Optimized for enterprise document collections and data lake ingestion
- **Resource cleanup**: Proper Ray shutdown procedures following best practices

This pipeline transforms unstructured data lake documents into structured, analytics-ready datasets suitable for enterprise data warehouse consumption, business intelligence tools, and advanced analytics workflows.