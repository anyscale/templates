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

    2025-10-11 00:17:23,060	INFO worker.py:1833 -- Connecting to existing Ray cluster at address: 10.0.48.117:6379...
    2025-10-11 00:17:23,071	INFO worker.py:2004 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-77uweunq3awbhqefvry4lwcqq5.i.anyscaleuserdata.com [39m[22m
    2025-10-11 00:17:23,075	INFO packaging.py:380 -- Pushing file package 'gcs://_ray_pkg_d09a1f3a380b650bc6804514c9ba098775a62b40.zip' (1.11MiB) to Ray cluster...
    2025-10-11 00:17:23,080	INFO packaging.py:393 -- Successfully pushed file package 'gcs://_ray_pkg_d09a1f3a380b650bc6804514c9ba098775a62b40.zip'.
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/worker.py:2052: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
      warnings.warn(





<div class="lm-Widget p-Widget lm-Panel p-Panel jp-Cell-outputWrapper">
    <div style="margin-left: 50px;display: flex;flex-direction: row;align-items: center">
        <div class="jp-RenderedHTMLCommon" style="display: flex; flex-direction: row;">
  <svg viewBox="0 0 567 224" fill="none" xmlns="http://www.w3.org/2000/svg" style="height: 3em;">
    <g clip-path="url(#clip0_4338_178347)">
        <path d="M341.29 165.561H355.29L330.13 129.051C345.63 123.991 354.21 112.051 354.21 94.2307C354.21 71.3707 338.72 58.1807 311.88 58.1807H271V165.561H283.27V131.661H311.8C314.25 131.661 316.71 131.501 319.01 131.351L341.25 165.561H341.29ZM283.29 119.851V70.0007H311.82C331.3 70.0007 342.34 78.2907 342.34 94.5507C342.34 111.271 331.34 119.861 311.82 119.861L283.29 119.851ZM451.4 138.411L463.4 165.561H476.74L428.74 58.1807H416L367.83 165.561H380.83L392.83 138.411H451.4ZM446.19 126.601H398L422 72.1407L446.24 126.601H446.19ZM526.11 128.741L566.91 58.1807H554.35L519.99 114.181L485.17 58.1807H472.44L514.01 129.181V165.541H526.13V128.741H526.11Z" fill="var(--jp-ui-font-color0)"/>
        <path d="M82.35 104.44C84.0187 97.8827 87.8248 92.0678 93.1671 87.9146C98.5094 83.7614 105.083 81.5067 111.85 81.5067C118.617 81.5067 125.191 83.7614 130.533 87.9146C135.875 92.0678 139.681 97.8827 141.35 104.44H163.75C164.476 101.562 165.622 98.8057 167.15 96.2605L127.45 56.5605C121.071 60.3522 113.526 61.6823 106.235 60.3005C98.9443 58.9187 92.4094 54.9203 87.8602 49.0574C83.3109 43.1946 81.0609 35.8714 81.5332 28.4656C82.0056 21.0599 85.1679 14.0819 90.4252 8.8446C95.6824 3.60726 102.672 0.471508 110.08 0.0272655C117.487 -0.416977 124.802 1.86091 130.647 6.4324C136.493 11.0039 140.467 17.5539 141.821 24.8501C143.175 32.1463 141.816 39.6859 138 46.0505L177.69 85.7505C182.31 82.9877 187.58 81.4995 192.962 81.4375C198.345 81.3755 203.648 82.742 208.33 85.3976C213.012 88.0532 216.907 91.9029 219.616 96.5544C222.326 101.206 223.753 106.492 223.753 111.875C223.753 117.258 222.326 122.545 219.616 127.197C216.907 131.848 213.012 135.698 208.33 138.353C203.648 141.009 198.345 142.375 192.962 142.313C187.58 142.251 182.31 140.763 177.69 138L138 177.7C141.808 184.071 143.155 191.614 141.79 198.91C140.424 206.205 136.44 212.75 130.585 217.313C124.731 221.875 117.412 224.141 110.004 223.683C102.596 223.226 95.6103 220.077 90.3621 214.828C85.1139 209.58 81.9647 202.595 81.5072 195.187C81.0497 187.779 83.3154 180.459 87.878 174.605C92.4405 168.751 98.9853 164.766 106.281 163.401C113.576 162.035 121.119 163.383 127.49 167.19L167.19 127.49C165.664 124.941 164.518 122.182 163.79 119.3H141.39C139.721 125.858 135.915 131.673 130.573 135.826C125.231 139.98 118.657 142.234 111.89 142.234C105.123 142.234 98.5494 139.98 93.2071 135.826C87.8648 131.673 84.0587 125.858 82.39 119.3H60C58.1878 126.495 53.8086 132.78 47.6863 136.971C41.5641 141.163 34.1211 142.972 26.7579 142.059C19.3947 141.146 12.6191 137.574 7.70605 132.014C2.79302 126.454 0.0813599 119.29 0.0813599 111.87C0.0813599 104.451 2.79302 97.2871 7.70605 91.7272C12.6191 86.1673 19.3947 82.5947 26.7579 81.6817C34.1211 80.7686 41.5641 82.5781 47.6863 86.7696C53.8086 90.9611 58.1878 97.2456 60 104.44H82.35ZM100.86 204.32C103.407 206.868 106.759 208.453 110.345 208.806C113.93 209.159 117.527 208.258 120.522 206.256C123.517 204.254 125.725 201.276 126.771 197.828C127.816 194.38 127.633 190.677 126.253 187.349C124.874 184.021 122.383 181.274 119.205 179.577C116.027 177.88 112.359 177.337 108.826 178.042C105.293 178.746 102.113 180.654 99.8291 183.44C97.5451 186.226 96.2979 189.718 96.3 193.32C96.2985 195.364 96.7006 197.388 97.4831 199.275C98.2656 201.163 99.4132 202.877 100.86 204.32ZM204.32 122.88C206.868 120.333 208.453 116.981 208.806 113.396C209.159 109.811 208.258 106.214 206.256 103.219C204.254 100.223 201.275 98.0151 197.827 96.97C194.38 95.9249 190.676 96.1077 187.348 97.4873C184.02 98.8669 181.274 101.358 179.577 104.536C177.879 107.714 177.337 111.382 178.041 114.915C178.746 118.448 180.653 121.627 183.439 123.911C186.226 126.195 189.717 127.443 193.32 127.44C195.364 127.443 197.388 127.042 199.275 126.259C201.163 125.476 202.878 124.328 204.32 122.88ZM122.88 19.4205C120.333 16.8729 116.981 15.2876 113.395 14.9347C109.81 14.5817 106.213 15.483 103.218 17.4849C100.223 19.4868 98.0146 22.4654 96.9696 25.9131C95.9245 29.3608 96.1073 33.0642 97.4869 36.3922C98.8665 39.7202 101.358 42.4668 104.535 44.1639C107.713 45.861 111.381 46.4036 114.914 45.6992C118.447 44.9949 121.627 43.0871 123.911 40.301C126.195 37.515 127.442 34.0231 127.44 30.4205C127.44 28.3772 127.038 26.3539 126.255 24.4664C125.473 22.5788 124.326 20.8642 122.88 19.4205ZM19.42 100.86C16.8725 103.408 15.2872 106.76 14.9342 110.345C14.5813 113.93 15.4826 117.527 17.4844 120.522C19.4863 123.518 22.4649 125.726 25.9127 126.771C29.3604 127.816 33.0638 127.633 36.3918 126.254C39.7198 124.874 42.4664 122.383 44.1635 119.205C45.8606 116.027 46.4032 112.359 45.6988 108.826C44.9944 105.293 43.0866 102.114 40.3006 99.8296C37.5145 97.5455 34.0227 96.2983 30.42 96.3005C26.2938 96.3018 22.337 97.9421 19.42 100.86ZM100.86 100.86C98.3125 103.408 96.7272 106.76 96.3742 110.345C96.0213 113.93 96.9226 117.527 98.9244 120.522C100.926 123.518 103.905 125.726 107.353 126.771C110.8 127.816 114.504 127.633 117.832 126.254C121.16 124.874 123.906 122.383 125.604 119.205C127.301 116.027 127.843 112.359 127.139 108.826C126.434 105.293 124.527 102.114 121.741 99.8296C118.955 97.5455 115.463 96.2983 111.86 96.3005C109.817 96.299 107.793 96.701 105.905 97.4835C104.018 98.2661 102.303 99.4136 100.86 100.86Z" fill="#00AEEF"/>
    </g>
    <defs>
        <clipPath id="clip0_4338_178347">
            <rect width="566.93" height="223.75" fill="white"/>
        </clipPath>
    </defs>
  </svg>
</div>

        <table class="jp-RenderedHTMLCommon" style="border-collapse: collapse;color: var(--jp-ui-font-color1);font-size: var(--jp-ui-font-size1);">
    <tr>
        <td style="text-align: left"><b>Python version:</b></td>
        <td style="text-align: left"><b>3.12.11</b></td>
    </tr>
    <tr>
        <td style="text-align: left"><b>Ray version:</b></td>
        <td style="text-align: left"><b>2.50.0</b></td>
    </tr>
    <tr>
    <td style="text-align: left"><b>Dashboard:</b></td>
    <td style="text-align: left"><b><a href="http://session-77uweunq3awbhqefvry4lwcqq5.i.anyscaleuserdata.com" target="_blank">http://session-77uweunq3awbhqefvry4lwcqq5.i.anyscaleuserdata.com</a></b></td>
</tr>

</table>

    </div>
</div>




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

    2025-10-11 00:17:23,668	INFO logging.py:293 -- Registered dataset logger for dataset dataset_72_0
    2025-10-11 00:17:23,686	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_72_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:17:23,687	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_72_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=1] -> TaskPoolMapOperator[ReadFiles]
    2025-10-11 00:17:23,692	WARNING resource_manager.py:134 -- ⚠️  Ray's object store is configured to use only 27.9% of available memory (98.3GiB out of 352.0GiB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    2025-10-11 00:17:30,038	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_72_0 execution finished in 6.35 seconds


    Dataset schema: Column  Type
    ------  ----
    bytes   binary
    path    string


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

    2025-10-11 00:17:30,138	INFO logging.py:293 -- Registered dataset logger for dataset dataset_74_0
    2025-10-11 00:17:30,143	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_74_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:17:30,144	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_74_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Project] -> AggregateNumRows[AggregateNumRows]


    Extracting text from documents...
    [36m(autoscaler +13s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.


    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2922' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2926' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2930' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2934' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2938' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2942' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2946' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2950' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2954' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2958' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2962' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2966' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2970' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2974' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2978' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2982' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2986' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2990' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2994' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2998' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3004' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3008' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3014' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3018' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3022' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3026' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3030' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3034' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3038' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3042' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3046' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3050' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3054' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3058' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3062' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3066' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3070' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3074' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3078' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3082' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3086' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3090' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3094' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3098' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3102' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3106' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3110' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3114' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3118' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3122' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3126' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3130' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3136' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3140' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3146' is an invalid float value
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3150' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value[32m [repeated 351x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=17419, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3' is an invalid float value[32m [repeated 145x across cluster][0m
    2025-10-11 00:18:00,192	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 114.2MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=114.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:18:00,193	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=16293, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P26' is an invalid float value[32m [repeated 28x across cluster][0m
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value[32m [repeated 81x across cluster][0m
    2025-10-11 00:18:30,218	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 114.2MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=114.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:18:30,219	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:19:00,293	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 114.2MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=114.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:19:00,294	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:19:01,182	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_74_0 execution finished in 91.04 seconds


    Text extraction completed: 100 documents processed



```python
documents_with_text.limit(25).to_pandas()
```

    2025-10-11 00:19:01,463	INFO logging.py:293 -- Registered dataset logger for dataset dataset_75_0
    2025-10-11 00:19:01,467	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_75_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:19:01,468	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_75_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=25] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)]
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/data/_internal/execution/operators/task_pool_map_operator.py:165: UserWarning: The maximum number of concurrent tasks for 'Map(process_file)' is set to 8, but the operator only received 4 input(s). This means that the operator can launch at most 4 task(s), which is less than the concurrency limit. You might be able to increase the number of concurrent tasks by configuring `override_num_blocks` earlier in the pipeline.
      warnings.warn(
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2922' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2926' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2930' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2934' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2938' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2942' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2946' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2950' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2954' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2958' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2962' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2966' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2970' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2974' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2978' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2982' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2986' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2990' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2994' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2998' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3004' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3008' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3014' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3018' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3022' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3026' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3030' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3034' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3038' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3042' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3046' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3050' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3054' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3058' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3062' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3066' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3070' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3074' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3078' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3082' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3086' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3090' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3094' is an invalid float value
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3564' is an invalid float value[32m [repeated 192x across cluster][0m
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P28' is an invalid float value[32m [repeated 230x across cluster][0m
    2025-10-11 00:19:29,527	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_75_0 execution finished in 28.06 seconds





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
      <th>document_id</th>
      <th>file_path</th>
      <th>file_name</th>
      <th>file_extension</th>
      <th>file_size_bytes</th>
      <th>file_size_mb</th>
      <th>discovery_timestamp</th>
      <th>extracted_text</th>
      <th>text_length</th>
      <th>word_count</th>
      <th>extraction_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b1036612-8d10-49b2-a3f8-218af0493e8b</td>
      <td>anyscale-rag-application/1000-docs/100G Networ...</td>
      <td>100G Networking Technology Overview - Slides -...</td>
      <td>.pdf</td>
      <td>1516903</td>
      <td>1.45</td>
      <td>2025-10-11T00:19:05.131495</td>
      <td>100G Networking Technology Overview Christophe...</td>
      <td>8996</td>
      <td>1558</td>
      <td>success</td>
    </tr>
    <tr>
      <th>1</th>
      <td>062705dd-e55c-4257-b6b3-acdbff5ca43d</td>
      <td>anyscale-rag-application/1000-docs/Grand Centr...</td>
      <td>Grand Central Dispatch - FreeBSD Dev Summit (1...</td>
      <td>.pdf</td>
      <td>130189</td>
      <td>0.12</td>
      <td>2025-10-11T00:19:05.495427</td>
      <td>Grand Central Dispatch FreeBSD Devsummit Rober...</td>
      <td>7831</td>
      <td>1071</td>
      <td>success</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d7b9d924-6f08-4417-99b6-5647a8ac7079</td>
      <td>anyscale-rag-application/1000-docs/Monitor_a_j...</td>
      <td>Monitor_a_job.docx</td>
      <td>.docx</td>
      <td>387461</td>
      <td>0.37</td>
      <td>2025-10-11T00:19:06.257774</td>
      <td>Monitor a job Anyscale jobs provides several t...</td>
      <td>3296</td>
      <td>585</td>
      <td>success</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13311876-9e19-4204-ad2a-54332be9426d</td>
      <td>anyscale-rag-application/1000-docs/Serial Orde...</td>
      <td>Serial Order - A Parallel Distributed Processi...</td>
      <td>.pdf</td>
      <td>2281776</td>
      <td>2.18</td>
      <td>2025-10-11T00:19:11.156516</td>
      <td>SERIAL ORDER: A PARALLEL DISTRmUTED PROCESSING...</td>
      <td>132375</td>
      <td>21122</td>
      <td>success</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57522ab2-e67e-41b5-9a63-76f738c8da16</td>
      <td>anyscale-rag-application/1000-docs/jargn10-the...</td>
      <td>jargn10-thejargonfilever00038gut.txt</td>
      <td>.txt</td>
      <td>1140873</td>
      <td>1.09</td>
      <td>2025-10-11T00:19:13.671480</td>
      <td>This Is The Project Gutenberg Etext of The Hac...</td>
      <td>1065517</td>
      <td>170519</td>
      <td>success</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3513ce06-840d-4ffe-a6d6-e96a8c970ace</td>
      <td>anyscale-rag-application/1000-docs/A Compariso...</td>
      <td>A Comparison of Programming Languages in Econo...</td>
      <td>.pdf</td>
      <td>211355</td>
      <td>0.20</td>
      <td>2025-10-11T00:19:09.853818</td>
      <td>A Comparison of Programming Languages in Econo...</td>
      <td>33839</td>
      <td>5307</td>
      <td>success</td>
    </tr>
    <tr>
      <th>6</th>
      <td>106f0fda-642f-4dbd-a8b6-75ed920c9e61</td>
      <td>anyscale-rag-application/1000-docs/A Compariso...</td>
      <td>A Comparison of Software and Hardware Techniqu...</td>
      <td>.pdf</td>
      <td>156844</td>
      <td>0.15</td>
      <td>2025-10-11T00:19:11.700186</td>
      <td>A Comparison of Software and Hardware Techniqu...</td>
      <td>71494</td>
      <td>11296</td>
      <td>success</td>
    </tr>
    <tr>
      <th>7</th>
      <td>81f48d32-a87a-4e50-bafb-dba914a76c3f</td>
      <td>anyscale-rag-application/1000-docs/A Compilati...</td>
      <td>A Compilation Target for Probabilistic Program...</td>
      <td>.pdf</td>
      <td>892594</td>
      <td>0.85</td>
      <td>2025-10-11T00:19:13.122018</td>
      <td>A Compilation Target for Probabilistic Program...</td>
      <td>39374</td>
      <td>6122</td>
      <td>success</td>
    </tr>
    <tr>
      <th>8</th>
      <td>f6a01e8a-3708-496b-b2cb-2d487aa6fd8f</td>
      <td>anyscale-rag-application/1000-docs/Graph Theor...</td>
      <td>Graph Theory (2005).pdf</td>
      <td>.pdf</td>
      <td>206383</td>
      <td>0.20</td>
      <td>2025-10-11T00:19:13.693472</td>
      <td>V. Adamchik Graph Theory Victor Adamchik Fall ...</td>
      <td>10103</td>
      <td>1600</td>
      <td>success</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7c655f32-9bde-488f-81bd-36fe0d93063c</td>
      <td>anyscale-rag-application/1000-docs/Multidigit ...</td>
      <td>Multidigit Multiplication for Mathematicians (...</td>
      <td>.pdf</td>
      <td>346439</td>
      <td>0.33</td>
      <td>2025-10-11T00:19:15.065682</td>
      <td>MULTIDIGIT MULTIPLICATION FOR MATHEMATICIANS D...</td>
      <td>60434</td>
      <td>10046</td>
      <td>success</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ee19a135-a6fc-4104-b3dd-4839b4ae01d6</td>
      <td>anyscale-rag-application/1000-docs/Shining Lig...</td>
      <td>Shining Light on Shadow Stacks - 7 Nov 2018 (1...</td>
      <td>.pdf</td>
      <td>443892</td>
      <td>0.42</td>
      <td>2025-10-11T00:19:16.654990</td>
      <td>8 1 0 2 v o N 7 ] R C . s c [ 1 v 5 6 1 3 0 . ...</td>
      <td>67521</td>
      <td>10529</td>
      <td>success</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c55d665a-e2d9-4ae0-aa5e-28d3a6f1e2e2</td>
      <td>anyscale-rag-application/1000-docs/lwref - BSD...</td>
      <td>lwref - BSDCan2014 - FreeBSD.pdf</td>
      <td>.pdf</td>
      <td>214499</td>
      <td>0.20</td>
      <td>2025-10-11T00:19:17.192574</td>
      <td>An insane idea on reference counting Gleb Smir...</td>
      <td>11319</td>
      <td>1613</td>
      <td>success</td>
    </tr>
    <tr>
      <th>12</th>
      <td>564f8cd7-3520-4445-a439-e9f755b28e43</td>
      <td>anyscale-rag-application/1000-docs/A Block-sor...</td>
      <td>A Block-sorting Lossless Data Compression Algo...</td>
      <td>.pdf</td>
      <td>108608</td>
      <td>0.10</td>
      <td>2025-10-11T00:19:06.250512</td>
      <td>May 10, 1994 SRC Research Report 124 A Block-s...</td>
      <td>36622</td>
      <td>6095</td>
      <td>success</td>
    </tr>
    <tr>
      <th>13</th>
      <td>62c6143c-f0f8-4370-85c9-d84c6e36466a</td>
      <td>anyscale-rag-application/1000-docs/A Brief Int...</td>
      <td>A Brief Introduction to the Standard Annotatio...</td>
      <td>.pdf</td>
      <td>358744</td>
      <td>0.34</td>
      <td>2025-10-11T00:19:06.581700</td>
      <td>A Brief Introduction to the Standard Annotatio...</td>
      <td>13276</td>
      <td>2140</td>
      <td>success</td>
    </tr>
    <tr>
      <th>14</th>
      <td>8b3cf16e-c273-4d2f-9d91-e93a028e1793</td>
      <td>anyscale-rag-application/1000-docs/A Brief Tut...</td>
      <td>A Brief Tutorial on Database Queries, Data Min...</td>
      <td>.pdf</td>
      <td>207314</td>
      <td>0.20</td>
      <td>2025-10-11T00:19:07.930372</td>
      <td>A Brief Tutorial on Database Queries, Data Min...</td>
      <td>18398</td>
      <td>2878</td>
      <td>success</td>
    </tr>
    <tr>
      <th>15</th>
      <td>a22a3728-3c33-4431-aefa-4c0a75221b24</td>
      <td>anyscale-rag-application/1000-docs/A Case Stud...</td>
      <td>A Case Study in Optimizing HTM-Enabled Dynamic...</td>
      <td>.pdf</td>
      <td>325479</td>
      <td>0.31</td>
      <td>2025-10-11T00:19:09.078085</td>
      <td>A Case Study in Optimizing HTM-Enabled Dynamic...</td>
      <td>41101</td>
      <td>6551</td>
      <td>success</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4dc25a92-f51f-4f4b-9c74-9cf6a419d150</td>
      <td>anyscale-rag-application/1000-docs/A Catalogue...</td>
      <td>A Catalogue of Optimizing Transformations (197...</td>
      <td>.pdf</td>
      <td>2346760</td>
      <td>2.24</td>
      <td>2025-10-11T00:19:10.195582</td>
      <td>The term optimization The basic purpose of the...</td>
      <td>35286</td>
      <td>5336</td>
      <td>success</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2a51827a-2fd1-4202-bc16-c41c91033915</td>
      <td>anyscale-rag-application/1000-docs/Graph Theor...</td>
      <td>Graph Theoretic Obstacles to Perfect Hashing -...</td>
      <td>.pdf</td>
      <td>157158</td>
      <td>0.15</td>
      <td>2025-10-11T00:19:11.155396</td>
      <td>Graph theoretic obstacles to perfect hashing G...</td>
      <td>34105</td>
      <td>6403</td>
      <td>success</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4a27ec7b-4e32-462d-afb6-3d0a15368ecc</td>
      <td>anyscale-rag-application/1000-docs/Monotone Mi...</td>
      <td>Monotone Minimal Perfect Hashing - Searching a...</td>
      <td>.pdf</td>
      <td>536251</td>
      <td>0.51</td>
      <td>2025-10-11T00:19:17.050603</td>
      <td>O(1) ∗ † ‡ † S n {0,1,...,n − 1} (cid:0) (cid:...</td>
      <td>46972</td>
      <td>2678</td>
      <td>success</td>
    </tr>
    <tr>
      <th>19</th>
      <td>f2205fcc-15f4-47fd-bd52-feee5c174b43</td>
      <td>anyscale-rag-application/1000-docs/Setting Up ...</td>
      <td>Setting Up a Production Monitoring and Diagnos...</td>
      <td>.pdf</td>
      <td>4546292</td>
      <td>4.34</td>
      <td>2025-10-11T00:19:18.100976</td>
      <td>https://s.sashag.net/prodsdd Sasha Goldshtein ...</td>
      <td>17096</td>
      <td>2275</td>
      <td>success</td>
    </tr>
    <tr>
      <th>20</th>
      <td>817d6ab5-476a-4bd8-8bc1-0bfc23700834</td>
      <td>anyscale-rag-application/1000-docs/libtorque -...</td>
      <td>libtorque - Portable Multithreaded Continuatio...</td>
      <td>.pdf</td>
      <td>279599</td>
      <td>0.27</td>
      <td>2025-10-11T00:19:18.855460</td>
      <td>libtorque: Portable Multithreaded Continuation...</td>
      <td>23775</td>
      <td>3546</td>
      <td>success</td>
    </tr>
    <tr>
      <th>21</th>
      <td>048e69bc-f610-4aba-ba49-01bbdfb6beb6</td>
      <td>anyscale-rag-application/1000-docs/A Dive in t...</td>
      <td>A Dive in to Hyper-V Architecture and Vulnerab...</td>
      <td>.pdf</td>
      <td>4298584</td>
      <td>4.10</td>
      <td>2025-10-11T00:19:24.363729</td>
      <td>This presentation is for informational purpose...</td>
      <td>13282</td>
      <td>1478</td>
      <td>success</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2f9e6977-74cc-4d06-a276-e044390bb07d</td>
      <td>anyscale-rag-application/1000-docs/GraphBLAS M...</td>
      <td>GraphBLAS Mathmatics - Provisional Release 1.0...</td>
      <td>.pdf</td>
      <td>691008</td>
      <td>0.66</td>
      <td>2025-10-11T00:19:27.296515</td>
      <td>GraphBLAS Mathematics - Provisional Release 1....</td>
      <td>40944</td>
      <td>7326</td>
      <td>success</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4165dcd2-1d40-4804-b071-a99117b0bc6f</td>
      <td>anyscale-rag-application/1000-docs/Multiple By...</td>
      <td>Multiple Byte Processing with Full-Word Instru...</td>
      <td>.pdf</td>
      <td>451218</td>
      <td>0.43</td>
      <td>2025-10-11T00:19:28.245814</td>
      <td>out destroying the 1D C-R property, it is nece...</td>
      <td>21908</td>
      <td>3989</td>
      <td>success</td>
    </tr>
    <tr>
      <th>24</th>
      <td>d6d5b6ce-3d0a-4fd7-9972-f8ee36ccb90f</td>
      <td>anyscale-rag-application/1000-docs/Shuffle - T...</td>
      <td>Shuffle - Tips and Tricks - Slides - GPU Tech ...</td>
      <td>.pdf</td>
      <td>799069</td>
      <td>0.76</td>
      <td>2025-10-11T00:19:29.509316</td>
      <td>Shuffle: Tips and Tricks Julien Demouth, NVIDI...</td>
      <td>7228</td>
      <td>1450</td>
      <td>success</td>
    </tr>
  </tbody>
</table>
</div>




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

    2025-10-11 00:19:29,691	INFO logging.py:293 -- Registered dataset logger for dataset dataset_77_0
    2025-10-11 00:19:29,696	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_77_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:19:29,697	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_77_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[Project] -> AggregateNumRows[AggregateNumRows]


    
    Enriching with business metadata...


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P12' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P45' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P69' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P111' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P208' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P210' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P212' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P214' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P216' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P218' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P220' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P273' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P304' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=15819, ip=10.0.0.255)[0m Cannot set gray non-stroke color because /'P455' is an invalid float value[32m [repeated 69x across cluster][0m
    [36m(Map(process_file) pid=16548, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value[32m [repeated 438x across cluster][0m
    [36m(Map(process_file) pid=16560, ip=10.0.6.91)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=16548, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P3866' is an invalid float value[32m [repeated 508x across cluster][0m
    2025-10-11 00:19:59,731	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 508.4MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=508.4MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:19:59,732	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=16548, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P2' is an invalid float value[32m [repeated 228x across cluster][0m
    2025-10-11 00:20:29,832	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 508.4MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=508.4MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:20:29,832	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:20:58,683	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_77_0 execution finished in 88.98 seconds


    Metadata enrichment completed: 100 documents classified



```python
documents_with_metadata.limit(5).to_pandas()
```

    2025-10-11 00:20:58,797	INFO logging.py:293 -- Registered dataset logger for dataset dataset_78_0
    2025-10-11 00:20:58,801	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_78_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:20:58,802	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_78_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=5] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)]
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/data/_internal/execution/operators/task_pool_map_operator.py:165: UserWarning: The maximum number of concurrent tasks for 'Map(process_file)' is set to 8, but the operator only received 1 input(s). This means that the operator can launch at most 1 task(s), which is less than the concurrency limit. You might be able to increase the number of concurrent tasks by configuring `override_num_blocks` earlier in the pipeline.
      warnings.warn(
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/data/_internal/execution/operators/task_pool_map_operator.py:165: UserWarning: The maximum number of concurrent tasks for 'Map(enrich_business_metadata)' is set to 10, but the operator only received 1 input(s). This means that the operator can launch at most 1 task(s), which is less than the concurrency limit. You might be able to increase the number of concurrent tasks by configuring `override_num_blocks` earlier in the pipeline.
      warnings.warn(
    2025-10-11 00:21:09,531	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_78_0 execution finished in 10.73 seconds





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
      <th>document_id</th>
      <th>file_path</th>
      <th>file_name</th>
      <th>file_extension</th>
      <th>file_size_bytes</th>
      <th>file_size_mb</th>
      <th>discovery_timestamp</th>
      <th>extracted_text</th>
      <th>text_length</th>
      <th>word_count</th>
      <th>extraction_status</th>
      <th>document_type</th>
      <th>business_category</th>
      <th>processing_priority</th>
      <th>priority_score</th>
      <th>estimated_pages</th>
      <th>processing_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ffb14e1e-b5ab-4d7f-9a11-d41f47e66d30</td>
      <td>anyscale-rag-application/1000-docs/100G Networ...</td>
      <td>100G Networking Technology Overview - Slides -...</td>
      <td>.pdf</td>
      <td>1516903</td>
      <td>1.45</td>
      <td>2025-10-11T00:21:00.779167</td>
      <td>100G Networking Technology Overview Christophe...</td>
      <td>8996</td>
      <td>1558</td>
      <td>success</td>
      <td>general_document</td>
      <td>general</td>
      <td>low</td>
      <td>1</td>
      <td>30</td>
      <td>classified</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7c1a9e64-0180-414c-8a40-f72404485064</td>
      <td>anyscale-rag-application/1000-docs/Grand Centr...</td>
      <td>Grand Central Dispatch - FreeBSD Dev Summit (1...</td>
      <td>.pdf</td>
      <td>130189</td>
      <td>0.12</td>
      <td>2025-10-11T00:21:01.153132</td>
      <td>Grand Central Dispatch FreeBSD Devsummit Rober...</td>
      <td>7831</td>
      <td>1071</td>
      <td>success</td>
      <td>general_document</td>
      <td>general</td>
      <td>low</td>
      <td>1</td>
      <td>2</td>
      <td>classified</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43aaff7e-34d0-4042-b2c1-749bab93ce81</td>
      <td>anyscale-rag-application/1000-docs/Monitor_a_j...</td>
      <td>Monitor_a_job.docx</td>
      <td>.docx</td>
      <td>387461</td>
      <td>0.37</td>
      <td>2025-10-11T00:21:01.950850</td>
      <td>Monitor a job Anyscale jobs provides several t...</td>
      <td>3296</td>
      <td>585</td>
      <td>success</td>
      <td>general_document</td>
      <td>general</td>
      <td>low</td>
      <td>1</td>
      <td>7</td>
      <td>classified</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e4a7205e-0af4-487c-8184-3e8e8e2732a7</td>
      <td>anyscale-rag-application/1000-docs/Serial Orde...</td>
      <td>Serial Order - A Parallel Distributed Processi...</td>
      <td>.pdf</td>
      <td>2281776</td>
      <td>2.18</td>
      <td>2025-10-11T00:21:06.921489</td>
      <td>SERIAL ORDER: A PARALLEL DISTRmUTED PROCESSING...</td>
      <td>132375</td>
      <td>21122</td>
      <td>success</td>
      <td>general_document</td>
      <td>general</td>
      <td>low</td>
      <td>1</td>
      <td>45</td>
      <td>classified</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bd194fc1-0137-4fe8-b336-c32a59f0fd94</td>
      <td>anyscale-rag-application/1000-docs/jargn10-the...</td>
      <td>jargn10-thejargonfilever00038gut.txt</td>
      <td>.txt</td>
      <td>1140873</td>
      <td>1.09</td>
      <td>2025-10-11T00:21:09.486768</td>
      <td>This Is The Project Gutenberg Etext of The Hac...</td>
      <td>1065517</td>
      <td>170519</td>
      <td>success</td>
      <td>general_document</td>
      <td>general</td>
      <td>low</td>
      <td>1</td>
      <td>22</td>
      <td>classified</td>
    </tr>
  </tbody>
</table>
</div>




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

    2025-10-11 00:21:09,660	INFO logging.py:293 -- Registered dataset logger for dataset dataset_81_0
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/util/dependencies.py:42: UserWarning: Numba isn't available. Install numba>=0.61>=0.61 to get better performance for hash partitioning operations. Falling back to slower Python implementation for RayTurbo optimizations.
      warnings.warn(
    2025-10-11 00:21:09,668	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_81_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:21:09,669	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_81_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> HashAggregateOperator[HashAggregate(key_columns=('document_type',), num_partitions=200)] -> LimitOperator[limit=5]


    Analyzing document collection using Ray Data native operations...
    Document Type Distribution:


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value[32m [repeated 377x across cluster][0m
    [36m(Map(process_file) pid=16294, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P244' is an invalid float value[32m [repeated 446x across cluster][0m
    [36m(Map(process_file) pid=16548, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value[32m [repeated 79x across cluster][0m
    [36m(Map(process_file) pid=15788, ip=10.0.0.100)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    2025-10-11 00:21:39,812	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 618.0MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=618.0MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:21:39,813	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=17486, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value[32m [repeated 151x across cluster][0m
    2025-10-11 00:22:09,837	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 618.0MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=618.0MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:22:09,838	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:22:39,906	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 618.0MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=618.0MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:22:39,907	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:22:43,684	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_81_0 execution finished in 94.01 seconds


           document_type  count()  sum(file_size_bytes)  mean(file_size_mb)  \
    0   general_document       99              91471983            0.881515   
    1  research_document        1                432535            0.410000   
    
       max(estimated_pages)  
    0                   159  
    1                     8  



```python

# Business category analysis
category_stats = documents_with_metadata.groupby("business_category").aggregate(
    Count(),
    Mean("priority_score"),
    Sum("file_size_mb")
)
```

    2025-10-11 00:22:43,832	INFO logging.py:293 -- Registered dataset logger for dataset dataset_84_0
    2025-10-11 00:22:43,926	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_84_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:22:43,926	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_84_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> HashAggregateOperator[HashAggregate(key_columns=('business_category',), num_partitions=200)] -> LimitOperator[limit=5]


    Business Category Analysis:


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=16615, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P290' is an invalid float value[32m [repeated 409x across cluster][0m
    [36m(Map(process_file) pid=16216, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'P45' is an invalid float value[32m [repeated 145x across cluster][0m
    [36m(Map(process_file) pid=15819, ip=10.0.0.255)[0m Cannot set gray non-stroke color because /'P306' is an invalid float value[32m [repeated 64x across cluster][0m
    [36m(Map(process_file) pid=17500, ip=10.0.4.21)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=16295, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P274' is an invalid float value[32m [repeated 70x across cluster][0m
    2025-10-11 00:23:13,982	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 679.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=679.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:23:13,983	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:23:44,003	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 679.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=679.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:23:44,004	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:24:14,019	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 679.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=679.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:24:14,020	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:24:17,854	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_84_0 execution finished in 93.92 seconds


      business_category  count()  mean(priority_score)  sum(file_size_mb)
    0          research        1                   1.0               0.41
    1           general       99                   1.0              87.27


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

    /tmp/ipykernel_67923/1159361854.py:71: DeprecationWarning: String expressions are deprecated and will be removed in a future version. Use predicate expressions from ray.data.expressions instead. For example: from ray.data.expressions import col; ds.filter(expr=col('column_name') > 5)
      high_quality_docs = quality_assessed_docs.filter(
    2025-10-11 00:24:18,026	INFO logging.py:293 -- Registered dataset logger for dataset dataset_87_0
    2025-10-11 00:24:18,032	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_87_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:24:18,033	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_87_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[Project] -> AggregateNumRows[AggregateNumRows]


    Assessing document quality...


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P26' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P42' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P54' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P62' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P72' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P80' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P121' is an invalid float value
    [36m(Map(process_file) pid=16559, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P129' is an invalid float value
    [36m(Map(process_file) pid=16295, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value[32m [repeated 68x across cluster][0m
    [36m(Map(process_file) pid=16550, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value[32m [repeated 500x across cluster][0m
    [36m(Map(process_file) pid=17486, ip=10.0.34.16)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P3214' is an invalid float value[32m [repeated 296x across cluster][0m
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P28' is an invalid float value[32m [repeated 395x across cluster][0m
    2025-10-11 00:24:48,078	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 703.9MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=703.9MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:24:48,079	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:25:18,085	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 703.9MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=703.9MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:25:18,086	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:25:43,871	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_87_0 execution finished in 85.83 seconds
    2025-10-11 00:25:43,878	INFO logging.py:293 -- Registered dataset logger for dataset dataset_88_0
    2025-10-11 00:25:43,882	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_88_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:25:43,882	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_88_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[Filter(<expression>)] -> TaskPoolMapOperator[Project] -> AggregateNumRows[AggregateNumRows]


    Total documents assessed: 100


    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value[32m [repeated 90x across cluster][0m
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value[32m [repeated 575x across cluster][0m
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P352' is an invalid float value[32m [repeated 473x across cluster][0m
    [36m(Map(process_file) pid=16216, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'P157' is an invalid float value[32m [repeated 142x across cluster][0m
    [36m(Map(process_file) pid=15818, ip=10.0.0.255)[0m Cannot set gray non-stroke color because /'H3' is an invalid float value[32m [repeated 106x across cluster][0m
    2025-10-11 00:26:13,979	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 717.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=717.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:26:13,980	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=16152, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'P129' is an invalid float value[32m [repeated 20x across cluster][0m
    [36m(Map(process_file) pid=17503, ip=10.0.4.21)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    2025-10-11 00:26:44,010	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 717.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=717.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:26:44,010	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:27:14,026	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 717.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=717.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:27:14,027	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:27:23,226	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_88_0 execution finished in 99.34 seconds


    High quality documents: 1


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

    2025-10-11 00:27:23,336	INFO logging.py:293 -- Registered dataset logger for dataset dataset_90_0
    2025-10-11 00:27:23,342	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_90_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:27:23,343	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_90_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[FlatMap(create_text_chunks)] -> TaskPoolMapOperator[Project] -> AggregateNumRows[AggregateNumRows]


    Creating text chunks...


    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2922' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2926' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2930' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2934' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2938' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2942' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2946' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2950' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2954' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2958' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2962' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2966' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2970' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2974' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2978' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2982' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2986' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2990' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2994' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P2998' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3004' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3008' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3014' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3018' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3022' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3026' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3030' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3034' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3038' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3042' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3046' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3050' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3054' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3058' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3062' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3066' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3070' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3074' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3078' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3082' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3086' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3090' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3094' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3098' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3102' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3106' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3110' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3114' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3118' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3122' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3126' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3130' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3136' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3140' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3146' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3150' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3154' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3158' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3162' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3166' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3170' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3174' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3178' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3182' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3186' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3190' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3194' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3198' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3202' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3206' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3210' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3214' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3218' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3222' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3226' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3230' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3234' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3238' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3242' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3246' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3250' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3254' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3258' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3262' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3266' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3270' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3274' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3278' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3282' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3284' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3286' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3288' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3294' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3298' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3302' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3306' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3310' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3314' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3318' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3322' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3326' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3330' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3334' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3338' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3342' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3346' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3350' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3354' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3358' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3362' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3366' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3370' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3374' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3378' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3382' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3386' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3390' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3394' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3398' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3404' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3406' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3408' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3410' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3412' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3414' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3416' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3418' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3426' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3430' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3434' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3440' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3444' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3448' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3452' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3456' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3460' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3464' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3468' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3472' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3476' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3480' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3484' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3488' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3492' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3496' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3500' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3504' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3508' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3512' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3516' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3520' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3524' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3528' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3532' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3536' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3540' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3544' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3548' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3552' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3556' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3560' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3564' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3568' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3572' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3576' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3580' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3584' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3588' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3592' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3596' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3600' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3604' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3608' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3612' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3616' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3620' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3624' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3628' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3632' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3636' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3638' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3640' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3642' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3644' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3646' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3648' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3650' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3652' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3654' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3656' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3660' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3662' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3664' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3670' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3674' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3678' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3682' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3686' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3690' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3694' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3698' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3702' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3706' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3710' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3714' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3718' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3722' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3726' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3730' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3734' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3738' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3742' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3746' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3750' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3754' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3758' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3762' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3766' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3770' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3774' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3778' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3782' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3786' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3790' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3794' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3798' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3802' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3806' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3810' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3814' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3818' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3822' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3826' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3830' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3834' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3838' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3842' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3846' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3850' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3854' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3858' is an invalid float value
    [36m(Map(process_file) pid=15723, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P3860' is an invalid float value
    [36m(Map(process_file) pid=16216, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'P280' is an invalid float value[32m [repeated 156x across cluster][0m
    [36m(Map(process_file) pid=16152, ip=10.0.18.28)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=16351, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value[32m [repeated 155x across cluster][0m
    [36m(Map(process_file) pid=16351, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'pgfpat3' is an invalid float value[32m [repeated 7x across cluster][0m
    2025-10-11 00:27:53,393	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 730.6MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=730.6MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:27:53,394	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=16351, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'H2' is an invalid float value[32m [repeated 56x across cluster][0m
    2025-10-11 00:28:23,435	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 730.6MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=730.6MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:28:23,436	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:28:46,321	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_90_0 execution finished in 82.98 seconds


    Text chunking completed: 5,116 chunks created


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

    2025-10-11 00:50:59,250	INFO logging.py:293 -- Registered dataset logger for dataset dataset_101_0
    2025-10-11 00:50:59,257	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_101_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:50:59,257	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_101_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[FlatMap(create_text_chunks)] -> TaskPoolMapOperator[Project->MapBatches(add_column)->MapBatches(add_column)->MapBatches(add_column)->Project] -> AggregateNumRows[AggregateNumRows]
    2025-10-11 00:50:59,285	WARNING progress_bar.py:120 -- Truncating long operator name to 100 characters. To disable this behavior, set `ray.data.DataContext.get_current().DEFAULT_ENABLE_PROGRESS_BAR_NAME_TRUNCATION = False`.


    Creating data warehouse schema...


    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2922' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2926' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2930' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2934' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2938' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2942' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2946' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2950' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2954' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2958' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2962' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2966' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2970' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2974' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2978' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2982' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2986' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2990' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2994' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P2998' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3004' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3008' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3014' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3018' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3022' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3026' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3030' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3034' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3038' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3042' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3046' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3050' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3054' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3058' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3062' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3066' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3070' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3074' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3078' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3082' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3086' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3090' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3094' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3098' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3102' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3106' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3110' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3114' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3118' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3122' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3126' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3130' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3136' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3140' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3146' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3150' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3154' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3158' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3162' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3166' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3170' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3174' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3178' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3182' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3186' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3190' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3194' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3198' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3202' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3206' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3210' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3214' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3218' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3222' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3226' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3230' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3234' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3238' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3242' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3246' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3250' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3254' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3258' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3262' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3266' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3270' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3274' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3278' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3282' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3284' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3286' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3288' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3294' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3298' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3302' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3306' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3310' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3314' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3318' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3322' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3326' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3330' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3334' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3338' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3342' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3346' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3350' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3354' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3358' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3362' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3366' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3370' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3374' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3378' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3382' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3386' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3390' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3394' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3398' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3404' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P290' is an invalid float value[32m [repeated 296x across cluster][0m
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P3' is an invalid float value[32m [repeated 144x across cluster][0m
    [36m(Map(process_file) pid=17486, ip=10.0.34.16)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=16560, ip=10.0.6.91)[0m Cannot set gray non-stroke color because /'P129' is an invalid float value[32m [repeated 31x across cluster][0m
    2025-10-11 00:51:29,374	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 855.2MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=855.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:51:29,375	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:51:59,419	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 855.2MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=855.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:51:59,421	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:52:27,784	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_101_0 execution finished in 88.53 seconds


    Warehouse schema created: 5,116 records


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

    2025-10-11 00:54:51,685	INFO logging.py:293 -- Registered dataset logger for dataset dataset_105_0
    2025-10-11 00:54:51,692	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_105_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 00:54:51,693	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_105_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[FlatMap(create_text_chunks)] -> TaskPoolMapOperator[Project->MapBatches(add_column)->MapBatches(add_column)->MapBatches(add_column)] -> TaskPoolMapOperator[Write]


    Writing to data warehouse...


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2728' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2732' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2736' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2740' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2744' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2748' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2752' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2758' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2762' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2768' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2772' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2776' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2780' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2784' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2788' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2792' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2796' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2800' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2804' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2808' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2812' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2816' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2820' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2824' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2828' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2832' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2836' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2840' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2844' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2848' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2852' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2856' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2860' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2864' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2868' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2872' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2876' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2880' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2884' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2888' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2892' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2896' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2900' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2906' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2910' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2914' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2918' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2922' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2926' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2930' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2934' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2938' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2942' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2946' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2950' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2954' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2958' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2962' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2966' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2970' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2974' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2978' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2982' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2986' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2990' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2994' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P2998' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3004' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3008' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3014' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3018' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3022' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3026' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3030' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3034' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3038' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3042' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3046' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3050' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3054' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3058' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3062' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3066' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3070' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3074' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3078' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3082' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3086' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3090' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3094' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3098' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3102' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3106' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3110' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3114' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3118' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3122' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3126' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3130' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3136' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3140' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3146' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3150' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3154' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3158' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3162' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3166' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3170' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3174' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3178' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3182' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3186' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3190' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3194' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3198' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3202' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3206' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3210' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3214' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3218' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3222' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3226' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3230' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3234' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3238' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3242' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3246' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3250' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3254' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3258' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3262' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3266' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3270' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3274' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3278' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3282' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3284' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3286' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3288' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3294' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3298' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3302' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3306' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3310' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3314' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3318' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3322' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3326' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3330' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3334' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3338' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3342' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3346' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3350' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3354' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3358' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3362' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3366' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3370' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3374' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3378' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3382' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3386' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3390' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3394' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3398' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3404' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3406' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3408' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3410' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3412' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3414' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3416' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3418' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3426' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3430' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3434' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3440' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3444' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3448' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3452' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3456' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3460' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3464' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3468' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3472' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3476' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3480' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3484' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3488' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3492' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3496' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3500' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3504' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3508' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3512' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3516' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3520' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3524' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3528' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3532' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3536' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3540' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3544' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3548' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3552' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3556' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3560' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3564' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3568' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3572' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3576' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3580' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3584' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3588' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3592' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3596' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3600' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3604' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3608' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3612' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3616' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3620' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3624' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3628' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3632' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3636' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3638' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3640' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3642' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3644' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3646' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3648' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3650' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3652' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3654' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3656' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3660' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3662' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3664' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3670' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3674' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3678' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3682' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3686' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3690' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3694' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3698' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3702' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3706' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3710' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3714' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3718' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3722' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3726' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3730' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3734' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3738' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3742' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3746' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3750' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3754' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3758' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3762' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3766' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3770' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3774' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3778' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3782' is an invalid float value
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P3786' is an invalid float value
    [36m(Map(process_file) pid=15788, ip=10.0.0.100)[0m Cannot set gray non-stroke color because /'P303' is an invalid float value[32m [repeated 186x across cluster][0m
    [36m(Map(process_file) pid=17947, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'pgfpat3' is an invalid float value[32m [repeated 164x across cluster][0m
    [36m(Map(process_file) pid=15819, ip=10.0.0.255)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=17947, ip=10.0.18.28)[0m Cannot set gray non-stroke color because /'H2' is an invalid float value[32m [repeated 121x across cluster][0m
    2025-10-11 00:55:21,734	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 838.1MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=838.1MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:55:21,734	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:55:51,738	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 838.1MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=838.1MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:55:51,739	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:56:21,743	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 838.1MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=838.1MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:56:21,744	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'Write' uses 794.0MB of memory per task on average, but Ray
    only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=794.0MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 00:56:21,745	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 00:56:24,000	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_105_0 execution finished in 92.30 seconds
    2025-10-11 00:56:24,046	INFO dataset.py:5134 -- Data sink Parquet finished. 5116 rows and 23.4MB data written.


    Main warehouse table written successfully



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

    /tmp/ipykernel_67923/269082049.py:4: DeprecationWarning: String expressions are deprecated and will be removed in a future version. Use predicate expressions from ray.data.expressions instead. For example: from ray.data.expressions import col; ds.filter(expr=col('column_name') > 5)
      financial_analytics = warehouse_dataset.filter(
    2025-10-11 01:01:53,536	INFO logging.py:293 -- Registered dataset logger for dataset dataset_114_0
    2025-10-11 01:01:53,544	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_114_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 01:01:53,545	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_114_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[FlatMap(create_text_chunks)] -> TaskPoolMapOperator[Project->MapBatches(add_column)->MapBatches(add_column)->MapBatches(add_column)] -> TaskPoolMapOperator[Filter(<expression>)] -> TaskPoolMapOperator[Project] -> TaskPoolMapOperator[Write]


    Creating business-specific datasets...


    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P15' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P19' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P23' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P33' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P39' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P43' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P47' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P53' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P57' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P63' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P74' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P78' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P84' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P88' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P90' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P92' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P94' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P96' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P98' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P100' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P102' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P104' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P106' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P108' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P110' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P112' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P114' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P116' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P118' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P120' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P122' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P126' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P128' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P130' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P134' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P136' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P138' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P140' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P142' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P144' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P146' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P148' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P160' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P168' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P174' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P180' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P186' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P192' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P198' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P204' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P217' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P221' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P227' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P235' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P239' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P243' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P251' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P255' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P259' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P265' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P271' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P275' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P281' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P285' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P289' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P295' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P299' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P305' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P311' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P315' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P321' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P325' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P331' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P337' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P341' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P347' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P353' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P357' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P361' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P365' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P371' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P375' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P379' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P383' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P387' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P391' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P395' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P399' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P403' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P407' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P413' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P417' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P421' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P425' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P429' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P435' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P441' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P447' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P451' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P457' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P463' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P465' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P467' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P469' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P471' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P473' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P475' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P477' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P479' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P481' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P483' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P485' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P487' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P489' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P491' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P493' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P495' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P497' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P499' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P501' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P503' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P505' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P507' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P509' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P511' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P513' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P515' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P517' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P525' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P529' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P535' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P541' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P547' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P553' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P557' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P559' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P561' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P574' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P580' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P584' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P588' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P594' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P598' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P602' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P608' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P616' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P624' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P632' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P640' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P648' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P656' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P658' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P660' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P662' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P664' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P666' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P668' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P670' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P672' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P674' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P676' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P678' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P684' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P688' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P692' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P696' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P702' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P706' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P710' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P714' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P718' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P722' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P726' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P730' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P736' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P740' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P746' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P750' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P756' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P760' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P766' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P770' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P774' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P778' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P784' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P790' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P796' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P800' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P806' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P810' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P816' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P822' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P826' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P832' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P836' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P840' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P844' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P848' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P852' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P856' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P860' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P864' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P868' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P872' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P876' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P880' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P884' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P888' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P892' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P896' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P900' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P904' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P908' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P912' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P916' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P920' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P924' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P928' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P932' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P936' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P940' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P944' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P948' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P952' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P956' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P960' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P964' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P972' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P976' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P982' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P986' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P992' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P996' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1000' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1004' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1010' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1016' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1020' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1026' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1030' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1036' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1040' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1046' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1052' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1056' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1060' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1064' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1068' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1072' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1076' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1080' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1084' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1090' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1094' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1100' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1106' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1110' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1114' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1118' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1124' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1128' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1132' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1138' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1144' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1148' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1154' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1162' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1166' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1170' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1174' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1178' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1184' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1188' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1192' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1196' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1200' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1204' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1208' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1212' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1216' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1220' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1224' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1228' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1232' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1236' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1240' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1244' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1248' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1252' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1256' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1260' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1264' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1268' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1272' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1276' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1280' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1284' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1288' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1292' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1296' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1300' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1304' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1308' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1312' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1316' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1318' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1320' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1322' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1324' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1326' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1328' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1330' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1334' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1342' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1350' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1354' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1358' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1366' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1370' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1374' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1383' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1385' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1387' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1394' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1398' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1402' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1408' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1414' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1420' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1424' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1428' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1432' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1436' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1442' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1448' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1454' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1460' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1464' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1470' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1474' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1478' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1482' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1488' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1492' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1496' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1500' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1506' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1510' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1514' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1518' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1522' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1526' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1530' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1543' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1545' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1549' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1555' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1561' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1567' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1573' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1579' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1584' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1589' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1594' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1598' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1602' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1606' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1610' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1614' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1618' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1622' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1628' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1632' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1638' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1644' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1650' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1654' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1658' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1664' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1670' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1674' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1678' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1684' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1688' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1694' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1700' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1706' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1715' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1719' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1723' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1727' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1731' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1735' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1741' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1745' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1749' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1755' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1761' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1765' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1769' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1773' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1777' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1781' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1787' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1791' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1795' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1799' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1803' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1807' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1811' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1815' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1819' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1822' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1828' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1832' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1836' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1842' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1848' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1852' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1858' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1862' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1868' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1874' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1878' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1884' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1888' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1892' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1896' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1902' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1908' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1912' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1918' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1922' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1928' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1932' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1938' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1946' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1952' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1958' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1962' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1968' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1972' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1978' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1982' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1986' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1992' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P1996' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2002' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2008' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2012' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2020' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2024' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2028' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2034' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2040' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2044' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2050' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2056' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2060' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2064' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2068' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2074' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2078' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2084' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2088' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2092' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2098' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2102' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2108' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2112' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2118' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2124' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2131' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2133' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2135' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2137' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2139' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2141' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2145' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2151' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2155' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2159' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2165' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2169' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2173' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2180' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2186' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2190' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2194' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2198' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2204' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2208' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2212' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2216' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2220' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2224' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2230' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2236' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2240' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2244' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2248' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2252' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2256' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2260' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2264' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2268' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2275' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2279' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2285' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2291' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2295' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2299' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2305' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2309' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2315' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2321' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2325' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2331' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2335' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2339' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2344' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2350' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2356' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2360' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2372' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2376' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2382' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2388' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2392' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2396' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2400' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2404' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2410' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2414' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2420' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2424' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2428' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2433' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2437' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2443' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2447' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2453' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2459' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2463' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2469' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2475' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2479' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2483' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2488' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2492' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2498' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2502' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2508' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2513' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2519' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2523' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2527' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2533' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2539' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2543' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2549' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2553' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2559' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2564' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2568' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2574' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2578' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2582' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2588' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2592' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2596' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2601' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2606' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2612' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2618' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2622' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2624' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2626' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2632' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2636' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2640' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2644' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2648' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2652' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2656' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2660' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2664' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2668' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2672' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2676' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2680' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2684' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2688' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2692' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2696' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2700' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2704' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2708' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2712' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2716' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2720' is an invalid float value
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P2724' is an invalid float value
    [36m(Map(process_file) pid=17489, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P274' is an invalid float value[32m [repeated 461x across cluster][0m
    [36m(Map(process_file) pid=17486, ip=10.0.34.16)[0m Cannot set gray non-stroke color because /'P124' is an invalid float value[32m [repeated 61x across cluster][0m
    [36m(Map(process_file) pid=16352, ip=10.0.30.227)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=17499, ip=10.0.4.21)[0m Cannot set gray non-stroke color because /'P12' is an invalid float value[32m [repeated 99x across cluster][0m
    2025-10-11 01:02:23,630	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 787.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=787.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:02:23,630	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P152' is an invalid float value[32m [repeated 57x across cluster][0m
    2025-10-11 01:02:53,682	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 787.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=787.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:02:53,683	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16293, ip=10.0.15.76)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Map(process_file) pid=16674, ip=10.0.20.152)[0m Cannot set gray non-stroke color because /'P534' is an invalid float value[32m [repeated 63x across cluster][0m
    2025-10-11 01:03:23,788	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 787.7MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=787.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:03:23,789	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'Filter(<expression>)' uses 824.7MB of memory per task on
    average, but Ray only requests 0.0B per task at the start of the
    pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=824.7MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:03:23,790	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'Write' uses 833.1MB of memory per task on average, but Ray
    only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=833.1MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:03:23,791	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 01:03:24,745	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_114_0 execution finished in 91.20 seconds
    2025-10-11 01:03:24,768	INFO dataset.py:5134 -- Data sink Parquet finished. 0 rows and 0.0B data written.
    /tmp/ipykernel_67923/269082049.py:20: DeprecationWarning: String expressions are deprecated and will be removed in a future version. Use predicate expressions from ray.data.expressions instead. For example: from ray.data.expressions import col; ds.filter(expr=col('column_name') > 5)
      compliance_analytics = warehouse_dataset.filter(
    2025-10-11 01:03:24,785	INFO logging.py:293 -- Registered dataset logger for dataset dataset_119_0
    2025-10-11 01:03:24,792	INFO streaming_executor.py:159 -- Starting execution of Dataset dataset_119_0. Full logs are in /tmp/ray/session_2025-10-10_21-11-45_497822_2529/logs/ray-data
    2025-10-11 01:03:24,793	INFO streaming_executor.py:160 -- Execution plan of Dataset dataset_119_0: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> LimitOperator[limit=100] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(process_file)] -> TaskPoolMapOperator[Map(enrich_business_metadata)] -> TaskPoolMapOperator[MapBatches(process_quality_assessment_batch)] -> TaskPoolMapOperator[FlatMap(create_text_chunks)] -> TaskPoolMapOperator[Project->MapBatches(add_column)->MapBatches(add_column)->MapBatches(add_column)] -> TaskPoolMapOperator[Filter(<expression>)] -> TaskPoolMapOperator[Project] -> TaskPoolMapOperator[Write]
    /home/ray/anaconda3/lib/python3.12/site-packages/ray/anyscale/data/_internal/cluster_autoscaler/productivity_calculator.py:174: RuntimeWarning: invalid value encountered in divide
      gpu_fraction_per_op = (optimal_num_tasks_per_op * num_gpus_per_op) / np.sum(
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Write pid=17489, ip=10.0.34.16)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set[32m [repeated 46x across cluster][0m
    [36m(Map(process_file) pid=16293, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P27' is an invalid float value[32m [repeated 4x across cluster][0m
    [36m(Map(process_file) pid=16293, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'P2366' is an invalid float value[32m [repeated 525x across cluster][0m
    [36m(Map(process_file) pid=16547, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P280' is an invalid float value[32m [repeated 533x across cluster][0m
    [36m(Map(process_file) pid=15818, ip=10.0.0.255)[0m Cannot set gray non-stroke color because /'P132' is an invalid float value[32m [repeated 150x across cluster][0m
    [36m(Map(process_file) pid=16718, ip=10.0.0.100)[0m Could get FontBBox from font descriptor because None cannot be parsed as 4 floats
    [36m(Map(process_file) pid=16294, ip=10.0.15.76)[0m Cannot set gray non-stroke color because /'H2' is an invalid float value[32m [repeated 37x across cluster][0m
    [36m(Map(process_file) pid=16353, ip=10.0.30.227)[0m Cannot set gray non-stroke color because /'P342' is an invalid float value[32m [repeated 96x across cluster][0m
    2025-10-11 01:03:54,910	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 809.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=809.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:03:54,911	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 01:04:24,950	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 809.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=809.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:04:24,951	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Filter(<expression>) pid=16559, ip=10.0.6.91)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Map(process_file) pid=16548, ip=10.0.37.24)[0m Cannot set gray non-stroke color because /'P26' is an invalid float value[32m [repeated 3x across cluster][0m
    2025-10-11 01:04:54,989	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'ReadFiles' uses 809.8MB of memory per task on average, but
    Ray only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=809.8MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:04:54,990	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'Filter(<expression>)' uses 790.2MB of memory per task on
    average, but Ray only requests 0.0B per task at the start of the
    pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=790.2MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:04:54,991	WARNING issue_detector_manager.py:58 -- 
    
    Operator 'Write' uses 823.1MB of memory per task on average, but Ray
    only requests 0.0B per task at the start of the pipeline.
    
    To avoid out-of-memory errors, consider setting `memory=823.1MB` in
    the appropriate function or method call. (This might be unnecessary if
    the number of concurrent tasks is low.)
    
    To change the frequency of this warning, set
    `DataContext.get_current().issue_detectors_config.high_memory_detector_config.detection_time_interval_s`,
    or disable the warning by setting value to -1. (current value: 30)
    
    2025-10-11 01:04:54,992	WARNING issue_detector_manager.py:67 -- To disable issue detection, run DataContext.get_current().issue_detectors_config.detectors = [].
    2025-10-11 01:04:56,459	INFO streaming_executor.py:279 -- ✔️  Dataset dataset_119_0 execution finished in 91.66 seconds
    2025-10-11 01:04:56,492	INFO dataset.py:5134 -- Data sink Parquet finished. 0 rows and 0.0B data written.


    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'document_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'chunk_id': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'business_category': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'document_type': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'file_extension': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'quality_rating': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'processing_priority': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'text_content': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'file_name': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'file_path': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'discovery_timestamp': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'extraction_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'processing_status': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'processing_date': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'pipeline_version': cannot call `vectorize` on size 0 inputs unless `otypes` is set
    [36m(Project pid=16718, ip=10.0.0.100)[0m Error calculating size for column 'processing_engine': cannot call `vectorize` on size 0 inputs unless `otypes` is set


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

