# Ray Data Templates Classification

## Template Type Classification for Rule Application

This document classifies each Ray Data template according to the rule decision framework to determine which rules apply.

---

## Template Types

### 1. **standard-ray-data** (Use native operations)
Templates teaching standard Ray Data patterns and best practices.

**Rules:** Strict application - use native readers/writers, <50 line code blocks

#### Templates:
1. **ray-data-etl-optimization** (README.md)
   - Type: standard-ray-data
   - Focus: ETL pipelines with native Ray Data operations
   - Rules: 151, 160, 161 apply strictly

2. **ray-data-batch-inference-optimization** (01, 02, 03)
   - Type: standard-ray-data
   - Focus: Batch ML inference optimization
   - Rules: 151, 160, 161 apply strictly

3. **ray-data-unstructured-ingestion** (README.md)
   - Type: standard-ray-data
   - Focus: Document processing pipeline
   - Rules: 151, 160, 161 apply strictly

4. **ray-data-ml-feature-engineering** (README.md)
   - Type: standard-ray-data
   - Focus: Feature engineering pipelines
   - Rules: 151, 160, 161 apply strictly

5. **ray-data-data-quality-monitoring** (README.md)
   - Type: standard-ray-data
   - Focus: Data quality checks and validation
   - Rules: 151, 160, 161 apply strictly

6. **ray-data-geospatial-analysis** (README.md)
   - Type: standard-ray-data
   - Focus: Geospatial data processing
   - Rules: 151, 160, 161 apply strictly

7. **ray-data-nlp-text-analytics** (README, 01, 02)
   - Type: standard-ray-data
   - Focus: NLP processing with Ray Data
   - Rules: 151, 160, 161 apply strictly

8. **ray-data-log-ingestion** (README.md)
   - Type: standard-ray-data
   - Focus: Log processing and analysis
   - Rules: 151, 160, 161 apply strictly

9. **ray-data-financial-forecasting** (01, 02)
   - Type: standard-ray-data
   - Focus: Financial data analysis
   - Rules: 151, 160, 161 apply strictly

---

### 2. **custom-datasource** (Build FileBasedDatasource)
Templates teaching how to extend Ray Data with custom connectors.

**Rules:** Inverted - DON'T use native readers, BUILD FileBasedDatasource, allow >50 line classes

#### Templates:
1. **ray-data-medical-connectors** (README, 01, 02)
   - Type: custom-datasource
   - Focus: FileBasedDatasource for HL7/DICOM formats
   - Rules: 151, 160, 161 are INVERTED
   - Exceptions: Code blocks can exceed 50 lines for complete class definitions
   - Educational goal: Teach FileBasedDatasource extension pattern

---

## Classification Summary

| Template | Type | Native Ops? | Code Size Limit | Key Focus |
|----------|------|-------------|-----------------|-----------|
| etl-optimization | standard | ✅ Yes | <50 lines | Native operations |
| batch-inference | standard | ✅ Yes | <50 lines | ML inference |
| unstructured-ingestion | standard | ✅ Yes | <50 lines | Document processing |
| ml-feature-engineering | standard | ✅ Yes | <50 lines | Feature engineering |
| data-quality-monitoring | standard | ✅ Yes | <50 lines | Data validation |
| geospatial-analysis | standard | ✅ Yes | <50 lines | Geospatial processing |
| nlp-text-analytics | standard | ✅ Yes | <50 lines | NLP processing |
| log-ingestion | standard | ✅ Yes | <50 lines | Log analysis |
| financial-forecasting | standard | ✅ Yes | <50 lines | Financial analysis |
| **medical-connectors** | **custom-datasource** | ❌ **No** | **<150 lines OK** | **FileBasedDatasource** |

---

## Improvement Priorities by Template Type

### For Standard Templates (9 templates):
**Focus:**
1. ✅ Use native Ray Data operations (read_parquet, map_batches, write_parquet)
2. ✅ Keep code blocks <50 lines
3. ✅ Add expected output blocks
4. ✅ Fix missing newlines
5. ✅ Convert bullet walls to tables
6. ✅ Add tip/note callouts
7. ✅ Achieve 40/35/25 content balance
8. ✅ Remove buzzwords
9. ✅ Professional presentation

### For Custom Datasource Templates (1 template):
**Focus:**
1. ✅ Show FileBasedDatasource implementation
2. ✅ Allow complete class definitions (>50 lines)
3. ✅ Implement _read_stream() method
4. ✅ Build custom parsers
5. ✅ Document why custom (not native) approach
6. ✅ Already completed (2,787+ improvements)

---

## Action Plan Distribution

### Total: 300+ Actions Across Templates

**Standard Templates (280 actions):**
- etl-optimization: 30 actions
- batch-inference (3 parts): 60 actions (20 each)
- unstructured-ingestion: 30 actions
- ml-feature-engineering: 30 actions
- data-quality-monitoring: 20 actions (already improved)
- geospatial-analysis: 30 actions
- nlp-text-analytics (3 parts): 40 actions (already improved)
- log-ingestion: 20 actions
- financial-forecasting (2 parts): 40 actions (20 each)

**Custom Datasource Templates (20 actions):**
- medical-connectors: 20 validation actions (already 2,787+ improvements)

---

## Improvement Tracking

### Completed:
- ✅ medical-connectors: 2,787+ improvements (custom-datasource)
- ✅ data-quality-monitoring: 225+ improvements (standard)
- ✅ nlp-text-analytics: 620+ improvements (standard)

### Remaining:
- ⏳ 7 standard templates need improvements
- ⏳ Total: ~280 actions remaining

---

**Generated**: 2025-10-06  
**Framework**: Based on rule-decision-framework.mdc  
**Purpose**: Guide systematic template improvements with appropriate rules

