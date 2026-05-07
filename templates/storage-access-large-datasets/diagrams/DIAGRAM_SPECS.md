# Diagram Generation Specifications
# Template: Storage Access and Large Datasets on Anyscale

This document specifies the diagrams to be generated for this template using the template-diagram generation skills.

## Diagram 1: Anyscale Storage Architecture

**Name:** anyscale-storage-architecture
**Type:** Architecture-level
**Target Section:** Anyscale Storage Fundamentals
**Skill:** generate-diagram
**Iterations:** 3 (Medium complexity)
**Size:** 80% width

### Content Description

A diagram showing the three Anyscale storage mount points and their characteristics:

**Components:**
1. `/mnt/cluster_storage/` - Shared across cluster nodes, persists across workspace restarts, ephemeral for Jobs/Services
2. `/mnt/shared_storage/` - Cross-cluster shared storage, permanent until deleted
3. `/mnt/user_storage/` - User-specific storage, permanent until deleted

**Visual Elements:**
- Three storage tier boxes with different colors/shading to indicate persistence levels
- Arrows showing access patterns (workspace → cluster_storage, workspace/job → shared_storage)
- Labels indicating "Ephemeral for Jobs" vs "Permanent" storage
- Icons for workspace, jobs, and services accessing different storage tiers

**Embedding Location:**
After the paragraph: "Anyscale provides three types of storage mounts:" in the Anyscale Storage Fundamentals section.

---

## Diagram 2: Cloud Storage to Ray Data Pipeline

**Name:** cloud-storage-pipeline
**Type:** Architecture-level
**Target Section:** Introduction
**Skill:** generate-diagram
**Iterations:** 3 (Medium complexity)
**Size:** 80% width

### Content Description

A multi-cloud data access architecture showing how Ray Data integrates with different cloud storage providers.

**Components:**
1. Cloud Storage Tier (left):
   - Amazon S3 bucket (with AWS logo indicator)
   - Google Cloud Storage bucket (with GCP logo indicator)
   - Azure Blob Storage container (with Azure logo indicator)

2. Filesystem Layer (middle):
   - PyArrow S3FileSystem (for S3)
   - gcsfs.GCSFileSystem (for GCS)
   - adlfs.AzureBlobFileSystem (for Azure)

3. Ray Data Layer (right):
   - ray.data.read_parquet()
   - ray.data.read_csv()
   - ray.data.read_json()
   - Unified Dataset interface

**Data Flow:**
Arrows showing: Cloud Storage → Filesystem Interface → Ray Data API → Dataset

**Authentication Notes:**
- S3: IAM roles, credentials file
- GCS: Service account JSON
- Azure: Account key, SAS token

**Embedding Location:**
After the paragraph: "Ray Data provides a unified interface..." in the Introduction section.

---

## Diagram 3: Ray Data Transformation Pipeline

**Name:** ray-data-transformation-pipeline
**Type:** Section-specific
**Target Section:** Data Transformations
**Skill:** generate-diagram
**Iterations:** 2 (Simple complexity)
**Size:** 60% width

### Content Description

A linear pipeline showing Ray Data's lazy execution model with transformation stages.

**Pipeline Stages (left to right):**

1. **Read** - `ray.data.read_parquet()`
   - Status: "Logical plan created"
   - Dataset icon with file symbol

2. **Filter** - `.filter()`
   - Status: "Operation queued"
   - Filter funnel icon

3. **Transform** - `.map_batches()`
   - Status: "Operation queued"
   - Transformation gear icon

4. **Select** - `.select_columns()`
   - Status: "Operation queued"
   - Column selection icon

5. **Write** - `.write_parquet()`
   - Status: "Execution triggered"
   - Output file symbol with execution indicator

**Visual Elements:**
- Dashed lines between stages 1-4 indicating "lazy evaluation (not executed yet)"
- Solid arrow from stage 4 to 5 with label "Execution triggered by write()"
- Small notes: "No data movement until terminal operation"

**Embedding Location:**
After the paragraph: "Ray Data provides powerful transformation capabilities..." in the Data Transformations section.

---

## Generation Instructions

For each diagram, the template-diagram agent should:

1. Create iteration subdirectories: `iter-1/`, `iter-2/`, etc.
2. Generate both PNG and DrawIO source files
3. Use consistent color scheme:
   - Cloud providers: Their brand colors
   - Ray Data: Ray blue (#1E88E5)
   - Anyscale platform: Anyscale teal (#00BCD4)
4. Include clear labels and arrows with descriptive text
5. Optimize for readability at specified width (60% or 80%)

## Success Criteria

Each diagram should:
- ✓ Accurately represent the technical architecture/flow described in the template code
- ✓ Use clear, professional visual design
- ✓ Be legible when rendered at target width in the notebook
- ✓ Include alt text for accessibility
- ✓ Have both editable (.drawio) and final (.png) versions
