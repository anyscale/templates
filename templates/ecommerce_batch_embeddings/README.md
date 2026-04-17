# E-Commerce Batch Product Embedding Pipeline

**⏱️ Time to complete**: 30 min
### Anyscale Technical Demo — Ray Data on Anyscale Jobs

---

## Business Context

An e-commerce company with **10M+ SKUs** needs to regenerate product embeddings weekly to power:
- Semantic product search
- "You might also like" recommendations
- Personalized catalog ranking

**Today:** A Spark pipeline that takes **14+ hours**, can't use GPUs efficiently, and if it fails at hour 13 — you start over.

**With Ray Data on Anyscale:** The same job runs in minutes, on heterogeneous CPU+GPU hardware, with automatic fault tolerance and full observability.

---

## Architecture

```
Anyscale Job
|
+- [Ray Data] read_parquet ------------- Distributed parallel read
+- [map_batches / CPU] preprocess ------- Text cleaning + field combine
|       m5.4xlarge x 2-10 nodes
+- [map_batches / GPU] ProductEmbedder -- Sentence embedding inference
|       g4dn.xlarge x 1-4 nodes (T4 GPU, all-MiniLM-L6-v2, 384-dim)
+- [write_parquet] ---------------------- Embeddings to /mnt/cluster_storage/
```

Data streams between stages — no intermediate disk writes, no idle CPUs waiting on GPUs.


## Step 1: Connect to the Ray Cluster

The next cell initializes a connection to the Anyscale-managed Ray cluster and prints available resources (CPUs, GPUs, nodes). This is the zero-ops experience -- Anyscale provisions and manages the heterogeneous cluster for you. No Terraform, no YARN config, no Docker orchestration.

**Key differentiator:** Unlike Spark, Ray natively schedules work across both CPU and GPU nodes in a single cluster, so your preprocessing and inference share one unified resource pool.


```python
import os
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

!pip install -q -r requirements.txt
```


```python
# Cell 2 - Setup: connect to Ray cluster and inspect resources
import sys, os

# Point to the batch-embeddings root so src.* and scripts.* are importable
DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray

# In Anyscale Workspace, Ray is pre-initialized.
# runtime_env working_dir ensures Ray workers can import src.* from this repo.
ray.init(
    ignore_reinit_error=True,
    runtime_env={"working_dir": DEMO_ROOT},
)

resources = ray.cluster_resources()
print("Ray cluster resources:")
for resource, count in sorted(resources.items()):
    if not resource.startswith('node:'):
        print(f"  {resource:<20} {count}")

nodes = ray.nodes()
print(f"\nCluster nodes: {len(nodes)}")
for n in nodes:
    res = ', '.join(f"{k}={v}" for k, v in n['Resources'].items() if not k.startswith('node:'))
    print(f"  {n['NodeManagerAddress']:<20} alive={n['Alive']}  {res}")
```

## Step 2: Load the Product Catalog with Ray Data

The next cell generates a synthetic product catalog (100K SKUs at "medium" scale) and loads it as a Ray Dataset using `read_parquet`. Ray Data reads in parallel across cluster nodes -- no single-driver bottleneck like `pandas.read_parquet` or Spark's driver-side schema inference.

**Business value:** For a real catalog of 10M+ SKUs, this distributed read scales linearly. You point it at your S3 path and Ray handles partitioning automatically -- no manual repartitioning or tuning shuffle configs.


```python
# Cell 3 - Generate and inspect the product catalog
from src.generate_data import generate_catalog, SCALE_MAP
import ray.data

INPUT_PATH  = "/mnt/cluster_storage/ecommerce-demo/raw/products_medium.parquet"
OUTPUT_PATH = "/mnt/cluster_storage/ecommerce-demo/embeddings/medium/"
SCALE       = "medium"  # 100K products

# Generate synthetic catalog (skips if file already exists)
if not os.path.exists(INPUT_PATH):
    generate_catalog(SCALE_MAP[SCALE], INPUT_PATH)
else:
    print(f"Using existing catalog at {INPUT_PATH}")

# Load with Ray Data and inspect
ds = ray.data.read_parquet(INPUT_PATH)

print(f"Dataset size:  {ds.count():,} products")
print(f"Schema:\n{ds.schema()}")

print("\nSample rows:")
for row in ds.take(3):
    print(f"\n  product_id : {row['product_id']}")
    print(f"  title      : {row['title']}")
    print(f"  category   : {row['category']} | brand: {row['brand']} | price: ${row['price']:.2f}")
    print(f"  description: {row['description'][:120]}...")

```

## Step 3: CPU Preprocessing with `map_batches`

The next cell runs text cleaning (HTML stripping, field combination) across CPU workers using `map_batches`. This is pure CPU work -- no GPUs needed. Ray Data automatically schedules these batches across your `m5.4xlarge` nodes while keeping GPU nodes free for the embedding stage.

**Why this matters:** In Spark, you'd either over-provision GPU instances for a CPU-only stage or build a separate preprocessing pipeline. Ray Data lets you pin each stage to the right hardware in a single pipeline definition.


```python
# Cell 4 - Run preprocessing stage (CPU workers only)
# Shows text cleaning, HTML stripping, and field combination
from src.preprocess import preprocess_product
from src.utils import timer

with timer("CPU preprocessing (sample)"):
    ds_preprocessed = ds.map_batches(
        preprocess_product,
        batch_size=1024,
        num_cpus=1,
        batch_format="numpy",
    )
    sample_preprocessed = ds_preprocessed.take(3)

print("\nPost-preprocessing combined_text (first 200 chars):")
for row in sample_preprocessed:
    print(f"\n  {row['combined_text'][:200]}...")

print("\nTIP: Check the Ray Dashboard 'Cluster' tab to see CPU workers active.")

```

## Step 4: Full Streaming Pipeline -- Preprocess, Embed, Write

This is the core of the demo. The next cell runs the end-to-end pipeline: CPU preprocessing streams directly into GPU embedding, which streams directly into Parquet writes. There are **no intermediate disk materializations** between stages -- data flows through memory buffers.

**Technical differentiator:** Ray Data's streaming execution means your GPU workers start embedding as soon as the first preprocessed batch is ready. CPU and GPU stages run concurrently, maximizing hardware utilization. Open the Ray Dashboard to watch this live.

**Business value:** This architecture cuts a 14-hour Spark job down to minutes and eliminates the "fail at hour 13, restart from scratch" problem -- failed batches retry individually.


```python
# Cell 5 - Run full pipeline: preprocess + embed + write
# CPU workers preprocess while GPU workers embed — data streams between stages
from src.pipeline import run_embedding_pipeline

print("Starting full pipeline...")
print(f"  Input:   {INPUT_PATH}")
print(f"  Output:  {OUTPUT_PATH}")
print(f"  GPUs:    2 workers (concurrency=2)\n")
print("TIP: Open Ray Dashboard -> Jobs to watch:")
print("  - CPU workers handling preprocessing")
print("  - T4 GPU workers activating for embedding")
print("  - Data streaming between stages (no intermediate writes)\n")

metrics = run_embedding_pipeline(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    num_gpus=2,
)

```

## Step 5: Validate Embeddings with Semantic Search

The next cell loads the output embeddings and runs cosine similarity searches against real product queries. This proves the pipeline produced high-quality 384-dimensional embeddings that capture semantic meaning -- "wireless bluetooth headphones" should match headphone products, not cables.

**Why this matters:** Embedding quality directly drives search relevance and recommendation accuracy. This validation step is what you'd show stakeholders to prove the pipeline output is production-ready before plugging it into your vector store or search index.


```python
# Cell 6 - Validate outputs: embedding shape + cosine similarity search
from scripts.validate_outputs import load_embeddings, print_embedding_stats, similarity_search

df = load_embeddings(OUTPUT_PATH)
print_embedding_stats(df)

DEMO_QUERIES = [
    "wireless bluetooth headphones noise cancelling",
    "women's running shoes lightweight breathable",
    "daily face moisturizer SPF sensitive skin",
]

print("\nSemantic similarity search results:")
for query in DEMO_QUERIES:
    print(f"\n{'─' * 65}")
    print(f"  Query: \"{query}\"")
    print(f"{'─' * 65}")
    results = similarity_search(query, df, top_k=5)
    for _, row in results.iterrows():
        print(f"  [{row['score']:.4f}]  {row['title'][:55]:<55}  ${row['price']:.2f}")

```

## Step 6: Metrics Summary and Path to Production

The final cell prints pipeline performance metrics and shows how to promote this exact notebook logic to a scheduled **Anyscale Job** with a single CLI command. Same code, same cluster config -- no rewrite needed.

**Key takeaway:** What you just ran interactively in a Workspace becomes a production cron job with `anyscale job submit`. Scale from 100K to 10M products by changing one flag. Anyscale handles autoscaling, fault tolerance, and cluster lifecycle automatically.


```python
# Cell 7 - Metrics summary and path to production
from src.utils import print_metrics_table

print_metrics_table(metrics)

print("Path to production as an Anyscale Job:")
print("""
  # Submit from CLI (run from batch-embeddings/ directory):
  anyscale job submit --config-file job_config.yaml

  # Scale to 2M products:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/run_pipeline.py --scale large --num-gpus 4'
""")

print("Key differentiators vs. Spark:")
print("  Heterogeneous compute  -- CPU preprocessing + GPU inference in one pipeline")
print("  Streaming execution    -- no intermediate disk writes between stages")
print("  Fault tolerance        -- failed batches retry automatically, no full restart")
print("  Autoscaling            -- GPU workers scale to demand, then scale to zero")
print("  One codebase           -- same code runs in Workspace and as a scheduled Job")

```
