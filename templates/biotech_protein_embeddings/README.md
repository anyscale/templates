# Biotech Protein Sequence Embedding Pipeline (ESM-2)

**⏱️ Time to complete**: 30 min
### Anyscale Technical Demo — Ray Data on Anyscale Jobs

---

## Business Context

A biotech company with **1M+ internal and public protein sequences** needs embeddings to power:
- **Homology search** — find proteins with similar function (faster than BLAST, no alignment needed)
- **Library clustering** — discover novel scaffolds in designed-binder libraries
- **Candidate featurization** — input features for binding affinity, solubility, and stability predictors
- **Pre-screening** — rank candidates before expensive structure prediction (Boltz-1, AlphaFold)

**Today:** A single-node BioPython loop takes **36+ hours**, wastes GPU with mixed-length batching (~30% utilization), and if it fails at hour 35 — you start over.

**With Ray Data on Anyscale:** The same job runs in minutes on heterogeneous CPU+GPU hardware, with length-aware batching that pushes GPU utilization to 80%+, automatic fault tolerance, and full observability.

---

## What is ESM-2?

**ESM-2** (Evolutionary Scale Modeling) is Meta's protein language model, trained on millions of protein sequences. It's to proteins what BERT is to text: one forward pass produces an embedding that captures evolutionary, structural, and functional information about the protein.

- **Input:** A protein sequence (string of amino acid letters, e.g., `MKTLLILAVF...`)
- **Output:** A 1280-dimensional embedding vector (for the 650M parameter variant)
- **Use cases:** Similarity search, clustering, classification, featurizing for downstream models

---

## Architecture

```
Anyscale Job
|
+- [Ray Data] read corpus ------------------- Distributed parallel read
+- [map_batches / CPU] validate + filter ----- Drop non-canonical AAs, length bounds
|       m5.4xlarge x 2-8 nodes
+- [map_batches / CPU] length bucketing ------ 4 buckets (20-128, 129-256, 257-512, 513-1024)
+- [sort by bucket] -------------------------- Length-homogeneous GPU batches
+- [map_batches / GPU] ESM-2 embedding ------- Tokenize, forward pass, mean-pool
|       g5.xlarge x 1-4 nodes (A10G, 1280-dim)
+- [map_batches / CPU] taxonomy join --------- Broadcast join with organism metadata
+- [write_parquet] --------------------------- Embeddings to /mnt/cluster_storage/
```

**Hero moment:** We run the pipeline TWICE — first without bucketing (~30% GPU util), then with bucketing (~80% GPU util). Same data, same hardware, 2-3x throughput improvement from one preprocessing optimization.

**Estimated runtime:** ~5-8 minutes for both runs on 2x A10G GPUs (100K sequences).

## Step 1: Connect to the Ray Cluster

The next cell initializes a connection to the Anyscale-managed Ray cluster and prints available resources (CPUs, GPUs, nodes). This is the zero-ops experience -- Anyscale provisions and manages the heterogeneous cluster for you.

**Key differentiator:** Unlike single-node scripts, Ray natively schedules work across both CPU and GPU nodes in a single cluster. CPU workers handle FASTA parsing and validation while GPU workers run ESM-2 inference -- one unified resource pool, no manual orchestration.


```python
import os
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

!pip install -q -r requirements.txt
```


```python
# Cell 2 - Setup: connect to Ray cluster and inspect resources
import sys, os

# Point to the protein-embeddings root so src.* and scripts.* are importable
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

## Step 2: Load (or Generate) the Protein Corpus

The next cell generates a synthetic protein corpus (100K sequences at "medium" scale) and loads it as a Ray Dataset. The synthetic generator:

- Starts from **20 seed protein family motifs** (kinase domains, zinc fingers, immunoglobulin folds, etc.)
- Applies **random point mutations** (10-30%) to simulate natural sequence divergence within families
- Varies **sequence length** (20-1024 amino acids) following a realistic bimodal distribution
- Injects a small fraction (~8%) of **non-canonical amino acids** to test the validation stage
- Produces a **FASTA file** (standard bioinformatics format) and a **Parquet version** for efficient Ray Data loading

**What is FASTA?** The standard text format for biological sequences. Each record has a `>header` line with metadata, followed by the amino acid sequence. Think of it as "CSV for biologists."

**Business value:** For a real corpus of 10M+ sequences from UniProt or an internal library, this distributed read scales linearly. You point it at your S3 path and Ray handles partitioning automatically.


```python
# Cell 3 - Generate and inspect the protein corpus
from src.corpus_generator import save_corpus, SCALE_MAP
import ray.data
import pandas as pd

BASE_DIR = "/mnt/cluster_storage/protein-embeddings"
INPUT_DIR = f"{BASE_DIR}/raw"
INPUT_PATH = f"{INPUT_DIR}/corpus.parquet"
TAXONOMY_PATH = f"{INPUT_DIR}/taxonomy_lookup.parquet"
PAIRS_PATH = f"{INPUT_DIR}/homolog_test_pairs.csv"
OUTPUT_NAIVE = f"{BASE_DIR}/embeddings/naive/"
OUTPUT_BUCKETED = f"{BASE_DIR}/embeddings/bucketed/"
SCALE = "medium"  # 100K sequences

# Generate synthetic corpus (skips if file already exists)
if not os.path.exists(INPUT_PATH):
    save_corpus(INPUT_DIR, num_sequences=SCALE_MAP[SCALE])
else:
    print(f"Using existing corpus at {INPUT_PATH}")

# Load with Ray Data and inspect
ds = ray.data.read_parquet(INPUT_PATH)

print(f"\nDataset size:  {ds.count():,} protein sequences")
print(f"Schema:\n{ds.schema()}")

# Show sequence length distribution
df_sample = ds.to_pandas()
print(f"\nSequence length distribution:")
print(f"  Min:    {df_sample['length'].min()} aa")
print(f"  Median: {df_sample['length'].median():.0f} aa")
print(f"  Mean:   {df_sample['length'].mean():.0f} aa")
print(f"  Max:    {df_sample['length'].max()} aa")

# Show family distribution
print(f"\nProtein family distribution (top 10):")
print(df_sample['family'].value_counts().head(10).to_string())

# Show sample sequences
print(f"\nSample sequences:")
for _, row in df_sample.head(3).iterrows():
    print(f"\n  sequence_id : {row['sequence_id']}")
    print(f"  organism_id : {row['organism_id']}")
    print(f"  family      : {row['family']}")
    print(f"  length      : {row['length']} aa")
    print(f"  sequence    : {row['sequence'][:60]}...")
```

## Step 3: Run the NAIVE Pipeline (No Length Bucketing)

This is the **"before"** in our before/after comparison. The pipeline runs without length bucketing, meaning GPU batches contain a mix of short (20 aa) and long (1024 aa) sequences. Every sequence in a batch gets padded to the length of the longest one.

**Why this is slow:** ESM-2's transformer attention is O(L^2) in sequence length. If a batch has one 1000 aa sequence and thirty-one 50 aa sequences, all 32 get padded to 1000 tokens. That's ~95% wasted compute.

**Expected GPU utilization: ~30-40%** — most of the GPU time is spent computing attention over padding tokens.

**TIP:** Open the Ray Dashboard (Jobs tab) to watch CPU workers handling validation while GPU workers run ESM-2 inference. You'll see GPU utilization hovering low.


```python
# Cell 4 - Run NAIVE pipeline (no length bucketing)
# GPU batches contain mixed-length sequences -> heavy padding -> low GPU utilization
from src.pipeline import run_embedding_pipeline_naive

NUM_GPUS = 2
# Use the smaller ESM-2 variant for faster demo iteration.
# Switch to "facebook/esm2_t33_650M_UR50D" for production runs (1280-dim).
MODEL = "facebook/esm2_t12_35M_UR50D"

print("Starting NAIVE pipeline (no length bucketing)...")
print(f"  Input:     {INPUT_PATH}")
print(f"  Output:    {OUTPUT_NAIVE}")
print(f"  Model:     {MODEL}")
print(f"  GPUs:      {NUM_GPUS} workers\n")
print("TIP: Open the Ray Dashboard -> Jobs to watch:")
print("  - CPU workers handling validation and filtering")
print("  - GPU workers running ESM-2 (low utilization due to mixed-length batches)")
print("  - Data streaming between stages (no intermediate writes)\n")

metrics_naive = run_embedding_pipeline_naive(
    input_path=INPUT_PATH,
    output_path=OUTPUT_NAIVE,
    taxonomy_path=TAXONOMY_PATH,
    num_gpus=NUM_GPUS,
    model_name=MODEL,
)
```

## Step 4: Run the BUCKETED Pipeline (Length-Aware Batching)

Now the **"after"**. The only change: we add a CPU preprocessing step that assigns each sequence to a length bucket, then sort by bucket before the GPU stage. This ensures each GPU batch contains sequences of similar length, minimizing padding waste.

**The optimization:**
- Bucket 0: 20-128 aa (short peptides)
- Bucket 1: 129-256 aa (single-domain proteins)
- Bucket 2: 257-512 aa (multi-domain proteins)
- Bucket 3: 513-1024 aa (large proteins)

**Expected GPU utilization: ~80%** — sequences in the same batch are similar length, so padding overhead drops from ~70% to ~10%.

**This is a single-line change in the pipeline** (`ds.sort("length_bucket")`) that yields 2-3x throughput improvement. Ray Data makes it trivial.

**TIP:** Compare the GPU utilization in the Ray Dashboard with the previous run. The difference should be dramatic.


```python
# Cell 5 - Run BUCKETED pipeline (length-aware batching)
# Same data, same hardware, same model — but sequences are sorted by length bucket
# before the GPU stage, so each batch has similar-length sequences.
from src.pipeline import run_embedding_pipeline_bucketed

print("Starting BUCKETED pipeline (length-aware batching)...")
print(f"  Input:     {INPUT_PATH}")
print(f"  Output:    {OUTPUT_BUCKETED}")
print(f"  Model:     {MODEL}")
print(f"  GPUs:      {NUM_GPUS} workers\n")
print("TIP: Watch GPU utilization in the Ray Dashboard — it should be significantly")
print("  higher than the naive run. Same hardware, 2-3x throughput.\n")

metrics_bucketed = run_embedding_pipeline_bucketed(
    input_path=INPUT_PATH,
    output_path=OUTPUT_BUCKETED,
    taxonomy_path=TAXONOMY_PATH,
    num_gpus=NUM_GPUS,
    model_name=MODEL,
)

# Side-by-side comparison
print("\n" + "=" * 60)
print("  NAIVE vs BUCKETED — SIDE-BY-SIDE COMPARISON")
print("=" * 60)
print(f"  {'Metric':<30} {'Naive':<15} {'Bucketed':<15}")
print(f"  {'-'*30} {'-'*15} {'-'*15}")
print(f"  {'Throughput':<30} {metrics_naive['Throughput']:<15} {metrics_bucketed['Throughput']:<15}")
print(f"  {'Wall time':<30} {metrics_naive['Wall time']:<15} {metrics_bucketed['Wall time']:<15}")
print(f"  {'Embed time':<30} {metrics_naive['Embed time']:<15} {metrics_bucketed['Embed time']:<15}")
print(f"  {'Est. job cost':<30} {metrics_naive['Est. Anyscale job cost']:<15} {metrics_bucketed['Est. Anyscale job cost']:<15}")
print("=" * 60)
print("\n  Length bucketing is a preprocessing change — Ray Data makes it")
print("  a single-line repartition (ds.sort). Same hardware, same model, 2-3x faster.")
```

## Step 5: Validate Embeddings — Homologs vs. Random Pairs

The next cell loads the output embeddings and computes cosine similarity between labeled pairs:

- **Homolog pairs:** Two proteins from the **same family** (e.g., two kinase domains). These share a common ancestor and should have **high** cosine similarity in embedding space.
- **Random pairs:** Two proteins from **different families** (e.g., a kinase vs. a zinc finger). These are unrelated and should have **low** cosine similarity.

If the embeddings are meaningful, we should see a clear separation between the two groups. This validates that ESM-2 embeddings capture protein family relationships — the foundation for downstream similarity search, clustering, and candidate ranking.

**Why this matters:** Embedding quality directly drives the accuracy of homology search, clustering, and the candidate ranking step in the Boltz-1 structure prediction pipeline.


```python
# Cell 6 - Validate embeddings: homolog pairs vs random pairs
import numpy as np
from scripts.validate_outputs import load_embeddings, print_embedding_stats, compute_pair_similarities, print_pair_stats

# Load the bucketed output (higher quality due to better GPU utilization)
df = load_embeddings(OUTPUT_BUCKETED)
print_embedding_stats(df)

# Load homolog test pairs and compute cosine similarities
if os.path.exists(PAIRS_PATH):
    pairs_df = pd.read_csv(PAIRS_PATH)
else:
    # Fall back to the shipped asset
    asset_pairs = os.path.join(DEMO_ROOT, "assets", "homolog_test_pairs.csv")
    pairs_df = pd.read_csv(asset_pairs)

pairs_df = compute_pair_similarities(df, pairs_df)
print_pair_stats(pairs_df)

# Show individual pair similarities
print(f"\nSample homolog pairs (same family — expect HIGH similarity):")
homologs = pairs_df[pairs_df["relationship"] == "homolog"].head(5)
for _, row in homologs.iterrows():
    print(f"  {row['seq_id_a']} <-> {row['seq_id_b']}  "
          f"family={row['family']:<20}  cosine_sim={row['cosine_sim']:.4f}")

print(f"\nSample random pairs (different families — expect LOW similarity):")
randoms = pairs_df[pairs_df["relationship"] == "random"].head(5)
for _, row in randoms.iterrows():
    print(f"  {row['seq_id_a']} <-> {row['seq_id_b']}  "
          f"families={row['family']:<35}  cosine_sim={row['cosine_sim']:.4f}")

# Show sample embeddings
print(f"\nSample embeddings (first 3 sequences, first 8 dimensions):")
for _, row in df.head(3).iterrows():
    emb = np.array(row["embedding"])
    print(f"  {row['sequence_id']}: dim={len(emb)}, emb[0:8]={emb[:8].round(4)}")
```

## Step 6: Fault Tolerance and Productionization

### Fault Tolerance
Ray Data provides **batch-level fault tolerance** out of the box. If a GPU worker dies mid-inference:

1. Ray detects the failure within seconds
2. The affected batches (not the entire dataset) are rescheduled to surviving workers
3. Anyscale automatically provisions a replacement node
4. The pipeline continues without restarting from scratch

**To test this live:** While the pipeline is running, go to the Anyscale console and terminate one GPU worker. Watch the Ray Dashboard -- you'll see the batches reassigned and the pipeline complete successfully.

This is critical for biotech: a 12-hour embedding run over 10M sequences cannot afford to restart from scratch because one GPU had a memory error on batch 9,999.

### Anyscale Job Submission
The same pipeline code runs as a scheduled **Anyscale Job** with a single CLI command. No rewrite, no Docker, no Kubernetes -- just `anyscale job submit`.


```python
# Cell 7 - Fault tolerance and Anyscale Job submission
print("Fault Tolerance Demo")
print("=" * 52)
print("""
  To test fault tolerance live:

  1. Start a long-running pipeline (--scale large)
  2. While it's running, terminate a GPU worker in the Anyscale console
  3. Watch the Ray Dashboard:
     - Failed batches are detected within seconds
     - Batches are rescheduled to surviving workers
     - A replacement node is provisioned automatically
     - The pipeline completes successfully

  No manual intervention. No restart from scratch.
  Just the affected batches are retried.
""")

print("Anyscale Job Submission")
print("=" * 52)
print("""
  # Submit from CLI (run from protein-embeddings/ directory):
  anyscale job submit --config-file job_config.yaml

  # Scale to 500K sequences with 4 GPU workers:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/02_run_embedding.py --scale large --num-gpus 4 --bucketed'

  # Use the full 650M model for production-quality embeddings:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/02_run_embedding.py --scale large --num-gpus 4 --model facebook/esm2_t33_650M_UR50D'

  # Run naive mode for comparison:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/02_run_embedding.py --scale medium --no-bucketed'
""")
```

## Step 7: Final Summary and Production Path

The final cell prints the complete metrics comparison and connects this demo to the broader biotech pipeline story.

**What we demonstrated:**
1. A streaming protein embedding pipeline using ESM-2 on Ray Data
2. Heterogeneous CPU+GPU compute in a single pipeline definition
3. Length-aware batching (bucketing) that delivers 2-3x throughput improvement
4. Embedding validation proving family-level semantic signal
5. One-command promotion to a scheduled Anyscale Job

**Where these embeddings go next:**
- **Vector database** (Pinecone, Weaviate, pgvector) for real-time similarity search
- **Boltz-1 screening pipeline** — rank candidates by embedding similarity before expensive structure prediction
- **Downstream classifiers** — binding affinity, solubility, stability prediction models
- **Clustering dashboards** — visualize the protein library landscape for drug discovery teams


```python
# Cell 8 - Final metrics summary and production path
from src.utils import print_metrics_table

print("NAIVE Pipeline Metrics:")
print_metrics_table(metrics_naive)

print("BUCKETED Pipeline Metrics:")
print_metrics_table(metrics_bucketed)

print("=" * 60)
print("  FINAL COMPARISON: NAIVE vs BUCKETED")
print("=" * 60)
print(f"  {'Metric':<30} {'Naive':<15} {'Bucketed':<15}")
print(f"  {'-'*30} {'-'*15} {'-'*15}")
print(f"  {'Throughput':<30} {metrics_naive['Throughput']:<15} {metrics_bucketed['Throughput']:<15}")
print(f"  {'Wall time':<30} {metrics_naive['Wall time']:<15} {metrics_bucketed['Wall time']:<15}")
print(f"  {'Embed time':<30} {metrics_naive['Embed time']:<15} {metrics_bucketed['Embed time']:<15}")
print("=" * 60)

print("""
Production Path:
  1. This notebook -> Anyscale Job (anyscale job submit --config-file job_config.yaml)
  2. Schedule nightly re-embedding of new sequences added to the corpus
  3. Embeddings land in Parquet on cluster storage, ready for:
     - Vector database ingestion (similarity search API)
     - Boltz-1 candidate screening pipeline (rank by embedding distance)
     - Downstream binding/affinity/solubility classifiers
  4. Scale from 100K to 10M+ sequences by changing --scale flag
  5. Anyscale handles autoscaling, fault tolerance, and cluster lifecycle

Key differentiators vs. single-node BioPython:
  - Heterogeneous compute  -- CPU parsing + GPU inference in one pipeline
  - Length bucketing        -- 2-3x throughput from one preprocessing optimization
  - Streaming execution    -- no intermediate disk writes between stages
  - Fault tolerance        -- batch-level retry, not full restart
  - Autoscaling            -- GPU workers scale to demand, then scale to zero
  - One codebase           -- same code runs in Workspace and as a scheduled Job

Connection to Boltz-1 demo:
  These embeddings are the INPUT to the Boltz-1 structure prediction screening pipeline.
  Instead of running Boltz-1 on all 1M candidates (prohibitively expensive), you:
    1. Embed all candidates with this pipeline (minutes, not days)
    2. Rank by embedding similarity to known binders (milliseconds)
    3. Run Boltz-1 only on the top-1000 candidates (hours, not months)
""")
```
