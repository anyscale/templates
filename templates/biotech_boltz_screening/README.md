# Biotech Protein Interaction Screening

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/biotech_boltz_screening"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/biotech_boltz_screening" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 30 min

> **The scores in this template are simulated.** The Ray Data pipeline is real and is the point: CPU feature prep streaming into GPU actors streaming into CPU post-processing, autoscaling, with no intermediate materialization. The GPU stage stands in for a structure predictor so the template runs anywhere in minutes with no checkpoint download. `src/structure_scorer.py` is the one class you replace to screen for real, and it documents the contract.

### Anyscale Technical Demo — Ray Data on Anyscale Jobs

---

## Business Context

A biotech team is designing a binder against a cancer target. They have **1,000 designed candidate proteins**. To rank them they need to fold each target+binder complex through a structure predictor such as **Boltz** (MIT license, AlphaFold3-competitive quality). On one A100 at ~30 seconds per complex, that's **8+ hours** — and a single CUDA OOM kills the run.

**Today:** Researchers run the predictor in a `for` loop on a single GPU. No fault tolerance, no observability, no way to burst.

**With Ray Data on Anyscale:** Wrap the existing Python entry point in a Ray Data actor, fan out across an autoscaling GPU pool, stream features in and confidence scores out, and produce a ranked candidate list — in minutes.

---

## Architecture

```
Anyscale Job / Workspace
|
+- [Ray Data] read_parquet ---------------------- Distributed parallel read
+- [map_batches / CPU] feature_prep ------------- Parse sequences, build Boltz input
|       m5.2xlarge x 1-4 nodes
+- [map_batches / GPU] SimulatedStructureScorer - Scoring stage (swap in a predictor)
|       g5.2xlarge x 1-8 nodes (A10G, 1 GPU per actor)
+- [map_batches / CPU] classify_and_filter ------ Confidence tiers + ranking
+- [write_parquet] ------------------------------ Scored results, tagged with the scorer used
```

Data streams between stages — no intermediate disk writes, no idle CPUs waiting on GPUs.

---

**Timing note**: With a real predictor at ~30s per complex, 500 complexes on 4x A10G finishes in roughly 4 minutes against 4+ hours sequentially — 4 GPUs in parallel, CPU feature prep streaming into GPU inference with no idle time between stages, and Ray Data load-balancing across workers. The simulated scorer returns immediately, so what you are watching here is the pipeline's scheduling and streaming behaviour, not that speedup. Open the Ray Dashboard during the run to see CPU and GPU workers in action.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/biotech_boltz_screening
```

## Step 1: Connect to the Ray Cluster

The next cell initializes a connection to the Anyscale-managed Ray cluster and prints available resources (CPUs, GPUs, nodes). This is the zero-ops experience -- Anyscale provisions and manages the heterogeneous cluster for you. No Terraform, no Kubernetes, no Docker orchestration.

**Key differentiator:** Unlike traditional HPC schedulers, Ray natively schedules work across both CPU and GPU nodes in a single cluster, so your sequence parsing and structure prediction share one unified resource pool.


```python
import os
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```


```python
# Cell 2 - Setup: connect to Ray cluster and inspect resources
import sys, os

# Point to the boltz-screening root so src.* and scripts.* are importable
DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray

# In Anyscale Workspace, Ray is pre-initialized.
# runtime_env working_dir ensures Ray workers can import src.* from this repo.
ray.init(
    ignore_reinit_error=True,
    runtime_env={
        "working_dir": DEMO_ROOT,
        # Install the locked deps into the Ray worker runtime_env so that
        # worker actors/tasks (not just the driver) can import them.
        "pip": os.path.join(DEMO_ROOT, "python_depset.lock"),
    },
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

## Step 2: Load or Generate Candidate Data

The next cell generates a synthetic screening dataset of protein-protein complexes:
- **1 fixed target protein** (~150 amino acids) — a short receptor extracellular domain
- **500 candidate binder proteins** (~50-80 aa each) — created by mutating a seed scaffold with varying mutation rates

This simulates a real binder design campaign where the target is fixed and the binders are diversified computationally. The mutation rate spread ensures a realistic distribution of binding potential (most won't bind, a few will).

**Business value:** In production, this Parquet file would come from your computational design pipeline (e.g., RFdiffusion, ProteinMPNN). The Ray Data pipeline doesn't care where the candidates come from — it just screens them.


```python
# Cell 3 - Generate and inspect candidate complexes
from src.candidate_generator import generate_candidates, SCALE_MAP, TARGET_SEQUENCE
import ray.data
import pandas as pd

INPUT_PATH  = "/mnt/cluster_storage/boltz-screening/candidates/medium_pp.parquet"
OUTPUT_PATH = "/mnt/cluster_storage/boltz-screening/results/medium/"
SCALE       = "medium"  # 500 protein-protein complexes
NUM_GPUS    = 4

# Generate synthetic candidates (skips if file already exists)
if not os.path.exists(INPUT_PATH):
    generate_candidates(INPUT_PATH, num_candidates=SCALE_MAP[SCALE], complex_type="pp")
else:
    print(f"Using existing candidates at {INPUT_PATH}")

# Load with Ray Data and inspect
ds = ray.data.read_parquet(INPUT_PATH)

print(f"Dataset size:  {ds.count():,} candidate complexes")
print(f"Schema:\n{ds.schema()}")

# Show the fixed target protein
print(f"\nTarget protein ({len(TARGET_SEQUENCE)} residues):")
print(f"  {TARGET_SEQUENCE[:80]}...")

# Show sample binder candidates
print("\nSample binder candidates:")
for row in ds.take(3):
    seq = row['binder_seq']
    print(f"\n  complex_id : {row['complex_id']}")
    print(f"  binder_seq : {seq[:60]}... ({len(seq)} aa)")
    print(f"  type       : {row['complex_type']}")
```

## Step 3: Walk the Pipeline Code

Before running the full pipeline, let's look at the **SimulatedStructureScorer** callable class. This is the core GPU stage — one actor per A10G GPU:

- **`__init__`**: Runs once per actor. A real predictor loads its weights here, which is what amortizes a ~30s model load across every batch that actor handles; the stand-in loads nothing.
- **`__call__`**: Processes a batch of complexes, emitting confidence metrics (pLDDT, ipTM) per complex, plus a `scorer` column recording where the numbers came from.

**Swapping in a real predictor** means replacing this one class and keeping the rest. The contract Ray Data depends on — constructor args, input columns, output columns — is written out at the top of `src/structure_scorer.py`.

**Key metrics:**
- **pLDDT** (predicted Local Distance Difference Test): Per-residue confidence in the structure. 0-100, >70 is reliable.
- **ipTM** (interface predicted Template Modeling): Confidence in the interaction interface. 0-1, >0.8 is high.
- **confidence**: the aggregate of the two. This is what we rank by.

The pipeline chains 5 stages: read → feature prep (CPU) → score (GPU) → classify (CPU) → write.


```python
# Cell 4 - Inspect the pipeline code and the scoring class
import inspect
from src.structure_scorer import SimulatedStructureScorer
from src.pipeline import run_screening_pipeline

# Show the scoring class signature — this is the swap point for a real predictor
print("SimulatedStructureScorer — Ray Data callable class (1 actor per GPU):")
print(f"  __init__ args: {inspect.signature(SimulatedStructureScorer.__init__)}")
print(f"  __call__ args: {inspect.signature(SimulatedStructureScorer.__call__)}")
print()

# Show the pipeline function signature
print("run_screening_pipeline signature:")
print(f"  {inspect.signature(run_screening_pipeline)}")
print()

# Show the pipeline stages
print("Pipeline stages:")
print("  [1/5] Read candidate Parquet       — distributed parallel read")
print("  [2/5] Feature prep (CPU)           — parse sequences, build predictor input dicts")
print("  [3/5] Scoring (GPU, simulated)     — 1 actor per A10G")
print("  [4/5] Post-processing (CPU)        — classify confidence tiers, filter")
print("  [5/5] Write scored Parquet          — results + scorer provenance")
```

## Step 4: Run the Screening Pipeline

This is the core of the demo. The next cell runs the full end-to-end pipeline: CPU feature prep streams directly into the GPU scoring stage, which streams directly into CPU post-processing and Parquet writes. There are **no intermediate disk materializations** between stages.

**Technical differentiator:** Ray Data's streaming execution means your GPU workers start scoring as soon as the first feature-prepped batch is ready. CPU and GPU stages run concurrently, maximizing hardware utilization. Open the Ray Dashboard to watch this live.

**TIP:** Open the Ray Dashboard -> Jobs tab to watch:
- CPU workers handling sequence parsing and feature prep
- A10G GPU workers activating for the scoring stage
- Data streaming between stages (no intermediate writes)
- Per-stage throughput and backpressure


```python
# Cell 5 - Run the full screening pipeline
from src.pipeline import run_screening_pipeline

print("Starting screening pipeline (simulated scorer)...")
print(f"  Input:   {INPUT_PATH}")
print(f"  Output:  {OUTPUT_PATH}")
print(f"  GPUs:    {NUM_GPUS} workers (concurrency={NUM_GPUS})\n")
print("TIP: Open Ray Dashboard -> Jobs to watch:")
print("  - CPU workers parsing sequences and building predictor inputs")
print("  - A10G GPU workers activating for the scoring stage")
print("  - Data streaming between stages (no intermediate writes)\n")

metrics = run_screening_pipeline(
    candidates_path=INPUT_PATH,
    output_path=OUTPUT_PATH,
    num_gpus=NUM_GPUS,
)
```

## Step 5: Analyze Results — Confidence Distribution and Top Candidates

The next cell loads the scored output and shows:
1. **Confidence tier distribution** — how many candidates fall into high/medium/low tiers
2. **Top-10 candidates** ranked by confidence score, with ipTM and pLDDT

This is the key demo moment: the pipeline has ranked every candidate and the top of the list is immediately available. With a real predictor wired into `src/structure_scorer.py`, this is the table a wet-lab team would triage from. **With the simulated scorer these rankings carry no biological meaning** — the `scorer` column on every row records that.

**Business value:** what used to require a week of single-GPU compute and ad-hoc post-processing scripts becomes a ranked Parquet table produced in minutes, from the same model code.


```python
# Cell 6 - Load results, show confidence distribution and top-10
import numpy as np
from scripts.validate_results import load_results, print_confidence_distribution, print_top_candidates

df = load_results(OUTPUT_PATH)
print(f"Loaded {len(df):,} scored complexes\n")

# Confidence distribution
print_confidence_distribution(df)

# Top-10 candidates
print_top_candidates(df, k=10)

# Quick histogram of confidence scores
print("\nConfidence score distribution:")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
counts, _ = np.histogram(df["confidence"].values, bins=bins)
for lo, hi, count in zip(bins[:-1], bins[1:], counts):
    bar = "█" * (count // max(1, len(df) // 50))
    print(f"  {lo:.1f}-{hi:.1f}  {count:>5}  {bar}")
```

## Step 6: 3D Structure Visualization and Fault Tolerance

The next cell demonstrates two things:

1. **3D structure rendering** — Using `py3Dmol` to render a predicted CIF structure inline, which is what a researcher would inspect before advancing a binder. The simulated scorer emits a placeholder instead of real atomic coordinates, so the cell below reports that rather than rendering; a real predictor's CIF bytes flow through the pipeline unchanged and render here.

2. **Fault tolerance** — Ray Data automatically retries failed batches. If a GPU worker crashes mid-prediction (CUDA OOM, hardware fault), only the affected batch is re-run on another worker. The pipeline does not restart from scratch — this is critical for long-running screens.

**Why this matters:** In a single-GPU `for` loop, a crash at complex 6,432 of 10,000 loses all progress. With Ray Data, only that batch retries. The ranked output is unchanged.


```python
# Cell 7 - 3D structure visualization (py3Dmol) + fault tolerance explanation

# ── 3D Structure Render ──────────────────────────────────────────────────
# Show the predicted structure of the top-ranked candidate.
# py3Dmol renders CIF/PDB structures inline in Jupyter notebooks. The simulated
# scorer emits a placeholder, so this reports rather than rendering nothing.
top_candidate = df.sort_values("confidence", ascending=False).iloc[0]
print(f"Top candidate: {top_candidate['complex_id']}")
print(f"  Confidence: {top_candidate['confidence']:.4f}")
print(f"  ipTM:       {top_candidate['iptm']:.4f}")
print(f"  pLDDT:      {top_candidate['plddt_mean']:.1f}")
print()

is_simulated = top_candidate.get("scorer") == "simulated"

try:
    import py3Dmol
    cif_data = top_candidate.get("cif_bytes", b"")
    if is_simulated:
        print("Scorer was 'simulated' — the cif_bytes column holds a placeholder, not coordinates.")
        print("Wire a real predictor into src/structure_scorer.py to render a structure here.")
    elif isinstance(cif_data, bytes) and len(cif_data) > 0:
        cif_str = cif_data.decode("utf-8") if isinstance(cif_data, bytes) else cif_data
        view = py3Dmol.view(width=600, height=400)
        view.addModel(cif_str, "cif")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        view.show()
        print("3D structure rendered above (py3Dmol).")
    else:
        print("No CIF structure data available for inline rendering.")
except ImportError:
    print("py3Dmol not installed. To render structures inline:")
    print("  pip install py3Dmol")
    print("  Then re-run this cell to see the 3D structure.")

# ── Fault Tolerance ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FAULT TOLERANCE — RAY DATA AUTOMATIC RETRY")
print("=" * 60)
print("""
  Ray Data provides per-batch fault tolerance out of the box:

  1. If a GPU worker crashes (CUDA OOM, hardware fault, spot
     instance preemption), Ray detects the failure automatically.

  2. Only the affected batch is retried on another available
     GPU worker. Completed batches are NOT re-processed.

  3. The pipeline continues without manual intervention.
     The final ranked output is identical to an error-free run.

  To test: kill a GPU worker pod mid-run and watch the Ray
  Dashboard show the retry. The pipeline completes normally.

  Config: max_retries=2 in job_config.yaml ensures the Job
  itself restarts if the driver node fails.
""")
print("=" * 60)
```

## Step 7: Metrics Summary and Path to Production

The final cell prints pipeline performance metrics and shows how to promote this exact notebook logic to a scheduled **Anyscale Job** with a single CLI command. Same code, same cluster config -- no rewrite needed.

**Key takeaway:** What you just ran interactively in a Workspace becomes a production nightly screen with `anyscale job submit`. Scale from 500 to 50,000 complexes by changing one flag. Anyscale handles autoscaling, fault tolerance, and cluster lifecycle automatically. You now have an internal AlphaFold-as-a-service.


```python
# Cell 8 - Metrics summary and path to production
from src.utils import print_metrics_table

print_metrics_table(metrics)

print("Path to production as an Anyscale Job:")
print("""
  # Submit from CLI (run from boltz-screening/ directory):
  anyscale job submit --config-file job_config.yaml

  # Scale to 2,000 complexes with 8 GPUs:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/02_run_screening.py --scale large --num-gpus 8'

  # Run protein-ligand screening instead:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/02_run_screening.py --scale medium --complex-type pl'
""")

print("Key differentiators vs. single-GPU loop:")
print("  Heterogeneous compute  -- CPU feature prep + GPU scoring in one pipeline")
print("  Streaming execution    -- no intermediate disk writes between stages")
print("  Fault tolerance        -- failed batches retry automatically, no full restart")
print("  Autoscaling            -- GPU workers scale to demand, then scale to zero")
print("  One codebase           -- same code runs in Workspace and as a scheduled Job")
print()
print("Same model code. Ray Data scaling. GPUs autoscale to zero when idle.")
print("Swap src/structure_scorer.py for a real predictor and this becomes an")
print("internal folding service on the same pipeline.")
```
