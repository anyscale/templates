"""
Ray Data protein embedding pipeline.

Two modes:
  - run_embedding_pipeline_naive:    No length bucketing (GPU util ~30-40%)
  - run_embedding_pipeline_bucketed: With length bucketing via sort (GPU util ~80%)

The side-by-side comparison is the hero moment of this demo — same data, same hardware,
same model, but 2-3x throughput difference from a single preprocessing optimization.

Orchestrates: read -> CPU validate -> [CPU bucket + sort] -> GPU embed -> CPU postprocess -> write
"""
import time

import pandas as pd
import ray
import ray.data

from src.fasta_loader import read_parquet_corpus
from src.cpu_transforms import validate_and_filter, assign_length_bucket
from src.esm_embedder import ESMEmbedder
from src.postprocess import join_taxonomy, flag_near_duplicates
from src.utils import (
    calc_throughput, print_metrics_table, format_number,
    estimate_single_node_time, estimate_job_cost,
)


def run_embedding_pipeline_naive(
    input_path: str,
    output_path: str,
    taxonomy_path: str,
    num_gpus: int = 2,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> dict:
    """
    Naive embedding pipeline WITHOUT length bucketing.

    GPU batches contain mixed-length sequences, forcing heavy padding.
    GPU utilization ~30-40%. This is the "before" in the before/after comparison.

    Stages:
      1. read_parquet       — distributed parallel read
      2. map_batches(CPU)   — validate + filter sequences
      3. map_batches(GPU)   — ESM-2 embedding (no bucketing)
      4. map_batches(CPU)   — taxonomy join
      5. write_parquet      — embeddings to output path
    """
    pipeline_start = time.time()
    stage_times = {}

    # ── Stage 1: Read ──────────────────────────────────────────────────────
    print(f"\n[1/5] Reading protein corpus from {input_path}")
    t0 = time.time()
    ds = read_parquet_corpus(input_path, num_blocks=num_gpus * 8)
    total_sequences = ds.count()
    stage_times["read"] = time.time() - t0
    print(f"  Sequences loaded: {format_number(total_sequences)}")

    # ── Stage 2: CPU Validation + Filtering ────────────────────────────────
    print(f"\n[2/5] Validating sequences — filtering non-canonical AAs and length bounds")
    t0 = time.time()
    ds = ds.map_batches(
        validate_and_filter,
        batch_size=2048,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["validate"] = time.time() - t0

    # ── Stage 3: GPU ESM-2 Embedding (NO bucketing) ───────────────────────
    print(f"\n[3/5] Embedding with ESM-2 — {num_gpus} GPU worker(s), NO length bucketing")
    print(f"  (Mixed-length batches cause heavy padding — expect ~30-40% GPU utilization)")
    t0 = time.time()
    ds = ds.map_batches(
        ESMEmbedder,
        fn_constructor_kwargs={"model_name": model_name},
        batch_size=32,
        num_gpus=1,
        concurrency=num_gpus,
        batch_format="numpy",
    )
    stage_times["embed"] = time.time() - t0

    # ── Stage 4: CPU Post-processing ──────────────────────────────────────
    print(f"\n[4/5] Joining taxonomy metadata (broadcast join)")
    t0 = time.time()
    tax_ref = ray.put(pd.read_parquet(taxonomy_path))
    ds = ds.map_batches(
        lambda b: join_taxonomy(b, ray.get(tax_ref)),
        batch_size=4096,
        num_cpus=1,
        batch_format="numpy",
    )
    ds = ds.map_batches(
        flag_near_duplicates,
        batch_size=4096,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["postprocess"] = time.time() - t0

    # ── Stage 5: Write ────────────────────────────────────────────────────
    print(f"\n[5/5] Writing embeddings to {output_path}")
    t0 = time.time()
    ds.write_parquet(output_path)
    stage_times["write"] = time.time() - t0

    # ── Metrics ───────────────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_sequences, wall_time)

    metrics = {
        "Mode": "NAIVE (no bucketing)",
        "Total sequences processed": format_number(total_sequences),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.0f} sequences/sec",
        "GPU workers": str(num_gpus),
        "Model": model_name.split("/")[-1],
        "Read time": f"{stage_times['read']:.1f}s",
        "Validate time": f"{stage_times['validate']:.1f}s",
        "Embed time": f"{stage_times['embed']:.1f}s",
        "Post-process time": f"{stage_times['postprocess']:.1f}s",
        "Write time": f"{stage_times['write']:.1f}s",
        "Output path": output_path,
        "Est. single-node time": estimate_single_node_time(total_sequences),
        "Est. Anyscale job cost": estimate_job_cost(wall_time, num_gpu_workers=num_gpus),
    }

    print_metrics_table(metrics)
    return metrics


def run_embedding_pipeline_bucketed(
    input_path: str,
    output_path: str,
    taxonomy_path: str,
    num_gpus: int = 2,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
) -> dict:
    """
    Optimized embedding pipeline WITH length bucketing.

    Sequences are sorted by length bucket before the GPU stage, so each GPU batch
    contains sequences of similar length. This minimizes padding waste and boosts
    GPU utilization from ~30% to ~80%.

    This is the "after" in the before/after comparison — the hero moment of the demo.

    Stages:
      1. read_parquet       — distributed parallel read
      2. map_batches(CPU)   — validate + filter sequences
      3. map_batches(CPU)   — assign length buckets
      4. sort(length_bucket)— repartition for length-homogeneous GPU batches
      5. map_batches(GPU)   — ESM-2 embedding (bucketed)
      6. map_batches(CPU)   — taxonomy join
      7. write_parquet      — embeddings to output path
    """
    pipeline_start = time.time()
    stage_times = {}

    # ── Stage 1: Read ──────────────────────────────────────────────────────
    print(f"\n[1/7] Reading protein corpus from {input_path}")
    t0 = time.time()
    ds = read_parquet_corpus(input_path, num_blocks=num_gpus * 8)
    total_sequences = ds.count()
    stage_times["read"] = time.time() - t0
    print(f"  Sequences loaded: {format_number(total_sequences)}")

    # ── Stage 2: CPU Validation + Filtering ────────────────────────────────
    print(f"\n[2/7] Validating sequences — filtering non-canonical AAs and length bounds")
    t0 = time.time()
    ds = ds.map_batches(
        validate_and_filter,
        batch_size=2048,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["validate"] = time.time() - t0

    # ── Stage 3: CPU Length Bucketing ──────────────────────────────────────
    print(f"\n[3/7] Assigning length buckets (4 buckets: 20-128, 129-256, 257-512, 513-1024)")
    t0 = time.time()
    ds = ds.map_batches(
        assign_length_bucket,
        batch_size=2048,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["bucket"] = time.time() - t0

    # ── Stage 4: Sort by Bucket ───────────────────────────────────────────
    print(f"\n[4/7] Sorting by length bucket — GPU batches will be length-homogeneous")
    t0 = time.time()
    ds = ds.sort("length_bucket")
    stage_times["sort"] = time.time() - t0

    # ── Stage 5: GPU ESM-2 Embedding (WITH bucketing) ─────────────────────
    print(f"\n[5/7] Embedding with ESM-2 — {num_gpus} GPU worker(s), WITH length bucketing")
    print(f"  (Length-homogeneous batches minimize padding — expect ~80% GPU utilization)")
    t0 = time.time()
    ds = ds.map_batches(
        ESMEmbedder,
        fn_constructor_kwargs={"model_name": model_name},
        batch_size=32,
        num_gpus=1,
        concurrency=num_gpus,
        batch_format="numpy",
    )
    stage_times["embed"] = time.time() - t0

    # ── Stage 6: CPU Post-processing ──────────────────────────────────────
    print(f"\n[6/7] Joining taxonomy metadata (broadcast join)")
    t0 = time.time()
    tax_ref = ray.put(pd.read_parquet(taxonomy_path))
    ds = ds.map_batches(
        lambda b: join_taxonomy(b, ray.get(tax_ref)),
        batch_size=4096,
        num_cpus=1,
        batch_format="numpy",
    )
    ds = ds.map_batches(
        flag_near_duplicates,
        batch_size=4096,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["postprocess"] = time.time() - t0

    # ── Stage 7: Write ────────────────────────────────────────────────────
    print(f"\n[7/7] Writing embeddings to {output_path}")
    t0 = time.time()
    ds.write_parquet(output_path)
    stage_times["write"] = time.time() - t0

    # ── Metrics ───────────────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_sequences, wall_time)

    metrics = {
        "Mode": "BUCKETED (length-aware batching)",
        "Total sequences processed": format_number(total_sequences),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.0f} sequences/sec",
        "GPU workers": str(num_gpus),
        "Model": model_name.split("/")[-1],
        "Read time": f"{stage_times['read']:.1f}s",
        "Validate time": f"{stage_times['validate']:.1f}s",
        "Bucket time": f"{stage_times['bucket']:.1f}s",
        "Sort time": f"{stage_times['sort']:.1f}s",
        "Embed time": f"{stage_times['embed']:.1f}s",
        "Post-process time": f"{stage_times['postprocess']:.1f}s",
        "Write time": f"{stage_times['write']:.1f}s",
        "Output path": output_path,
        "Est. single-node time": estimate_single_node_time(total_sequences),
        "Est. Anyscale job cost": estimate_job_cost(wall_time, num_gpu_workers=num_gpus),
    }

    print_metrics_table(metrics)
    return metrics
