"""
Ray Data embedding pipeline.
Orchestrates: read → CPU preprocess → GPU embed → write Parquet
"""
import time

import ray
import ray.data
from ray.data import ActorPoolStrategy

from src.preprocess import preprocess_product
from src.embed import ProductEmbedder
from src.utils import calc_throughput, print_metrics_table, format_number, estimate_single_node_time, estimate_job_cost


def run_embedding_pipeline(
    input_path: str,
    output_path: str,
    num_gpus: int = 2,
) -> dict:
    """
    End-to-end Ray Data embedding pipeline.

    Stages:
      1. read_parquet       — distributed parallel read
      2. map_batches(CPU)   — text cleaning and field combination
      3. map_batches(GPU)   — sentence embedding generation
      4. write_parquet      — distributed parallel write

    Returns a metrics dict for display.
    """
    pipeline_start = time.time()

    # ── Stage 1: Read ──────────────────────────────────────────────────────
    print(f"\n[1/4] Reading product catalog from {input_path}")
    # override_num_blocks ensures data is split across all GPU actors (not just 1 block)
    ds = ray.data.read_parquet(input_path, override_num_blocks=num_gpus * 8)
    total_products = ds.count()
    print(f"  Products loaded: {format_number(total_products)}")
    print(f"  Schema: {ds.schema()}")

    # ── Stage 2: CPU Preprocessing ─────────────────────────────────────────
    print(f"\n[2/4] Preprocessing — CPU workers (batch_size=1024)")
    ds = ds.map_batches(
        preprocess_product,
        batch_size=1024,
        num_cpus=1,
        batch_format="numpy",
    )

    # ── Stage 3: GPU Embedding ─────────────────────────────────────────────
    print(f"\n[3/4] Embedding — {num_gpus} GPU worker(s) (batch_size=256)")
    ds = ds.map_batches(
        ProductEmbedder,
        batch_size=256,
        num_gpus=1,
        compute=ActorPoolStrategy(size=num_gpus),
        batch_format="numpy",
    )

    # ── Stage 4: Write ─────────────────────────────────────────────────────
    print(f"\n[4/4] Writing embeddings to {output_path}")
    ds.write_parquet(output_path)

    # ── Metrics ────────────────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_products, wall_time)

    metrics = {
        "Total products processed": format_number(total_products),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.0f} products/sec",
        "Embedding dimensions": "384 (all-MiniLM-L6-v2)",
        "GPU workers": str(num_gpus),
        "Output path": output_path,
        "Estimated single-node time": estimate_single_node_time(total_products, throughput),
        "Estimated job cost": estimate_job_cost(wall_time),
    }

    print_metrics_table(metrics)
    return metrics
