"""
Ray Data VLM batch enrichment pipeline.

Three modes:
  - run_pipeline_naive:           Single GPU stage does fetch + decode + infer.
                                  GPU is blocked on I/O. The "before" picture.
  - run_pipeline_heterogeneous:   CPU pool fetches + decodes; GPU pool only
                                  runs the VLM (via ray.data.llm). The "after".
  - run_with_checkpoints:         Same heterogeneous pipeline, sharded with
                                  atomic per-shard commits → resumable across
                                  cluster restarts. The production-grade path.

Catalog for the simple modes is cached at ``catalog_path``; for the sharded
mode it's pre-split into ``input_dir/shard_NNNN.parquet`` files (see
``src.load_catalog.shard_catalog_to_parquet``).
"""
import os
import time
import uuid

import ray
import ray.data
from ray.data import ActorPoolStrategy

from src.enrich import NaiveVLMEnricher, build_heterogeneous_processor
from src.load_catalog import load_amazon_reviews_2023
from src.observability import (
    finish_wandb,
    init_ray_metrics,
    log_event,
    maybe_init_wandb,
    record_shard_complete,
)
from src.preprocess import fetch_and_decode
from src.shard import (
    cleanup_stale_tmp,
    commit_shard,
    list_completed_shards,
    list_input_shards,
    shard_input_path,
    shard_tmp_path,
)
from src.utils import (
    calc_throughput,
    estimate_job_cost,
    estimate_single_node_time,
    format_number,
    print_metrics_table,
)


def ensure_catalog(category: str, n_rows: int, catalog_path: str) -> None:
    """Build the normalized parquet cache on first run; reuse on subsequent runs."""
    if os.path.exists(catalog_path):
        print(f"[load] Reusing existing catalog at {catalog_path}")
        return
    print(f"[load] Building catalog at {catalog_path} ({n_rows:,} rows from {category})")
    load_amazon_reviews_2023(category, n_rows=n_rows).write_parquet(catalog_path)


def run_pipeline_naive(catalog_path: str, output_path: str, num_gpus: int = 2) -> dict:
    """Single GPU stage does HTTP fetch + image decode + VLM generate."""
    pipeline_start = time.time()

    print(f"\n[1/3] Reading catalog from {catalog_path}")
    ds = ray.data.read_parquet(catalog_path, override_num_blocks=num_gpus * 8)
    total = ds.count()
    print(f"  Products loaded: {format_number(total)}")

    print(f"\n[2/3] NAIVE enrichment — {num_gpus} GPU actor(s) doing fetch + decode + generate")
    print(f"  (GPU will sit idle while each image downloads — expect low utilization)")
    ds = ds.map_batches(
        NaiveVLMEnricher,
        batch_size=8,
        num_gpus=1,
        compute=ActorPoolStrategy(size=num_gpus),
        batch_format="numpy",
    )

    print(f"\n[3/3] Writing enriched catalog to {output_path}")
    ds.write_parquet(output_path)

    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total, wall_time)

    # Real per-operator timings live in ds.stats() — Ray Data's internal
    # streaming executor accumulates them as it runs. Manual time.time()
    # around .map_batches() / .write_parquet() is misleading because Ray
    # Data is lazy; the work all attributes to the terminal operator.
    print("\n--- Ray Data per-operator stats ---")
    print(ds.stats())

    print("\n--- Sample enriched rows ---")
    print(ray.data.read_parquet(output_path).take(3))

    metrics = {
        "Mode": "NAIVE (GPU does fetch + decode + infer)",
        "Total products processed": format_number(total),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.2f} products/sec",
        "GPU workers": str(num_gpus),
        "Model": "Qwen2.5-VL-3B-Instruct",
        "Output path": output_path,
        "Est. single-node time": estimate_single_node_time(total, throughput),
        "Est. Anyscale job cost": estimate_job_cost(wall_time, num_cpu_workers=0, num_gpu_workers=num_gpus),
    }
    print_metrics_table(metrics)
    return metrics


def run_pipeline_heterogeneous(
    catalog_path: str,
    output_path: str,
    num_gpus: int = 2,
    cpu_concurrency: int = 8,
) -> dict:
    """CPU pool fetches + decodes; ray.data.llm processor runs only the VLM."""
    pipeline_start = time.time()

    print(f"\n[1/4] Reading catalog from {catalog_path}")
    ds = ray.data.read_parquet(catalog_path, override_num_blocks=cpu_concurrency * 4)
    total = ds.count()
    print(f"  Products loaded: {format_number(total)}")

    print(f"\n[2/4] CPU image fetch + decode — {cpu_concurrency} CPU workers")
    ds = ds.map_batches(
        fetch_and_decode,
        batch_size=16,
        concurrency=cpu_concurrency,
        batch_format="numpy",
    )

    print(f"\n[3/4] VLM enrichment via ray.data.llm — {num_gpus} GPU worker(s)")
    print(f"  (GPU only runs inference; CPU pool keeps it fed)")
    processor = build_heterogeneous_processor(num_gpus=num_gpus, batch_size=8)
    ds = processor(ds)

    print(f"\n[4/4] Writing enriched catalog to {output_path}")
    ds.write_parquet(output_path)

    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total, wall_time)

    # Real per-operator timings live in ds.stats() — Ray Data's streaming
    # executor accumulates them as it runs. Manual time.time() around lazy
    # operators is misleading because the work attributes to the terminal
    # write_parquet rather than the upstream stages it actually came from.
    print("\n--- Ray Data per-operator stats ---")
    print(ds.stats())

    print("\n--- Sample enriched rows ---")
    print(ray.data.read_parquet(output_path).take(3))

    metrics = {
        "Mode": "HETEROGENEOUS (CPU pool fetches; GPU only infers)",
        "Total products processed": format_number(total),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.2f} products/sec",
        "CPU workers": str(cpu_concurrency),
        "GPU workers": str(num_gpus),
        "Model": "Qwen2.5-VL-3B-Instruct",
        "Output path": output_path,
        "Est. single-node time": estimate_single_node_time(total, throughput),
        "Est. Anyscale job cost": estimate_job_cost(
            wall_time, num_cpu_workers=cpu_concurrency, num_gpu_workers=num_gpus
        ),
    }
    print_metrics_table(metrics)
    return metrics


def run_with_checkpoints(
    input_dir: str,
    output_dir: str,
    num_gpus: int = 2,
    batch_size: int = 8,
    cpu_concurrency: int = 8,
    run_name: str = None,
    max_shards_this_run: int = None,
) -> dict:
    """Process input shards with atomic per-shard commits.

    - Skips already-committed shards (resume semantics).
    - Cleans up half-written .tmp/ dirs from a prior crashed run.
    - Builds the vLLM processor once so engine actors stay warm across shards.
    - On SIGTERM (caller raises KeyboardInterrupt), the in-flight shard's
      .tmp/ stays behind; the next run cleans it and reprocesses.
    """
    run_name = run_name or f"run-{uuid.uuid4().hex[:8]}"
    pipeline_start = time.time()

    os.makedirs(output_dir, exist_ok=True)

    stale = cleanup_stale_tmp(output_dir)
    if stale:
        log_event("cleanup_stale_tmp", removed=stale)

    all_inputs = list_input_shards(input_dir)
    completed = list_completed_shards(output_dir)
    remaining = sorted(set(all_inputs) - completed)

    log_event(
        "checkpoint_scan",
        input_shards=len(all_inputs),
        already_completed=len(completed),
        remaining=len(remaining),
        run_name=run_name,
    )

    if not remaining:
        print(f"All {len(all_inputs)} shards already completed at {output_dir} — nothing to do.")
        return {"status": "no-op", "completed": len(completed), "total": len(all_inputs)}

    if max_shards_this_run is not None and max_shards_this_run < len(remaining):
        log_event("max_shards_capped", remaining_total=len(remaining), processing=max_shards_this_run)
        remaining = remaining[:max_shards_this_run]

    print(f"\n[checkpoint] Resuming run: {len(completed)}/{len(all_inputs)} done, "
          f"{len(remaining)} shards to process this invocation\n")

    init_ray_metrics()
    maybe_init_wandb(run_name=run_name, total_shards=len(all_inputs))

    # Build the vLLM processor ONCE — actor pool persists across shards so
    # the model loads only on the first shard.
    print(f"[engine] Building Qwen2.5-VL-3B vLLM processor ({num_gpus} GPU workers)...")
    processor = build_heterogeneous_processor(num_gpus=num_gpus, batch_size=batch_size)

    total_products = 0
    shard_times = []

    for shard_id in remaining:
        shard_start = time.time()
        in_path = shard_input_path(input_dir, shard_id)
        tmp_dir = shard_tmp_path(output_dir, shard_id)

        print(f"[shard {shard_id:04d}] Reading {in_path}")
        ds = ray.data.read_parquet(in_path)
        n_in = ds.count()

        ds = ds.map_batches(
            fetch_and_decode,
            batch_size=16,
            concurrency=cpu_concurrency,
            batch_format="numpy",
        )
        ds = processor(ds)
        ds.write_parquet(tmp_dir)

        # Atomic commit. If the process dies before this rename, the next run
        # cleans up the .tmp/ and reprocesses this shard from scratch.
        commit_shard(output_dir, shard_id)

        elapsed = time.time() - shard_start
        total_products += n_in
        shard_times.append(elapsed)

        completed_now = len(completed) + len(shard_times)
        record_shard_complete(
            shard_id=shard_id,
            num_products=n_in,
            elapsed_seconds=elapsed,
            total_completed=completed_now,
            total_shards=len(all_inputs),
        )
        print(f"[shard {shard_id:04d}] ✓ {n_in} products in {elapsed:.1f}s "
              f"({completed_now}/{len(all_inputs)} shards done)")

    finish_wandb()

    print("\n--- Sample enriched rows ---")
    print(ray.data.read_parquet(output_dir).take(3))

    wall_time = time.time() - pipeline_start
    avg_shard = sum(shard_times) / len(shard_times) if shard_times else 0
    throughput = calc_throughput(total_products, wall_time)

    metrics = {
        "Run name": run_name,
        "Mode": "Sharded + checkpointed (ray.data.llm + Qwen2.5-VL-3B)",
        "Shards completed this run": format_number(len(shard_times)),
        "Shards committed total": format_number(len(completed) + len(shard_times)),
        "Shards remaining": format_number(len(all_inputs) - len(completed) - len(shard_times)),
        "Products processed this run": format_number(total_products),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Avg time per shard": f"{avg_shard:.1f}s",
        "Throughput": f"{throughput:,.2f} products/sec",
        "CPU concurrency": str(cpu_concurrency),
        "GPU workers": str(num_gpus),
        "Output dir": output_dir,
        "Est. Anyscale cost (this run)": estimate_job_cost(
            wall_time, num_cpu_workers=cpu_concurrency, num_gpu_workers=num_gpus
        ),
    }
    print_metrics_table(metrics, title="VLM ENRICHMENT JOB — RUN COMPLETE")
    return metrics
