"""
Ray Data Boltz screening pipeline.
Orchestrates: read → CPU feature prep → GPU structure prediction → CPU classify → write
"""
import time

import ray
import ray.data
from ray.data import SaveMode

from src.feature_prep import build_boltz_input_batch
from src.boltz_predictor import SCORER_NAME, BoltzPredictor, ensure_weights
from src.postprocess import classify_and_filter
from src.utils import (
    calc_throughput, print_dataset_stats, print_metrics_table, format_number,
    estimate_single_node_time, estimate_job_cost,
)


def run_screening_pipeline(
    candidates_path: str,
    output_path: str,
    cache_dir: str = "/mnt/cluster_storage/boltz_cache",
    target_msa_path: str = "empty",
    num_gpus: int = 4,
    batch_size: int = 32,
) -> dict:
    """
    End-to-end Ray Data Boltz screening pipeline.

    Stages:
      1. read_parquet           — distributed parallel read of candidate complexes
      2. map_batches (CPU)      — parse sequences, build Boltz input YAML
      3. map_batches (GPU)      — Boltz structure prediction (1 actor per GPU)
      4. map_batches (CPU)      — classify confidence tiers, filter
      5. write_parquet          — scored results + CIF bytes to output

    Stages 2-4 only add operators to a plan. Ray Data is lazy: nothing runs until the
    terminal `write_parquet()`, so a wall-clock around each `map_batches()` call would
    measure plan construction, not work. Real per-operator timings come from
    `ds.stats()` after the write.

    Returns a metrics dict for display.
    """
    # Once here, rather than every actor racing for the same cache.
    ensure_weights(cache_dir)

    pipeline_start = time.time()

    # ── Stage 1: Read ──────────────────────────────────────────────────────
    print(f"\n[1/5] Reading candidates from {candidates_path}")
    ds = ray.data.read_parquet(candidates_path, override_num_blocks=num_gpus * 4)
    total_complexes = ds.count()
    print(f"  Candidates found: {format_number(total_complexes)}")

    # ── Stage 2: CPU Feature Prep ──────────────────────────────────────────
    # Parse sequences, validate amino acids, build Boltz YAML input dicts.
    # Attaches pre-computed MSA for the target, MSA-free for binder candidates.
    print("\n[2/5] Queueing feature prep — CPU workers (parse sequences, build Boltz inputs)")
    ds = ds.map_batches(
        lambda batch: build_boltz_input_batch(batch, target_msa_path),
        batch_size=64,
        num_cpus=1,
        batch_format="numpy",
    )

    # ── Stage 3: GPU Structure Prediction ──────────────────────────────────
    # One Boltz actor per GPU; concurrency = num_gpus saturates the autoscaled
    # pool. Each batch becomes a single `boltz predict` call, which costs ~31s of
    # model load plus ~11s per complex on an L4 — so a large batch_size is what
    # keeps that startup from being paid over and over.
    print(f"\n[3/5] Queueing Boltz structure prediction — {num_gpus} GPU worker(s)")
    ds = ds.map_batches(
        BoltzPredictor,
        fn_constructor_kwargs={"cache_dir": cache_dir},
        batch_size=batch_size,
        num_gpus=1,
        concurrency=num_gpus,
        batch_format="numpy",
    )

    # ── Stage 4: CPU Post-processing ───────────────────────────────────────
    # Classify confidence tiers (high/medium/low), add passed_filter flag.
    print("\n[4/5] Queueing post-processing — classify confidence tiers, filter")
    ds = ds.map_batches(
        classify_and_filter,
        batch_size=256,
        num_cpus=1,
        batch_format="numpy",
    )

    # ── Stage 5: Execute and Write ─────────────────────────────────────────
    # The terminal operation. Stages 2-4 built a plan and ran nothing; they
    # execute here, streamed, as write_parquet() pulls batches through it.
    print(f"\n[5/5] Executing the plan, writing scored results to {output_path}")
    ds.write_parquet(output_path, mode=SaveMode.OVERWRITE)

    # ── Metrics ────────────────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_complexes, wall_time)

    metrics = {
        "Total complexes screened": format_number(total_complexes),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:.2f} complexes/sec",
        "GPU workers": str(num_gpus),
        "Output path": output_path,
        "Scorer": SCORER_NAME,
        "Est. single-GPU time": estimate_single_node_time(total_complexes),
        "Est. Anyscale job cost": estimate_job_cost(wall_time, num_gpu_workers=num_gpus),
    }

    print_metrics_table(metrics)

    # Reading these: operators aren't 1:1 with the stages above (Ray Data fuses
    # neighbours sharing a compute strategy), and their wall times overlap rather than
    # partition the run, so they sum to more than the pipeline's.
    print_dataset_stats(ds.stats())
    return metrics
