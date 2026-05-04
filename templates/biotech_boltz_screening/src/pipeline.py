"""
Ray Data Boltz-1 screening pipeline.
Orchestrates: read → CPU feature prep → GPU structure prediction → CPU classify → write
"""
import time

import ray
import ray.data

from src.feature_prep import build_boltz_input_batch
from src.boltz_predictor import BoltzPredictor
from src.postprocess import classify_and_filter
from src.utils import (
    calc_throughput, print_metrics_table, format_number,
    estimate_single_node_time, estimate_job_cost,
)


def run_screening_pipeline(
    candidates_path: str,
    output_path: str,
    weights_path: str = "/mnt/cluster_storage/boltz/boltz1.ckpt",
    target_msa_path: str = "/mnt/cluster_storage/boltz-screening/assets/target_msa.a3m",
    num_gpus: int = 4,
) -> dict:
    """
    End-to-end Ray Data Boltz-1 screening pipeline.

    Stages:
      1. read_parquet           — distributed parallel read of candidate complexes
      2. map_batches (CPU)      — parse sequences, build Boltz-1 input dicts
      3. map_batches (GPU)      — Boltz-1 structure prediction (1 actor per GPU)
      4. map_batches (CPU)      — classify confidence tiers, filter
      5. write_parquet          — scored results + CIF bytes to output

    Returns a metrics dict for display.
    """
    pipeline_start = time.time()
    stage_times = {}

    # ── Stage 1: Read ──────────────────────────────────────────────────────
    print(f"\n[1/5] Reading candidates from {candidates_path}")
    t0 = time.time()
    ds = ray.data.read_parquet(candidates_path, override_num_blocks=num_gpus * 4)
    total_complexes = ds.count()
    stage_times["read"] = time.time() - t0
    print(f"  Candidates loaded: {format_number(total_complexes)}")

    # ── Stage 2: CPU Feature Prep ──────────────────────────────────────────
    # Parse sequences, validate amino acids, build Boltz-1 YAML-style input dicts.
    # Attaches pre-computed MSA for the target, MSA-free for binder candidates.
    print(f"\n[2/5] Feature prep — CPU workers (parse sequences, build Boltz inputs)")
    t0 = time.time()
    ds = ds.map_batches(
        lambda batch: build_boltz_input_batch(batch, target_msa_path),
        batch_size=64,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["feature_prep"] = time.time() - t0

    # ── Stage 3: GPU Structure Prediction ──────────────────────────────────
    # One Boltz-1 actor per A10G GPU. Each actor loads the model once and
    # processes all batches assigned to it. concurrency = num_gpus ensures
    # we saturate the autoscaled GPU pool.
    print(f"\n[3/5] Boltz-1 structure prediction — {num_gpus} GPU worker(s)")
    t0 = time.time()
    ds = ds.map_batches(
        BoltzPredictor,
        fn_constructor_kwargs={"weights_path": weights_path},
        batch_size=4,
        num_gpus=1,
        concurrency=num_gpus,
        batch_format="numpy",
    )
    stage_times["prediction"] = time.time() - t0

    # ── Stage 4: CPU Post-processing ───────────────────────────────────────
    # Classify confidence tiers (high/medium/low), add passed_filter flag.
    print(f"\n[4/5] Post-processing — classify confidence tiers, filter")
    t0 = time.time()
    ds = ds.map_batches(
        classify_and_filter,
        batch_size=256,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["postprocess"] = time.time() - t0

    # ── Stage 5: Write Results ─────────────────────────────────────────────
    print(f"\n[5/5] Writing scored results to {output_path}")
    t0 = time.time()
    ds.write_parquet(output_path)
    stage_times["write"] = time.time() - t0

    # ── Metrics ────────────────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_complexes, wall_time)

    metrics = {
        "Total complexes screened": format_number(total_complexes),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:.2f} complexes/sec",
        "GPU workers": str(num_gpus),
        "Read time": f"{stage_times['read']:.1f}s",
        "Feature prep time": f"{stage_times['feature_prep']:.1f}s",
        "Prediction time": f"{stage_times['prediction']:.1f}s",
        "Post-processing time": f"{stage_times['postprocess']:.1f}s",
        "Write time": f"{stage_times['write']:.1f}s",
        "Output path": output_path,
        "Est. single-GPU time": estimate_single_node_time(total_complexes),
        "Est. Anyscale job cost": estimate_job_cost(wall_time, num_gpu_workers=num_gpus),
    }

    print_metrics_table(metrics)
    return metrics
