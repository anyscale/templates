"""
Ray Data fraud scoring pipeline.
Orchestrates: read → join features → compute features → score → write
"""
import time

import numpy as np
import pandas as pd
import ray
import ray.data

from src.feature_engineering import join_user_features, join_merchant_features, compute_features
from src.fraud_scorer import FraudScorer
from src.utils import (
    calc_throughput, print_metrics_table, format_number,
    estimate_single_node_time, estimate_spark_time, estimate_job_cost,
)


def run_fraud_scoring_pipeline(
    input_path: str,
    output_path: str,
    model_path: str,
    user_features_path: str,
    merchant_features_path: str,
    num_workers: int = 10,
) -> dict:
    """
    End-to-end Ray Data fraud scoring pipeline.

    Stages:
      1. read_parquet           — distributed parallel read
      2. map_batches (CPU)      — join pre-computed user + merchant aggregates
      3. map_batches (CPU)      — feature engineering (21 features)
      4. map_batches (CPU)      — XGBoost fraud scoring + risk classification
      5. write_parquet          — scored transactions to output

    Returns a metrics dict for display.
    """
    pipeline_start = time.time()
    stage_times = {}

    # ── Stage 1: Read ──────────────────────────────────────────
    print(f"\n[1/5] Reading transactions from {input_path}")
    t0 = time.time()
    ds = ray.data.read_parquet(input_path, override_num_blocks=num_workers * 4)
    total_transactions = ds.count()
    stage_times["read"] = time.time() - t0
    print(f"  Transactions loaded: {format_number(total_transactions)}")

    # ── Broadcast lookups ──────────────────────────────────────
    print(f"\n[2/5] Joining pre-computed user + merchant aggregates (broadcast)")
    t0 = time.time()
    user_lookup_ref = ray.put(pd.read_parquet(user_features_path))
    merchant_lookup_ref = ray.put(pd.read_parquet(merchant_features_path))

    ds = ds.map_batches(
        lambda batch: join_user_features(batch, ray.get(user_lookup_ref)),
        batch_size=10_000, num_cpus=1, batch_format="numpy",
    )
    ds = ds.map_batches(
        lambda batch: join_merchant_features(batch, ray.get(merchant_lookup_ref)),
        batch_size=10_000, num_cpus=1, batch_format="numpy",
    )
    stage_times["join"] = time.time() - t0

    # ── Stage 3: Feature engineering ───────────────────────────
    print(f"\n[3/5] Computing 21 engineered features (CPU workers)")
    t0 = time.time()
    ds = ds.map_batches(
        compute_features,
        batch_size=10_000,
        num_cpus=1,
        batch_format="numpy",
    )
    stage_times["features"] = time.time() - t0

    # ── Stage 4: XGBoost scoring ───────────────────────────────
    print(f"\n[4/5] Scoring with XGBoost model ({num_workers} concurrent workers)")
    t0 = time.time()
    ds = ds.map_batches(
        FraudScorer,
        fn_constructor_kwargs={"model_path": model_path},
        batch_size=50_000,
        num_cpus=1,
        concurrency=num_workers,
        batch_format="numpy",
    )
    stage_times["scoring"] = time.time() - t0

    # ── Stage 5: Write results ─────────────────────────────────
    print(f"\n[5/5] Writing scored transactions to {output_path}")
    t0 = time.time()
    ds.write_parquet(output_path)
    stage_times["write"] = time.time() - t0

    # ── Metrics ────────────────────────────────────────────────
    wall_time = time.time() - pipeline_start
    throughput = calc_throughput(total_transactions, wall_time)

    metrics = {
        "Total transactions scored": format_number(total_transactions),
        "Wall time": f"{wall_time:.1f}s ({wall_time / 60:.1f} min)",
        "Throughput": f"{throughput:,.0f} transactions/sec",
        "Concurrent workers": str(num_workers),
        "Read time": f"{stage_times['read']:.1f}s",
        "Join time": f"{stage_times['join']:.1f}s",
        "Feature engineering time": f"{stage_times['features']:.1f}s",
        "Scoring time": f"{stage_times['scoring']:.1f}s",
        "Write time": f"{stage_times['write']:.1f}s",
        "Output path": output_path,
        "Est. single-node time": estimate_single_node_time(total_transactions),
        "Est. Spark pipeline": estimate_spark_time(total_transactions),
        "Est. Anyscale job cost": estimate_job_cost(wall_time),
    }

    print_metrics_table(metrics)
    return metrics
