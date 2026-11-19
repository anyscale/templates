"""
Observability for the sharded VLM enrichment job.

Three tiers, all wired through one set of helpers so the per-shard loop
only calls ``record_shard_complete()``:

1. **Structured JSON logs** (always-on) — one line per event to stdout,
   picked up by Anyscale's log aggregation. Searchable by ``event`` field.
2. **Ray metrics** (Counter / Gauge / Histogram) — surfaced via Ray's
   Prometheus exporter and visible in the Anyscale Grafana dashboard.
3. **W&B** (optional, gated on ``WANDB_API_KEY``) — per-shard step charts.
"""
import json
import logging
import os
import sys
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Structured logging (always-on)
# ---------------------------------------------------------------------------

_log_handler = logging.StreamHandler(sys.stdout)
_log_handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines
logger = logging.getLogger("vlm-enrichment")
logger.setLevel(logging.INFO)
logger.addHandler(_log_handler)
logger.propagate = False


def log_event(event: str, **fields) -> None:
    """Emit a single JSON line for downstream log aggregation."""
    payload = {"ts": time.time(), "event": event, **fields}
    logger.info(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# Ray metrics → Anyscale Prometheus → Grafana
# ---------------------------------------------------------------------------

_ray_metrics_initialized = False
_PRODUCTS_PROCESSED = None
_SHARDS_COMPLETED = None
_SHARD_LATENCY = None


def init_ray_metrics() -> None:
    """Lazy-init Ray metric handles. Must be called AFTER ray.init().

    Safe to call multiple times.
    """
    global _ray_metrics_initialized, _PRODUCTS_PROCESSED, _SHARDS_COMPLETED, _SHARD_LATENCY
    if _ray_metrics_initialized:
        return
    try:
        from ray.util.metrics import Counter, Gauge, Histogram

        _PRODUCTS_PROCESSED = Counter(
            "vlm_enrichment_products_processed_total",
            description="Total products enriched across all shards in this run",
            tag_keys=("shard",),
        )
        _SHARDS_COMPLETED = Gauge(
            "vlm_enrichment_shards_completed",
            description="Number of shards committed in this run",
        )
        _SHARD_LATENCY = Histogram(
            "vlm_enrichment_shard_seconds",
            description="Wall time per shard",
            boundaries=[10, 30, 60, 120, 300, 600, 1200, 3600],
        )
        _ray_metrics_initialized = True
    except Exception as e:
        log_event("ray_metrics_init_skipped", reason=str(e))


def record_shard_metrics(
    shard_id: int, num_products: int, elapsed_seconds: float, total_completed: int
) -> None:
    if not _ray_metrics_initialized:
        return
    _PRODUCTS_PROCESSED.inc(num_products, tags={"shard": str(shard_id)})
    _SHARDS_COMPLETED.set(total_completed)
    _SHARD_LATENCY.observe(elapsed_seconds)


# ---------------------------------------------------------------------------
# W&B (optional)
# ---------------------------------------------------------------------------

_wandb_run = None


def maybe_init_wandb(run_name: str, total_shards: int, project: Optional[str] = None) -> bool:
    """Initialize a W&B run if WANDB_API_KEY is set. No-op otherwise."""
    global _wandb_run
    if not os.environ.get("WANDB_API_KEY"):
        log_event("wandb_skipped", reason="WANDB_API_KEY not set")
        return False
    try:
        import wandb
    except ImportError:
        log_event("wandb_skipped", reason="wandb not installed")
        return False

    _wandb_run = wandb.init(
        project=project or os.environ.get("WANDB_PROJECT", "anyscale-vlm-enrichment-demo"),
        name=run_name,
        resume="allow",
        config={"total_shards": total_shards},
    )
    log_event("wandb_initialized", run_name=run_name, project=_wandb_run.project)
    return True


def log_to_wandb(**fields) -> None:
    if _wandb_run is None:
        return
    _wandb_run.log(fields)


def finish_wandb() -> None:
    global _wandb_run
    if _wandb_run is not None:
        _wandb_run.finish()
        _wandb_run = None


# ---------------------------------------------------------------------------
# Composite helper — call after each committed shard.
# ---------------------------------------------------------------------------

def record_shard_complete(
    shard_id: int,
    num_products: int,
    elapsed_seconds: float,
    total_completed: int,
    total_shards: int,
) -> None:
    throughput = num_products / elapsed_seconds if elapsed_seconds > 0 else 0
    log_event(
        "shard_complete",
        shard_id=shard_id,
        num_products=num_products,
        elapsed_seconds=round(elapsed_seconds, 2),
        throughput_per_sec=round(throughput, 2),
        progress=f"{total_completed}/{total_shards}",
    )
    record_shard_metrics(shard_id, num_products, elapsed_seconds, total_completed)
    log_to_wandb(
        shard_id=shard_id,
        shard_throughput=throughput,
        shard_wall_seconds=elapsed_seconds,
        shards_completed=total_completed,
        progress_pct=100.0 * total_completed / max(total_shards, 1),
    )
