"""
Timing, metrics, and display utilities for the fraud scoring pipeline demo.
"""
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    """Context manager that prints elapsed time for a labeled block."""
    print(f"\n⏱  {label}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"   ✓ {label} complete — {elapsed:.1f}s")


def calc_throughput(num_items: int, elapsed_seconds: float) -> float:
    return num_items / elapsed_seconds if elapsed_seconds > 0 else 0.0


def estimate_single_node_time(num_items: int) -> str:
    """Single-node XGBoost scoring estimate: ~5,000 txns/sec."""
    seconds = num_items / 5000
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hours"


def estimate_spark_time(num_items: int) -> str:
    """Estimated Spark pipeline time based on PRD benchmarks: ~280 txns/sec."""
    seconds = num_items / 280
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hours"


def print_metrics_table(metrics: dict):
    width = 58
    print("\n" + "=" * width)
    print("  PIPELINE METRICS SUMMARY")
    print("=" * width)
    for key, value in metrics.items():
        print(f"  {key:<34} {value}")
    print("=" * width + "\n")


def format_number(n: int) -> str:
    return f"{n:,}"


def estimate_job_cost(wall_time_seconds: float, num_cpu_workers: int = 10) -> str:
    hours = wall_time_seconds / 3600
    cost = num_cpu_workers * 0.768 * hours
    return f"~${cost:.2f} (est. on-demand, m5.4xlarge)"
