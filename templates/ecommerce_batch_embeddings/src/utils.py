"""
Timing, metrics, and display utilities for the embedding pipeline demo.
"""
import time
from contextlib import contextmanager
from typing import Optional


@contextmanager
def timer(label: str):
    """Context manager that prints elapsed time for a labeled block."""
    print(f"\n⏱  {label}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"   ✓ {label} complete — {elapsed:.1f}s")


def calc_throughput(num_items: int, elapsed_seconds: float) -> float:
    """Return items per second."""
    return num_items / elapsed_seconds if elapsed_seconds > 0 else 0.0


def estimate_single_node_time(num_items: int, throughput_per_sec: float) -> str:
    """
    Rough estimate of single-node CPU-only time.
    all-MiniLM-L6-v2 on CPU encodes ~150 sentences/sec on a modern 8-core machine.
    """
    CPU_THROUGHPUT = 150
    seconds = num_items / CPU_THROUGHPUT
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hours"


def print_metrics_table(metrics: dict):
    """Print a formatted metrics summary table."""
    width = 52
    print("\n" + "=" * width)
    print("  PIPELINE METRICS SUMMARY")
    print("=" * width)
    for key, value in metrics.items():
        print(f"  {key:<30} {value}")
    print("=" * width + "\n")


def format_number(n: int) -> str:
    return f"{n:,}"


def estimate_job_cost(wall_time_seconds: float, num_cpu_workers: int = 4, num_gpu_workers: int = 2) -> str:
    """
    Rough Anyscale cost estimate based on on-demand EC2 pricing.
    m5.4xlarge ~$0.768/hr, g4dn.xlarge (T4) ~$0.526/hr
    """
    hours = wall_time_seconds / 3600
    cpu_cost = num_cpu_workers * 0.768 * hours
    gpu_cost = num_gpu_workers * 0.526 * hours
    total = cpu_cost + gpu_cost
    return f"~${total:.2f} (est. on-demand, g4dn.xlarge T4)"
