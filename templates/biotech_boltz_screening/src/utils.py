"""
Timing, metrics, and display utilities for the Boltz-1 screening pipeline demo.
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
    """Return items per second."""
    return num_items / elapsed_seconds if elapsed_seconds > 0 else 0.0


def estimate_single_node_time(num_complexes: int) -> str:
    """
    Single-GPU Boltz-1 estimate: ~30 seconds per complex on one A100.
    This is the baseline researchers experience without Ray Data.
    """
    seconds = num_complexes * 30
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} hours"


def print_metrics_table(metrics: dict):
    """Print a formatted metrics summary table."""
    width = 52
    print("\n" + "═" * width)
    print("  PIPELINE METRICS SUMMARY")
    print("═" * width)
    for key, value in metrics.items():
        print(f"  {key:<30} {value}")
    print("═" * width + "\n")


def format_number(n: int) -> str:
    return f"{n:,}"


def estimate_job_cost(
    wall_time_seconds: float,
    num_cpu_workers: int = 2,
    num_gpu_workers: int = 4,
) -> str:
    """
    Rough Anyscale cost estimate based on on-demand EC2 pricing.
    m5.2xlarge ~$0.384/hr, g5.2xlarge (A10G) ~$1.212/hr
    """
    hours = wall_time_seconds / 3600
    cpu_cost = num_cpu_workers * 0.384 * hours
    gpu_cost = num_gpu_workers * 1.212 * hours
    total = cpu_cost + gpu_cost
    return f"~${total:.2f} (est. on-demand, g5.2xlarge A10G)"
