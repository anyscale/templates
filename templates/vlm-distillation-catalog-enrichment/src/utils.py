"""
Timing, metrics, and display utilities for the VLM enrichment pipeline.
"""
import time
from contextlib import contextmanager


@contextmanager
def timer(label: str):
    print(f"\n⏱  {label}...")
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"   ✓ {label} complete — {elapsed:.1f}s")


def calc_throughput(num_items: int, elapsed_seconds: float) -> float:
    return num_items / elapsed_seconds if elapsed_seconds > 0 else 0.0


def format_number(n: int) -> str:
    return f"{n:,}"


def print_metrics_table(metrics: dict, title: str = "PIPELINE METRICS SUMMARY"):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    for key, value in metrics.items():
        print(f"  {key:<32} {value}")
    print("=" * width + "\n")


def estimate_single_node_time(num_items: int, throughput_per_sec: float = 0.5) -> str:
    """
    Rough single-node estimate for Qwen2.5-VL-3B on a single A10G with no
    CPU offload of image fetch/decode. Empirically ~0.5 items/sec end-to-end
    when the GPU also fetches and decodes images.
    """
    seconds = num_items / max(throughput_per_sec, 1e-6)
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hours"


def estimate_job_cost(
    wall_time_seconds: float,
    num_cpu_workers: int = 4,
    num_gpu_workers: int = 2,
    cpu_hourly: float = 0.768,   # m5.4xlarge on-demand us-west-2
    gpu_hourly: float = 1.006,   # g5.xlarge A10G on-demand us-west-2
) -> str:
    hours = wall_time_seconds / 3600
    total = num_cpu_workers * cpu_hourly * hours + num_gpu_workers * gpu_hourly * hours
    return f"~${total:.2f} (est. on-demand, m5.4xlarge CPU + g5.xlarge A10G)"
