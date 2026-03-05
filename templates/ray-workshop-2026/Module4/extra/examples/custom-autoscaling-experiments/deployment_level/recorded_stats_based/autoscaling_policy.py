"""
Custom metrics autoscaling policy example.

This policy scales based on CPU and memory usage reported by replicas via
the `record_autoscaling_stats()` method, useful for resource-aware scaling.
"""

import logging
from typing import Any, Dict

from ray.serve.config import AutoscalingContext

logger = logging.getLogger("ray.serve")

# Resource thresholds (customize based on your workload)
CPU_HIGH = 75.0  # Scale up when any replica exceeds this
CPU_LOW = 25.0   # Scale down when all replicas below this
MEMORY_HIGH_MB = 500.0
MEMORY_LOW_MB = 200.0


def custom_metrics_autoscaling_policy(
    ctx: AutoscalingContext,
) -> tuple[int, Dict[str, Any]]:
    """Scale based on CPU and memory metrics from replicas."""
    cpu_metrics = ctx.aggregated_metrics.get("cpu_usage", {})
    memory_metrics = ctx.aggregated_metrics.get("memory_usage_mb", {})

    if not cpu_metrics or not memory_metrics:
        return ctx.target_num_replicas, {"reason": "No metrics available"}

    cpu_values = list(cpu_metrics.values())
    memory_values = list(memory_metrics.values())

    max_cpu = max(cpu_values)
    max_memory = max(memory_values)
    avg_cpu = sum(cpu_values) / len(cpu_values)
    avg_memory = sum(memory_values) / len(memory_values)

    # Scale up if any replica is overloaded
    if max_cpu > CPU_HIGH or max_memory > MEMORY_HIGH_MB:
        new_replicas = min(
            ctx.capacity_adjusted_max_replicas,
            ctx.target_num_replicas + 1,
        )
        return new_replicas, {
            "reason": f"High usage: CPU={max_cpu:.0f}%, Mem={max_memory:.0f}MB"
        }

    # Scale down if all replicas are underutilized
    if avg_cpu < CPU_LOW and avg_memory < MEMORY_LOW_MB:
        new_replicas = max(
            ctx.capacity_adjusted_min_replicas,
            ctx.target_num_replicas - 1,
        )
        return new_replicas, {
            "reason": f"Low usage: CPU={avg_cpu:.0f}%, Mem={avg_memory:.0f}MB"
        }

    return ctx.target_num_replicas, {
        "reason": f"Stable: CPU={avg_cpu:.0f}%, Mem={avg_memory:.0f}MB"
    }
