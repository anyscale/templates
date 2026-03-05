"""
Schedule-based autoscaling policy example.

This policy scales based on time of day, useful for predictable traffic patterns
(e.g., more replicas during business hours, fewer at night).
"""

from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

from ray.serve.config import AutoscalingContext


def scheduled_autoscaling_policy(
    ctx: AutoscalingContext,
) -> tuple[int, Dict[str, Any]]:
    """Scale based on time of day."""
    current_hour = datetime.now(ZoneInfo("America/Los_Angeles")).hour

    # Define replica targets by hour (customize for your traffic pattern)
    if 9 <= current_hour < 17:  # Business hours: 9am-5pm
        desired = 8
    elif 7 <= current_hour < 9 or 17 <= current_hour < 20:  # Shoulder hours
        desired = 4
    else:  # Off hours
        desired = 1

    # Respect configured bounds
    final = max(
        ctx.capacity_adjusted_min_replicas,
        min(ctx.capacity_adjusted_max_replicas, desired),
    )

    return final, {"reason": f"Hour {current_hour}: target={desired}"}
