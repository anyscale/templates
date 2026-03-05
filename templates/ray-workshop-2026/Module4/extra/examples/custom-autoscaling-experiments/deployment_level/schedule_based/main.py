"""
Example deployment using schedule-based autoscaling.

This deployment scales based on time of day for predictable traffic patterns.
"""

import asyncio

from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy


@serve.deployment(
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=12,
        upscale_delay_s=30,
        downscale_delay_s=60,
        policy=AutoscalingPolicy(
            policy_function="autoscaling_policy:scheduled_autoscaling_policy"
        ),
    ),
    max_ongoing_requests=5,
)
class ScheduledDeployment:
    async def __call__(self) -> str:
        await asyncio.sleep(0.5)  # Simulate processing
        return "Hello, world!"


app = ScheduledDeployment.bind()
