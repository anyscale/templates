"""
Example deployment using Prometheus-based autoscaling.

This deployment scales based on P90 latency metrics rather than request counts.
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
            policy_function="autoscaling_policy:p90_latency_autoscaling_policy"
        ),
    ),
    max_ongoing_requests=5,
)
class BatchProcessingDeployment:
    async def __call__(self) -> str:
        await asyncio.sleep(0.5)  # Simulate processing
        return "Hello, world!"


app = BatchProcessingDeployment.bind()
