"""
Example deployment using custom metrics (CPU/memory) for autoscaling.

This deployment reports resource usage via `record_autoscaling_stats()`,
which the custom policy uses to make scaling decisions.
"""

import asyncio
from typing import Dict

import psutil
from ray import serve
from ray.serve.config import AutoscalingConfig, AutoscalingPolicy


@serve.deployment(
    autoscaling_config=AutoscalingConfig(
        min_replicas=1,
        max_replicas=10,
        upscale_delay_s=30,
        downscale_delay_s=60,
        policy=AutoscalingPolicy(
            policy_function="autoscaling_policy:custom_metrics_autoscaling_policy"
        ),
    ),
    max_ongoing_requests=5,
)
class CustomMetricsDeployment:
    def __init__(self):
        self.process = psutil.Process()

    async def __call__(self) -> str:
        await asyncio.sleep(0.5)  # Simulate processing
        return "Hello, world!"

    def record_autoscaling_stats(self) -> Dict[str, float]:
        """Report custom metrics to the autoscaler.
        
        This method is called periodically by Ray Serve. The returned metrics
        are available in the policy via ctx.aggregated_metrics.
        """
        try:
            return {
                "cpu_usage": self.process.cpu_percent(interval=0.1),
                "memory_usage_mb": self.process.memory_info().rss / (1024 * 1024),
            }
        except Exception:
            return {"cpu_usage": 0.0, "memory_usage_mb": 0.0}


app = CustomMetricsDeployment.bind()
