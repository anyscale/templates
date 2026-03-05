"""
Prometheus-based autoscaling policy example.

This policy scales based on P90 latency metrics fetched from Prometheus,
useful when you want to maintain latency SLAs rather than targeting request counts.

NOTE: This implementation manually applies delay logic (upscale_delay_s, downscale_delay_s).
Once https://github.com/ray-project/ray/pull/58857 is merged, you can use the
`@apply_autoscaling_config` decorator to get this behavior automatically.
"""

import logging
import os
from typing import Any, Dict

import ray
import requests
from ray.serve.config import AutoscalingContext
from ray.serve._private.constants import CONTROL_LOOP_INTERVAL_S

logger = logging.getLogger("ray.serve")

PROMETHEUS_HOST = os.environ.get("RAY_PROMETHEUS_HOST", "http://localhost:9090")

# Latency thresholds (customize based on your SLA)
SCALE_UP_THRESHOLD_MS = 500
SCALE_DOWN_THRESHOLD_MS = 50


def get_p90_latency(app_name: str, deployment_name: str) -> float | None:
    """Fetch P90 latency from Prometheus."""
    session_name = ray._private.worker.global_worker.node.session_name
    query = (
        "histogram_quantile(0.9, "
        "sum(rate(ray_serve_deployment_processing_latency_ms_bucket{"
        f'application="{app_name}",'
        f'deployment="{deployment_name}",'
        f'SessionName="{session_name}"'
        "}[5m])) by (application, deployment, le))"
    )
    try:
        response = requests.get(
            f"{PROMETHEUS_HOST}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        if result["status"] == "success" and result["data"]["result"]:
            return float(result["data"]["result"][0]["value"][1])
    except Exception as e:
        logger.warning(f"Failed to query Prometheus: {e}")
    return None


def p90_latency_autoscaling_policy(
    ctx: AutoscalingContext,
) -> tuple[int, Dict[str, Any]]:
    """Scale based on P90 latency from Prometheus."""
    policy_state = ctx.policy_state or {}
    decision_counter = policy_state.get("decision_counter", 0)

    p90_latency_ms = get_p90_latency(ctx.app_name, ctx.deployment_name)

    if p90_latency_ms is None:
        return ctx.target_num_replicas, policy_state

    # Determine scaling direction based on latency
    if p90_latency_ms > SCALE_UP_THRESHOLD_MS:
        # Want to scale up
        desired = min(ctx.capacity_adjusted_max_replicas, ctx.target_num_replicas + 1)
        if desired > ctx.target_num_replicas:
            decision_counter = max(1, decision_counter + 1)
        else:
            decision_counter = 0

    elif p90_latency_ms < SCALE_DOWN_THRESHOLD_MS:
        # Want to scale down
        desired = max(ctx.capacity_adjusted_min_replicas, ctx.target_num_replicas - 1)
        if desired < ctx.target_num_replicas:
            decision_counter = min(-1, decision_counter - 1)
        else:
            decision_counter = 0
    else:
        # Latency acceptable - reset counter
        decision_counter = 0
        desired = ctx.target_num_replicas

    # Apply delay logic: only scale after sustained signal
    final_replicas = ctx.target_num_replicas
    upscale_threshold = int(ctx.config.upscale_delay_s / CONTROL_LOOP_INTERVAL_S)
    downscale_threshold = int(ctx.config.downscale_delay_s / CONTROL_LOOP_INTERVAL_S)

    if decision_counter > upscale_threshold:
        final_replicas = desired
        decision_counter = 0
    elif decision_counter < -downscale_threshold:
        final_replicas = desired
        decision_counter = 0

    policy_state["decision_counter"] = decision_counter
    return final_replicas, policy_state
