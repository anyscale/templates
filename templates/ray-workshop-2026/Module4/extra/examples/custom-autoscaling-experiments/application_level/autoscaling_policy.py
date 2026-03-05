"""
Application-level autoscaling policy example.

This policy coordinates scaling across multiple deployments in an application,
useful when deployments have dependencies (e.g., preprocessor -> model pipeline).
"""

from typing import Dict, Tuple
from ray.serve.config import AutoscalingContext
from ray.serve._private.common import DeploymentID


def coordinated_scaling_policy(
    contexts: Dict[DeploymentID, AutoscalingContext],
) -> Tuple[Dict[DeploymentID, int], Dict]:
    """Scale deployments proportionally based on a preprocessing -> model pipeline."""
    decisions = {}

    # Find deployments by name (safely)
    preprocessing_id = next((d for d in contexts if d.name == "Preprocessor"), None)
    model_id = next((d for d in contexts if d.name == "Model"), None)

    if preprocessing_id is None or model_id is None:
        # Return empty decisions if deployments not found
        return {}, {"error": "Required deployments not found"}

    preprocessing_ctx = contexts[preprocessing_id]
    model_ctx = contexts[model_id]

    # Scale preprocessor: 1 replica per 10 queued requests
    desired = max(1, int(preprocessing_ctx.total_num_requests // 10))
    # Apply user-defined bounds of min and max replicas
    preprocessing_replicas = min(desired, preprocessing_ctx.capacity_adjusted_max_replicas)
    preprocessing_replicas = max(preprocessing_replicas, preprocessing_ctx.capacity_adjusted_min_replicas)
    decisions[preprocessing_id] = preprocessing_replicas

    # Scale model: 2x preprocessor (model takes longer)
    desired = preprocessing_replicas * 2
    # Apply user-defined bounds of min and max replicas
    model_replicas = min(desired, model_ctx.capacity_adjusted_max_replicas)
    model_replicas = max(model_replicas, model_ctx.capacity_adjusted_min_replicas)
    decisions[model_id] = model_replicas

    return decisions, {}
