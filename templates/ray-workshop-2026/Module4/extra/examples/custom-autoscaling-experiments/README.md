# Ray Serve Custom Autoscaling Examples

Examples demonstrating custom autoscaling policies for Ray Serve deployments.

## Examples

### Deployment-Level Policies

These policies make scaling decisions for a single deployment.

| Example | Use Case |
|---------|----------|
| `prometheus_based/` | Scale based on P90 latency from Prometheus (latency SLA) |
| `recorded_stats_based/` | Scale based on CPU/memory from `record_autoscaling_stats()` |
| `schedule_based/` | Scale based on time of day (predictable traffic) |

### Application-Level Policies

| Example | Use Case |
|---------|----------|
| `application_level/` | Coordinate scaling across multiple deployments in a pipeline |

## Running the Examples

```bash
# Start Ray Serve with a deployment
serve run deployment_level/prometheus_based/main:app

# For Prometheus-based example, ensure Prometheus is running:
export RAY_PROMETHEUS_HOST=http://localhost:9090
```

## Key Concepts

1. **`AutoscalingContext`** - Provides metrics, replica state, and bounds to your policy
2. **`target_num_replicas`** - Use this (not `current_num_replicas`) for scaling decisions
3. **`capacity_adjusted_min/max_replicas`** - Always respect these bounds
4. **`policy_state`** - Persist state across policy invocations (e.g., for delay logic)

## Best Practices

- Use `ctx.target_num_replicas` as baseline (not `current_num_replicas`)
- Always respect `capacity_adjusted_min_replicas` and `capacity_adjusted_max_replicas`
- Use logging (`logging.getLogger("ray.serve")`) instead of `print()`
- Handle missing metrics gracefully by returning current state
