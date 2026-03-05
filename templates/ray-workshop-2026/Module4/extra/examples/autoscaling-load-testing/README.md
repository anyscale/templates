# Autoscaling Benchmark Example

This directory contains a practical example for benchmarking and tuning Ray Serve autoscaling using a ResNet50 image classification model.

## Directory Contents

| File | Purpose |
|------|---------|
| `resnet50_model.py` | Ray Serve deployment with ResNet50 model |
| `benchmark.yaml` | Configuration for single-replica benchmarking |
| `locustfile.py` | Load test definition with traffic patterns |
| `run_locust.sh` | Script to execute load tests |
| `scripts/deploy.sh` | Deployment helper script |
| `scripts/visualize_metrics.py` | Generate plots from load test results |

## How to Use This Example

This example supports the autoscaling tuning workflow described in the notebook. The key steps are:

### Step 1: Benchmark a Single Replica

Before configuring autoscaling, you need to understand how much load a single replica can handle. The `benchmark.yaml` configuration locks the deployment to one replica:

```yaml
autoscaling_config:
  min_replicas: 1
  max_replicas: 1
```

Deploy with:

```bash
serve run benchmark.yaml
```

### Step 2: Run Load Tests

The `locustfile.py` defines a traffic pattern that gradually increases load:

```python
stages = [
    {"cumulative_duration": 60, "users": 1, "spawn_rate": 1},      # Warm-up
    {"cumulative_duration": 180, "users": 20, "spawn_rate": 1/30}, # Ramp up
    {"cumulative_duration": 360, "users": 40, "spawn_rate": 1/30}, # Peak load
]
```

Run the load test:

```bash
bash run_locust.sh
```

Results are saved to `results.html`.

### Step 3: Observe Metrics

While the load test runs, monitor the Ray Dashboard:

```
http://127.0.0.1:8265
```

Navigate to Serve > Your Application > Metrics to observe:
- Ongoing requests per replica
- Request latency (P50, P90, P99)
- Queue depth

### Step 4: Determine Optimal Settings

From your benchmarks, identify:
- The number of ongoing requests where latency starts to degrade
- Use ~80% of this value as `target_ongoing_requests`
- Set `max_ongoing_requests` to 1.2-1.5x the target

Example: If latency degrades at 6 ongoing requests, configure:

```yaml
autoscaling_config:
  target_ongoing_requests: 4
  max_ongoing_requests: 6
```

## The Model

`resnet50_model.py` implements a simple image classification service:

```python
@serve.deployment
class Model:
    def __init__(self):
        self.resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # ... preprocessing and category loading
    
    async def __call__(self, request):
        uri = (await request.json())["uri"]
        # ... download, preprocess, inference
        return self.categories[predicted_index]
```

Test with:

```bash
curl -X POST http://127.0.0.1:8000 \
  -H "Content-Type: application/json" \
  -d '{"uri": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"}'
```

## Visualizing Results

Generate plots from Locust CSV output:

```bash
python scripts/visualize_metrics.py \
  --stats-history results_stats_history.csv \
  --output-dir plots/
```

This creates:
- `users_over_time.png` - Active users during the test
- `rps_over_time.png` - Requests per second
- `latency_over_time.png` - P50/P90/P99 latencies
- `dashboard.png` - Combined summary

## Key Autoscaling Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `target_ongoing_requests` | 2 | Desired requests per replica |
| `max_ongoing_requests` | 5 | Buffer before backpressure |
| `upscale_delay_s` | 30 | Seconds before adding replicas |
| `downscale_delay_s` | 600 | Seconds before removing replicas |
| `upscaling_factor` | 1.0 | Multiplier for scale-up decisions |
| `downscaling_factor` | 1.0 | Multiplier for scale-down decisions |

## Common Tuning Adjustments

**Scale up faster** (for bursty traffic):
```yaml
upscale_delay_s: 15
upscaling_factor: 1.5
```

**Scale down slower** (for slow-starting services):
```yaml
downscale_delay_s: 900
downscaling_factor: 0.8
```

**Handle more requests per replica** (higher utilization):
```yaml
target_ongoing_requests: 4
max_ongoing_requests: 6
```

## Prerequisites

```bash
pip install "ray[serve]" torch torchvision pillow locust matplotlib pandas
```
