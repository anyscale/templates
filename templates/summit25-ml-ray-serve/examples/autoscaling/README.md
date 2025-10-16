# Ray Serve Autoscaling: Hands-On Tutorial

This directory contains hands-on examples for understanding and tuning autoscaling in Ray Serve, using a ResNet50 image classification model.

## Overview

These examples demonstrate:
- Basic autoscaling configuration
- Aggressive upscaling for burst traffic
- Conservative downscaling for slow-initialization services
- Production-ready balanced configuration
- Load testing with Locust
- Metrics visualization

## Prerequisites

Install required dependencies:

```bash
pip install "ray[serve]" torch torchvision pillow locust matplotlib pandas
```

## Quick Start

### 1. Deploy the Service

Choose one of the autoscaling configurations:

```bash
# Basic configuration (default delays and factors)
bash scripts/deploy.sh configs/basic.yaml

# Aggressive upscaling (fast response to traffic spikes)
bash scripts/deploy.sh configs/aggressive_upscale.yaml

# Conservative downscaling (avoid replica churn)
bash scripts/deploy.sh configs/conservative_downscale.yaml

# Production configuration (balanced settings)
bash scripts/deploy.sh configs/production.yaml
```

The service will be available at `http://127.0.0.1:8000`.

### 2. Test the Deployment

Send a test request:

```bash
curl -X POST http://127.0.0.1:8000 \
  -H "Content-Type: application/json" \
  -d '{"uri": "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"}'
```

Expected response: `Samoyed`

### 3. Run Load Test

Execute the load test to observe autoscaling behavior:

```bash
bash scripts/run_load_test.sh
```

This runs a ~11 minute load test with varying traffic:
- 0-1 min: 1 user (warm-up)
- 1-3 min: 20 users (scale up)
- 3-6 min: 40 users (peak load)
- 6-9 min: 20 users (scale down)
- 9-11 min: 1 user (cool down)

Results are saved to `results/` directory.

### 4. Monitor Metrics

While the load test runs, monitor Ray Serve:

```bash
# Ray Dashboard
open http://127.0.0.1:8265

# View Serve metrics
# Navigate to: Dashboard → Serve → [Your Application] → Metrics
```

### 5. Visualize Results

Generate plots from load test results:

```bash
python scripts/visualize_metrics.py \
  --stats-history results/load_test_*_stats_history.csv \
  --output-dir plots/
```

This generates:
- `users_over_time.png` - Active users during test
- `rps_over_time.png` - Requests per second
- `latency_over_time.png` - P50/P90/P99 latencies
- `dashboard.png` - Combined summary view

## Autoscaling Configurations Explained

### Basic Configuration (`configs/basic.yaml`)

```yaml
autoscaling_config:
  target_ongoing_requests: 2
  min_replicas: 1
  max_replicas: 100
```

**Use case:** Default starting point, moderate scaling behavior.

**Characteristics:**
- Maintains ~2 ongoing requests per replica
- Default delays (upscale: 30s, downscale: 600s)
- Standard scaling factors (1.0)

### Aggressive Upscale (`configs/aggressive_upscale.yaml`)

```yaml
autoscaling_config:
  upscale_delay_s: 15
  metrics_interval_s: 5
  upscaling_factor: 1.5
```

**Use case:** Services that experience sudden traffic bursts.

**Characteristics:**
- Scales up quickly (15s delay vs 30s default)
- More frequent metrics collection (5s vs 10s)
- Amplified upscaling (1.5x factor)

**Trade-offs:**
- ✅ Faster response to traffic spikes
- ✅ Reduced latency during scale-up
- ⚠️ May over-provision if traffic is spiky
- ⚠️ Higher costs during transient spikes

### Conservative Downscale (`configs/conservative_downscale.yaml`)

```yaml
autoscaling_config:
  downscale_delay_s: 900
  downscaling_factor: 0.8
```

**Use case:** Services with slow initialization or unpredictable traffic.

**Characteristics:**
- Longer wait before scaling down (15 min vs 10 min)
- Gentler downscaling (0.8x factor)

**Trade-offs:**
- ✅ Avoids frequent replica churn
- ✅ Better for slow-loading models
- ✅ Handles traffic fluctuations better
- ⚠️ Higher idle costs during low traffic

### Production Configuration (`configs/production.yaml`)

```yaml
autoscaling_config:
  min_replicas: 2
  max_replicas: 50
  upscale_delay_s: 20
  downscale_delay_s: 600
  upscaling_factor: 1.2
  downscaling_factor: 0.9
```

**Use case:** Balanced production deployment.

**Characteristics:**
- Always maintains 2 replicas (high availability)
- Reasonable max limit (cost control)
- Moderate delays and factors
- Slightly more aggressive upscaling than downscaling

## Tuning Guide

### Step 1: Benchmark Single Replica

Lock to a single replica to understand baseline performance:

```yaml
autoscaling_config:
  min_replicas: 1
  max_replicas: 1
```

Measure:
- Maximum sustainable QPS per replica
- P50, P90, P99 latencies
- Average processing time
- Queue depth at various loads

### Step 2: Set Initial Parameters

Based on benchmarks, configure:

```python
# If single replica handles 10 RPS with ~100ms latency
# and you want ~2 requests per replica:
target_ongoing_requests = 2
max_ongoing_requests = 3  # 20-50% higher than target
```

### Step 3: Load Test and Observe

Run load tests with realistic traffic patterns:
- Gradual ramp-up
- Sustained load
- Traffic spikes
- Ramp-down

Monitor:
- Scale-up lag time
- Queue sizes during transitions
- Latency during scaling events
- Request failures or timeouts

### Step 4: Tune Based on Observations

#### If scale-up is too slow:
```yaml
upscale_delay_s: 15        # ↓ from 30
metrics_interval_s: 5      # ↓ from 10
upscaling_factor: 1.5      # ↑ from 1.0
```

#### If replicas scale down too aggressively:
```yaml
downscale_delay_s: 900     # ↑ from 600
downscaling_factor: 0.8    # ↓ from 1.0
```

#### If latency spikes during scaling:
```yaml
max_ongoing_requests: 8    # ↑ increase buffer
upscale_delay_s: 15        # ↓ scale faster
```

#### If cost is too high:
```yaml
downscale_delay_s: 300     # ↓ scale down sooner
downscaling_factor: 1.2    # ↑ scale down faster
max_replicas: 20           # ↓ limit max scale
```

### Step 5: Production Validation

1. **Gradual rollout:** Start with subset of traffic
2. **Monitor closely:** Watch metrics for first 24-48 hours
3. **Review regularly:** Daily check for anomalies
4. **Document:** Record optimal settings and patterns

## Key Metrics to Monitor

### During Load Testing

- **User count:** Matches expected traffic pattern
- **QPS:** Increases with users, plateaus near capacity
- **Replica count:** Should track with load (divided by target_ongoing_requests)
- **P50 latency:** Should remain relatively stable
- **P99 latency:** Watch for spikes during scale-up

### In Production

- **Queue depth:** Should stay low (< max_ongoing_requests)
- **Scaling events:** Frequency and magnitude
- **Error rate:** Should not increase during scaling
- **Cost metrics:** Replica-hours vs traffic served
- **SLA compliance:** Latency percentiles within bounds

## Common Issues and Solutions

### Issue: Replicas oscillate up and down

**Cause:** Delays are too short or factors too aggressive.

**Solution:**
```yaml
upscale_delay_s: 45        # Increase both delays
downscale_delay_s: 900
look_back_period_s: 45     # Match or exceed upscale_delay_s
```

### Issue: High latency during traffic spikes

**Cause:** Scale-up is too slow or max_ongoing_requests too low.

**Solution:**
```yaml
upscale_delay_s: 15
upscaling_factor: 1.5
max_ongoing_requests: 8    # Increase buffer
```

### Issue: Too many idle replicas during low traffic

**Cause:** Downscaling is too conservative.

**Solution:**
```yaml
downscale_delay_s: 300     # Reduce delay
downscaling_factor: 1.2    # More aggressive
```

### Issue: Requests timing out during scale-up

**Cause:** Not enough request buffering capacity.

**Solution:**
```yaml
max_ongoing_requests: 10   # Increase significantly
target_ongoing_requests: 5 # Maintain ratio (2:1)
```

## Advanced Patterns

### Cost-Optimized Configuration

Prioritize cost over latency (within SLA):

```yaml
autoscaling_config:
  target_ongoing_requests: 5      # Higher utilization
  upscale_delay_s: 45            # Wait longer
  downscale_delay_s: 300         # Scale down faster
  upscaling_factor: 1.0
  downscaling_factor: 1.2
```

### Latency-Optimized Configuration

Prioritize latency over cost:

```yaml
autoscaling_config:
  target_ongoing_requests: 1      # Lower utilization
  min_replicas: 5                # Always warm
  upscale_delay_s: 10            # Very fast upscale
  downscale_delay_s: 1200        # Rarely downscale
  upscaling_factor: 2.0
  downscaling_factor: 0.5
```

### Predictable Traffic Pattern

For services with known daily/weekly patterns:

```yaml
autoscaling_config:
  min_replicas: 2                # Handle baseline
  max_replicas: 20               # Cap peak
  upscale_delay_s: 30
  downscale_delay_s: 1800        # Long delay (30 min)
```

Consider using scheduled scaling or time-based min_replicas adjustments.

## Resources

- [Ray Serve Autoscaling Documentation](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html)
- [Ray Dashboard](http://127.0.0.1:8265)
- [Locust Documentation](https://docs.locust.io/)

## Troubleshooting

### Ray Serve won't start

```bash
# Check if Ray is running
ray status

# If not, start Ray
ray start --head

# Then deploy again
bash scripts/deploy.sh configs/basic.yaml
```

### Load test fails immediately

```bash
# Check service is responding
curl http://127.0.0.1:8000/healthz

# Check Ray Serve status
serve status
```

### Visualization fails

```bash
# Ensure pandas and matplotlib are installed
pip install pandas matplotlib

# Check CSV file exists
ls -lh results/
```

### Out of memory errors

Reduce `max_replicas` in configuration:

```yaml
autoscaling_config:
  max_replicas: 10  # Lower limit
```

Or increase Ray cluster resources:

```bash
ray start --head --num-cpus=16
```

