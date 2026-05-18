# Production Observability for Ray Serve

**⏱️ Time to complete**: 15-20 minutes

Learn how to monitor, debug, and maintain production Ray Serve applications using built-in metrics, custom instrumentation, structured logging, health checks, and distributed tracing.

**Learning Objectives:**
- Understand Ray Serve's built-in observability stack (metrics, logs, dashboard)
- Implement custom metrics for application-specific monitoring
- Configure logging for debugging and auditing
- Implement health checks for reliability
- Set up Grafana dashboards and alerts on Anyscale
- Use distributed tracing for request-level debugging

**Prerequisites:**
- Basic familiarity with Ray Serve deployments
- Access to an Anyscale workspace or local Ray cluster

---

## Setup

Import the required libraries and set up a simple deployment to demonstrate observability features.


```python
import ray
from ray import serve
import requests
import logging
import time
from typing import Dict
ray.init(ignore_reinit_error=True)

```

Define a simple counter deployment that we'll instrument with observability features throughout this template.


```python
@serve.deployment(num_replicas=2)
class CounterDeployment:
    def __init__(self):
        self.count = 0

    def __call__(self, request):
        self.count += 1
        return {"count": self.count, "message": "Request processed"}
```

Deploy the service locally.


```python
app = CounterDeployment.bind()
handle = serve.run(app, name="counter-app")
```

Send a test request to verify the deployment is working.


```python
response = requests.get("http://localhost:8000/")
print(response.json())
```

---

## Built-in Metrics & Dashboard

Ray Serve automatically collects a comprehensive set of metrics for every deployment. These metrics are exported to Prometheus and visible in the Ray Dashboard and Grafana.

**Key built-in metrics:**
- **HTTP Proxy metrics**: Requests per second (QPS), latency histograms (P50, P90, P95, P99), error rates
- **Request routing metrics**: Queue depths, router fulfillment time
- **Request processing metrics**: Replica processing latency, replica utilization
- **Replica lifecycle metrics**: Startup latency, health check failures
- **Autoscaling metrics**: Target replicas, scaling decisions, metric delays

Generate some load to populate metrics.


```python
# Generate 100 requests
for i in range(100):
    response = requests.get("http://localhost:8000/")
    time.sleep(0.01)  # Small delay to spread requests over time

print("Generated 100 requests")
```

Check deployment status programmatically using `serve.status()`.


```python
status = serve.status()
print(f"Application status: {status.applications['counter-app'].status}")
print(f"Deployments: {list(status.applications['counter-app'].deployments.keys())}")

# Access deployment details
deployment_status = status.applications['counter-app'].deployments['CounterDeployment']
print(f"Deployment status: {deployment_status.status}")
print(f"Replicas: {len(deployment_status.replica_states)}")
```

### Viewing Metrics in the Ray Dashboard

Access the Ray Dashboard at `http://localhost:8265` (or your Anyscale workspace URL). Navigate to the **Serve** page to view:

- **Application-level metrics**: Throughput and latency per application
- **Deployment-level metrics**: Latency, replica count, queue size per deployment
- **Request timeline**: Request flow through proxy → router → replica

For Anyscale workspaces, the dashboard is accessible via the **Monitoring** tab in your workspace UI.

---

## Custom Metrics

While built-in metrics provide infrastructure-level insights, custom metrics let you track application-specific behavior like model performance, cache hits, or business logic outcomes.

Ray Serve provides three metric types via `ray.serve.metrics`:
- **Counter**: Monotonically increasing count (e.g., total requests, cache hits)
- **Gauge**: Point-in-time value (e.g., current memory usage, queue depth)
- **Histogram**: Distribution of values (e.g., inference time, batch size)

Update the deployment to include a custom counter with tags.


```python
from ray.serve.metrics import Counter

@serve.deployment(num_replicas=2)
class CounterWithMetrics:
    def __init__(self):
        self.count = 0
        # Create a custom counter with a tag for tracking different request types
        self.request_counter = Counter(
            "custom_requests_total",
            description="Total number of requests processed by this deployment",
            tag_keys=("request_type",)
        )
        # Set default tags that apply to all metric increments
        self.request_counter.set_default_tags({"request_type": "standard"})

    def __call__(self, request):
        self.count += 1
        # Increment the custom counter
        self.request_counter.inc()
        return {"count": self.count, "message": "Request processed with custom metrics"}
```

Deploy the updated deployment with custom metrics.


```python
app_with_metrics = CounterWithMetrics.bind()
handle_with_metrics = serve.run(app_with_metrics, name="counter-metrics-app", route_prefix="/metrics")
```

Generate traffic to increment the custom metric.


```python
# Generate 50 requests to the new deployment
for i in range(50):
    response = requests.get("http://localhost:8000/metrics")
    time.sleep(0.01)

print("Generated 50 requests with custom metrics")
```

### Viewing Custom Metrics

Custom metrics are exported alongside built-in metrics. In Grafana or Prometheus, query for `custom_requests_total` to see your metric. The metric will include automatic tags for `deployment`, `replica`, and `application`, plus your custom `request_type` tag.

Example Prometheus query:
```
rate(custom_requests_total[1m])
```

This shows the per-second rate of custom requests over a 1-minute window.

---

## Logging Configuration

Ray Serve uses Python's standard logging framework. By default, logs are written at the INFO level in plain text format. For production environments, you can configure structured logging (JSON), adjust log levels, and control access logs.

**LoggingConfig parameters:**
- `log_level`: Log verbosity (DEBUG, INFO, WARNING, ERROR)
- `encoding`: Log format (TEXT or JSON for structured logging)
- `enable_access_log`: Whether to log HTTP access events
- `logs_dir`: Custom log directory (optional)

Define a deployment with debug-level JSON logging.


```python
from ray.serve.schema import LoggingConfig

logging_config = LoggingConfig(
    log_level="DEBUG",
    encoding="JSON",
    enable_access_log=True
)

@serve.deployment(num_replicas=1, logging_config=logging_config)
class LoggingDeployment:
    def __init__(self):
        # Get the ray.serve logger
        self.logger = logging.getLogger("ray.serve")
        self.logger.info("LoggingDeployment initialized")

    def __call__(self, request):
        self.logger.debug("Processing request")
        self.logger.info("Request handled successfully")
        return {"status": "ok", "message": "Check logs for structured output"}
```

Deploy with the logging configuration.


```python
app_with_logging = LoggingDeployment.bind()
handle_with_logging = serve.run(app_with_logging, name="logging-app", route_prefix="/logging")
```

Generate requests to trigger log output.


```python
# Send requests to generate logs
for i in range(5):
    response = requests.get("http://localhost:8000/logging")
    print(response.json())
```

### Viewing Logs

**Local development:**
Logs are written to `/tmp/ray/session_latest/logs/serve/`. Check the replica logs for structured JSON output:

```bash
# View logs in local Ray session
tail -f /tmp/ray/session_latest/logs/serve/*.log
```

**Anyscale workspaces:**
Navigate to the **Logs** tab in your workspace to view deployment logs. Use the deployment name filter to focus on specific deployments.

### Log Aggregation with Loki

For production deployments, Anyscale provides Grafana Loki integration for centralized log aggregation and filtering. Use [LogQL](https://grafana.com/docs/loki/latest/logql/) to query logs across deployments:

```
{deployment="LoggingDeployment"} |= "Request handled"
```

---

## Health Checks

Health checks ensure that replicas are functioning correctly. If a replica's health check fails 3 consecutive times, Ray Serve automatically replaces it with a new replica.

Implement the `check_health()` method in your deployment to define custom health check logic. This method should raise an exception if the replica is unhealthy.

Define a deployment with a health check that can be toggled for demonstration.


```python
@serve.deployment(
    num_replicas=1,
    health_check_period_s=5,  # Check health every 5 seconds
    health_check_timeout_s=2  # Health check must complete within 2 seconds
)
class HealthCheckDeployment:
    def __init__(self):
        self.healthy = True
        self.logger = logging.getLogger("ray.serve")
        self.logger.info("HealthCheckDeployment initialized")

    def check_health(self):
        """Custom health check method.

        Raises an exception if unhealthy, returns normally if healthy.
        """
        if not self.healthy:
            self.logger.error("Health check failed!")
            raise RuntimeError("Replica is unhealthy")
        self.logger.debug("Health check passed")

    def __call__(self, request):
        # Allow toggling health status via query parameter
        if request.query_params.get("fail_health") == "true":
            self.healthy = False
            return {"status": "Health will fail on next check"}
        return {"status": "healthy", "message": "Replica is functioning normally"}
```

Deploy the health-checked deployment.


```python
app_with_health = HealthCheckDeployment.bind()
handle_with_health = serve.run(app_with_health, name="health-app", route_prefix="/health")
```

Test the healthy state.


```python
response = requests.get("http://localhost:8000/health")
print(response.json())
```

Simulate a health check failure.


```python
# Trigger health check failure
response = requests.get("http://localhost:8000/health?fail_health=true")
print(response.json())

print("Health check will fail within 15 seconds (3 consecutive failures at 5-second intervals)")
print("Ray Serve will automatically replace the unhealthy replica")
```

### Observing Replica Replacement

After triggering the health check failure, monitor the deployment status:


```python
import time

# Wait for health check to fail and replica to be replaced
for i in range(6):
    time.sleep(5)
    status = serve.status()
    deployment_status = status.applications['health-app'].deployments['HealthCheckDeployment']
    print(f"[{i*5}s] Status: {deployment_status.status}, Replicas: {len(deployment_status.replica_states)}")
```

You'll see the replica transition from RUNNING → UNHEALTHY → replaced with a new RUNNING replica.

**Common health check patterns:**
- Database connection checks
- Model loading verification
- Dependency service availability
- Memory or resource thresholds

---

## Grafana Integration

Anyscale provides built-in Grafana dashboards for visualizing Ray Serve metrics. These dashboards aggregate metrics across all deployments and applications, making it easy to monitor production performance.

### Pre-built Dashboards

Navigate to the **Monitoring** tab in your Anyscale workspace to access Grafana dashboards:

**Application-level dashboards:**
- Throughput per application (requests/second)
- Latency percentiles (P50, P90, P95, P99) per application
- Error rates and HTTP status codes

**Deployment-level dashboards:**
- Latency per deployment
- Replica count per deployment
- Queue size per deployment
- Autoscaling activity

**Custom metrics dashboards:**
Your custom metrics (like `custom_requests_total`) appear alongside built-in metrics and can be added to custom panels.

### Setting Up Alerts

Grafana supports metric-based alerting with notification channels (email, Slack, PagerDuty). Common alert rules for Ray Serve:

- High latency: Alert when P99 latency exceeds threshold for 5 minutes
- Low replica count: Alert when replicas drop below minimum
- High queue depth: Alert when request queue grows beyond capacity
- Error rate spike: Alert when 5xx errors exceed threshold

For alert setup, see the [Grafana alerting documentation](https://grafana.com/docs/grafana/v7.5/alerting/).

---

## Distributed Tracing

For debugging complex request flows across multiple deployments, Anyscale provides distributed tracing via the Anyscale tracing service. Tracing captures the full request path: HTTP proxy → ingress → downstream deployments, with timing breakdowns for each hop.

**Use cases for tracing:**
- Identifying performance bottlenecks in multi-deployment applications
- Debugging request failures across service boundaries
- Understanding request routing and queuing behavior

### Enabling Tracing

Tracing is enabled at the workspace level in Anyscale. Once enabled, traces appear in the tracing UI for every HTTP request.

**Trace components:**
- Request ID (auto-generated or custom via `X-Request-ID` header)
- Span timeline (proxy, router, replica processing)
- Deployment hops for composed applications
- Queue wait times and execution times

For detailed setup and usage, see the [Anyscale tracing guide](https://docs.anyscale.com/monitoring/tracing/).

---

## Summary & Next Steps

You've learned how to instrument Ray Serve applications with production-grade observability:

- **Built-in metrics**: QPS, latency, queue depth, replica count — automatically collected and exported to Prometheus
- **Custom metrics**: Track application-specific behavior with Counter, Gauge, and Histogram
- **Structured logging**: Configure log levels, JSON encoding, and access logs for debugging
- **Health checks**: Implement custom health logic to ensure replica reliability
- **Grafana dashboards**: Visualize metrics and set up alerts for production monitoring
- **Distributed tracing**: Debug request flows across multi-deployment applications

### Next Steps

- **Production deployment**: Deploy your instrumented application as an [Anyscale Service](https://docs.anyscale.com/services/) for production workloads
- **Alerting**: Set up Grafana alerts for critical metrics (latency, error rate, replica health)
- **Log aggregation**: Configure [Loki](https://grafana.com/docs/loki/latest/logql/) for centralized log filtering and search
- **Advanced metrics**: Explore histogram boundaries, custom tag strategies, and metric cardinality management
- **Monitoring best practices**: See the [Ray observability guide](https://docs.ray.io/en/latest/ray-observability/getting-started.html) for comprehensive monitoring patterns
