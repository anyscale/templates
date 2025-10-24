# Serving ML Models With Ray Serve: Best Practices and Scalable Production Patterns

## 1. Foundations
- What is Ray Serve?  
- Why use Ray Serve for model serving?  
- Comparison with Kubernetes-native serving (K8s, Knative, KFServing, etc.)  
- When to choose Ray Serve  

## 2. Architecture
- Key components of Ray Serve
- How Ray Serve utilitizes Ray Core
- Lifetime of a request
- Request routing

## 3. Designing Ray Serve Applications
- Structuring Ray Serve code  
- Recommended Python package layout  
- Deployment config files & patterns  
- Testing and development workflow  
  - Local test mode (`enable_local_test_mode`)  
  - Using decorators and dependency injection for testability  
- FastAPI + Ray Serve: integration patterns  
  - Basic routing vs. advanced use cases  
- Key considerations  
  - Serialization/deserialization overhead  
  - Error handling and retries  
  - Background tasks and dependency management  

## 4. Observability & Reliability
- Health checks (built-in and custom)  
- Metrics, tracing, and alerting (Prometheus, Grafana, Datadog)  
- Debugging tips for production workloads  

## 5. Autoscaling with Ray Serve
- Configuring autoscaling policies in Ray Serve  
- Load testing with Locust (hands-on exercise)  
- Finding and tuning the “right” autoscaling config  