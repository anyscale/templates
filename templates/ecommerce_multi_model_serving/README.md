# E-Commerce PyTorch Ranker Serving

**⏱️ Time to complete**: 30 min
### Anyscale Technical Demo — Ray Serve on Anyscale Services

---

## The Problem

You trained a ranking model. Now your platform team needs to deploy it:
containerize it, write Kubernetes manifests, configure HPA, health checks,
rolling deploys. Weeks of work. When traffic spikes — manual intervention.

## What We're Building

A single `POST /recommend` endpoint backed by one Ray Serve deployment:

```
POST /recommend
    → ProductRanker  (Ray Serve, GPU)
        1. Encode query    → all-MiniLM-L6-v2 (PyTorch)    → 384-dim vector
        2. Retrieve        → FAISS IndexFlatIP (in-process)  → top-100 products
        3. Re-rank         → cross-encoder (PyTorch)         → top-10 results
    ← JSON: results + encode_ms / retrieve_ms / rerank_ms
```

The same Python class that runs here becomes the production service.
No Dockerfile. No Kubernetes YAML. One command to deploy.


### Step 1: Connect to the Ray Cluster

The cell below initializes a connection to the Ray cluster running in this Anyscale workspace. Notice we pass `runtime_env` with a `working_dir` -- this tells Ray to automatically ship our project code to every worker node. **No Docker image rebuild required when you change code.**

After connecting, we print the cluster's available resources (CPUs, GPUs, memory) and list every node. This is the same heterogeneous cluster (CPU head + GPU workers) that will back our production service.


```python
import os
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

!pip install -q -r requirements.txt
```


```python
import sys, os

DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray

ray.init(
    ignore_reinit_error=True,
    runtime_env={"working_dir": DEMO_ROOT},
)

resources = ray.cluster_resources()
print("Ray cluster resources:")
for resource, count in sorted(resources.items()):
    if not resource.startswith('node:'):
        print(f"  {resource:<20} {count}")

nodes = ray.nodes()
print(f"\nCluster nodes: {len(nodes)}")
for n in nodes:
    res = ', '.join(f"{k}={v}" for k, v in n['Resources'].items() if not k.startswith('node:'))
    print(f"  {n['NodeManagerAddress']:<20} alive={n['Alive']}  {res}")
```

### Step 2: Build the Product Embedding Index

Next we generate a FAISS vector index over 50K synthetic products. The key detail: `ray.remote(num_gpus=1)` dispatches this work to a GPU worker node automatically. Ray's scheduler finds a node with a free GPU, ships the code there, and runs it -- **no SSH, no kubectl, no container orchestration.**

This is the same pattern your data engineers would use for any batch GPU workload (embedding, fine-tuning, inference). One decorator turns any Python function into a distributed GPU task.


```python
# Build the FAISS index: generate 50K products → embed → FAISS index.
# Dispatched to a GPU worker via Ray so embedding runs on T4, not the head node CPU.
# Skips entirely if both output files already exist.
import ray
from src.build_index import build_index

# Wrap in a Ray remote task to run on a GPU worker node
build_index_remote = ray.remote(num_gpus=1)(build_index)
ray.get(build_index_remote.remote())

print("\nTIP: Ray Dashboard -> Cluster tab shows GPU utilization during embedding.")

```

### Step 3: Deploy the Ranking Service with Ray Serve

`serve.run()` deploys our `ProductRanker` as a live HTTP endpoint inside this workspace. A single replica loads three components onto one GPU: the sentence encoder, the FAISS index, and the cross-encoder reranker. **This is the exact same `app` object that gets deployed to production via `service_config.yaml`** -- no translation layer, no repackaging.

Ray Serve gives you native Python composition of models behind a single endpoint, with built-in autoscaling, health checks, and request batching -- capabilities that would require stitching together multiple Kubernetes primitives.


```python
# Deploy the ProductRanker in the workspace Ray cluster.
# serve.run() is the same call used by `serve run deploy:app` on the CLI.
# The same app binding goes into service_config.yaml for production.
from ray import serve
from deploy import app

handle = serve.run(app, name="product-ranker")

print("Service started at http://localhost:8000")
print("\nProductRanker replica has loaded:")
print("  - FAISS index (50K product vectors)")
print("  - all-MiniLM-L6-v2 encoder (GPU)")
print("  - cross-encoder/ms-marco-MiniLM-L-6-v2 reranker (GPU)")
print("\nTIP: Ray Dashboard -> Serve tab shows the deployment and replica count.")

```

### Step 4: Run Live Queries

We now send real HTTP requests to the `/recommend` endpoint. Each request flows through the full pipeline: encode the query, retrieve candidates via FAISS, then rerank with the cross-encoder. The response includes **per-stage latency breakdowns** -- this observability comes for free, no external instrumentation needed.

Notice the filters (`max_price`, `category_filter`) are handled in application logic within the same deployment. This shows how Ray Serve lets you keep complex business logic in Python rather than spreading it across microservices.


```python
import requests

BASE_URL = "http://localhost:8000"

DEMO_QUERIES = [
    {"query": "wireless bluetooth headphones noise cancelling"},
    {"query": "women's running shoes lightweight breathable"},
    {"query": "daily face moisturizer SPF sensitive skin"},
    {"query": "wireless headphones", "max_price": 100.0},
    {"query": "running gear", "category_filter": "sports"},
]

# Warmup: first request loads nothing new (models already loaded at init)
# but warms up the FastAPI request path
print("Sending warmup request...")
requests.post(f"{BASE_URL}/recommend", json={"query": "warmup"}, timeout=60)
print("Ready.\n")

for payload in DEMO_QUERIES:
    resp = requests.post(f"{BASE_URL}/recommend", json=payload, timeout=30)
    data = resp.json()
    stages = data['stages']

    print(f"{'─' * 68}")
    label = data['query']
    if payload.get('max_price'):
        label += f"  [max_price=${payload['max_price']:.0f}]"
    if payload.get('category_filter'):
        label += f"  [category={payload['category_filter']}]"
    print(f"Query : {label}")
    print(f"\nLatency  {data['latency_ms']:.0f}ms total")
    print(f"  encode_ms    {stages['encode_ms']:>6.1f}  (query → 384-dim vector)")
    print(f"  retrieve_ms  {stages['retrieve_ms']:>6.1f}  (FAISS top-100)")
    print(f"  rerank_ms    {stages['rerank_ms']:>6.1f}  (cross-encoder scoring)")
    print(f"\nTop-5 results:")
    for r in data['results'][:5]:
        print(f"  [{r['relevance_score']:>7.3f}]  {r['title'][:55]:<55}  ${r['price']:.2f}")
    print()

```

### Step 5: Load Test and Autoscaling

This is where it gets interesting. We fire 100 requests at 20 concurrent connections. **Open the Ray Dashboard Serve tab** to watch Ray Serve's autoscaler detect the load spike and spin up a second GPU replica automatically. No HPA tuning, no custom metrics pipelines, no PodDisruptionBudgets.

The autoscaling config (`min_replicas`, `max_replicas`, `target_ongoing_requests`) is declared in Python alongside the deployment -- one place to reason about scaling behavior, versioned with your model code.


```python
# Ramp up concurrent requests and watch the autoscaler in the Ray Dashboard.
# As queue depth grows, a second replica starts on a new GPU worker node.
from scripts.load_test import run_load_test

print("TIP: Keep Ray Dashboard -> Serve tab open to watch replica count scale up.\n")

run_load_test(
    url="http://localhost:8000",
    concurrency=20,
    total_requests=100,
)

```

### Step 6: From Notebook to Production in One Command

Everything we just ran -- the same model code, the same `app` binding, the same autoscaling config -- ships to Anyscale Services with a single CLI command. Anyscale handles TLS termination, DNS, authentication, multi-AZ fault tolerance, and zero-downtime canary rollouts.

**The comparison table below is the punchline:** every row represents weeks of platform engineering work on Kubernetes that Ray Serve and Anyscale replace with a Python decorator or a CLI flag. This is how ML teams ship to production without waiting on infra.


```python
print("Path to production — one command from the CLI:")
print("""
  anyscale service deploy -f service_config.yaml --working-dir ./
""")

print("To update the service (zero-downtime canary rollout):")
print("""
  anyscale service deploy -f service_config.yaml --working-dir ./ --canary-percent 10
""")

print("What Ray Serve handles vs. what you'd manage with Kubernetes:")
rows = [
    ("Autoscaling replicas",    "serve.deployment(autoscaling_config=...)", "HPA + custom metrics"),
    ("GPU allocation",          "ray_actor_options={'num_gpus': 1}",        "GPU resource limits + node selectors"),
    ("Health checks",           "automatic",                                "Liveness + readiness probes"),
    ("Rolling deploy",          "--canary-percent flag",                    "Deployment strategy YAML"),
    ("Per-stage observability", "built into response",                      "Custom instrumentation"),
]
print(f"\n  {'Concern':<28} {'Ray Serve':<40} {'Kubernetes'}")
print(f"  {'─'*28} {'─'*40} {'─'*30}")
for concern, serve_answer, k8s_answer in rows:
    print(f"  {concern:<28} {serve_answer:<40} {k8s_answer}")

```
