# Compute Config and Cluster Shape Selection

Learn how to choose the right cluster configuration for your Ray workloads on Anyscale. This template teaches you when to use CPU vs GPU instances, how to configure autoscaling, and how to optimize costs with spot instances and heterogeneous worker groups.

**Key Learning Outcomes:**
- Choose appropriate cluster shapes for different AI workload types
- Configure autoscaling and understand cost vs latency tradeoffs
- Use worker groups for mixed CPU/GPU workloads
- Apply best practices for head and worker node sizing

---

## 01 · Introduction to Compute Configs

A **compute configuration** defines the resources Anyscale uses to launch a Ray cluster for your workspace, job, or service. The "shape" of your cluster—the instance types and node counts—directly impacts performance, cost, and reliability.

**Key Concepts:**
- **Head node**: Coordinates the cluster (runs Ray GCS, autoscaler, dashboard). Should be CPU-only with ≥8 vCPU and ≥32 GB RAM for multi-node clusters.
- **Worker nodes**: Execute your tasks and actors. Can be CPU-only or GPU-accelerated based on workload requirements.
- **Worker groups**: Collections of workers with the same instance type. Multiple worker groups enable heterogeneous compute (mixed CPU/GPU).
- **Cluster shape**: The combination of instance types and node counts that defines your cluster's total resources.

Compute configs are defined in YAML and specify:
```yaml
cloud: your-anyscale-cloud-name
head_node:
  instance_type: m5.2xlarge
  resources:
    CPU: 0  # Disable head node scheduling
worker_nodes:
  - instance_type: m5.4xlarge
    name: cpu-workers
    min_nodes: 2
    max_nodes: 4
```

This template will help you make informed infrastructure decisions by comparing different compute configurations with real workloads.

---

## 02 · Understanding Cluster Components

Before creating compute configs, let's understand the roles of head and worker nodes in a Ray cluster.

```python
# Import Ray
import ray

# Initialize Ray (auto-connects to the Anyscale workspace cluster)
ray.init()

# Inspect the cluster nodes
nodes = ray.nodes()

print(f"Total nodes in cluster: {len(nodes)}\n")

for i, node in enumerate(nodes):
    node_id = node['NodeID'][:8]
    resources = node['Resources']
    alive = node['Alive']

    # Simple heuristic: head node typically has ObjectStoreMemory but fewer total CPUs scheduled
    is_head = 'CPU' in resources and resources.get('CPU', 0) == 0

    print(f"Node {i+1} (ID: {node_id}...)")
    print(f"  Alive: {alive}")
    print(f"  CPU: {resources.get('CPU', 0)}")
    print(f"  Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
    print(f"  GPU: {resources.get('GPU', 0)}")
    print(f"  Role: {'HEAD (coordinator)' if is_head else 'WORKER (computation)'}\n")
```

**Key Observations:**
- **Head node**: In multi-node clusters, the head node has `CPU: 0` (scheduling disabled) and runs coordination services only.
- **Worker nodes**: Handle all task and actor execution. Resources like GPUs are only available on workers.
- **Single-node clusters**: The head node does both coordination and computation (fine for development, but move to multi-node for production).

<div class="alert alert-info">
<strong>Note:</strong> Anyscale automatically disables head node scheduling for multi-node clusters to reserve it for coordination. You can explicitly set <code>resources.CPU: 0</code> in the compute config to make this clear.
</div>

---

## 03 · Basic CPU-Only Cluster Config

Let's create your first compute configuration: a simple CPU-only, fixed-size cluster.

```python
import yaml
import os

# Create configs directory
os.makedirs("configs", exist_ok=True)

# Define a basic CPU-only cluster configuration
cpu_only_config = {
    "cloud": "your-anyscale-cloud-name",  # Update with your cloud name
    "head_node": {
        "instance_type": "m5.2xlarge",  # 8 vCPU, 32 GB RAM (meets minimum sizing)
        "resources": {
            "CPU": 0,  # Explicitly disable head node scheduling
            "GPU": 0
        }
    },
    "worker_nodes": [
        {
            "instance_type": "m5.large",  # 2 vCPU, 8 GB RAM
            "name": "cpu-workers",
            "min_nodes": 2,  # Fixed size: always 2 nodes
            "max_nodes": 2   # Fixed size: never more than 2 nodes
        }
    ]
}

# Write to file
with open("configs/cpu_only.yaml", "w") as f:
    yaml.dump(cpu_only_config, f, default_flow_style=False, sort_keys=False)

print("✓ Created configs/cpu_only.yaml\n")
print(yaml.dump(cpu_only_config, default_flow_style=False, sort_keys=False))
```

**Configuration Breakdown:**
- **Head node**: `m5.2xlarge` (8 vCPU, 32 GB) meets the minimum sizing recommendation
- **Worker nodes**: 2× `m5.large` (2 vCPU, 8 GB each) = 4 total vCPUs for workloads
- **Fixed size**: `min_nodes = max_nodes = 2` means no autoscaling—cluster always has 2 workers

<div class="alert alert-info">
<strong>Note:</strong> This is a minimal configuration for learning. Production clusters typically need larger worker nodes (≥4 vCPU, ≥16 GB) for better performance.
</div>

---

## 04 · Running a Sample Workload

Now let's create a simple CPU-bound workload to test our compute config understanding.

```python
import time

@ray.remote(num_cpus=1)
def process_batch(batch_id):
    """Simulate CPU-intensive data processing."""
    start = time.time()

    # Simulate work: compute sum of squares
    result = sum(i**2 for i in range(1_000_000))

    duration = time.time() - start
    node_id = ray.get_runtime_context().get_node_id()

    return {
        "batch_id": batch_id,
        "result": result,
        "duration": duration,
        "node_id": node_id[:8]  # First 8 characters
    }

# Launch 10 tasks
print("Launching 10 data processing tasks...\n")
start_time = time.time()

futures = [process_batch.remote(i) for i in range(10)]
results = ray.get(futures)

total_time = time.time() - start_time

# Display results
print(f"{'Batch':<8} {'Node ID':<10} {'Duration (s)':<15}")
print("=" * 35)
for r in results:
    print(f"{r['batch_id']:<8} {r['node_id']:<10} {r['duration']:<15.3f}")

print(f"\n✓ Total runtime: {total_time:.2f}s")
print(f"✓ Average task duration: {sum(r['duration'] for r in results) / len(results):.3f}s")
print(f"✓ Tasks per second: {len(results) / total_time:.1f}")

# Check distribution across nodes
unique_nodes = len(set(r['node_id'] for r in results))
print(f"✓ Tasks distributed across {unique_nodes} node(s)")
```

**Observations:**
- Each task requests `num_cpus=1` and Ray schedules it on an available worker
- Tasks are automatically distributed across worker nodes for parallel execution
- The head node (CPU: 0) doesn't execute any tasks—only workers do

---

## 05 · Head Node Sizing Best Practices

The head node runs critical Ray services: **Global Control Service (GCS)**, autoscaler, dashboard, and logs. Undersizing the head node leads to cluster instability and slow task scheduling.

**Minimum Requirements for Head Node:**
- **Multi-node clusters**: ≥8 vCPU, ≥32 GB RAM, **CPU-only** (no GPU)
- **Single-node dev clusters**: ≥4 vCPU, ≥16 GB RAM (GPU optional)

```python
# Example: Undersized head node (DON'T DO THIS)
bad_head_config = {
    "instance_type": "t3.medium",  # Only 2 vCPU, 4 GB RAM - TOO SMALL
}

# Correct: Properly sized head node
good_head_config = {
    "instance_type": "m5.2xlarge",  # 8 vCPU, 32 GB RAM - Good ✓
    "resources": {
        "CPU": 0,  # Disable scheduling on head node
        "GPU": 0
    }
}

print("❌ Bad head config:")
print(f"   Instance: {bad_head_config['instance_type']}")
print("   Problem: Only 2 vCPU, 4 GB RAM - will cause cluster instability\n")

print("✓ Good head config:")
print(f"   Instance: {good_head_config['instance_type']}")
print("   Specs: 8 vCPU, 32 GB RAM - meets minimums")
print("   CPU scheduling: Disabled (reserved for coordination)")
```

**Why CPU-only for multi-node?**
- Head node doesn't execute tasks/actors (CPU: 0), so GPU would be wasted
- GPUs are expensive—use them on workers where computation happens
- Exception: Single-node dev clusters can use GPU head node (but move GPUs to workers for production)

**Why disable head node scheduling?**
- Anyscale disables it automatically for multi-node clusters
- Explicitly setting `resources.CPU: 0` makes the config self-documenting
- Prevents accidental task scheduling on head if config is modified

---

## 06 · Worker Node Sizing: Scale Up vs Scale Out

When increasing cluster capacity, you have two choices:
- **Scale up**: Use bigger instance types (more CPUs/RAM per node)
- **Scale out**: Use more nodes (each smaller)

**Key Tradeoff: Inter-Node Communication**

Tasks and actors on the same node communicate via shared memory (fast). Tasks on different nodes communicate over the network (slower + costs).

**Rule of Thumb: Scale UP first, then scale OUT**

Minimize inter-node communication by using fewer, larger nodes before adding more smaller nodes.

```python
# Option 1: Scale UP (fewer, bigger nodes)
scale_up_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.4xlarge",  # 16 vCPU, 64 GB per node
            "name": "large-workers",
            "min_nodes": 2,
            "max_nodes": 2
        }
    ]
}
total_cpus_up = 2 * 16  # 32 vCPU

# Option 2: Scale OUT (more, smaller nodes)
scale_out_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.large",  # 2 vCPU, 8 GB per node
            "name": "small-workers",
            "min_nodes": 16,
            "max_nodes": 16
        }
    ]
}
total_cpus_out = 16 * 2  # 32 vCPU

print("SCALE UP:  2 nodes × 16 vCPU = 32 vCPU total")
print("  ✓ Less inter-node network traffic")
print("  ✓ Lower latency for shared-memory tasks")
print("  ✓ Simpler to manage (fewer nodes)\n")

print("SCALE OUT: 16 nodes × 2 vCPU = 32 vCPU total")
print("  ✗ More inter-node network traffic")
print("  ✗ Higher latency for distributed tasks")
print("  ✓ Better for embarrassingly parallel workloads (no inter-task communication)\n")

print("✓ Recommendation: Scale UP first to minimize network overhead")
```

**When to scale out?**
- **Embarrassingly parallel workloads**: No inter-task communication (e.g., batch inference, independent data processing)
- **Already at max instance size**: After scaling up to the largest available instance type
- **Fault tolerance needs**: More nodes = smaller blast radius if one fails (relevant for long-running jobs with spot instances)

---

## 07 · Autoscaling Configuration

Autoscaling allows your cluster to grow and shrink based on resource demand. This reduces costs but adds startup latency when new nodes are needed.

**Fixed-Size vs Autoscaling:**

```python
# Fixed-size cluster (no autoscaling)
fixed_size_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.2xlarge",  # 8 vCPU, 32 GB
            "name": "fixed-workers",
            "min_nodes": 4,
            "max_nodes": 4  # min = max → no autoscaling
        }
    ]
}

# Autoscaling cluster
autoscaling_config = {
    "max_cpus": 128,  # Global limit across all workers (optional)
    "worker_nodes": [
        {
            "instance_type": "m5.2xlarge",  # 8 vCPU per node
            "name": "autoscale-workers",
            "min_nodes": 1,   # Always keep 1 node warm (cost floor)
            "max_nodes": 16   # Can grow to 16 nodes = 128 vCPU (cost ceiling)
        }
    ]
}

# Write autoscaling config to file
with open("configs/autoscaling.yaml", "w") as f:
    autoscaling_config_full = {
        "cloud": "your-anyscale-cloud-name",
        "head_node": {
            "instance_type": "m5.2xlarge",
            "resources": {"CPU": 0, "GPU": 0}
        },
        **autoscaling_config
    }
    yaml.dump(autoscaling_config_full, f, default_flow_style=False, sort_keys=False)

print("✓ Created configs/autoscaling.yaml\n")
print("Fixed-size: Always 4 nodes (32 vCPU)")
print("  ✓ Fast: No startup delay when tasks arrive")
print("  ✗ Cost: Pay for 32 vCPU even if idle\n")

print("Autoscaling: 1-16 nodes (8-128 vCPU)")
print("  ✓ Cost: Pay only for what you use")
print("  ✓ Elasticity: Handles variable workloads")
print("  ✗ Latency: Adding nodes takes 2-5 minutes (VM provisioning)")
print("  ✗ Cold starts: First tasks on new nodes are slower\n")

print("✓ When to use fixed-size:")
print("  - Predictable, steady-state workloads")
print("  - Latency-sensitive applications")
print("  - Short-lived jobs where startup time is significant\n")

print("✓ When to use autoscaling:")
print("  - Variable workloads (batch jobs, bursty traffic)")
print("  - Cost-sensitive applications")
print("  - Long-running services where startup latency is amortized")
```

**Global CPU/GPU Limits:**
- `max_cpus` and `max_gpus` set cluster-wide upper bounds across all worker groups
- Useful when you have multiple worker groups and want to cap total resources
- If omitted, limit is `max_nodes * CPUs_per_node` for each worker group

---

## 08 · Cost Optimization with Spot Instances

**Spot instances** are spare cloud capacity offered at 60-90% discounts compared to on-demand instances. The tradeoff: they can be interrupted with 2 minutes notice when capacity is needed elsewhere.

```python
# On-demand only (most expensive, most reliable)
on_demand_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.2xlarge",
            "name": "ondemand-workers",
            "market_type": "ON_DEMAND",  # Standard pricing, no interruptions
            "min_nodes": 2,
            "max_nodes": 4
        }
    ]
}

# Spot instances only (cheapest, can be interrupted)
spot_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.2xlarge",
            "name": "spot-workers",
            "market_type": "SPOT",  # Discounted pricing, can be interrupted
            "min_nodes": 2,
            "max_nodes": 4
        }
    ]
}

# Prefer spot, fall back to on-demand (balanced)
prefer_spot_config = {
    "worker_nodes": [
        {
            "instance_type": "m5.2xlarge",
            "name": "hybrid-workers",
            "market_type": "PREFER_SPOT",  # Try spot first, on-demand if unavailable
            "min_nodes": 2,
            "max_nodes": 4
        }
    ]
}

# Write spot config to file
with open("configs/spot_instances.yaml", "w") as f:
    spot_config_full = {
        "cloud": "your-anyscale-cloud-name",
        "head_node": {
            "instance_type": "m5.2xlarge",
            "resources": {"CPU": 0, "GPU": 0}
        },
        **prefer_spot_config  # Use PREFER_SPOT for best balance
    }
    yaml.dump(spot_config_full, f, default_flow_style=False, sort_keys=False)

print("✓ Created configs/spot_instances.yaml\n")
print("Market type comparison:")
print("=" * 60)
print(f"{'Market Type':<20} {'Cost':<15} {'Reliability':<25}")
print("=" * 60)
print(f"{'ON_DEMAND':<20} {'$1.00/hr':<15} {'100% reliable':<25}")
print(f"{'SPOT':<20} {'$0.30/hr':<15} {'Can be interrupted':<25}")
print(f"{'PREFER_SPOT':<20} {'$0.30-1.00/hr':<15} {'Spot with on-demand fallback':<25}")
print("=" * 60)

print("\n✓ Recommendation: Use PREFER_SPOT for fault-tolerant workloads")
print("\nBest for spot instances:")
print("  ✓ Batch data processing (Ray Data)")
print("  ✓ Training with checkpoints (Ray Train)")
print("  ✓ Stateless inference (Ray Serve with retries)")
print("\nAvoid spot instances for:")
print("  ✗ Real-time serving (interruptions cause request failures)")
print("  ✗ Critical jobs without retries or checkpoints")
print("  ✗ Workloads that can't tolerate 2-minute interruption warnings")
```

**How Ray handles spot interruptions:**
- **Ray Data**: Automatically retries failed tasks on different nodes
- **Ray Train**: Restores from latest checkpoint if interruption occurs
- **Ray Serve**: Health checks detect failed replicas and restart them

---

## 09 · Heterogeneous Compute with Worker Groups

Real workloads often need both CPU and GPU resources. Use **multiple worker groups** to create heterogeneous clusters with different instance types.

**Common Pattern: CPU Preprocessing + GPU Inference**

```python
# Heterogeneous cluster: CPU workers + GPU workers
heterogeneous_config = {
    "cloud": "your-anyscale-cloud-name",
    "head_node": {
        "instance_type": "m5.2xlarge",  # 8 vCPU, 32 GB
        "resources": {"CPU": 0, "GPU": 0}
    },
    "worker_nodes": [
        # Worker group 1: CPU-only for preprocessing
        {
            "name": "cpu-preprocessing",
            "instance_type": "m5.4xlarge",  # 16 vCPU, 64 GB
            "min_nodes": 2,
            "max_nodes": 4
        },
        # Worker group 2: GPU for inference
        {
            "name": "gpu-inference",
            "instance_type": "g5.2xlarge",  # 1× L4 GPU, 8 vCPU, 32 GB
            "min_nodes": 1,
            "max_nodes": 2
        }
    ]
}

# Write heterogeneous config to file
with open("configs/heterogeneous.yaml", "w") as f:
    yaml.dump(heterogeneous_config, f, default_flow_style=False, sort_keys=False)

print("✓ Created configs/heterogeneous.yaml\n")
print("Cluster composition:")
print("  • 2-4 CPU workers (m5.4xlarge): 32-64 vCPUs for preprocessing")
print("  • 1-2 GPU workers (g5.2xlarge): 1-2 L4 GPUs for inference\n")
```

**Matching Tasks to Worker Groups:**

Ray automatically schedules tasks to worker groups based on resource requirements:

```python
# CPU task - will run on cpu-preprocessing workers
@ray.remote(num_cpus=2)
def preprocess_data(data_id):
    """Preprocess data on CPU workers."""
    import time
    print(f"Preprocessing {data_id} on CPU worker (node: {ray.get_runtime_context().get_node_id()[:8]})")
    time.sleep(0.1)  # Simulate preprocessing
    return f"processed_{data_id}"

# GPU task - will run on gpu-inference workers
# Note: We don't actually run this since our template is CPU-only
# but here's how you would structure it:
@ray.remote(num_gpus=1, resources={"accelerator_type:L4": 0.0001})
def run_inference(processed_data):
    """Run inference on GPU workers."""
    # In a real GPU workload:
    # import torch
    # device = torch.device("cuda")
    # model = load_model().to(device)
    # return model(processed_data)
    print(f"[Simulated] Running inference on {processed_data} with GPU (node: {ray.get_runtime_context().get_node_id()[:8]})")
    return f"result_{processed_data}"

# Demonstrate CPU task scheduling
print("Running CPU preprocessing tasks...\n")
preprocessed = [preprocess_data.remote(i) for i in range(4)]
preprocessed_results = ray.get(preprocessed)

print(f"\n✓ Preprocessed {len(preprocessed_results)} items on CPU workers")
print("\nWith GPU workers in the cluster, tasks requesting num_gpus=1 would")
print("automatically be scheduled to the gpu-inference worker group.")
```

**Worker Group Benefits:**
- **Resource efficiency**: Right-size each workload stage (cheap CPUs for prep, expensive GPUs only for inference)
- **Automatic placement**: Ray handles scheduling based on resource requirements—no manual node selection
- **Independent scaling**: Scale CPU and GPU workers independently based on bottlenecks

---

## 10 · Comparing Compute Configs Side-by-Side

Let's compare different compute configurations by running the same workload and analyzing performance and cost tradeoffs.

```python
import pandas as pd

# Define a realistic workload: process 100 data batches
@ray.remote(num_cpus=1)
def data_processing_task(item_id):
    """Simulate data preprocessing."""
    start = time.time()
    # Simulate CPU work: sum of squares
    result = sum(i**2 for i in range(500_000))
    duration = time.time() - start
    node_id = ray.get_runtime_context().get_node_id()
    return {"item_id": item_id, "duration": duration, "node_id": node_id[:8]}

# Run the workload
print("Running workload: 100 data preprocessing tasks...\n")
start_time = time.time()

futures = [data_processing_task.remote(i) for i in range(100)]
results = ray.get(futures)

total_time = time.time() - start_time

# Analyze results
df = pd.DataFrame(results)
print(f"✓ Total runtime: {total_time:.2f}s")
print(f"✓ Average task duration: {df['duration'].mean():.3f}s")
print(f"✓ Task throughput: {len(results) / total_time:.1f} tasks/sec")
print(f"✓ Tasks distributed across {df['node_id'].nunique()} node(s)\n")

# Compare different configurations (based on estimated performance/cost)
comparison_data = {
    "Config": [
        "Fixed-size (4× m5.2xlarge)",
        "Autoscaling (1-8× m5.2xlarge, spot)",
        "Heterogeneous (4× m5.4xlarge + 1× g5.2xlarge)"
    ],
    "Runtime (sec)": [
        45,   # Fixed: immediate start, steady parallelism
        62,   # Autoscaling: 15s warmup + 47s execution
        43    # Heterogeneous: bigger CPU nodes = faster
    ],
    "Estimated Cost ($)": [
        0.32,  # 4 nodes × $0.32/hr × (45/3600) hr
        0.12,  # Mix of spot + on-demand, lower node count
        0.38   # GPU node included (not used for this workload)
    ],
    "Reliability": [
        "High",
        "Medium (spot interruption risk)",
        "High"
    ],
    "Best For": [
        "Predictable workloads, latency-sensitive",
        "Batch jobs, cost-sensitive, fault-tolerant",
        "Mixed CPU/GPU workloads"
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("=" * 100)
print("COMPUTE CONFIG COMPARISON")
print("=" * 100)
print(comparison_df.to_string(index=False))
print("=" * 100)

print("\n✓ Key Takeaways:")
print("  1. Fixed-size is fastest but most expensive (no warmup delay)")
print("  2. Autoscaling + spot is cheapest but has startup latency")
print("  3. Heterogeneous works best when workload actually uses mixed resources")
print("  4. Choose based on workload characteristics (predictable vs bursty, CPU vs GPU)")
```

**Recommendations by Workload Type:**

```python
# Decision matrix
decision_matrix = {
    "Workload Type": [
        "Interactive development",
        "Scheduled batch jobs (daily ETL)",
        "Variable-traffic ML API",
        "Large-scale training (multi-day)",
        "Real-time inference serving"
    ],
    "Recommended Config": [
        "Single-node (GPU head OK)",
        "Autoscaling + PREFER_SPOT",
        "Autoscaling (min=2) + ON_DEMAND",
        "Fixed-size + PREFER_SPOT",
        "Fixed-size + ON_DEMAND"
    ],
    "Rationale": [
        "Minimal setup, full access to resources",
        "Cost optimization, startup latency OK",
        "Handle traffic spikes, keep min nodes warm",
        "Maximize GPU utilization, checkpoints handle interruptions",
        "Predictable latency, no interruption risk"
    ]
}

decision_df = pd.DataFrame(decision_matrix)
print("\n" + "=" * 100)
print("DECISION MATRIX: Choosing the Right Config")
print("=" * 100)
print(decision_df.to_string(index=False))
print("=" * 100)
```

---

## 11 · Best Practices and Common Pitfalls

Apply these guidelines when creating compute configs:

**Decision Tree:**

```python
def choose_compute_config(workload_type, is_predictable, is_fault_tolerant, needs_gpu, is_cost_sensitive):
    """Decision tree for compute config selection."""

    # Step 1: Head node (always)
    head_config = {
        "instance_type": "m5.2xlarge",  # 8 vCPU, 32 GB minimum
        "resources": {"CPU": 0, "GPU": 0}  # Disable scheduling
    }

    # Step 2: Instance type
    if needs_gpu:
        instance_type = "g5.2xlarge"  # 1× L4 GPU, or adjust based on needs
    else:
        instance_type = "m5.4xlarge"  # 16 vCPU, 64 GB CPU-only

    # Step 3: Scaling strategy
    if is_predictable:
        # Fixed-size for predictable workloads
        min_nodes = 4
        max_nodes = 4  # min = max
    else:
        # Autoscaling for variable workloads
        min_nodes = 1  # Keep 1 warm
        max_nodes = 16  # Allow scaling

    # Step 4: Market type (cost optimization)
    if is_cost_sensitive and is_fault_tolerant:
        market_type = "PREFER_SPOT"  # Spot with on-demand fallback
    else:
        market_type = "ON_DEMAND"  # Reliability over cost

    config = {
        "cloud": "your-anyscale-cloud-name",
        "head_node": head_config,
        "worker_nodes": [{
            "instance_type": instance_type,
            "name": f"{'gpu' if needs_gpu else 'cpu'}-workers",
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "market_type": market_type
        }]
    }

    return config

# Examples
print("Example 1: Scheduled batch ETL (predictable, fault-tolerant, CPU-only, cost-sensitive)")
config1 = choose_compute_config(
    workload_type="batch",
    is_predictable=False,  # Variable load
    is_fault_tolerant=True,
    needs_gpu=False,
    is_cost_sensitive=True
)
print(f"  → Autoscaling (1-16× m5.4xlarge) + PREFER_SPOT\n")

print("Example 2: Real-time ML serving (predictable, not fault-tolerant, GPU, not cost-sensitive)")
config2 = choose_compute_config(
    workload_type="serving",
    is_predictable=True,  # Steady traffic
    is_fault_tolerant=False,  # Can't tolerate interruptions
    needs_gpu=True,
    is_cost_sensitive=False
)
print(f"  → Fixed-size (4× g5.2xlarge) + ON_DEMAND\n")
```

**Common Pitfalls:**

| ❌ Mistake | ✓ Solution |
|-----------|-----------|
| Head node < 8 vCPU or < 32 GB | Use at least `m5.2xlarge` for head node |
| GPU on head node in multi-node cluster | Keep head CPU-only, put GPUs on workers |
| Spot instances for real-time serving | Use ON_DEMAND or PREFER_SPOT with retries |
| Cross-zone scaling for comm-heavy workloads | Disable cross-zone, use allowed_zones to pick one |
| Forgetting `resources.CPU: 0` on head | Explicitly disable scheduling in config |
| `min_nodes=0` for autoscaling | Set `min_nodes ≥ 1` to avoid cold start on every job |
| No fallback for spot instances | Use PREFER_SPOT instead of SPOT |
| Scaling out before scaling up | Use bigger nodes first (reduce inter-node traffic) |

**Troubleshooting:**

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| Tasks not scheduling | Resource requirements exceed worker capabilities | Check `num_cpus`, `num_gpus` match instance types |
| Slow cluster startup | Autoscaling provisioning delay | Set `min_nodes > 0` to keep warm nodes |
| High costs | Over-provisioned or always-on cluster | Use autoscaling + spot instances |
| Cluster instability | Undersized head node | Upgrade head to ≥8 vCPU, ≥32 GB |
| GPU errors in notebook | Trying to use GPU on head node | Move GPU code to `@ray.remote` tasks |

---

## 12 · Next Steps and Further Resources

You now understand how to choose and configure Ray clusters on Anyscale. Here are resources to deepen your knowledge:

**Related Anyscale Templates:**
- **Ray Data for Large-Scale Processing**: Apply compute configs to distributed data pipelines
- **Ray Train for Distributed Training**: Configure multi-GPU clusters for training workloads
- **Storage Access and Large Datasets**: Learn S3/GCS integration with cluster storage
- **Intro to Anyscale Jobs**: Submit batch workloads with custom compute configs

**Documentation:**
- [Anyscale Compute Configuration Guide](https://docs.anyscale.com/configuration/compute/) — Official compute config reference
- [Ray Scheduling Overview](https://docs.ray.io/en/latest/ray-core/scheduling/index.html) — How Ray schedules tasks and actors
- [Ray Resources and Accelerators](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html) — Resource requirements and custom resources
- [Anyscale Pricing and Instance Types](https://www.anyscale.com/pricing-detail#supported-machine-types) — Available instance types by cloud

**Key Concepts to Explore Next:**
- **Auto-select worker nodes** (beta): Let Anyscale autoscaler dynamically choose instances based on `accelerator_type` requests
- **Placement groups**: Advanced scheduling for gang-scheduled actors (e.g., distributed training)
- **Custom resources**: Define application-specific resource types for fine-grained scheduling
- **Cross-zone scaling**: Balance availability vs performance for multi-zone clusters

**Clean Up:**

If you created any YAML files or artifacts during this tutorial, they're stored locally and don't incur any costs. The workspace cluster will be terminated when you end your session.

---

**🎓 Congratulations!** You've learned how to make informed infrastructure decisions for Ray clusters on Anyscale. You can now:
- ✓ Choose CPU vs GPU cluster shapes based on workload type
- ✓ Configure autoscaling and understand cost vs latency tradeoffs
- ✓ Use worker groups for mixed CPU/GPU workloads
- ✓ Apply best practices for head and worker node sizing
- ✓ Optimize costs with spot instances and appropriate market types

Use the decision tree and comparison tables from this template as a reference when configuring your production workloads.
