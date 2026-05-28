# Introduction to Anyscale Jobs

**⏱️ Time to complete**: 15-20 minutes

Welcome to Anyscale Jobs! This hands-on tutorial teaches you how to submit, configure, monitor, and debug batch workloads on Anyscale.

## What You'll Learn

By the end of this tutorial, you'll be able to:

1. Submit jobs from an Anyscale workspace
2. Configure job runtime environments and compute resources
3. Monitor job status and inspect logs
4. Debug failing jobs and resubmit them
5. Submit jobs programmatically using the Python SDK

## What are Anyscale Jobs?

Anyscale Jobs run discrete workloads on dedicated Ray clusters, separate from your workspace. They're ideal for:

- **Batch inference** — Process large datasets through ML models
- **Model training** — Distribute training across GPUs or nodes
- **Data processing** — Transform and prepare datasets using Ray Data
- **Scheduled workflows** — Run periodic tasks like model retraining

**Key difference from workspaces:** While workspaces are for interactive development, jobs are for production batch execution with automatic retries, resource management, and comprehensive monitoring.

<img src="https://docs.anyscale.com/img/jobs/intro-to-jobs.png" width="80%" alt="Introduction to Anyscale Jobs overview diagram">

Jobs leverage Ray's distributed execution model to run your code across a cluster. For a deeper understanding of how Ray Jobs work under the hood, see the [Ray Jobs Overview](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html).

## Prerequisites

This tutorial assumes you're running in an Anyscale workspace. If you haven't worked with workspaces yet, check out the "Introduction to Workspaces" template first.

---

# Setup & Prerequisites

First, let's verify the Anyscale CLI is available in our workspace:


```python
!anyscale --version
```

Great! The Anyscale CLI comes pre-installed in Anyscale workspaces. If you're running this notebook outside of Anyscale, install the CLI with:

```bash
pip install -U anyscale
anyscale login
```

---

# Create a Simple Ray Task

Let's start by creating a Ray function that we'll run as a job. Ray's `@ray.remote` decorator turns regular Python functions into distributed tasks.


```python
import ray
import time


@ray.remote
def process(x):
    """
    A simple Ray task that squares a number.
    The @ray.remote decorator tells Ray this function can run on any worker node.
    """
    print(f"Processing {x}")
    time.sleep(0.1)  # Simulate work
    return x**2
```

The `@ray.remote` decorator transforms our function into a **remote function**. When we call it, instead of running locally, it gets scheduled on a worker node in the Ray cluster.

**Why use `@ray.remote`?**
- **Parallel execution**: Multiple tasks run simultaneously across cluster nodes
- **Scalability**: Distribute workload without changing code structure
- **Resource isolation**: Each task runs in its own process

Now let's see how to invoke this remote function:


```python
# Call .remote() to create a task
# This returns immediately with an ObjectRef (like a future/promise)
obj_ref = process.remote(5)
print(f"Task submitted. ObjectRef: {obj_ref}")
```

The `.remote()` method submits the task to Ray and returns an `ObjectRef` immediately — it doesn't wait for the result.

To actually get the result, we use `ray.get()`:


```python
# ray.get() blocks until the task completes and returns the result
result = ray.get(obj_ref)
print(f"Result: {result}")  # Prints: Result: 25
```

**Key pattern**: `function.remote(args)` → submit task, `ray.get(ObjectRef)` → retrieve result.

Let's run multiple tasks in parallel:


```python
# Submit 10 tasks in parallel
object_refs = [process.remote(i) for i in range(10)]

# Wait for all results
results = ray.get(object_refs)
print(f"All results: {results}")
```

Perfect! We're running 10 tasks in parallel across the cluster. In the next section, we'll package this into a script and run it as a job.

---

# Run the Task Locally in Workspace

Before submitting as a job, let's run our code directly in the workspace to verify it works. First, we'll create a standalone Python script:


```python
# Write our Ray script to disk
script_content = """import ray
import time

@ray.remote
def process(x):
    print(f"Processing {x}")
    time.sleep(0.1)
    return x ** 2

# Run 100 tasks in parallel
results = ray.get([process.remote(i) for i in range(100)])
print(f"Processed {len(results)} numbers")
print(f"Sum of squares (0-99): {sum(results)}")
"""

with open("main.py", "w") as f:
    f.write(script_content)

print("✓ Created main.py")
```

Now let's execute this script in our workspace cluster:


```python
!python main.py
```

**What just happened?**
The script ran on the **workspace cluster** — the same Ray cluster your notebook is connected to. The tasks were distributed across worker nodes, but everything stayed within your workspace environment.

**Workspace execution characteristics:**
- Runs on your persistent workspace cluster
- Shares the same Ray cluster as your notebook
- Good for development and testing
- Resources are shared with other workspace activities

In the next section, we'll submit this same script as an Anyscale Job — it will run on a **dedicated cluster** provisioned just for that job.

---

# Submit Your First Job from Workspace

Now let's submit our script as an Anyscale Job. Jobs run on dedicated Ray clusters separate from your workspace, ideal for production workloads.


```python
!anyscale job submit --name my-first-job --wait -- python main.py
```

**What's happening here?**

1. `anyscale job submit` — Submit a new job
2. `--name my-first-job` — Give the job a friendly name
3. `--wait` — Block until the job completes (useful for tutorials; omit for async submission)
4. `-- python main.py` — The command to run (everything after `--` is the entrypoint)

**Key difference from workspace execution:**
This job runs on a **new Ray cluster** provisioned specifically for it. Anyscale automatically:
- Provisions compute resources
- Uploads your working directory
- Starts the Ray cluster
- Executes your entrypoint
- Terminates the cluster when done

Let's inspect the job in the Anyscale console:

**View your job:**
1. Navigate to **Home > Jobs** in the Anyscale console
2. Find your job "my-first-job"
3. Click to see execution details, logs, and Ray Dashboard

![Job UI Screenshot](https://raw.githubusercontent.com/anyscale/templates/main/templates/job-intro/assets/anyscale-job.png)

**Job states:**
- **STARTING** → Cluster is provisioning
- **RUNNING** → Job is executing
- **SUCCEEDED** → Job completed successfully
- **FAILED** → Job encountered an error

Congratulations! You've submitted your first Anyscale Job. In the next sections, we'll dive into job configuration options.

---

# Understanding Job Configuration

So far, we've used minimal configuration. Now let's explore how to configure jobs for real-world scenarios.

## Why Configuration Matters

When you ran `anyscale job submit --name my-first-job -- python main.py`, Anyscale used **default settings** for everything:

- **Container image**: Latest Anyscale Ray image
- **Compute resources**: Default autoscaling configuration
- **Dependencies**: Whatever's already in the image
- **Retry behavior**: No automatic retries

For production workloads, you'll want explicit control over these settings. Configuration determines how your job runs, what resources it uses, and how it handles failures. The [Anyscale Jobs tutorial](https://docs.anyscale.com/jobs/tutorial/) covers the full workflow, and the [Job API Reference](https://docs.anyscale.com/reference/job-api/) provides complete field documentation.

Let's start with the basics.

## The Three Ways to Configure Jobs

There are three approaches to configuring Anyscale Jobs:

1. **CLI flags** — Quick, inline configuration for simple jobs
2. **YAML config file** — Structured, version-controlled configuration (recommended)
3. **Python SDK** — Programmatic submission from code

We'll focus on YAML configs first, as they're the most common pattern for production jobs.

## Basic Job Configuration Fields

Let's create a `job.yaml` file with the essential configuration fields:


```python
job_yaml = """# job.yaml - Basic Anyscale Job configuration

name: my-configured-job

# What command to run when the job starts
entrypoint: python main.py

# Container image (we'll use the slim Ray image)
image_uri: anyscale/ray:2.55.1-slim-py313-cu129

# How many times to retry if the job fails
max_retries: 2
"""

with open("job.yaml", "w") as f:
    f.write(job_yaml)

print("✓ Created job.yaml")
```

Let's break down each field:

### `name` (required)
**What it does:** Identifies your job in the Anyscale console
**Why it matters:** Makes it easy to find your job in job history and logs
**Tip:** Use descriptive names like `batch-inference-2024-05-04` instead of generic names like `job1`

### `entrypoint` (required)
**What it does:** The command Anyscale runs when your job starts
**Why it matters:** This is literally what gets executed — like running a script from your terminal
**Common patterns:**
- `python main.py` — Run a script
- `python -m my_module.train` — Run a module
- `python main.py --epochs 10 --lr 0.001` — Pass arguments

### `image_uri` (optional, but recommended)
**What it does:** Specifies the Docker container image for your job
**Why it matters:** Different images have different pre-installed packages
- `slim` images: Ray, pandas, numpy, PyArrow (smaller, faster to start)
- `llm` images: Above + torch, transformers, vLLM (for ML workloads)

**If you don't specify this:** Anyscale uses the latest image, which can change over time. Pin your image for reproducibility.

### `max_retries` (optional, default: 0)
**What it does:** How many times to automatically restart the job if it fails
**Why it matters:** Handles transient failures (network issues, spot instance preemption)
**Common values:**
- `0` — No retries (fail fast during development)
- `2-3` — Standard production setting
- `5+` — For jobs that must succeed despite transient issues

Now let's submit a job using this config:


```python
!anyscale job submit -f job.yaml --wait
```

**What changed?**
- Job now retries automatically if it fails
- Running on a specific, pinned image version
- Has a descriptive name for easy identification

But we're still missing some crucial config: **dependencies** and **environment variables**. Let's tackle those next.

---

# Configure Runtime Environment

When your job runs on a new cluster, it needs to know:
1. **What code to run** (working directory)
2. **What packages to install** (dependencies)
3. **What secrets/config to use** (environment variables)

Let's configure all three.

## Working Directory

The `working_dir` field tells Anyscale which local files to upload to the job cluster.


```python
job_with_workdir = """name: job-with-dependencies
entrypoint: python main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129

# Upload the current directory to the job cluster
# Anyscale automatically syncs these files to /home/ray on worker nodes
working_dir: .

max_retries: 2
"""

with open("job_workdir.yaml", "w") as f:
    f.write(job_with_workdir)
```

**What is `.` (current directory)?**
When you submit a job from a workspace, `.` means "upload everything in my current workspace directory." Anyscale:
1. Compresses your files
2. Uploads them to cloud storage
3. Extracts them on every worker node at `/home/ray/`

**Why this matters:**
Your job cluster doesn't have access to your workspace filesystem. Any Python files, config files, or data files your script imports/reads must be explicitly included via `working_dir`.

**Common patterns:**
- `working_dir: .` — Upload everything (convenient for small projects)
- `working_dir: ./src` — Upload just the src folder
- Use `.gitignore`-style exclusions with `excludes` field for large directories

## Python Dependencies

Most jobs need additional packages beyond what's in the base image. The `requirements` field handles this. Ray's runtime environment system installs these dependencies on each worker node before your job runs. For more details on managing dependencies in Ray, see the [Environment Dependencies guide](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html).


```python
# Create a requirements.txt for our job
requirements_content = """# requirements.txt
emoji==2.8.0
"""

with open("requirements.txt", "w") as f:
    f.write(requirements_content)

print("✓ Created requirements.txt")
```

Now let's update our job config to install dependencies:


```python
job_with_deps = """name: job-with-dependencies
entrypoint: python main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129

working_dir: .

# Install Python dependencies before running the job
# Can be a path to requirements.txt or an inline list
requirements:
  - emoji==2.8.0

max_retries: 2
"""

with open("job_with_deps.yaml", "w") as f:
    f.write(job_with_deps)
```

**Two ways to specify requirements:**

1. **Inline list** (shown above) — Good for a few packages
2. **Path to file** — `requirements: requirements.txt` — Better for many packages

**Why pin versions?**
`emoji==2.8.0` instead of `emoji` ensures your job uses the exact version you tested with. Unpinned dependencies can break when packages update.

**What happens during job startup?**
1. Cluster provisions
2. Anyscale runs `pip install emoji==2.8.0` on all nodes
3. Your entrypoint executes

This means job startup is slightly slower with dependencies, but once installed, execution is fast.

## Environment Variables

Many jobs need secrets (API keys) or configuration (URLs, model names). The `env_vars` field provides these:


```python
job_with_env = """name: job-with-environment
entrypoint: python main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129

working_dir: .

requirements:
  - emoji==2.8.0

# Environment variables available to your job
env_vars:
  EXAMPLE_ENV_VAR: "production"
  LOG_LEVEL: "INFO"
  # For secrets like HF_TOKEN, set them in workspace Dependencies tab instead

max_retries: 2
"""

with open("job_with_env.yaml", "w") as f:
    f.write(job_with_env)
```

**How to access these in your code:**


```python
import os

env_var = os.environ.get("EXAMPLE_ENV_VAR", "default")
log_level = os.environ.get("LOG_LEVEL", "INFO")
```

**Best practices for secrets:**
- ❌ **Don't** put secrets in YAML files (they're version controlled)
- ✅ **Do** set secrets in the Anyscale workspace **Dependencies tab** (encrypted, not logged)
- ✅ **Do** use `env_vars` for non-sensitive configuration

Now let's test our fully configured job:


```python
!anyscale job submit -f job_with_env.yaml --wait
```

**Checkpoint:** Your job now has:
- ✅ Explicit container image
- ✅ Code files uploaded
- ✅ Dependencies installed
- ✅ Environment variables set
- ✅ Retry policy configured

In the next section, we'll configure compute resources (CPUs, GPUs, node types).

---

# Configure Compute Resources

So far, we've used Anyscale's default compute configuration. For production workloads, you'll want explicit control over:
- Node types (CPU vs GPU)
- Number of workers
- Autoscaling behavior

## Why Compute Configuration Matters

**Cost control:** Smaller nodes save money
**Performance:** GPUs accelerate training/inference
**Scalability:** More workers = more parallelism

The `compute_config` block lets you specify the exact cluster topology for your job. You can control instance types, autoscaling behavior, and resource allocation. For a comprehensive guide to compute configuration options, see [Compute configuration on Anyscale](https://docs.anyscale.com/configuration/compute/) and the [Compute Config API Reference](https://docs.anyscale.com/reference/compute-config-api/).

Let's create a custom compute configuration:


```python
job_with_compute = """name: job-with-custom-compute
entrypoint: python main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129

working_dir: .
requirements:
  - emoji==2.8.0

# Custom compute configuration
compute_config:
  # Head node (runs job driver code)
  head_node:
    instance_type: m5.2xlarge

  # Worker nodes (run Ray tasks)
  worker_nodes:
    - instance_type: m5.xlarge
      min_nodes: 1      # Always keep 1 worker running
      max_nodes: 5      # Scale up to 5 workers under load
      # Anyscale autoscales between min and max based on workload

max_retries: 2
"""

with open("job_with_compute.yaml", "w") as f:
    f.write(job_with_compute)
```

**Breaking down compute_config:**

### `head_node.instance_type`
**What it does:** Specifies the machine type for the head node
**Why it matters:** The head node runs your entrypoint script and coordinates workers
**Common choices:**
- `m5.2xlarge` (8 CPUs, 32GB RAM) — Good default for CPU jobs
- `m5.8xlarge` (32 CPUs, 128GB RAM) — Large jobs with complex coordination

### `worker_nodes`
**What it does:** Defines the worker pool that executes Ray tasks
**Why it matters:** Workers do the actual compute work (training, inference, data processing)

### `instance_type` (worker)
**Common patterns:**
- **CPU-only jobs:** `m5.xlarge`, `m5.2xlarge` (cost-effective for data processing)
- **GPU jobs:** `g4dn.xlarge` (1x T4 GPU), `p3.2xlarge` (1x V100 GPU)
- **Large models:** `g5.12xlarge` (4x A10G GPUs), `p4d.24xlarge` (8x A100 GPUs)

### `min_nodes` / `max_nodes`
**What it does:** Autoscaling boundaries
**Why it matters:** Balance cost (fewer nodes) vs. speed (more nodes)

**Example scenarios:**
- **Fixed cluster:** `min_nodes: 3, max_nodes: 3` — Always 3 workers, no scaling
- **Cost-optimized:** `min_nodes: 0, max_nodes: 10` — Start with 0, scale as needed, terminate when idle
- **Performance-first:** `min_nodes: 5, max_nodes: 20` — Keep 5 warm, burst to 20 under load

**How autoscaling works:**
1. Job starts → Anyscale provisions `min_nodes` workers
2. Workload increases → Anyscale adds workers up to `max_nodes`
3. Workload decreases → Anyscale removes idle workers back to `min_nodes`
4. Job finishes → Entire cluster terminates

Let's submit this job with custom compute:


```python
!anyscale job submit -f job_with_compute.yaml --wait
```

**For GPU jobs** (not needed for this tutorial, but good to know):

```yaml
compute_config:
  worker_nodes:
    - instance_type: g4dn.xlarge  # 1x T4 GPU
      min_nodes: 2
      max_nodes: 8
```

Then in your Python code, request GPUs:


```python
@ray.remote(num_gpus=1)
def gpu_task():
    import torch

    print(torch.cuda.is_available())  # True
```

**Checkpoint:** You now know how to configure:
- ✅ Node types (head and workers)
- ✅ Autoscaling boundaries
- ✅ GPU resources (when needed)

Next, we'll monitor job execution and inspect logs.

---

# Monitor and Inspect Jobs

Once a job is running, you need to track its progress and troubleshoot issues. Anyscale provides several ways to monitor jobs through the CLI, Python SDK, and web console.

<img src="https://docs.anyscale.com/img/jobs/job-details.png" width="80%" alt="Job detail page showing job configuration and status">

The Anyscale console provides real-time metrics, logs, and cluster health visualization. For comprehensive monitoring guidance, see [Monitor a job](https://docs.anyscale.com/jobs/monitor/).

## Check Job Status (Python SDK)

Let's use the Anyscale Python SDK to programmatically check job status:


```python
import anyscale

# Get status of our most recent job
status = anyscale.job.status(name="job-with-custom-compute")

print(f"Job Name: {status.name}")
print(f"State: {status.state}")  # RUNNING, SUCCEEDED, FAILED, etc.
print(f"Job ID: {status.id}")
```

**Common job states:**
- `STARTING` — Cluster is provisioning
- `RUNNING` — Job is executing
- `SUCCEEDED` — Job completed successfully
- `FAILED` — Job encountered an error
- `TERMINATED` — Job was manually stopped

## Wait for Job Completion

The `wait()` function blocks until a job reaches a terminal state:


```python
# Submit a job and wait for it to finish
!anyscale job submit -f job_with_compute.yaml --name monitored-job --wait

# Check final status after completion
status = anyscale.job.status(name="monitored-job")
print(f"Final state: {status.state}")
```

**When to use `wait()`:**
- In CI/CD pipelines (fail the build if job fails)
- For sequential workflows (job B depends on job A)
- During development (see results immediately)

**When NOT to use `wait()`:**
- Long-running batch jobs (hours or days)
- When you need to monitor multiple jobs in parallel

## View Job Logs

Logs are essential for debugging. Let's retrieve logs programmatically:


```python
# Get logs for a specific job
logs = anyscale.job.get_logs(name="monitored-job", mode="TAIL")

print("Job logs:")
print(logs)
```

**Log modes:**
- `HEAD` — First 1000 lines (useful for startup issues)
- `TAIL` — Last 1000 lines (useful for final errors)

**Best practice:** View full logs in the Anyscale console (**Home > Jobs > [Job Name] > Logs tab**) for comprehensive debugging.

## Monitoring Best Practices

1. **Use meaningful job names** — `batch-inference-2024-05-04` beats `job-1234`
2. **Check status before debugging** — Saves time vs. diving into logs immediately
3. **Monitor Ray Dashboard** — View task timeline, resource usage, worker health
4. **Set up email alerts** — Get notified when jobs fail (configure in Anyscale console)

In the next section, we'll intentionally break a job and debug it using these monitoring tools.

---

# Debug a Failing Job

Let's simulate a real-world debugging scenario: submitting a job that fails, inspecting the error, fixing it, and resubmitting.

## Step 1: Create a Broken Script

First, let's modify our script to introduce a common error:


```python
broken_script = """import ray

@ray.remote
def process(x):
    # Intentional bug: divide by zero when x == 5
    result = 100 / (x - 5)
    return result

# This will fail when we reach x=5
results = ray.get([process.remote(i) for i in range(10)])
print(f"Results: {results}")
"""

with open("broken_main.py", "w") as f:
    f.write(broken_script)

print("✓ Created broken_main.py (will fail at x=5)")
```

## Step 2: Submit the Failing Job


```python
failing_job_yaml = """name: debug-example-failing
entrypoint: python broken_main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129
working_dir: .
max_retries: 0  # Don't retry, we want to see the failure
"""

with open("failing_job.yaml", "w") as f:
    f.write(failing_job_yaml)

# Submit the job (it will fail)
!anyscale job submit -f failing_job.yaml --wait
```

**Expected output:** Job will fail with a `ZeroDivisionError`.

## Step 3: Inspect the Failure

Now let's use our monitoring skills to debug:


```python
# Check job status
status = anyscale.job.status(name="debug-example-failing")
print(f"State: {status.state}")  # Should be FAILED

# Get the logs to see the error
logs = anyscale.job.get_logs(name="debug-example-failing", mode="TAIL")
print("\nJob logs (last 1000 lines):")
print(logs)
```

**What to look for in logs:**
- Python tracebacks (most important)
- `ZeroDivisionError: division by zero`
- Line numbers pointing to the error

**Common debugging patterns:**

| Error Type | What to Check |
|------------|---------------|
| `ModuleNotFoundError` | Missing package in `requirements` |
| `FileNotFoundError` | File not included in `working_dir` |
| `OutOfMemoryError` | Increase node size in `compute_config` |
| `ZeroDivisionError` | Logic bug in code (like our example) |
| Ray task failures | Check Ray Dashboard for task-level errors |

## Step 4: Fix the Bug

Let's fix the division by zero error:


```python
fixed_script = """import ray

@ray.remote
def process(x):
    # Fixed: avoid division by zero
    if x == 5:
        return 0  # Handle the edge case
    result = 100 / (x - 5)
    return result

# Now this will succeed
results = ray.get([process.remote(i) for i in range(10)])
print(f"Results: {results}")
print("Job completed successfully!")
"""

with open("fixed_main.py", "w") as f:
    f.write(fixed_script)

print("✓ Created fixed_main.py")
```

## Step 5: Resubmit the Fixed Job


```python
fixed_job_yaml = """name: debug-example-fixed
entrypoint: python fixed_main.py
image_uri: anyscale/ray:2.55.1-slim-py313-cu129
working_dir: .
max_retries: 0
"""

with open("fixed_job.yaml", "w") as f:
    f.write(fixed_job_yaml)

# Resubmit with the fix
!anyscale job submit -f fixed_job.yaml --wait
```

**Expected output:** Job succeeds this time!

## Verify the Fix


```python
# Confirm the job succeeded
status = anyscale.job.status(name="debug-example-fixed")
print(f"State: {status.state}")  # Should be SUCCEEDED

# View successful logs
logs = anyscale.job.get_logs(name="debug-example-fixed", mode="TAIL")
print("\nSuccess logs:")
print(logs)
```

**Debugging workflow summary:**
1. ❌ Job fails
2. 🔍 Check status to confirm failure
3. 📋 Read logs to find error
4. 🔧 Fix the code
5. ✅ Resubmit and verify success

**Pro tip:** Use `max_retries: 2` in production to automatically retry transient failures, but set `max_retries: 0` during debugging so you see failures immediately.

---

# Programmatic Job Submission (SDK)

So far, we've submitted jobs using YAML configs. For CI/CD pipelines and dynamic workflows, you'll want programmatic submission using the Python SDK.

## Why Use the Python SDK?

**YAML configs are great for:**
- Static, version-controlled job definitions
- Manual submissions
- Simple, repeatable workflows

**Python SDK is better for:**
- Parameterized jobs (different configs per run)
- CI/CD automation
- Multi-job orchestration
- Dynamic resource allocation

The Python SDK provides programmatic control over job lifecycle, allowing you to build complex workflows. For complete SDK documentation, see the [Python SDK Overview](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html) and [Python SDK API Reference](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/jobs-package-ref.html).

Let's see how to submit jobs programmatically.

## Basic SDK Submission


```python
from anyscale.job.models import JobConfig
import anyscale

# Define job configuration in Python
config = JobConfig(
    name="sdk-submitted-job",
    entrypoint="python main.py",
    image_uri="anyscale/ray:2.55.1-slim-py313-cu129",
    working_dir=".",
    requirements=["emoji==2.8.0"],
    max_retries=2,
)

# Submit the job
job_id = anyscale.job.submit(config)

print(f"✓ Job submitted with ID: {job_id}")
```

**What happened:**
We defined the same configuration we had in YAML, but in Python. The `submit()` function returns a job ID immediately (non-blocking).

## Wait for Completion


```python
# Wait for the job to complete
anyscale.job.wait(name="sdk-submitted-job")

# Check final status
status = anyscale.job.status(name="sdk-submitted-job")

if status.state == "SUCCEEDED":
    print("✓ Job completed successfully!")
    logs = anyscale.job.get_logs(name="sdk-submitted-job", mode="TAIL")
    print("\nJob output:")
    print(logs)
else:
    print(f"❌ Job failed with state: {status.state}")
```

## Dynamic Configuration Example

The real power of the SDK is dynamic job generation:


```python
import datetime

# Generate a unique job name with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_name = f"batch-inference-{timestamp}"

# Parameterized configuration
config = JobConfig(
    name=job_name,
    entrypoint="python main.py",
    image_uri="anyscale/ray:2.55.1-slim-py313-cu129",
    working_dir=".",
    requirements=["emoji==2.8.0"],
    env_vars={"BATCH_SIZE": "1000", "MODEL_NAME": "my-model-v2", "LOG_LEVEL": "INFO"},
    max_retries=3,
)

# Submit and track
job_id = anyscale.job.submit(config)
print(f"✓ Submitted {job_name}")
print(f"  Job ID: {job_id}")
```

**Use cases for dynamic submission:**
- **Nightly batch jobs:** Generate new job name each night
- **A/B testing:** Submit multiple jobs with different hyperparameters
- **Data pipelines:** Chain jobs that depend on each other
- **Auto-scaling:** Submit jobs based on queue depth or incoming data volume

## Inline Compute Config (SDK)

You can also specify compute configuration programmatically:


```python
config = JobConfig(
    name="sdk-custom-compute",
    entrypoint="python main.py",
    image_uri="anyscale/ray:2.55.1-slim-py313-cu129",
    working_dir=".",
    compute_config={
        "head_node": {"instance_type": "m5.2xlarge"},
        "worker_nodes": [
            {"instance_type": "m5.xlarge", "min_nodes": 2, "max_nodes": 8}
        ],
    },
)

job_id = anyscale.job.submit(config)
print(f"✓ Job submitted with custom compute: {job_id}")
```

**Checkpoint:** You now know how to:
- ✅ Submit jobs programmatically with Python
- ✅ Use `JobConfig` for structured configuration
- ✅ Dynamically generate job parameters
- ✅ Inline compute configs in code

Next, let's cover job termination and cleanup.

---

# Job Termination and Cleanup

Sometimes you need to stop a running job — maybe it's taking too long, or you found a bug and want to cancel it.

## Terminate a Running Job

Let's submit a long-running job and then terminate it:


```python
# First, create a long-running script
long_script = """import ray
import time

@ray.remote
def slow_task(i):
    time.sleep(60)  # Sleep for 1 minute per task
    return i

# This will take ~10 minutes total (100 tasks × 60s ÷ number of workers)
results = ray.get([slow_task.remote(i) for i in range(100)])
print(f"Completed {len(results)} tasks")
"""

with open("long_main.py", "w") as f:
    f.write(long_script)

print("✓ Created long_main.py (runs for ~10 minutes)")
```

Submit the long job (don't wait for it):


```python
from anyscale.job.models import JobConfig
import anyscale

config = JobConfig(
    name="long-running-job",
    entrypoint="python long_main.py",
    image_uri="anyscale/ray:2.55.1-slim-py313-cu129",
    working_dir=".",
)

job_id = anyscale.job.submit(config)
print(f"✓ Submitted long-running job (ID: {job_id})")
print("  This job will run for ~10 minutes...")
```

Wait a few seconds for it to start, then terminate it:


```python
import time

# Give the job a moment to start
time.sleep(10)

# Check status before terminating
status = anyscale.job.status(name="long-running-job")
print(f"Current state: {status.state}")

# Terminate the job
anyscale.job.terminate(name="long-running-job")
print("✓ Job termination requested")
```

Verify termination:


```python
# Wait a moment for termination to complete
time.sleep(5)

# Check final status
final_status = anyscale.job.status(name="long-running-job")
print(f"Final state: {final_status.state}")  # Should be TERMINATED
```

## When to Terminate Jobs

**Common scenarios:**
- **Development:** Realized you have a bug, want to fix and resubmit
- **Cost control:** Job is taking longer than expected, want to stop charges
- **Resource management:** Need to free up quota for higher-priority jobs
- **Stale jobs:** Job is stuck or unresponsive

## Cleanup Patterns

**Jobs automatically clean up when they finish:**
- Cluster is terminated
- Temporary storage is deleted
- Resources are released

**You typically don't need manual cleanup**, but here are patterns for when you do:

### Pattern 1: Terminate All Test Jobs


```python
# List all jobs
!anyscale job list

# Terminate any test jobs by name
# Example: !anyscale job terminate -n test-job-name
print("✓ Use 'anyscale job terminate -n <job-name>' to clean up test jobs")
```

### Pattern 2: Timeout Safety


```python
import anyscale
from anyscale.job.models import JobConfig

# Submit a job with a timeout
config = JobConfig(
    name="timeout-safe-job",
    entrypoint="python main.py",
    timeout_s=300,  # Automatically terminate after 5 minutes
)

job_id = anyscale.job.submit(config)
```

**`timeout_s` is useful when:**
- You know the job should finish in X minutes
- You want hard limits on costs
- You're running untested code

---

# Summary & Next Steps

Congratulations! You've learned the complete Anyscale Jobs workflow:

## What You Accomplished

✅ **Ray Basics**
- Created distributed tasks with `@ray.remote`
- Used `ray.get()` to retrieve results
- Ran parallel workloads across clusters

✅ **Job Submission**
- Submitted jobs from workspaces
- Used CLI (`anyscale job submit`)
- Used Python SDK (`anyscale.job.submit`)

✅ **Configuration**
- Configured container images (`image_uri`)
- Set up runtime environments (`working_dir`, `requirements`, `env_vars`)
- Customized compute resources (`compute_config`)
- Implemented retry policies (`max_retries`)

✅ **Monitoring & Debugging**
- Checked job status programmatically
- Retrieved and analyzed logs
- Debugged failing jobs and resubmitted fixes

✅ **Job Management**
- Terminated running jobs
- Implemented timeout safety
- Understood job lifecycle and cleanup

## Key Takeaways

1. **Jobs vs. Workspaces:** Jobs run on dedicated clusters, workspaces run on shared clusters. Use jobs for production batch workloads.

2. **Configuration is crucial:** Always pin `image_uri`, specify dependencies explicitly, and configure compute resources for production jobs.

3. **Debugging workflow:** Status → Logs → Fix → Resubmit. Use `max_retries: 0` during development.

4. **SDK for automation:** Use `JobConfig` and `anyscale.job.submit()` for CI/CD and dynamic workflows.

5. **Compute defaults are conservative:** Customize `compute_config` to match your workload's CPU/GPU needs.

## Next Steps

Ready to go deeper? Explore these advanced topics:

### Production Features
- **[Job Schedules](https://docs.anyscale.com/jobs/schedules)** — Run jobs on cron schedules for recurring workloads
- **[Job Queues](https://docs.anyscale.com/jobs/queues)** — Submit multiple jobs to the same cluster to reduce startup times
- **[Monitoring & Alerts](https://docs.anyscale.com/jobs/monitor)** — Set up email alerts and Grafana dashboards

### Scaling Up
- **[Ray Train](https://docs.ray.io/en/latest/train/train.html)** — Distributed model training at scale
- **[Ray Data](https://docs.ray.io/en/latest/data/data.html)** — Large-scale data processing and ETL
- **[Ray Serve](https://docs.ray.io/en/latest/serve/index.html)** — Deploy models as online services

### Integration Patterns
- **CI/CD with Jobs** — Automate training and inference pipelines
- **Multi-job workflows** — Chain jobs with dependencies
- **Cross-cluster data sharing** — Use `/mnt/shared_storage/` for workspace → job data handoff

Happy job submitting! 🚀

---

## Additional Resources

### Anyscale Jobs
- [What are Anyscale jobs?](https://docs.anyscale.com/jobs/) — Platform overview and key concepts
- [Get started with jobs](https://docs.anyscale.com/jobs/tutorial/) — Comprehensive tutorial
- [Create and manage jobs](https://docs.anyscale.com/jobs/manage/) — CLI and SDK workflows
- [Submit jobs to persistent job queues](https://docs.anyscale.com/jobs/queues/) — Reduce cold-start times
- [Job schedules](https://docs.anyscale.com/jobs/schedules/) — Automate recurring workloads

### Ray Jobs
- [Ray Jobs Overview](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) — Ray's job submission architecture
- [Quickstart using the Ray Jobs CLI](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html) — Open-source Ray CLI guide
- [Ray Jobs CLI API Reference](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html) — Command-line reference

### Configuration & APIs
- [Compute configuration on Anyscale](https://docs.anyscale.com/configuration/compute/) — Instance types, autoscaling, resource allocation
- [Job API Reference](https://docs.anyscale.com/reference/job-api/) — Complete field documentation
- [Compute Config API Reference](https://docs.anyscale.com/reference/compute-config-api/) — Cluster topology configuration
