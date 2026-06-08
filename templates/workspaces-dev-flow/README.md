# Workspaces and Development Flow on Anyscale

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/workspaces-dev-flow"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/workspaces-dev-flow" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: 20 min

Welcome to Anyscale Workspaces! This tutorial will guide you through the complete developer workflow — from setting up your workspace to deploying production workloads. Think of this as your "day-zero" onboarding to the Anyscale platform.

## Introduction & What are Workspaces

Anyscale Workspaces provide a fully managed development environment that combines the familiarity of local development with the power of cloud compute. Each workspace includes:

- **Jupyter notebook interface** with file browser and terminal access
- **Ray cluster** (head node + worker nodes) for distributed compute
- **Persistent storage** with multiple mount points for different persistence scopes
- **Port forwarding** for web apps (automatically detects services like TensorBoard)
- **Git integration** for version control workflows
- **VS Code / Cursor integration** via Anyscale extension
- **Seamless handoff** to Anyscale Jobs and Services for production deployment

Workspaces are ideal for:
- Interactive development and debugging
- Iterative experimentation with datasets and models
- Testing distributed code before productionizing
- Collaborating on shared storage
- Prototyping APIs and web services

### Architecture Overview

An Anyscale Workspace consists of:

1. **Workspace UI** — Jupyter notebook interface running on the head node
2. **Head node** — CPU-only node that hosts the Jupyter server and executes notebook cells (driver code)
3. **Worker nodes** — Optional nodes (CPU or GPU) that execute distributed Ray tasks
4. **Ray cluster** — Always running and ready to accept tasks
5. **Storage mounts** — Multiple persistence scopes for different use cases

This architecture lets you write code in notebooks while leveraging distributed compute for heavy workloads.

## Getting Started with Your Workspace

Let's verify your workspace is running and explore the Ray cluster.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/workspaces-dev-flow
```


```python
import ray
import os
from pathlib import Path

# Ensure Ray is initialized. Anyscale Workspaces auto-starts Ray, but other
# environments (e.g., Anyscale Jobs, local Ray, CI) don't — ignore_reinit_error
# makes this call idempotent.
ray.init(ignore_reinit_error=True)

# Print cluster resources
resources = ray.cluster_resources()
print("Available cluster resources:")
for resource, count in sorted(resources.items()):
    print(f"  {resource}: {count}")
```

You should see output showing available CPUs, memory, and other resources. The `ray.init(ignore_reinit_error=True)` call above is a no-op when Ray is already running (as in Anyscale Workspaces) and bootstraps Ray when it isn't (e.g., when running this notebook outside a Workspace).


```python
# Check how many nodes are in the cluster
nodes = ray.nodes()
print(f"\nCluster has {len(nodes)} node(s)")

# List storage mounts
print("\nAvailable storage mounts:")
!ls -d /mnt/*/
```

Your workspace comes with a Ray cluster pre-configured and ready to use. The resources shown above are what's available for your distributed tasks.

## Understanding Storage

Anyscale Workspaces provide multiple storage mounts, each with different persistence and sharing characteristics.

### Note on Storage

This template uses shared storage paths that are accessible from all nodes in the cluster. In a multi-node cluster, Ray workers on different nodes cannot access the head node's local file system. Use a [shared storage solution](https://docs.anyscale.com/configuration/storage#shared) accessible from every node.

### Storage Types

| Mount Path | Scope | Persistence | Use Case |
|-----------|-------|-------------|----------|
| `/home/ray/default` | Workspace | While running | Temporary scratch space |
| `/mnt/cluster_storage/` | Cluster | Workspace lifetime | Working files, checkpoints |
| `/mnt/user_storage/` | User | Indefinitely | Personal datasets, configs |
| `/mnt/shared_storage/` | Organization | Indefinitely | Team-wide shared data |

**Best practices:**
- Use `/mnt/cluster_storage/` for active work during development
- Save important results to `/mnt/user_storage/` or `/mnt/shared_storage/`
- Local storage (`/home/ray/default`) is lost when the workspace terminates

Let's test the storage mounts.


```python
# Create our working directory
workspace_dir = Path("/mnt/cluster_storage/workspace_tutorial")
workspace_dir.mkdir(parents=True, exist_ok=True)

# Write a test file to cluster storage
test_file = workspace_dir / "test.txt"
test_file.write_text("Hello from Anyscale Workspace!")

print(f"Created file: {test_file}")
print(f"Contents: {test_file.read_text()}")
```


```python
# Create a file in user storage (persists across workspaces)
user_dir = Path("/mnt/user_storage")
user_file = user_dir / "persistent_data.txt"
user_file.write_text("This file persists across workspace sessions")

print(f"Created persistent file: {user_file}")
!ls -lh /mnt/user_storage/
```

Files in `/mnt/cluster_storage/` persist while your workspace is running but are deleted when you terminate it. Files in `/mnt/user_storage/` persist indefinitely — you'll see them in future workspaces.

## Managing Dependencies

Workspaces support multiple dependency management strategies depending on whether you're developing interactively or preparing for production deployment.

### Runtime Installation (Development)

For quick experimentation, install packages at runtime using `pip install`.


```python
# Install a package for this session
!pip install emoji
```


```python
# Test the installed package
import emoji
result = emoji.emojize("Python is :fire:")
print(f"Emoji test: {result}")
```

Runtime installations are session-only — they're lost when the workspace restarts. For production workloads (Jobs and Services), use a `requirements.txt` file.

### Requirements File (Production)

For Jobs and Services, specify dependencies in `requirements.txt` so they're available when your code runs on separate clusters.


```python
# View the requirements file for this template
!cat requirements.txt
```

### Environment Variables

Sensitive credentials (API keys, tokens) should be set as environment variables, not hardcoded in notebooks.


```python
# Check for common environment variables
hf_token = os.environ.get("HF_TOKEN")
wandb_key = os.environ.get("WANDB_API_KEY")

print(f"HF_TOKEN set: {hf_token is not None}")
print(f"WANDB_API_KEY set: {wandb_key is not None}")
```

**To set environment variables:** Use the **Dependencies** tab in the workspace UI. Variables set there are available in your notebook session and can be propagated to Jobs and Services.

## Distributed Computing with Ray

Now let's demonstrate Ray's distributed computing capabilities. Remember: notebook cells execute on the head node (CPU-only), but Ray tasks execute on worker nodes.


```python
# Define a remote function
@ray.remote
def square(x):
    import time
    time.sleep(0.1)  # simulate some work
    return x ** 2

# Execute 10 tasks in parallel
print("Launching 10 parallel tasks...")
futures = [square.remote(i) for i in range(10)]
results = ray.get(futures)

print(f"Results: {results}")
print(f"Sum: {sum(results)}")
```

These tasks ran in parallel on the worker nodes. Without Ray, the sequential execution would take 1 second (10 tasks × 0.1 sec each). With Ray's parallelism, the total time is closer to 0.1 seconds (limited by the slowest task).


```python
# Check the Ray Dashboard for task execution details
print("View task execution in the Ray Dashboard:")
print("  → Click 'Dashboard' in the workspace UI sidebar")
print("  → Navigate to 'Jobs' tab to see task details")
```

The Ray Dashboard provides real-time visibility into distributed execution: which tasks ran on which nodes, resource utilization, and performance metrics.

## Git Integration & Version Control

Workspaces support standard Git workflows. You can clone repositories, make changes, commit, and push — all from the workspace terminal. For automated workflows or non-interactive execution, configure SSH keys or use the git:// protocol to avoid credential prompts.


```python
# Demonstrate Git clone workflow
# Note: Using a shallow clone (--depth 1) for faster execution
repo_dir = Path("/mnt/cluster_storage/examples")

print("Git clone example:")
print(f"  git clone --depth 1 https://github.com/anyscale/anyscale-examples.git {repo_dir}")
print("\nIn an interactive workspace, you can run this command in the terminal.")
print("For this demo, we'll skip the actual clone to avoid timeout issues.")
```

### Git Workflow Example

For typical development, you'd follow this pattern:

```bash
# Navigate to your project directory
cd /mnt/cluster_storage/your-project

# Create a new branch
git checkout -b feature/new-experiment

# Make changes, then commit
git add .
git commit -m "Add new experiment code"

# Push to remote
git push origin feature/new-experiment
```

**Note:** SSH keys stored in `~/.ssh/` persist in user storage, so you only need to configure Git credentials once.

## Port Forwarding for Web Apps

Workspaces automatically detect web services running on the head node and provide public URLs to access them. This is useful for tools like TensorBoard, Streamlit, or custom web apps.

**Note:** Background processes using `&` are not fully supported in Jupyter notebook cells. To demonstrate port forwarding interactively, run the command below in the workspace terminal instead. The code is shown here for reference.


```python
# Demonstrate port forwarding by creating a sample HTML file
# Actual HTTP server must be run from the terminal (Jupyter doesn't support background processes)

sample_html = workspace_dir / "index.html"
sample_html.write_text("""
<html>
<head><title>Workspace Demo</title></head>
<body>
    <h1>Hello from Anyscale Workspace!</h1>
    <p>This page is served from /mnt/cluster_storage/workspace_tutorial/</p>
</body>
</html>
""")

print(f"Created sample HTML file: {sample_html}")
print("\nTo test port forwarding:")
print("  1. Open the workspace Terminal tab")
print(f"  2. Run: cd {workspace_dir} && python -m http.server 8000")
print("  3. Click the 'Ports' tab in the workspace UI")
print("  4. Find port 8000 and click the URL to access your web app")
print("\nNote: HTTP server runs in the terminal, not in notebook cells")
```

Anyscale automatically detects services running on any port and creates forwarding URLs. This works for TensorBoard, Jupyter Lab extensions, FastAPI apps, and any other HTTP service.

## IDE Integration (VS Code / Cursor)

Develop locally while running code remotely using the Anyscale VS Code or Cursor extension.

### Setup Steps

1. **Install the extension**
   - Open VS Code or Cursor
   - Search for "Anyscale" in the extensions marketplace
   - Install and reload

2. **Connect to your workspace**
   - Sign in with your Anyscale credentials
   - Click "Connect to Workspace" in the extension sidebar
   - Select this workspace from the list

3. **Open remote folder**
   - Choose "Open Remote Folder"
   - Navigate to `/home/ray/default` or `/mnt/cluster_storage/`

### Benefits

- **Local editing** — Use your familiar local editor with all its extensions
- **Remote execution** — Code runs on the workspace cluster with full compute resources
- **IntelliSense** — Code completion works against the remote Python environment
- **Terminal access** — Integrated terminal connects directly to the workspace
- **File sync** — Changes sync automatically between local and remote

For detailed setup instructions, see the [VS Code integration documentation](https://docs.anyscale.com/platform/workspaces/vscode/).

## Submitting Jobs from Workspace

Once your code is stable, you can submit it as an Anyscale Job for production batch execution. Jobs run on separate clusters and don't require an active workspace.

### What are Anyscale Jobs?

- **Batch workloads** — Fire-and-forget execution for data processing, training, ETL
- **Separate clusters** — Jobs spin up dedicated clusters and terminate when done
- **CLI submission** — Submit from workspace terminal or external CI/CD
- **Automatic retries** — Configure retry policies for fault tolerance

Let's create a simple job and submit it.


```python
# View the job script
!cat main.py
```


```python
# View the job configuration
!cat job.yaml
```

The `job.yaml` file defines the job's configuration:
- **name**: Job identifier
- **entrypoint**: Command to execute (e.g., `python main.py`)
- **compute_config**: Worker nodes (auto-selected based on workload)
- **runtime_env**: Working directory, dependencies, environment variables

### Submit the Job

```bash
# Submit the job and wait for completion
anyscale job submit -f job.yaml --wait
```

The `--wait` flag blocks until the job finishes. For long-running jobs, omit `--wait` and check status separately.

### Monitor Job Progress

```bash
# Check job status
anyscale job status --name workspace-tutorial-job

# View job logs
anyscale job logs --name workspace-tutorial-job
```

Jobs run independently of your workspace — you can close the workspace and the job continues running. This makes Jobs ideal for long-running batch workloads.

## Deploying Services from Workspace

For long-running APIs or model serving, deploy your code as an Anyscale Service. Services provide HTTPS endpoints and scale automatically based on traffic.

### What are Anyscale Services?

- **Long-running deployments** — Always-on services with HTTPS endpoints
- **Autoscaling** — Automatically scale replicas based on request load
- **Production-ready** — Built-in health checks, metrics, logging
- **Ray Serve integration** — Deploy Ray Serve applications with one command

Let's deploy a simple echo service.


```python
# View the service application code
!cat serve_app.py
```


```python
# View the service configuration
!cat service.yaml
```

The `service.yaml` file defines:
- **name**: Service identifier
- **applications**: Ray Serve apps to deploy (import path points to `serve_app.py`)
- **compute_config**: Auto-selected worker nodes
- **runtime_env**: Working directory and dependencies

### Deploy the Service

```bash
# Deploy the service
anyscale service deploy -f service.yaml
```

Deployment takes 2-4 minutes for the first cluster spin-up. Once deployed, Anyscale provides an HTTPS endpoint URL.

### Test the Service

After deployment completes, you'll receive a service URL. Test it with a request like this:


```python
import os
import requests

# Set SERVICE_URL to the URL you got from "anyscale service deploy" above.
# Without it, the cell prints the example call and exits.
service_url = os.environ.get("SERVICE_URL", "")
if service_url:
    response = requests.post(
        f"{service_url}/echo",
        json={"message": "Hello from workspace!"},
    )
    print(f"Service response: {response.json()}")
else:
    print('Set SERVICE_URL to test against a deployed service, e.g.:')
    print('  export SERVICE_URL="https://your-service.anyscale.com"')

```

Expected response:
```json
{
    "echo": "Hello from workspace!",
    "timestamp": "2026-05-05T05:30:00.123456",
    "service": "workspace-tutorial-service"
}
```

Services remain running even after you terminate your workspace. They're independent production deployments with their own clusters.

## Monitoring & Debugging

Anyscale provides multiple tools for monitoring and debugging your workloads.

### Ray Dashboard

The Ray Dashboard is automatically available in every workspace:

- **Jobs tab** — View task execution, resource utilization, and performance metrics
- **Actors tab** — Monitor long-lived actor processes
- **Logs tab** — Access driver and worker logs
- **Metrics tab** — System-level metrics (CPU, memory, network, disk)

**Access the dashboard:** Click "Dashboard" in the workspace UI sidebar.

### Workspace Logs

View workspace-specific logs directly in the UI:

- **Workspace logs** — Driver code output, notebook cell execution
- **Cluster logs** — Node startup, autoscaling events
- **Ray logs** — Task execution, object store activity

### Common Troubleshooting Tips

**Storage path errors:**
- Ensure paths start with `/mnt/cluster_storage/` or `/mnt/user_storage/` for multi-node access
- Avoid relative paths (`. /`, `../`) — they resolve differently on each node

**Dependency conflicts:**
- Use `pip list` to verify installed packages and versions
- Check for conflicts between base image packages and `pip install` additions

**Port conflicts:**
- Ports are shared across all notebook cells — stop old servers before starting new ones
- Use `!pkill -f "process-name"` to stop background processes

For comprehensive debugging guidance, see the [Anyscale monitoring and debugging guide](https://docs.anyscale.com/monitoring/).

## Summary & Next Steps

Congratulations! You've learned the complete Anyscale Workspace workflow:

✅ **Workspace setup** — Verified cluster, explored storage mounts
✅ **Dependency management** — Runtime pip installs, requirements.txt, env vars
✅ **Distributed compute** — Ray remote tasks for parallel execution
✅ **Git integration** — Standard Git workflows (clone, commit, push)
✅ **Port forwarding** — Served web apps with automatic URL creation
✅ **IDE integration** — Connected VS Code/Cursor for local development
✅ **Jobs** — Submitted batch workloads for production execution
✅ **Services** — Deployed long-running APIs with HTTPS endpoints
✅ **Monitoring** — Ray Dashboard, logs, troubleshooting tips

### Advanced Topics

Now that you've mastered the basics, explore these advanced capabilities:

- **Custom workspace templates** — Create reusable workspace configurations for your team
  [Learn more](https://docs.anyscale.com/platform/workspaces/custom-templates/)

- **Job queues and scheduling** — Submit jobs to persistent queues with cron-style schedules
  [Learn more](https://docs.anyscale.com/jobs/queues/)

- **Service autoscaling** — Configure request-based scaling and multi-replica deployments
  [Learn more](https://docs.anyscale.com/services/manage/)

- **Multi-environment patterns** — Set up dev → staging → production workflows
  [Learn more](https://docs.anyscale.com/best-practices/)

### Documentation

- [Anyscale Workspaces documentation](https://docs.anyscale.com/platform/workspaces/)
- [Anyscale Jobs documentation](https://docs.anyscale.com/jobs/)
- [Anyscale Services documentation](https://docs.anyscale.com/services/)
- [Ray documentation](https://docs.ray.io/)

**Happy building!** 🚀
