# Robotics simulation at scale with Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/groot-ray-serve"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/groot-ray-serve" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

In this module you will see Ray Core and Ray Serve combined to run a humanoid robotics workload end to end:

- A **3B-parameter vision-language-action model** (NVIDIA GR00T-N1.7) deployed behind an HTTP endpoint with **Ray Serve**
- **NVIDIA Isaac Lab** physics simulation running on a separate GPU, parallelizable across GPUs with **Ray Core**
- A live rollout that runs in the background while we keep talking

The same primitives that scale LLM inference scale cleanly to robotics. By the end of this notebook you will have:

1. A live Ray Serve deployment serving the GR00T policy on a GPU worker
2. A real round-trip from notebook to policy showing ~200 ms–1.5 s latency
3. A simulation rollout that runs on a Ray Core actor on a different GPU
4. A clear picture of how to swap models, fan out simulators, and scale replicas. All one-line changes.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/groot-ray-serve
```


> **One shared policy fleet. Many independent simulators.**
>
> Ray Serve owns the model-serving layer.  
> Ray Core fans out the expensive simulation layer.


![GR00T-N1.7 humanoid policy running a zero-shot pick-and-place rollout in NVIDIA Isaac Lab, served by Ray Serve](https://raw.githubusercontent.com/anyscale/templates/main/templates/groot-ray-serve/g1_groot_n17_zeroshot.gif)

## Architecture

<div align="center">

<svg xmlns="http://www.w3.org/2000/svg" width="900" viewBox="0 0 720 460" style="max-width: 100%; height: auto;">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="#666" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>

  <rect x="20" y="40" width="680" height="260" rx="14" fill="none" stroke="#888780" stroke-width="0.5" stroke-dasharray="4 4"/>
  <text x="40" y="64" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#2C2C2A">Anyscale Ray cluster</text>

  <rect x="50" y="160" width="170" height="120" rx="10" fill="#FAEEDA" stroke="#854F0B" stroke-width="0.5"/>
  <text x="135" y="188" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#633806">Head node</text>
  <text x="135" y="208" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#854F0B">Jupyter notebook</text>
  <text x="135" y="226" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#854F0B">Ray driver</text>
  <text x="135" y="244" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#854F0B">Schedules tasks</text>
  <text x="135" y="262" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#854F0B">Renders the GIF</text>

  <rect x="270" y="150" width="200" height="160" rx="10" fill="#EEEDFE" stroke="#534AB7" stroke-width="0.5"/>
  <text x="370" y="178" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#3C3489">GPU worker A</text>
  <text x="370" y="198" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#534AB7">running Ray Serve</text>
  <rect x="290" y="212" width="160" height="84" rx="6" fill="#CECBF6" stroke="#534AB7" stroke-width="1" fill-opacity="0.5"/>
  <text x="370" y="232" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#3C3489">GR00T-N1.7-3B</text>
  <text x="370" y="250" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#534AB7">VLA policy on cuda:0</text>
  <text x="370" y="268" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#534AB7">FastAPI ingress</text>
  <text x="370" y="286" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#534AB7">POST /predict</text>

  <rect x="500" y="150" width="180" height="160" rx="10" fill="#E1F5EE" stroke="#0F6E56" stroke-width="0.5"/>
  <text x="590" y="178" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#085041">GPU worker B</text>
  <text x="590" y="198" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#0F6E56">running Ray actor</text>
  <rect x="518" y="212" width="144" height="84" rx="6" fill="#9FE1CB" stroke="#0F6E56" stroke-width="1" fill-opacity="0.5"/>
  <text x="590" y="232" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="14" font-weight="500" fill="#085041">Isaac Lab</text>
  <text x="590" y="250" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#0F6E56">Isaac Sim 5.1</text>
  <text x="590" y="268" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#0F6E56">Unitree G1 humanoid</text>
  <text x="590" y="286" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#0F6E56">pick-and-place scene</text>

  <line x1="220" y1="180" x2="266" y2="180" stroke="#666" stroke-width="1.2" fill="none" marker-end="url(#arrow)"/>
  <text x="243" y="173" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#5F5E5A">deploy</text>

  <path d="M 135 156 L 135 110 L 590 110 L 590 146" stroke="#666" stroke-width="1.2" fill="none" marker-end="url(#arrow)"/>
  <text x="362" y="103" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#5F5E5A">launch sim actor</text>

  <line x1="496" y1="188" x2="474" y2="188" stroke="#666" stroke-width="1.2" fill="none" marker-end="url(#arrow)"/>
  <text x="485" y="181" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#5F5E5A">obs</text>

  <line x1="474" y1="270" x2="496" y2="270" stroke="#666" stroke-width="1.2" fill="none" marker-end="url(#arrow)"/>
  <text x="485" y="264" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#5F5E5A">actions</text>

  <text x="485" y="227" text-anchor="middle" font-family="-apple-system, sans-serif" font-size="12" fill="#888780">HTTP</text>
</svg>

</div>

The notebook on the head node deploys the GR00T policy to a GPU worker via Ray Serve, then launches a Ray actor on a different GPU worker that runs Isaac Lab and queries the policy over HTTP.

## Step 0: Hugging Face authentication

GR00T-N1.7 uses `nvidia/Cosmos-Reason2-2B` as its vision-language backbone. The Cosmos model is gated on Hugging Face, so a token with access is required.

**Before running:** accept the terms at https://huggingface.co/nvidia/Cosmos-Reason2-2B (must be logged in), then create a read token at https://huggingface.co/settings/tokens.


```python
import os
import getpass

# Prefer an already-set env var; fall back to a secure prompt.
# The token is never written to the notebook.
HF_TOKEN = os.environ.get("HF_TOKEN") or getpass.getpass("Paste your Hugging Face token: ")
os.environ["HF_TOKEN"] = HF_TOKEN

assert HF_TOKEN.startswith("hf_"), "Token should start with 'hf_'"
print(f"HF token loaded (ends in ...{HF_TOKEN[-4:]})")
```

## Step 1: Connect to the Ray cluster

Attach to the running Anyscale cluster. The `runtime_env` ensures every Ray task and Ray Serve replica inherits the HF token.


```python
import ray

ray.init(
    address="auto",
    ignore_reinit_error=True,
    runtime_env={"env_vars": {"HF_TOKEN": HF_TOKEN}},
)

resources = ray.cluster_resources()
print("Ray cluster connected.")
print(f"Available GPUs:         {int(resources.get('GPU', 0))}")
print(f"Available CPUs:         {int(resources.get('CPU', 0))}")
print(f"Object store memory:    {resources.get('object_store_memory', 0) / 1e9:.0f} GB")
print()
print("Translation: we are no longer running a notebook cell.")
print("We are controlling a distributed system.")
```

## Step 2: Pre-warm the model cache (Ray Core in action)

GR00T-N1.7 loads two checkpoints at deploy time:

- `nvidia/GR00T-N1.7-3B` is the policy itself (~6 GB)
- `nvidia/Cosmos-Reason2-2B` is the gated VLM backbone (~5 GB)

Pre-downloading both to every GPU worker means Ray Serve can land the replica on any worker without a cold-start surprise. **This cell is your first Ray Core pattern**: `@ray.remote` tasks fan out across the cluster in parallel.


```python
@ray.remote(num_gpus=1, runtime_env={"env_vars": {"HF_TOKEN": HF_TOKEN}})
def prewarm_models():
    import os, time
    from huggingface_hub import snapshot_download
    t0 = time.time()
    snapshot_download("nvidia/GR00T-N1.7-3B")
    snapshot_download("nvidia/Cosmos-Reason2-2B")
    return f"{os.uname().nodename}: ready in {time.time()-t0:.0f}s"

n_gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"Pre-warming GR00T and Cosmos on {n_gpus} GPU workers in parallel...")
print()
for r in ray.get([prewarm_models.remote() for _ in range(n_gpus)]):
    print(f"  {r}")
```

## Step 3: Deploy GR00T behind an HTTP endpoint (Ray Serve)

`GR00TPolicyServer` is a Python class wrapped in `@serve.deployment` and `@serve.ingress(FastAPI())`. The decorators live in `policy_server.py`. With one `serve.run` call, Ray Serve:

- Schedules a replica on a GPU worker
- Loads the 3B parameter GR00T-N1.7-3B model onto cuda:0
- Stands up an HTTP server with `POST /predict` and `GET /stats`
- Returns a stable URL the rest of the cluster can reach

Loading a 3B parameter model to GPU memory takes longer than Ray Serve's default 30 second health-check timeout, so `health_check_timeout_s` is extended.

**Expected runtime with pre-cached weights:** 60 to 90 seconds.


```python
import sys
import time
from ray import serve

# policy_server.py lives in the notebook's working directory (already on sys.path).
from policy_server import GR00TPolicyServer

print("Deploying GR00TPolicyServer to Ray Serve...")

deployment = GR00TPolicyServer.options(
    num_replicas=1,
    health_check_timeout_s=300,
    health_check_period_s=120,
    graceful_shutdown_timeout_s=10,
    ray_actor_options={
        "num_gpus": 1,
        "runtime_env": {"env_vars": {"HF_TOKEN": HF_TOKEN}},
    },
).bind(
    model_path="nvidia/GR00T-N1.7-3B", #nvidia/GR00T-N1.6-G1-PnPAppleToPlate
    embodiment_tag="REAL_G1",
)

# Bind the HTTP proxy to 0.0.0.0 so sim workers on OTHER GPU nodes can reach the
# policy (Serve's default binds loopback only). This is load-bearing, not redundant
# with serve.run.
serve.start(detached=False, http_options={"host": "0.0.0.0", "port": 8000})
serve.run(deployment, name="gr00t-policy")

import socket
def _head_ip():
    try:
        return ray.get_runtime_context().gcs_address.split(":")[0]
    except Exception:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

POLICY_URL = f"http://{_head_ip()}:8000"
print()
print(f"Policy is live at {POLICY_URL}")
```

## Step 4: Send the policy a real observation

A real observation in GR00T's `REAL_G1` schema includes:

- A short stack of camera frames (`video.ego_view`)
- The robot's joint state across both arms, hands, and waist
- A natural language instruction

The policy returns a **40-step action chunk** covering arms, hands, waist, base height, and navigation commands. This is the round-trip Isaac Lab makes once per chunk during a rollout.


```python
import numpy as np
import pickle
import requests

identity_pose = np.array([0.3, 0.0, 0.0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
dummy_obs = {
    "video": {"ego_view": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8)},
    "state": {
        "left_wrist_eef_9d":  identity_pose[None, None, :].copy(),
        "right_wrist_eef_9d": identity_pose[None, None, :].copy(),
        "left_arm":   np.zeros((1, 1, 7), dtype=np.float32),
        "right_arm":  np.zeros((1, 1, 7), dtype=np.float32),
        "left_hand":  np.zeros((1, 1, 7), dtype=np.float32),
        "right_hand": np.zeros((1, 1, 7), dtype=np.float32),
        "waist":      np.zeros((1, 1, 3), dtype=np.float32),
    },
    "language": {
        "annotation.human.task_description": [["pick up the apple and place it on the plate"]]
    },
}

t0 = time.time()
r = requests.post(f"{POLICY_URL}/predict", data=pickle.dumps(dummy_obs), timeout=180)
r.raise_for_status()
resp = pickle.loads(r.content)
latency_ms = (time.time() - t0) * 1000

print(f"Round trip: {latency_ms:.0f} ms")
print()
print("Action chunk:")
for k, v in resp["action"].items():
    print(f"  {k:24s} {np.asarray(v).shape}")
```

## Step 5: Kick off a live sim rollout in the background

We drive the simulator from a **Ray Actor**. The actor gives us a stable handle for launching rollouts on a dedicated GPU worker. Each `run_rollout` call boots Isaac Lab in a short-lived **subprocess**.

> Why a subprocess? Isaac Sim's Omniverse Kit insists on owning its process's main thread and asyncio event loop (it installs signal handlers and pumps its own loop), so it cannot be booted directly inside a Ray worker. 

The first cell below stages `sim_worker.py` and its supporting files into `/mnt/cluster_storage/groot_demo/`. That path is shared across all nodes in the workspace, so the actor's subprocess can find the files no matter which GPU worker it lands on.

`sim_actor.run_rollout.remote()` returns immediately with a **future**. This a placeholder for a result that does not exist yet. The actor boots Isaac Lab on its assigned GPU and runs the rollout in the background. We collect the result in Step 6.

**`--max-steps 8`** keeps the rollout short so the result is ready by the time we finish talking about Step 6.

The rollout count is set by **`N_ROLLOUTS`** in the cell below (`1` for this walkthrough). Each simulator is its own `SimActor` holding one GPU, so raising `N_ROLLOUTS` fans the work out across the cluster. On Anyscale the autoscaler can add GPU nodes to fit (a few minutes to provision, capped by the cluster's max-worker setting).


```python
import os, shutil

# Stage workshop files to shared cluster storage so sim workers can find them.
# /mnt/cluster_storage is shared across all nodes in the Anyscale workspace,
# but starts empty per workspace. We copy from the notebook's working dir.
WORKER_DIR = "/mnt/cluster_storage/groot_demo"
os.makedirs(WORKER_DIR, exist_ok=True)
for fname in ["sim_worker.py", "g1_env.py", "policy_server.py"]:
    src = os.path.abspath(fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(WORKER_DIR, fname))


@ray.remote(num_gpus=1, runtime_env={"env_vars": {"HF_TOKEN": HF_TOKEN}})
class SimActor:
    def __init__(self, policy_url: str, worker_dir: str):
        import os
        self.policy_url = policy_url
        self.worker_dir = worker_dir
        self.output_dir = "/mnt/cluster_storage/groot_demo/live_rollout_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.rollout_count = 0
    
    def run_rollout(self, seed: int = 1337, max_steps: int = 8, worker_id: int = 99):
        import subprocess, os, time
        results_file = f"/mnt/cluster_storage/groot_demo/live_rollout_w{worker_id}_{int(time.time())}.json"
        self.rollout_count += 1

        # Per-worker GIF name so fanned-out rollouts don't clobber each other.
        # Delete any GIF from a previous rollout up front: Step 6 decides
        # live-vs-fallback by file existence, so a stale GIF could otherwise be
        # mistaken for this run's output if the subprocess fails.
        gif_path = os.path.join(self.output_dir, f"worker{worker_id}_ep0.gif")
        if os.path.exists(gif_path):
            os.remove(gif_path)

        # No shell: pass args as a list, set the working dir with cwd=, and use
        # subprocess's own timeout instead of the `timeout` binary. This removes
        # the need for any bash layer (python -> python, not python -> bash -> python) 
        # and avoids shell quoting/injection; --instruction keeps its spaces as one
        # arg without manual quoting.
        cmd = [
            "python", "-u", "sim_worker.py",
            "--worker-id", str(worker_id),
            "--policy-url", self.policy_url,
            "--task", "Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
            "--instruction", "pick up the apple and place it on the plate",
            "--episodes", "1",
            "--max-steps", str(max_steps),
            "--action-horizon", "8",
            "--save-frames-every", "1",
            "--output-dir", self.output_dir,
            "--seed", str(seed),
            "--results-file", results_file,
        ]
        # subprocess's timeout RAISES (unlike the `timeout` binary's exit code),
        # so catch it to preserve "always return a result dict".
        try:
            r = subprocess.run(
                cmd, cwd=self.worker_dir,
                capture_output=True, text=True, timeout=600,
            )
            returncode, stdout, stderr = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as e:
            returncode = -1
            stdout = e.stdout or ""
            stderr = (e.stderr or "") + "\n[run_rollout] timed out after 600s"

        return {
            "worker_id": worker_id,
            "gif_path": gif_path if os.path.exists(gif_path) else None,
            "exit_code": returncode,
            "stderr_tail": "\n".join(stderr.splitlines()[-10:]) if returncode != 0 else "",
            "stdout_tail": "\n".join(stdout.splitlines()[-10:]) if stdout else "",
            "rollout_count": self.rollout_count,
        }


# Number of simulators to fan out. N_ROLLOUTS = 1 is the single background
# rollout this workshop walks through. Bump it to fan out: each SimActor holds
# one GPU, so on Anyscale the autoscaler provisions more GPU nodes to fit
# (new nodes take a few minutes to come up, and the count is capped by the
# cluster's max-worker setting).
N_ROLLOUTS = 1

print(f"Spawning {N_ROLLOUTS} sim actor(s) on GPU workers...")
sim_actors = [SimActor.remote(POLICY_URL, WORKER_DIR) for _ in range(N_ROLLOUTS)]

print(f"Launching {N_ROLLOUTS} rollout(s)...")
live_futures = [
    a.run_rollout.remote(seed=1337 + i, max_steps=8, worker_id=i)
    for i, a in enumerate(sim_actors)
]
print(f"Submitted {len(live_futures)} rollout(s). Isaac Lab is booting on each assigned GPU now.")
print("We will collect the result(s) in Step 6.")
```

## Step 6: Collect the live rollout

In Step 5 we launched the simulator(s) on Ray actor(s). Now we call `ray.get(live_futures)` to collect the result(s). This blocks until the rollout finishes.

What is actually happening on that other GPU worker:

1. Isaac Lab booted and loaded the Unitree G1 in a pick-and-place scene
2. Every few steps, it captured camera frames and joint state
3. Those observations went to the Ray Serve endpoint deployed in Step 3
4. The 40-step action chunks came back and stepped the physics
5. Frames stacked into a GIF saved to the worker's disk

Ray brings the GIF back to us when we ask for the result.

If the live run does not produce a GIF in time, the cell falls back to a pre-recorded one.


```python
from IPython.display import Image, Video, display
import os, shutil, time, threading

print(f"Collecting results from {len(live_futures)} background sim rollout(s)")
print("  Isaac Lab boot + rollout typically takes 200-300s on a cold worker")
print("  (faster if a worker's Isaac Sim caches are already warm)")
print()

result_holder = {}
def _collect():
    try:
        result_holder["results"] = ray.get(live_futures)
    except Exception as e:
        result_holder["results"] = [{"gif_path": None, "exit_code": -1,
                                     "stderr_tail": f"{type(e).__name__}: {e}"}]

t = threading.Thread(target=_collect, daemon=True)
t.start()
time.sleep(0.5)

elapsed = 0
while t.is_alive():
    time.sleep(2)
    elapsed += 2
    if elapsed % 30 == 0:                       # progress every ~30s, not every 2s
        print(f"  ...still running ({elapsed}s elapsed)", flush=True)

t.join()
results = result_holder["results"]
print(f"\n{len(results)} rollout(s) finished in ~{elapsed}s.")

# Only trust GIFs that THIS run actually produced. run_rollout deletes any stale
# GIF before launching, so a non-zero exit code or a missing file means that
# rollout failed -> report it, and fall back only if NONE succeeded.
ok = []
for i, r in enumerate(results):
    if r.get("exit_code") == 0 and r.get("gif_path") and os.path.exists(r["gif_path"]):
        ok.append(r)
    else:
        print(f"  rollout {i}: no fresh GIF (exit_code={r.get('exit_code')})")
        tail = r.get("stderr_tail") or ""
        if tail:
            print("   " + tail.replace("\n", "\n   "))
print(f"  {len(ok)}/{len(results)} produced a fresh GIF.")
print()

if ok:
    # Display the first successful rollout; all fanned-out GIFs are saved on the
    # workers' shared storage as worker<id>_ep0.gif.
    src = ok[0]["gif_path"]
    dst_gif = "./live_rollout.gif"
    dst_mp4 = "./live_rollout.mp4"
    shutil.copy(src, dst_gif)

    import imageio
    frames = imageio.mimread(dst_gif)
    imageio.mimsave(dst_mp4, frames, fps=15, codec="libx264")

    display(Video(dst_mp4, embed=True, html_attributes="controls autoplay loop muted"))
else:
    print("  No live GIF this run — falling back to the pre-recorded clip.")
    display(Image(filename="g1_groot_n17_zeroshot.gif"))
```


```python
# from IPython.display import Video, display
# display(Video("g1_groot_n17_zeroshot.mp4", embed=True, html_attributes="autoplay loop muted"))
```

### Reading the rollout

Three things to notice:

- **The robot is in the right scene with the right embodiment.** Isaac Lab loaded the Unitree G1 at the pick-and-place table with a target apple, rendering camera frames at the resolution GR00T expects.
- **Every motion is the policy's own output.** No scripted joint trajectory, no replay. The arms move because GR00T returned a 40-step action chunk and the simulator stepped through it.
- **The motion is exploratory rather than task-completing.** The arms search the workspace but do not yet land a clean grasp on the apple.

This is honest: a **zero-shot rollout from the GR00T-N1.7-3B base model**. The base model has never been trained on this exact task.

### What a fine-tuned rollout looks like

The Ray Serve + Ray Core infrastructure above is the constant. The model checkpoint is the variable. NVIDIA has published rollout videos of fine-tuned GR00T policies completing this exact task on the same Unitree G1. Open these in a new tab to see:

- 🍎 **[GR00T N1.5: "Pick the apple from table to plate"](https://research.nvidia.com/labs/gear/gr00t-n1_5/)**, the exact task this notebook runs, fine-tuned policy completing the grasp
- 🤖 **[GR00T N1.6 research page](https://research.nvidia.com/labs/gear/gr00t-n1_6/)**, the next-generation model with G1-specific fine-tunes including `GR00T-N1.6-G1-PnPAppleToPlate`

**Same robot, same task, same policy architecture, same Ray Serve infrastructure, different checkpoint.** Swapping to either fine-tune is a one-line change to the `.bind(model_path=...)` call in Step 3. Ray Serve handles the rest: replica scheduling, GPU placement, HTTP serving.

## Step 7: Inspect the policy server

Ray Serve exposes the policy as a real HTTP service. The stats endpoint reports latency and call count over the deployment's lifetime.


```python
import json
r = requests.get(f"{POLICY_URL}/stats", timeout=10)
print(json.dumps(r.json(), indent=2))
```

## Going further: what to try next

### Scale the policy server horizontally

```python
deployment = GR00TPolicyServer.options(num_replicas=4).bind(...)
```

Ray Serve schedules each replica on its own GPU and load-balances requests across them automatically. Mind the resource math: each replica needs a GPU, and every concurrent simulator needs one too — so `num_replicas=4` alongside parallel simulators requires a cluster with enough GPUs. (This 2-GPU demo cluster runs one policy replica and one simulator at a time.)

### Run many sim rollouts in parallel

Each `SimActor` holds its own GPU, so you fan out by creating several actors and launching a rollout on each:

```python
actors = [SimActor.remote(POLICY_URL, WORKER_DIR) for _ in range(N)]
results = ray.get([a.run_rollout.remote(seed=i) for i, a in enumerate(actors)])
```

Every actor queries the shared policy fleet and saves its own GIF. **This is the Module's headline pattern**: Ray Core fans out heavy simulators around a shared Ray Serve policy fleet, with no manual orchestration. `N` is bounded by the GPUs available after the policy replicas take theirs.

### Swap to the G1 fine-tune

This notebook used GR00T-N1.7-3B base. The repo also includes Path B, which loads NVIDIA's published G1 pick-and-place fine-tune `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`:

```bash
bash path_b_file_bridge/orchestrate_n16.sh
```

The fine-tune lives at https://huggingface.co/nvidia/GR00T-N1.6-G1-PnPAppleToPlate.

## Cleanup

Tear down the Ray Serve deployment **and** the `SimActor`(s). They are long-lived Ray actors bound to this kernel — `serve.shutdown()` does **not** reclaim them, so each keeps holding a GPU + CPU until explicitly killed (or the kernel exits). The Ray cluster itself keeps running.


```python
serve.shutdown()
print("Ray Serve stopped.")

# The SimActors are long-lived Ray actors bound to this kernel; serve.shutdown()
# does NOT reclaim them. Kill them explicitly to release their reserved GPU + CPU.
try:
    for a in sim_actors:
        ray.kill(a)
    print(f"Killed {len(sim_actors)} SimActor(s); their GPU + CPU are released.")
except NameError:
    print("No sim_actors in scope — nothing to kill.")
except Exception as e:
    print(f"SimActor(s) already gone ({type(e).__name__}).")
```
