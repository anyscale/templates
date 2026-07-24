"""Run the kMoL GPU microbench on an autoscaled L4 worker — WITHOUT needing a
kMoL image or a GPU head.

The trick (from the fintech-FM workspace playbook): the workspace's GPU workers
(g6.2xlarge / L4) autoscale on the MANAGED Ray 2.56 cluster, which kMoL's
2.51.2/py3.9 env cannot join. So we use a managed-cluster Ray task purely as a
*vehicle* to land on a GPU node, then shell out to kMoL's own python (from shared
/mnt storage) to run the pure-torch microbench. No Ray inside the microbench → no
version conflict; Ray only schedules the GPU.

Run with the BASE (managed-cluster) python:
    python scripts/launcher_gpu_microbench.py
"""

import ray

KMOL_SERVING = "/mnt/cluster_storage/kmol_serving"
KMOL_PY = "/mnt/cluster_storage/kmol_env/bin/python"

ray.init(address="auto")
print("managed ray:", ray.__version__)


@ray.remote(num_gpus=1, num_cpus=6)
def bench():
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{KMOL_SERVING}/stubs:{KMOL_SERVING}:{KMOL_SERVING}/kmol/src"
    # Ray sets CUDA_VISIBLE_DEVICES for this task; the subprocess inherits it.
    r = subprocess.run(
        [KMOL_PY, "scripts/gpu_microbench.py", "configs/ensemble_serve.example.json"],
        cwd=KMOL_SERVING,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    return f"[returncode={r.returncode}]\n{r.stdout}\n---STDERR (tail)---\n{r.stderr[-3000:]}"


print(ray.get(bench.remote()))
