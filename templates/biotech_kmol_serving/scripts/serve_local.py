"""Local / in-workspace dev harness: run the kMoL ensemble Serve app on an ISOLATED
Ray cluster (so it never collides with an Anyscale workspace's managed Ray).

Verified working in an Anyscale workspace (CPU): serves single + batched molecule
predictions from the 5-model ensemble on Ray 2.51.2 / Python 3.9.

Why this exists (vs `serve run serve_app:app`):
  Anyscale workspaces inject `RAY_RUNTIME_ENV_HOOK` / `RAY_RUNTIME_ENV_PLUGINS`
  (pointing at base-image-only plugins like `cgroup_runtime_plugin`) and a default
  `RAY_ADDRESS` for the managed cluster. To run the kMoL (2.51.2) app we clear those
  IN-PROCESS before importing ray, and start a fresh local cluster in its own temp
  dir. For a real deployment use a dedicated Service image instead (service.yaml).

Run:
    export PYTHONPATH="$PWD/stubs:$PWD:$PWD/kmol/src"
    export KMOL_CONFIG_PATH=configs/ensemble_serve.example.json KMOL_NUM_GPUS=0
    python scripts/make_synthetic_checkpoints.py $KMOL_CONFIG_PATH   # if no real weights
    python scripts/serve_local.py
    # then: curl -s -X POST localhost:8000/ -d '{"smiles":"CCO"}'
"""

import os

# Clear Anyscale-injected runtime-env hooks/plugins BEFORE importing ray.
for _k in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS", "RAY_ADDRESS"):
    os.environ.pop(_k, None)

import time

import ray
from ray import serve

from src.kmol_ensemble import build_app

NUM_CPUS = int(os.environ.get("KMOL_LOCAL_NUM_CPUS", "6"))
KEEPALIVE_S = int(os.environ.get("KMOL_LOCAL_KEEPALIVE_S", "600"))

ray.init(
    _temp_dir="/tmp/kmolray_local",
    include_dashboard=False,
    num_cpus=NUM_CPUS,
    num_gpus=0,
    logging_level="warning",
)
print("RAY_INIT_OK", ray.__version__, flush=True)

serve.run(build_app(os.environ["KMOL_CONFIG_PATH"]), blocking=False)
print("SERVE_UP — POST molecules to http://localhost:8000/", flush=True)

time.sleep(KEEPALIVE_S)
