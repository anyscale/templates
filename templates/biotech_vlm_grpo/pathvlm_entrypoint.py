"""
Custom SkyRL entrypoint that registers the `nct_crc` environment.

Modeled on examples/train/geometry3k/geometry3k_entrypoint.py, with one
deliberate difference: geometry3k lives *inside* the SkyRL repo, so it can
register its env with a dotted import path. This template lives in a separate
repo and stays there -- nothing is ever copied into the SkyRL checkout.

Why that needs care. SkyRL's entrypoint is a Ray task, and on this cluster it
always runs on a *worker* (the head node is configured with CPU=0 schedulable,
so a num_cpus=1 task cannot be placed there). Ray's uv integration uploads
os.getcwd() as the runtime working_dir and validates that pyproject.toml lives
inside it, which pins the launch directory to the SkyRL repo root -- so the
worker's sys.path never contains this template directory. Verified on the
worker: `import env` raises ModuleNotFoundError.

The fix is to stop depending on the worker importing anything. Registering
`register_pickle_by_value` on this module makes cloudpickle serialise the env
class *by value* into the task closure, so the definition travels with the task
instead of being looked up by name. Verified end to end on the worker: the
by-value class instantiates and scores correctly while `import env` still fails.
"""

import multiprocessing as mp
import os
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

# This file is executed by path from the SkyRL repo root, so its own directory
# is not on sys.path automatically.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import env as _env_module  # noqa: E402

# Must happen before any task is submitted.
ray.cloudpickle.register_pickle_by_value(_env_module)

from env import NctCrcEnv  # noqa: E402

mp.set_start_method("spawn", force=True)

# SkyRL's prepare_runtime_environment() forwards only a hardcoded allowlist into
# ray.init's runtime_env -- there is no general passthrough -- so exporting these
# in the launching shell does not reach the worker that actually trains.
# TENSORBOARD_DIR fails silently when missing: the tracker falls back to a
# relative path inside Ray's ephemeral runtime directory and every metric is lost.
FORWARDED_ENV_VARS = ("TENSORBOARD_DIR", "HF_TOKEN", "HF_HOME")


@ray.remote(num_cpus=1)
def pathvlm_entrypoint(cfg: SkyRLTrainConfig, forwarded_env: dict):
    os.environ.update(forwarded_env)

    # The class object, not a dotted path -- it arrives by value with this task.
    register(id="nct_crc", entry_point=NctCrcEnv)

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    forwarded_env = {k: os.environ[k] for k in FORWARDED_ENV_VARS if os.environ.get(k)}
    ray.get(pathvlm_entrypoint.remote(cfg, forwarded_env))


if __name__ == "__main__":
    main()
