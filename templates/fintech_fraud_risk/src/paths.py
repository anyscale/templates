"""Demo data locations: shared cluster storage (Ray workers) vs local project dir."""

import os

_CLUSTER_MOUNT = "/mnt/shared_storage"
_CLUSTER_DEMO = os.path.join(_CLUSTER_MOUNT, "fintech-demo")


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_demo_base_dir() -> str:
    """Return a base path all Ray workers can read.

    On Anyscale, ``/mnt/shared_storage`` is accessible by any node in the Anyscale Cloud. Writing generated
    Parquet only under the job's extracted ``working_dir`` (e.g. ``demo_data/``)
    leaves data on the head node, so distributed ``read_parquet`` fails on workers.

    Fallback ``<project>/demo_data`` is for local runs without cluster storage.
    """
    override = os.environ.get("FINTECH_DEMO_BASE")
    if override:
        return override
    if os.path.isdir(_CLUSTER_MOUNT):
        os.makedirs(_CLUSTER_DEMO, exist_ok=True)
        return _CLUSTER_DEMO
    return os.path.join(get_project_root(), "demo_data")
