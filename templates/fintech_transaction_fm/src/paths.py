"""Resolve a base directory for demo artifacts.

In an Anyscale Workspace we prefer shared cluster storage so every worker sees
the same paths. On a fresh Anyscale Job (no workspace mount) or locally we fall
back to a directory under the current working dir.
"""

import os

# Scale -> number of *cards* (each card produces a variable-length sequence of
# transactions). Kept small so `smoke` runs in CI on CPU in a couple minutes.
SCALE_MAP = {
    "smoke": 2_000,      # ~60K transactions  — CI / CPU sanity
    "small": 20_000,     # ~600K transactions — single GPU
    "medium": 200_000,   # ~6M transactions   — multi-GPU distributed story
}


def get_demo_base_dir() -> str:
    """Return a writable base dir for all artifacts, creating it if needed."""
    candidates = [
        "/mnt/cluster_storage/transaction-fm",  # workspace shared storage
        os.path.join(os.getcwd(), "demo_data"),  # local / job fallback
    ]
    for path in candidates:
        parent = os.path.dirname(path.rstrip("/"))
        if os.path.isdir(parent) or parent == "":
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except OSError:
                continue
    # Last resort: cwd.
    path = os.path.join(os.getcwd(), "demo_data")
    os.makedirs(path, exist_ok=True)
    return path


def artifact_paths(base_dir: str, scale: str) -> dict:
    """Canonical artifact locations for a given scale."""
    return {
        "raw": f"{base_dir}/raw/{scale}/transactions.parquet",
        "tokenized": f"{base_dir}/tokenized/{scale}/",
        "vocab": f"{base_dir}/tokenized/{scale}/vocab.json",
        "checkpoint": f"{base_dir}/model/{scale}/",
        "embeddings": f"{base_dir}/embeddings/{scale}/",
        "downstream": f"{base_dir}/downstream/{scale}/",
    }
