"""Resolve base directories for demo artifacts.

Two storage tiers, both visible to every worker:

* **ephemeral** (``/mnt/cluster_storage``) — fast, shared across the cluster's
  nodes, but wiped when the cluster is torn down. Regenerated intermediates
  (tokenized windows, checkpoint, embeddings, downstream metrics) live here.
* **persistent** (``/mnt/user_storage``) — survives cluster teardown, so a new
  cluster reuses the 266MB TabFormer download cache and the per-scale raw
  Parquet instead of re-downloading + re-normalizing. The durable inputs
  (``source``/``raw``/``splits``) live here.

On a fresh Anyscale Job (no workspace mount) or locally both tiers fall back to
a directory under the current working dir.
"""

import os

# Scale -> number of *cards* sampled. Derived from configs/<scale>.yaml (the
# single source of truth for per-scale settings); kept here because the README
# walkthrough imports it.
def _scale_map() -> dict:
    from .scale_config import load_scales

    return {name: cfg["data"]["num_cards"] for name, cfg in load_scales().items()}


SCALE_MAP = _scale_map()


def _first_writable(candidates: list) -> str:
    """Return the first candidate whose parent exists and that we can create."""
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


def get_demo_base_dir() -> str:
    """Ephemeral, cluster-shared base for regenerated intermediates."""
    return _first_writable([
        "/mnt/cluster_storage/transaction-fm",  # workspace shared storage (per-cluster)
        os.path.join(os.getcwd(), "demo_data"),  # local / job fallback
    ])


def get_persistent_base_dir() -> str:
    """Cross-cluster persistent base for the download cache + raw inputs.

    ``/mnt/user_storage`` survives cluster teardown, so a new cluster reuses the
    TabFormer download and per-scale raw Parquet instead of re-fetching them.
    """
    return _first_writable([
        "/mnt/user_storage/transaction-fm",  # persists across clusters
        os.path.join(os.getcwd(), "demo_data"),  # local / job fallback
    ])


# Keys whose outputs live on persistent storage and must NOT be cleaned between
# runs — they are the cross-cluster cache the pipeline skips re-creating.
PERSISTENT_KEYS = ("source", "raw", "splits")


def artifact_paths(base_dir: str, scale: str, persistent_dir: str | None = None) -> dict:
    """Canonical artifact locations for a given scale.

    ``persistent_dir`` holds the durable inputs (``source``/``raw``/``splits``);
    it defaults to ``base_dir`` so existing single-tier callers are unchanged.
    """
    p = persistent_dir or base_dir
    return {
        "source": f"{p}/source/",  # downloaded real-data cache (scale-independent)
        "raw": f"{p}/raw/{scale}/transactions.parquet",
        "splits": f"{p}/raw/{scale}/splits.json",
        "tokenized_pretrain": f"{base_dir}/tokenized/{scale}/pretrain/",
        "tokenized_eval": f"{base_dir}/tokenized/{scale}/eval/",
        "vocab": f"{base_dir}/tokenized/{scale}/vocab.json",
        "checkpoint": f"{base_dir}/model/{scale}/",
        "embeddings": f"{base_dir}/embeddings/{scale}/",
        "downstream": f"{base_dir}/downstream/{scale}/",
    }


def resolve_artifact_paths(scale: str, base_dir: str | None = None) -> dict:
    """Resolve paths with the two-tier split applied.

    With no override, intermediates land on ephemeral cluster storage and the
    durable inputs on persistent user storage. An explicit ``base_dir`` (local
    runs / tests) keeps every artifact under that one directory.
    """
    if base_dir:
        return artifact_paths(base_dir, scale, persistent_dir=base_dir)
    return artifact_paths(get_demo_base_dir(), scale, persistent_dir=get_persistent_base_dir())


def write_splits_meta(out_path: str, timestamps, is_fraud, source: str, n_cards: int) -> dict:
    """Persist temporal split cutoffs + dataset stats next to the raw Parquet.

    Cutoffs follow the NVIDIA transaction-FM blueprint: 80/10/10 by transaction
    time, so the test set holds the most recent transactions (no temporal
    leakage). The tokenizer and downstream stages read this file so every stage
    agrees on the same split.
    """
    import json

    import numpy as np

    ts = np.asarray(timestamps).astype("datetime64[s]").astype(np.int64)
    q80, q90 = np.quantile(ts, [0.8, 0.9])
    fraud = np.asarray(is_fraud).astype(np.int64)
    meta = {
        "protocol": "temporal 80/10/10 by transaction time; last-event fraud label",
        "train_end": str(np.datetime64(int(q80), "s")),
        "val_end": str(np.datetime64(int(q90), "s")),
        "n_transactions": int(len(ts)),
        "n_cards": int(n_cards),
        "fraud_rate": float(fraud.mean()),
        "source": source,
    }
    os.makedirs(os.path.dirname(out_path.rstrip("/")), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
