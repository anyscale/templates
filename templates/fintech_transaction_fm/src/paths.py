"""Resolve a base directory for demo artifacts.

In an Anyscale Workspace we prefer shared cluster storage so every worker sees
the same paths. On a fresh Anyscale Job (no workspace mount) or locally we fall
back to a directory under the current working dir.
"""

import os

# Scale -> number of *cards* sampled. Derived from configs/<scale>.yaml (the
# single source of truth for per-scale settings); kept here because the README
# walkthrough imports it.
def _scale_map() -> dict:
    from .scale_config import load_scales

    return {name: cfg["data"]["num_cards"] for name, cfg in load_scales().items()}


SCALE_MAP = _scale_map()


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


def tensorboard_root(base_dir: str | None) -> str | None:
    """TensorBoard events live in the run's artifact base: ``<base>/tensorboard``.

    They travel with the model artifacts, and each base-dir keeps its own set
    (so a fresh base-dir doesn't mix its curves with a prior run's). Durable
    when base_dir is on /mnt/user_storage; for a job writing to ephemeral
    /mnt/cluster_storage, sync_tensorboard_events copies them to S3.
    Point `tensorboard --logdir <base>/tensorboard` at a run.
    """
    return os.path.join(base_dir, "tensorboard") if base_dir else None


def artifact_paths(base_dir: str, scale: str) -> dict:
    """Canonical artifact locations for a given scale."""
    return {
        "source": f"{base_dir}/source/",  # downloaded real-data cache (scale-independent)
        "raw": f"{base_dir}/raw/{scale}/transactions.parquet",
        "splits": f"{base_dir}/raw/{scale}/splits.json",
        "tokenized_pretrain": f"{base_dir}/tokenized/{scale}/pretrain/",
        "tokenized_eval": f"{base_dir}/tokenized/{scale}/eval/",
        "vocab": f"{base_dir}/tokenized/{scale}/vocab.json",
        "checkpoint": f"{base_dir}/model/{scale}/",
        "embeddings": f"{base_dir}/embeddings/{scale}/",
        "downstream": f"{base_dir}/downstream/{scale}/",
    }


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
