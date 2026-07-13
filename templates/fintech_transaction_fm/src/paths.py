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


def artifact_paths(base_dir: str, scale: str) -> dict:
    """Canonical artifact locations for a given scale."""
    return {
        "source": f"{base_dir}/source/",  # downloaded real-data cache (scale-independent)
        # one-time CSV → seq-tagged parquet shards, the working format (scale-independent):
        "source_parquet": f"{base_dir}/source_parquet/",
        "raw": f"{base_dir}/raw/{scale}/transactions.parquet",
        "splits": f"{base_dir}/raw/{scale}/splits.json",
        # NVIDIA-faithful temporal split regenerated from the CSV (src/nvsplit.py):
        # native TabFormer columns → {train,val_eval,test_eval}.parquet for the
        # NVIDIA-tokenizer pipeline (nb 03/05/06).
        "nvsplit": f"{base_dir}/nvsplit/{scale}/",
        # NVIDIA-tokenizer pretrain corpus (src/nvcorpus.py): ids.npy/attn.npy/vocab.json
        "nvcorpus": f"{base_dir}/nvcorpus/{scale}/",
        "tokenized_pretrain": f"{base_dir}/tokenized/{scale}/pretrain/",
        "tokenized_eval": f"{base_dir}/tokenized/{scale}/eval/",
        "vocab": f"{base_dir}/tokenized/{scale}/vocab.json",
        "checkpoint": f"{base_dir}/model/{scale}/",
        # our pretrained decoder exported as a HuggingFace dir, for the embedder (nb 05)
        "hf": f"{base_dir}/model_hf/{scale}/",
        "embeddings": f"{base_dir}/embeddings/{scale}/",
        "downstream": f"{base_dir}/downstream/{scale}/",
        # Part 7 supervised fine-tune (beyond-blueprint extension): labeled token sets
        # + fine-tuned model checkpoints per variant.
        "finetune": f"{base_dir}/finetune/{scale}/",
    }


def _latest_mtime(path: str) -> float:
    """Newest modification time under `path` (a file or a directory tree).

    Artifacts here are Parquet *directories* written shard-by-shard, so a
    directory's own mtime is unreliable — walk it and take the newest file.
    Returns -1.0 if the path does not exist.
    """
    if not os.path.exists(path):
        return -1.0
    if os.path.isfile(path):
        return os.path.getmtime(path)
    newest = os.path.getmtime(path)
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return newest


def stale_or_missing(output: str, inputs) -> bool:
    """True if `output` must be rebuilt: it's absent, or older than any input.

    Lets a notebook's skip-guard be *content-aware* instead of existence-only —
    so a freshly regenerated upstream (e.g. a retrained model) can never be
    silently ignored by a cached downstream artifact. `inputs` is a path or a
    list of paths; inputs that don't exist are ignored (they can't be newer).
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    out_mtime = _latest_mtime(output)
    if out_mtime < 0:
        return True  # missing -> build
    newest_input = max((_latest_mtime(p) for p in inputs), default=-1.0)
    return newest_input > out_mtime


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
