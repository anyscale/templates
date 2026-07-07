"""Regenerate NVIDIA's temporal train/val/test split from the raw TabFormer CSV.

This mirrors NVIDIA's NB01 (``01_dataset_baseline.ipynb``) exactly, so the notebook
series can build the split itself instead of depending on a precomputed
``temporal_split/`` artifact:

* **80/10/10 temporal split** by *cumulative daily transaction count* (test = most
  recent transactions → no leakage), via ``find_cutoff_date``.
* **100K stratified eval subsets** of val/test (preserving the natural fraud rate),
  via sklearn ``train_test_split(stratify=...)`` — NVIDIA's ``val_eval``/``test_eval``.

Output columns are NVIDIA's **native** TabFormer schema (``User, Card, Year, Month,
Day, Time, Amount, Use Chip, Merchant Name, Merchant City, Merchant State, Zip, MCC,
Errors?, Is Fraud?``) — exactly what ``src.nvidia_tokenizer`` consumes downstream.

Writes ``train.parquet`` (full temporal train — feeds the pretrain corpus + the
balanced downstream train), ``val_eval.parquet`` and ``test_eval.parquet`` (100K
stratified each), plus a ``split_meta.json``. cuDF/GPU job (matches NVIDIA's NB01).
"""
import json
import os
import time

import numpy as np
import ray


def _wait_for_files(file_paths, timeout: float = 300.0) -> None:
    """Block until each path is visible on the caller's node — /mnt/cluster_storage is
    NFS/EFS-backed, so a worker's write can lag a driver read by a fraction of a second."""
    for p in file_paths:
        t0 = time.time()
        while not os.path.exists(p):
            if time.time() - t0 > timeout:
                raise TimeoutError(f"output not visible after {timeout}s: {p}")
            time.sleep(0.5)

# NVIDIA's 13 raw feature columns (NB01 FEATURE_COLS) — Hour is derived from Time.
FEATURE_COLS = [
    "User", "Card", "Year", "Month", "Day", "Hour", "Amount", "Use Chip",
    "Merchant Name", "Merchant City", "Merchant State", "Zip", "MCC",
]


@ray.remote(num_gpus=1, num_cpus=8)
def _build(csv_path: str, out_dir: str, eval_samples: int, max_users, seed: int) -> dict:
    import cudf
    import numpy as np
    from sklearn.model_selection import train_test_split

    gdf = cudf.read_csv(csv_path)
    gdf.columns = [c.strip() for c in gdf.columns]

    # mini/CI: keep only a deterministic subset of users (each user = a card holder)
    if max_users is not None:
        users = gdf["User"].unique().to_pandas().to_numpy()
        if len(users) > max_users:
            rng = np.random.RandomState(seed)
            keep = rng.choice(users, size=max_users, replace=False)
            gdf = gdf[gdf["User"].isin(cudf.Series(keep))].reset_index(drop=True)

    # date column for temporal splitting (NB01 cell 12)
    ys = gdf["Year"].astype(str)
    ms = gdf["Month"].astype(str).str.zfill(2)
    ds = gdf["Day"].astype(str).str.zfill(2)
    gdf["date"] = cudf.to_datetime(ys + "-" + ms + "-" + ds, format="%Y-%m-%d")

    # 80/10/10 cutoff dates by cumulative daily count (NB01 find_cutoff_date)
    daily = gdf.groupby("date").size().reset_index(name="count").sort_values("date")
    daily["cum"] = daily["count"].cumsum()
    total = int(daily["cum"].iloc[-1])

    def _cutoff(ratio: float):
        hit = daily[daily["cum"] >= total * ratio].head(1).to_pandas()
        return hit["date"].iloc[0]

    train_cutoff = _cutoff(0.8)
    test_cutoff = _cutoff(0.9)

    tr = gdf[gdf["date"] < np.datetime64(train_cutoff)].drop(columns=["date"]).reset_index(drop=True)
    va = gdf[(gdf["date"] >= np.datetime64(train_cutoff)) & (gdf["date"] < np.datetime64(test_cutoff))].drop(columns=["date"]).reset_index(drop=True)
    te = gdf[gdf["date"] >= np.datetime64(test_cutoff)].drop(columns=["date"]).reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    tr.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)

    def _fraud(pdf):
        return (pdf["Is Fraud?"].astype(str).str.lower() == "yes").astype(int)

    def _strat(df_gpu, name: str):
        pdf = df_gpu.to_pandas()
        y = _fraud(pdf)
        if eval_samples >= len(pdf):
            sub = pdf.reset_index(drop=True)
        else:
            _, keep_idx = train_test_split(
                np.arange(len(pdf)), test_size=eval_samples, stratify=y, random_state=seed)
            sub = pdf.iloc[keep_idx].reset_index(drop=True)
        sub.to_parquet(os.path.join(out_dir, f"{name}.parquet"), index=False)
        yf = _fraud(sub)
        return {"rows": int(len(sub)), "fraud": int(yf.sum()), "fraud_rate": float(yf.mean())}

    val_stats = _strat(va, "val_eval")
    test_stats = _strat(te, "test_eval")

    tr_f = _fraud(tr.to_pandas())
    meta = {
        "source": "tabformer_csv",
        "protocol": "NVIDIA NB01: 80/10/10 temporal by cumulative daily count + 100K stratified eval",
        "train_cutoff": str(train_cutoff)[:10],
        "test_cutoff": str(test_cutoff)[:10],
        "eval_samples": int(eval_samples),
        "max_users": (int(max_users) if max_users is not None else None),
        "seed": int(seed),
        "train": {"rows": int(len(tr)), "fraud": int(tr_f.sum()), "fraud_rate": float(tr_f.mean())},
        "val_eval": val_stats,
        "test_eval": test_stats,
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_temporal_split(csv_path: str, out_dir: str, eval_samples: int = 100_000,
                         max_users=None, seed: int = 42) -> dict:
    """Regenerate NVIDIA's temporal split from ``csv_path`` into ``out_dir`` (GPU task).

    ``max_users`` caps the number of card-holders (mini/CI); ``None`` = every user
    (full). ``eval_samples`` sizes the stratified val/test eval subsets.
    """
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_build.remote(csv_path, out_dir, eval_samples, max_users, seed))
    _wait_for_files([os.path.join(out_dir, f) for f in
                     ("train.parquet", "val_eval.parquet", "test_eval.parquet", "split_meta.json")])
    return meta
