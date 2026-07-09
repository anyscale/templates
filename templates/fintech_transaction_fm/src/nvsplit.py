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
stratified each), plus a ``split_meta.json``.

Two implementations of the same protocol:

* :func:`build_temporal_split` — NVIDIA's original shape: ONE cuDF task on ONE GPU.
  Kept verbatim as the reference implementation.
* :func:`build_temporal_split_distributed` — the same split as a Ray Data pipeline on
  CPU workers: distributed CSV parse + date derivation + filtering + train write; the
  two seeded, order-sensitive steps (cutoff selection over ~7K daily counts, and the
  100K stratified eval sampling) run exactly the reference code on collected data.
  ``preserve_order=True`` keeps row order deterministic (CSV order), which the identity
  check against the reference output requires. Train is written as a DIRECTORY of
  parquet shards; use :func:`train_parquet_files` to read it in order.
"""
import json
import os
import re
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

    Reference implementation (NVIDIA's single-GPU shape). ``max_users`` caps the number
    of card-holders (mini/CI); ``None`` = every user (full). ``eval_samples`` sizes the
    stratified val/test eval subsets.
    """
    ray.init(ignore_reinit_error=True)
    meta = ray.get(_build.remote(csv_path, out_dir, eval_samples, max_users, seed))
    _wait_for_files([os.path.join(out_dir, f) for f in
                     ("train.parquet", "val_eval.parquet", "test_eval.parquet", "split_meta.json")])
    return meta


# ---------------------------------------------------------------------------
# Ray Data implementation — same protocol, CPU workers, no GPU anywhere.
# ---------------------------------------------------------------------------

def normalize_batch(batch):
    """Per-batch mirror of the reference preamble: strip column names, derive 'date'."""
    import pandas as pd
    batch.columns = [c.strip() for c in batch.columns]
    date_str = (batch["Year"].astype(str) + "-"
                + batch["Month"].astype(str).str.zfill(2) + "-"
                + batch["Day"].astype(str).str.zfill(2))
    batch["date"] = pd.to_datetime(date_str, format="%Y-%m-%d")
    return batch


def cutoff_dates(daily_counts):
    """The reference ``find_cutoff_date`` on the collected ~7K daily counts: first date
    where the cumulative transaction count crosses 80% / 90% of the total."""
    cnt_col = [c for c in daily_counts.columns if c != "date"][0]
    daily = daily_counts.sort_values("date").reset_index(drop=True)
    cum = daily[cnt_col].cumsum()
    total = int(cum.iloc[-1])
    return (daily.loc[cum >= total * 0.8, "date"].iloc[0],
            daily.loc[cum >= total * 0.9, "date"].iloc[0])


def ordered_parquet_files(path: str):
    """Files of a Ray-Data-written parquet directory in written (row) order — sorted by
    the integer components of the filenames. A single-file path returns ``[path]``."""
    if os.path.isfile(path):
        return [path]
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    files.sort(key=lambda f: [int(x) for x in re.findall(r"\d+", f)] or [0])
    return [os.path.join(path, f) for f in files]


@ray.remote(num_cpus=4)
def _stratified_eval(rows_dir: str, out_path: str, eval_samples: int, seed: int) -> dict:
    """The reference eval sampling, verbatim, on the collected val/test subset (~2.4M
    rows). Seeded + order-sensitive, so it runs exactly once on exactly-ordered data."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    pdf = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(rows_dir)],
                    ignore_index=True)
    y = (pdf["Is Fraud?"].astype(str).str.lower() == "yes").astype(int)
    if eval_samples >= len(pdf):
        sub = pdf.reset_index(drop=True)
    else:
        _, keep_idx = train_test_split(
            np.arange(len(pdf)), test_size=eval_samples, stratify=y, random_state=seed)
        sub = pdf.iloc[keep_idx].reset_index(drop=True)
    sub.to_parquet(out_path, index=False)
    yf = (sub["Is Fraud?"].astype(str).str.lower() == "yes").astype(int)
    return {"rows": int(len(sub)), "fraud": int(yf.sum()), "fraud_rate": float(yf.mean())}


@ray.remote(num_cpus=2)
def _fraud_count(rows_dir: str) -> dict:
    """Count rows/frauds of a written split by streaming just the label column."""
    import pyarrow.parquet as pq
    rows = fraud = 0
    for f in ordered_parquet_files(rows_dir):
        col = pq.read_table(f, columns=["Is Fraud?"]).column(0).to_pylist()
        rows += len(col)
        fraud += sum(1 for v in col if str(v).lower() == "yes")
    return {"rows": rows, "fraud": fraud, "fraud_rate": fraud / max(rows, 1)}


def build_temporal_split_distributed(csv_path: str, out_dir: str,
                                     eval_samples: int = 100_000, max_users=None,
                                     seed: int = 42) -> dict:
    """NVIDIA's temporal split as a Ray Data pipeline on CPU workers.

    Same protocol and (verified) same output as :func:`build_temporal_split`; the
    difference is execution: the 24M-row CSV parse, date derivation, filtering and the
    19.5M-row train write are distributed, and no stage touches a GPU. Train is written
    to ``<out_dir>/train_parquet/`` as ordered shards (``ordered_parquet_files`` /
    ``train_parquet_files`` read it back in row order).
    """
    import shutil

    import pandas as pd
    import ray.data

    ray.init(ignore_reinit_error=True)
    # Row order IS the contract here: the seeded eval sampling and the identity check
    # against the reference output both depend on CSV order surviving the pipeline.
    ray.data.DataContext.get_current().execution_options.preserve_order = True

    os.makedirs(out_dir, exist_ok=True)
    norm_dir = os.path.join(out_dir, "_normalized_tmp")
    dirs = {"train": os.path.join(out_dir, "train_parquet"),
            "val": os.path.join(out_dir, "_val_rows_tmp"),
            "test": os.path.join(out_dir, "_test_rows_tmp")}
    for d in (norm_dir, *dirs.values()):
        if os.path.isdir(d):
            shutil.rmtree(d)

    # Parse the CSV once, distributed; land it as parquet so the three split filters
    # below re-read columnar data instead of re-parsing 2.3GB of CSV per pass.
    ds = ray.data.read_csv(csv_path).map_batches(normalize_batch, batch_format="pandas")
    if max_users is not None:
        users = sorted(ds.unique("User"))
        if len(users) > max_users:
            rng = np.random.RandomState(seed)
            keep = set(rng.choice(np.array(users), size=max_users, replace=False).tolist())
            ds = ds.map_batches(lambda b: b[b["User"].isin(keep)], batch_format="pandas")
    ds.write_parquet(norm_dir)

    ds = ray.data.read_parquet(ordered_parquet_files(norm_dir))
    train_cutoff, test_cutoff = cutoff_dates(ds.groupby("date").count().to_pandas())
    tr_cut, te_cut = pd.Timestamp(train_cutoff), pd.Timestamp(test_cutoff)

    parts = {
        "train": lambda b: b[b["date"] < tr_cut].drop(columns=["date"]),
        "val": lambda b: b[(b["date"] >= tr_cut) & (b["date"] < te_cut)].drop(columns=["date"]),
        "test": lambda b: b[b["date"] >= te_cut].drop(columns=["date"]),
    }
    for name, fn in parts.items():
        ray.data.read_parquet(ordered_parquet_files(norm_dir)) \
            .map_batches(fn, batch_format="pandas").write_parquet(dirs[name])

    val_stats, test_stats, train_stats = ray.get([
        _stratified_eval.remote(dirs["val"], os.path.join(out_dir, "val_eval.parquet"),
                                eval_samples, seed),
        _stratified_eval.remote(dirs["test"], os.path.join(out_dir, "test_eval.parquet"),
                                eval_samples, seed),
        _fraud_count.remote(dirs["train"]),
    ])
    shutil.rmtree(norm_dir)
    shutil.rmtree(dirs["val"])
    shutil.rmtree(dirs["test"])

    meta = {
        "source": "tabformer_csv",
        "protocol": "NVIDIA NB01: 80/10/10 temporal by cumulative daily count + 100K stratified eval",
        "impl": "ray-data-cpu",
        "train_cutoff": str(train_cutoff)[:10],
        "test_cutoff": str(test_cutoff)[:10],
        "eval_samples": int(eval_samples),
        "max_users": (int(max_users) if max_users is not None else None),
        "seed": int(seed),
        "train": train_stats,
        "val_eval": val_stats,
        "test_eval": test_stats,
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    _wait_for_files([os.path.join(out_dir, "val_eval.parquet"),
                     os.path.join(out_dir, "test_eval.parquet")])
    return meta


def train_parquet_files(split_dir_or_file: str):
    """Ordered parquet file list for a train split — accepts the reference single file
    (``train.parquet``) or the distributed directory (``train_parquet/``)."""
    single = os.path.join(split_dir_or_file, "train.parquet")
    sharded = os.path.join(split_dir_or_file, "train_parquet")
    if os.path.isfile(split_dir_or_file):
        return [split_dir_or_file]
    if os.path.isdir(sharded):
        return ordered_parquet_files(sharded)
    if os.path.isfile(single):
        return [single]
    raise FileNotFoundError(f"no train split under {split_dir_or_file}")
