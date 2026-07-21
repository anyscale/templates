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
  CPU workers: date derivation, filtering and the 19.5M-row train write distributed;
  the seeded, order-sensitive steps (cutoff selection over ~7K daily counts, the 100K
  stratified eval sampling) run exactly the reference code on collected data. Row order
  is carried EXPLICITLY via a ``__seq__`` column baked into the one-time CSV→parquet
  conversion (a streaming engine does not guarantee block placement — measured), and
  order-sensitive consumers sort by it. Train is a DIRECTORY of parquet shards; read it
  with :func:`train_parquet_files`. Output verified identical to the reference
  (scripts/verify_distributed_split.py, 2026-07-09).
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
#
# Ordering design: identity with the reference depends on CSV row order, but a
# streaming engine does not guarantee which blocks land in which output file
# (measured: preserve_order kept the stream ordered yet write tasks bundled
# non-contiguous blocks). So order is carried EXPLICITLY: the one-time CSV →
# parquet conversion tags every row with ``__seq__`` (its CSV position), every
# distributed pass lets blocks land wherever, and the few consumers that need
# exact order sort by ``__seq__``.
# ---------------------------------------------------------------------------

SEQ = "__seq__"


@ray.remote(num_cpus=4)
def _csv_to_shards(csv_path: str, out_dir: str, rows_per_shard: int) -> dict:
    """One-time streaming CSV → parquet-shards conversion (single sequential task, so
    ``__seq__`` is trivially the CSV row position). Pins ``Time`` (and the other string
    columns cuDF infers as strings) so no reader re-infers them differently."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    convert = pacsv.ConvertOptions(
        column_types={
            "Time": pa.string(), "Amount": pa.string(), "Use Chip": pa.string(),
            "Merchant City": pa.string(), "Merchant State": pa.string(),
            "Errors?": pa.string(), "Is Fraud?": pa.string(), "Zip": pa.float64()},
        # cuDF nulls empty CSV fields; Arrow defaults them to "" for strings. Identity
        # (verified): "" is the ONLY null token in this file — restrict to exactly that.
        null_values=[""], strings_can_be_null=True)
    os.makedirs(out_dir, exist_ok=True)
    reader = pacsv.open_csv(csv_path, convert_options=convert)
    seq = shard = 0
    buf = []
    buf_rows = 0

    def _flush():
        nonlocal shard, buf, buf_rows
        if not buf:
            return
        tbl = pa.concat_tables(buf)
        pq.write_table(tbl, os.path.join(out_dir, f"shard_{shard:05d}.parquet"))
        shard += 1
        buf, buf_rows = [], 0

    for batch in reader:
        tbl = pa.Table.from_batches([batch])
        tbl = tbl.rename_columns([c.strip() for c in tbl.column_names])
        tbl = tbl.append_column(SEQ, pa.array(np.arange(seq, seq + len(tbl)), pa.int64()))
        seq += len(tbl)
        buf.append(tbl)
        buf_rows += len(tbl)
        if buf_rows >= rows_per_shard:
            _flush()
    _flush()
    return {"rows": seq, "shards": shard}


def ensure_parquet_shards(csv_path: str, shards_dir: str,
                          rows_per_shard: int = 1_000_000) -> dict:
    """Convert the raw CSV to seq-tagged parquet shards once; reuse thereafter."""
    ray.init(ignore_reinit_error=True)
    marker = os.path.join(shards_dir, "_conversion_meta.json")
    if os.path.exists(marker):
        return json.load(open(marker))
    meta = ray.get(_csv_to_shards.remote(csv_path, shards_dir, rows_per_shard))
    with open(marker, "w") as f:
        json.dump(meta, f)
    return meta


def normalize_date_column(batch):
    """Per-batch mirror of the reference preamble: derive the 'date' column."""
    import pandas as pd
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
def stratified_eval(rows_dir: str, out_path: str, eval_samples: int, seed: int) -> dict:
    """The reference eval sampling, verbatim, on the collected val/test subset (~2.4M
    rows). Seeded + order-sensitive, so rows are restored to CSV order via ``__seq__``
    first; the sample then matches the reference selection exactly."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    pdf = pd.concat([pd.read_parquet(f) for f in ordered_parquet_files(rows_dir)],
                    ignore_index=True)
    pdf = pdf.sort_values(SEQ, kind="mergesort").drop(columns=[SEQ]).reset_index(drop=True)
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
def fraud_count(rows_dir: str) -> dict:
    """Count rows/frauds of a written split by streaming just the label column."""
    import pyarrow.parquet as pq
    rows = fraud = 0
    for f in ordered_parquet_files(rows_dir):
        col = pq.read_table(f, columns=["Is Fraud?"]).column(0).to_pylist()
        rows += len(col)
        fraud += sum(1 for v in col if str(v).lower() == "yes")
    return {"rows": rows, "fraud": fraud, "fraud_rate": fraud / max(rows, 1)}


def fresh_part_dirs(out_dir: str) -> dict:
    """Empty output dirs for one split build: the persistent ``train_parquet/`` plus the
    two temporary val/test row dirs that :func:`finalize_split` samples from and removes."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    dirs = {"train": os.path.join(out_dir, "train_parquet"),
            "val": os.path.join(out_dir, "_val_rows_tmp"),
            "test": os.path.join(out_dir, "_test_rows_tmp")}
    for d in dirs.values():
        if os.path.isdir(d):
            shutil.rmtree(d)
    return dirs


def part_filters(train_cutoff, test_cutoff) -> dict:
    """The three temporal-split filters (per-batch), keyed by part name."""
    import pandas as pd
    tr_cut, te_cut = pd.Timestamp(train_cutoff), pd.Timestamp(test_cutoff)
    return {
        "train": lambda b: b[b["date"] < tr_cut].drop(columns=["date"]),
        "val": lambda b: b[(b["date"] >= tr_cut) & (b["date"] < te_cut)].drop(columns=["date"]),
        "test": lambda b: b[b["date"] >= te_cut].drop(columns=["date"]),
    }


def load_normalized(shards_dir: str, max_users=None):
    """The seq-tagged source shards as a Ray dataset with the 'date' column derived.
    ``max_users`` (mini/CI) keeps users ``< max_users`` — deterministic subset."""
    import ray.data
    ds = ray.data.read_parquet(ordered_parquet_files(shards_dir)) \
                 .map_batches(normalize_date_column, batch_format="pandas")
    if max_users is not None:
        ds = ds.map_batches(lambda b: b[b["User"] < max_users], batch_format="pandas")
    return ds


def finalize_split(out_dir: str, dirs: dict, train_cutoff, test_cutoff,
                   eval_samples: int, max_users=None, seed: int = 42) -> dict:
    """After the three part writes: draw the seeded 100K stratified eval samples, count
    train frauds, remove the temp row dirs, and write ``split_meta.json``."""
    import shutil

    val_stats, test_stats, train_stats = ray.get([
        stratified_eval.remote(dirs["val"], os.path.join(out_dir, "val_eval.parquet"),
                               eval_samples, seed),
        stratified_eval.remote(dirs["test"], os.path.join(out_dir, "test_eval.parquet"),
                               eval_samples, seed),
        fraud_count.remote(dirs["train"]),
    ])
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


def build_temporal_split_distributed(csv_path: str, out_dir: str,
                                     eval_samples: int = 100_000, max_users=None,
                                     seed: int = 42) -> dict:
    """NVIDIA's temporal split as a Ray Data pipeline on CPU workers.

    Same protocol and (verified) same output as :func:`build_temporal_split`; the
    difference is execution: date derivation, filtering and the 19.5M-row train write
    run distributed on CPU workers, and no stage touches a GPU. Train lands at
    ``<out_dir>/train_parquet/`` as shards whose rows carry ``__seq__`` (CSV position) —
    consumers needing exact reference order sort by it.

    This is the headless composition of the same pieces Part 2 shows inline:
    ``ensure_parquet_shards → load_normalized → cutoff_dates → part_filters → finalize_split``.
    """
    ray.init(ignore_reinit_error=True)

    shards_dir = os.path.join(os.path.dirname(os.path.dirname(csv_path.rstrip("/"))),
                              "source_parquet")
    ensure_parquet_shards(csv_path, shards_dir)

    dirs = fresh_part_dirs(out_dir)
    train_cutoff, test_cutoff = cutoff_dates(
        load_normalized(shards_dir, max_users).groupby("date").count().to_pandas())
    for name, fn in part_filters(train_cutoff, test_cutoff).items():
        load_normalized(shards_dir, max_users) \
            .map_batches(fn, batch_format="pandas").write_parquet(dirs[name])
    return finalize_split(out_dir, dirs, train_cutoff, test_cutoff,
                          eval_samples, max_users, seed)


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
