"""Downstream fraud classification — the headline result, trained on the cluster.

Evaluation protocol matches NVIDIA's transaction-FM blueprint on TabFormer so
the numbers are directly comparable when run on the real data:

* **Temporal 80/10/10 split** by transaction time (cutoffs from splits.json —
  the same ones the tokenizer used). Train on the past, early-stop on val,
  report on the most recent 10%. No temporal leakage.
* **Per-transaction, last-event labels**: each sample is one target transaction
  scored from the window of history ending at it.
* **Metrics: AUC-ROC and PR-AUC (Average Precision)** — at ~0.1% fraud, AUC-ROC
  saturates and PR-AUC is the operationally meaningful number (NVIDIA frames it
  the same way).

We compare three feature sets with the SAME XGBoost recipe so the only variable
is the representation:

1. ``raw``    — the target transaction's tabular fields (the "what you have today" baseline)
2. ``fm``     — the FM embedding of the history window only (no raw fields at all)
3. ``fusion`` — embedding concatenated with raw fields (Nubank's joint fusion)

The lift of (2) and (3) over (1) is the story: a pretrained transaction FM lets
you drop or augment a hand-tuned feature pipeline.

**Why this stage is a Ray job.** At `small`/`full` the embeddings are millions of
rows × hundreds of dims — too big to pull onto one node and fit with in-process
XGBoost (that was minutes-to-tens-of-minutes and pinned to one CPU box). The
training distributes with ``ray.train.xgboost.XGBoostTrainer``: Ray Data shards
the Parquet across workers, each worker builds a ``DMatrix`` from its shard, and
XGBoost's collective communicator does the distributed boosting. Scaling from a
laptop-CPU smoke test to a multi-GPU fit is one ``ScalingConfig`` change — the
same knob the pretrain and embed stages turn.

Distributed boosting partitions the data across workers, so with
``num_workers > 1`` the histograms (and thus the exact metric values) differ
slightly from a single-process fit — reproducible *given the same worker count*,
not bit-identical across counts. The eval fingerprint below is unaffected: it
hashes which rows are scored, not how the model was trained.
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_SPLITS = ("train", "val", "test")

# The four raw target-transaction features, after the same log-amount transform
# the in-process baseline used. These are the ``raw`` feature set and the
# tabular half of ``fusion``.
RAW_FEATURE_COLS = ("f_log_amount", "raw_hour", "raw_dow", "raw_mcc")
# Columns carried through training that are NOT model features.
_NON_FEATURE_COLS = ("label", "weight", "split")


def expand_features(batch):
    """Ray Data UDF: explode the embedding list into one column per dimension.

    XGBoost needs a flat numeric matrix, but the embeddings land as a single
    ``embedding`` list column. We fan it out to ``emb_0..emb_{d-1}`` and apply
    the same ``sign·log1p(|amount|)`` transform the tabular baseline used, so all
    three feature sets are just column subsets of one expanded table. Runs as a
    ``map_batches`` so the fan-out happens distributed, never on the driver.
    """
    import pandas as pd

    emb = np.asarray(batch["embedding"].tolist(), dtype=np.float32)
    out = pd.DataFrame(
        emb, columns=[f"emb_{i}" for i in range(emb.shape[1])], index=batch.index
    )
    amt = batch["raw_amount"].to_numpy(np.float64)
    out["f_log_amount"] = (np.sign(amt) * np.log1p(np.abs(amt))).astype(np.float32)
    out["raw_hour"] = batch["raw_hour"].astype(np.float32)
    out["raw_dow"] = batch["raw_dow"].astype(np.float32)
    out["raw_mcc"] = batch["raw_mcc"].astype(np.float32)
    out["label"] = batch["label"].astype(np.int64)
    out["weight"] = batch["weight"].astype(np.float64)
    out["split"] = batch["split"].astype(str)
    return out


def feature_columns(name: str, all_cols) -> list:
    """Columns for one feature set, given the expanded table's column names."""
    emb = sorted(
        (c for c in all_cols if c.startswith("emb_")), key=lambda c: int(c[4:])
    )
    raw = list(RAW_FEATURE_COLS)
    return {"raw": raw, "fm": emb, "fusion": emb + raw}[name]


def xgb_params(scale_pos_weight: float, use_gpu: bool) -> dict:
    """The single shared XGBoost recipe — identical across all three feature sets
    so representation is the only variable. ``device='cuda'`` routes the
    histogram build to each worker's GPU; ``scale_pos_weight`` counteracts the
    fraud-rare class.
    """
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "seed": 0,  # pinned: subsampled fits at 0.1% prevalence vary a LOT
    }


def train_loop_per_worker(config: dict) -> None:
    """Runs on each Ray Train worker: build a DMatrix from this worker's shard
    and join XGBoost's distributed boosting round. ``RayTrainReportCallback``
    checkpoints each round so the driver can recover the best booster; XGBoost's
    collective (set up by ``XGBoostTrainer``) handles cross-worker gradients.
    """
    import logging

    import ray.train
    import xgboost as xgb
    from ray.train.xgboost import RayTrainReportCallback

    # Ray Train logs a line per boosting round at INFO from inside this worker
    # process; quiet it here (driver-side setLevel can't reach this process) so
    # the notebook output stays readable.
    logging.getLogger().setLevel(logging.WARNING)

    label = config["label_column"]

    def to_dmatrix(shard) -> "xgb.DMatrix":
        df = shard.materialize().to_pandas()
        y = df.pop(label)
        return xgb.DMatrix(df, label=y)

    dtrain = to_dmatrix(ray.train.get_dataset_shard("train"))
    dvalid = to_dmatrix(ray.train.get_dataset_shard("valid"))
    xgb.train(
        config["params"],
        dtrain,
        num_boost_round=config["num_boost_round"],
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=config["early_stopping_rounds"],
        callbacks=[RayTrainReportCallback()],
        verbose_eval=50,  # a metric line every 50 rounds: progress without the per-round flood
    )


def train_feature_set(
    train_ds, val_ds, feature_cols, scaling_config, scale_pos_weight, storage_path
):
    """Fit one feature set as a distributed XGBoost job; return the best booster.

    ``train_ds``/``val_ds`` are the temporal splits; we hand the trainer only the
    chosen feature columns + the label, so the same split datasets drive ``raw``,
    ``fm``, and ``fusion`` by column selection alone. ``storage_path`` must be a
    location every node can read/write (shared cluster storage) — workers on
    other nodes persist the final checkpoint there for the driver to load.
    """
    from ray.train import RunConfig
    from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

    cols = list(feature_cols) + ["label"]
    trainer = XGBoostTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "label_column": "label",
            "params": xgb_params(scale_pos_weight, scaling_config.use_gpu),
            "num_boost_round": 400,
            "early_stopping_rounds": 30,
        },
        scaling_config=scaling_config,
        run_config=RunConfig(
            name="transaction_fm_downstream", storage_path=storage_path
        ),
        datasets={
            "train": train_ds.select_columns(cols),
            "valid": val_ds.select_columns(cols),
        },
    )
    result = trainer.fit()
    return RayTrainReportCallback.get_model(result.checkpoint)


def evaluate(test_df, feature_cols, booster) -> tuple:
    """Score the held-out test split on the driver and compute weighted metrics.

    The test split is small (the most-recent 10%), so we pull it to the driver
    and predict in-process rather than spin up a distributed inference pass — the
    distributed work that matters is the training above. Weighted metrics undo
    the normal-downsampling: they estimate performance at the natural fraud
    prevalence (what NVIDIA's blueprint reports), not on the fraud-enriched
    sample we kept for compute reasons.
    """
    import xgboost as xgb

    booster.set_param({"device": "cpu"})
    proba = booster.predict(xgb.DMatrix(test_df[list(feature_cols)]))
    y = test_df["label"].to_numpy()
    w = test_df["weight"].to_numpy()
    metrics = {
        "auc_roc": float(roc_auc_score(y, proba, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, proba, sample_weight=w)),
        "pr_auc_sampled": float(average_precision_score(y, proba)),
    }
    return metrics, proba


def _eval_fingerprint(embeddings_path: str) -> str:
    """Order-independent hash of eval-set membership (card_id, ts, split, label).

    Metrics from two runs are comparable iff their fingerprints match.
    Eval sampling is deterministic (per-card seeded RNG in the tokenizer), so
    this changes only when the raw data or the sampling knobs change — never
    with model/training changes.
    """
    import hashlib

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    t = pads.dataset(embeddings_path, format="parquet").to_table(
        columns=["card_id", "raw_ts", "split", "label"]
    )
    cols = [
        t.column("card_id").to_numpy(zero_copy_only=False).astype(np.int64),
        t.column("raw_ts").to_numpy(zero_copy_only=False).astype(np.int64),
        pc.index_in(t.column("split"), value_set=pa.array(_SPLITS))
        .to_numpy(zero_copy_only=False)
        .astype(np.int64),
        t.column("label").to_numpy(zero_copy_only=False).astype(np.int64),
    ]
    order = np.lexsort(cols[::-1])
    return hashlib.sha256(np.column_stack(cols)[order].tobytes()).hexdigest()[:16]


def run_downstream(
    embeddings_path: str,
    output_dir: str,
    num_workers: int = 1,
    use_gpu: bool = False,
) -> dict:
    """Train + evaluate all three feature sets on the cluster; persist a summary.

    This is the headless composition the ``05_train_downstream`` job runs; the
    notebook composes the SAME helpers (``expand_features``, ``train_feature_set``,
    ``evaluate``) inline so the two can't drift.
    """
    import time

    import ray
    from ray.train import ScalingConfig

    ds = ray.data.read_parquet(embeddings_path).map_batches(
        expand_features, batch_format="pandas"
    )
    splits = {s: ds.filter(expr=f"split == '{s}'").materialize() for s in _SPLITS}
    for s, sds in splits.items():
        if sds.count() == 0:
            raise RuntimeError(
                f"split '{s}' is empty — re-run 01/02 so splits.json temporal "
                "cutoffs are written and applied during tokenization"
            )

    # Driver-side stats: label/weight per split are tiny next to the embeddings,
    # and scale_pos_weight + the summary need them anyway.
    meta = splits["train"].select_columns(["label", "weight"]).to_pandas()
    pos = float(meta["label"].sum())
    neg = float(len(meta) - pos)
    scaling = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    # Ray Train persists each fit's checkpoint here; must be reachable by every
    # node, so it lives under the (shared) output dir, not the head node's disk.
    storage_path = os.path.join(os.path.abspath(output_dir), "ray_results")
    os.makedirs(storage_path, exist_ok=True)

    test_df = splits["test"].to_pandas()
    all_cols = ds.schema().names
    print(
        f"[05] loaded  train={len(meta):,}  val={splits['val'].count():,}  "
        f"test={len(test_df):,}  (emb_dim={len(feature_columns('fm', all_cols))}); "
        f"fitting num_workers={num_workers} use_gpu={use_gpu}",
        flush=True,
    )
    results, test_proba = {}, {}
    for name in ("raw", "fm", "fusion"):
        cols = feature_columns(name, all_cols)
        print(f"[05] training '{name}' ({len(cols)} features) ...", flush=True)
        t0 = time.time()
        booster = train_feature_set(
            splits["train"], splits["val"], cols, scaling,
            neg / max(pos, 1.0), storage_path,
        )
        results[name], test_proba[name] = evaluate(test_df, cols, booster)
        print(
            f"[05]   '{name}' done in {time.time() - t0:>4.0f}s  "
            f"AUC-ROC={results[name]['auc_roc']:.4f}  PR-AUC={results[name]['pr_auc']:.4f}",
            flush=True,
        )

    stats = {
        s: splits[s].select_columns(["label", "weight"]).to_pandas() for s in _SPLITS
    }
    summary = {
        "protocol": (
            "temporal 80/10/10 split by transaction time; per-transaction "
            "last-event fraud labels; prevalence-weighted metrics on the "
            "held-out most-recent 10% (NVIDIA transaction-FM blueprint protocol)"
        ),
        "eval_fingerprint": _eval_fingerprint(embeddings_path),
        "scaling": {"num_workers": num_workers, "use_gpu": use_gpu},
        "n_samples": {s: int(len(d)) for s, d in stats.items()},
        "fraud_rate": {s: float(d["label"].mean()) for s, d in stats.items()},
        "natural_fraud_rate": {
            s: float((d["weight"] * d["label"]).sum() / d["weight"].sum())
            for s, d in stats.items()
        },
        "embedding_dim": len(feature_columns("fm", all_cols)),
        "results": results,
        "fm_lift_pr_auc": results["fm"]["pr_auc"] - results["raw"]["pr_auc"],
        "fusion_lift_pr_auc": results["fusion"]["pr_auc"] - results["raw"]["pr_auc"],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "downstream_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-sample test scores so ROC/PR curves can be rebuilt offline. Apply
    # `weight` when plotting to get natural-prevalence curves (same correction
    # as the metrics above).
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = list(test_proba)
    y_te = test_df["label"].to_numpy()
    w_te = test_df["weight"].to_numpy()
    pq.write_table(
        pa.table(
            {
                "feature_set": np.repeat(np.array(names), len(test_df)),
                "label": np.tile(y_te, len(names)),
                "proba": np.concatenate([test_proba[n] for n in names]),
                "weight": np.tile(w_te, len(names)),
            }
        ),
        os.path.join(output_dir, "test_predictions.parquet"),
    )
    print(f"[05] per-sample test scores -> {output_dir}/test_predictions.parquet")
    return summary


def print_summary(summary: dict) -> None:
    n = summary["n_samples"]
    nfr = summary["natural_fraud_rate"]
    sc = summary["scaling"]
    print(f"protocol: {summary['protocol']}")
    print(
        f"trained distributed: num_workers={sc['num_workers']} use_gpu={sc['use_gpu']}"
    )
    print(
        f"samples  train={n['train']:,}  val={n['val']:,}  test={n['test']:,} "
        f"(natural test fraud rate {nfr['test']:.4%})"
    )
    print(
        f"eval fingerprint: {summary['eval_fingerprint']} "
        "(metrics comparable across runs iff this matches)"
    )
    print(f"{'feature set':<10} {'AUC-ROC':>10} {'PR-AUC':>10}")
    print("-" * 32)
    for name, r in summary["results"].items():
        print(f"{name:<10} {r['auc_roc']:>10.4f} {r['pr_auc']:>10.4f}")
    print("-" * 32)
    print(f"FM-only PR-AUC lift vs raw:  {summary['fm_lift_pr_auc']:+.4f}")
    print(f"Fusion PR-AUC lift vs raw:   {summary['fusion_lift_pr_auc']:+.4f}")
