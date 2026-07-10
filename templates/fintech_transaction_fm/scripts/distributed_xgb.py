"""Scale-out fusion variant: distributed XGBoost on the raw FM embedding.

The single-node embed_xgb path (src/benchmark_downstream.py) assembles the
full embedding matrix in driver memory — at fulltest scale a 3.55M x 512
float32 matrix plus working copies, which is why job_fulltest.yaml has to
specify a 128GB head node. This variant removes that wall WITHOUT touching
the pinned protocol path:

* Ray Data streams the embeddings parquet; each Ray Train worker materializes
  only ITS shard into an ``xgboost.DMatrix`` (peak memory scales 1/workers).
* ``xgboost.train`` runs data-parallel (Ray Train wires the collective);
  same XGB_PARAMS_EMBED, same seed, same early stopping on val AUC.
* Test scoring streams through Ray Data ``map_batches`` — the driver only
  ever holds the (y, proba) vectors.

The single-node path stays the benchmark of record: distributed hist XGBoost
does not grow bit-identical trees, so this ships as the scale-out variant.
"Measures properly" = its test AP should land inside the single-node run's
bootstrap 95% CI (the comparison is printed at the end against
``downstream/<run>/{benchmark_metrics,bootstrap_ci}.json``).

    # self-contained smoke (synthetic data, planted signal, any cluster):
    python scripts/distributed_xgb.py --smoke
    # the real thing (xl = seq-1024 model, full-test-period embeddings):
    python scripts/distributed_xgb.py --base-dir $BASE --run xl_fulltest --num-workers 4
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nvidia_baseline import XGB_PARAMS_EMBED  # noqa: E402

REQUIRED_COLS = ("embedding", "label", "split")


def native_params(sklearn_params: dict) -> tuple[dict, int, int]:
    """XGB_PARAMS_EMBED is sklearn-API shaped; xgboost.train wants native keys.

    Single-sources the pinned hyperparameters instead of duplicating them.
    """
    p = dict(sklearn_params)
    num_boost_round = p.pop("n_estimators")
    early_stopping_rounds = p.pop("early_stopping_rounds")
    p["eta"] = p.pop("learning_rate")
    p["alpha"] = p.pop("reg_alpha")
    p["lambda"] = p.pop("reg_lambda")
    p["seed"] = p.pop("random_state")
    p["objective"] = "binary:logistic"
    return p, num_boost_round, early_stopping_rounds


def _as_matrix(col) -> np.ndarray:
    """Embedding column batch -> (n, dim) float32, tensor-ext or object dtype."""
    arr = np.asarray(col)
    if arr.dtype == object:
        arr = np.stack(list(arr))
    return arr.astype(np.float32, copy=False)


def train_func(config: dict):
    import xgboost as xgb

    import ray.train
    from ray.train.xgboost import RayTrainReportCallback

    def shard_matrix(name: str):
        Xs, ys = [], []
        for b in ray.train.get_dataset_shard(name).iter_batches(
            batch_size=65_536, batch_format="numpy"
        ):
            Xs.append(_as_matrix(b["embedding"]))
            ys.append(np.asarray(b["label"]).astype(np.int64))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        return xgb.DMatrix(X, label=y), len(y), int(y.sum())

    dtrain, n_tr, f_tr = shard_matrix("train")
    dval, n_va, f_va = shard_matrix("val")
    rank = ray.train.get_context().get_world_rank()
    print(f"[dist-xgb] worker {rank}: train {n_tr:,} rows ({f_tr:,} fraud), "
          f"val {n_va:,} rows ({f_va:,} fraud)")

    xgb.train(
        config["params"],
        dtrain,
        num_boost_round=config["num_boost_round"],
        evals=[(dval, "val")],
        early_stopping_rounds=config["early_stopping_rounds"],
        verbose_eval=50,
        callbacks=[RayTrainReportCallback()],
    )


class Scorer:
    """Ray Data callable: loads the booster once per replica, scores batches."""

    def __init__(self, model_path: str, best_iteration: int | None):
        import xgboost as xgb

        self.bst = xgb.Booster()
        self.bst.load_model(model_path)
        self.it_range = (0, best_iteration + 1) if best_iteration is not None else None

    def __call__(self, batch: dict) -> dict:
        X = _as_matrix(batch["embedding"])
        kw = {"iteration_range": self.it_range} if self.it_range else {}
        return {
            "y": np.asarray(batch["label"]).astype(np.int64),
            "proba": self.bst.inplace_predict(X, **kw).astype(np.float32),
        }


def make_smoke_parquet(path: str, n: int = 40_000, dim: int = 32, seed: int = 0) -> None:
    """Synthetic embeddings with a planted linear signal — smoke the REAL
    entry point end-to-end (train -> early stop -> streamed scoring)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    logits = X[:, :4] @ np.array([2.0, -1.5, 1.0, 0.5]) - 2.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    split = np.where(np.arange(n) < 0.7 * n, "train",
                     np.where(np.arange(n) < 0.85 * n, "val", "test"))
    pd.DataFrame({"embedding": list(X), "label": y, "split": split}).to_parquet(
        path, index=False
    )
    print(f"[dist-xgb] smoke data: {n:,} rows, dim {dim}, "
          f"fraud rate {y.mean():.3%} -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=None)
    p.add_argument("--run", default="xl_fulltest",
                   help="embeddings/<run> to train on; compare vs downstream/<run>")
    p.add_argument("--embeddings-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpus-per-worker", type=int, default=4)
    p.add_argument("--smoke", action="store_true",
                   help="self-contained synthetic run; no artifacts needed")
    args = p.parse_args()

    import ray
    from ray.data.expressions import col
    from ray.train import RunConfig, ScalingConfig
    from ray.train.xgboost import RayTrainReportCallback, XGBoostTrainer

    ray.init(ignore_reinit_error=True)

    if args.smoke:
        import tempfile

        work = tempfile.mkdtemp(prefix="distxgb_smoke_")
        emb_path = os.path.join(work, "embeddings.parquet")
        make_smoke_parquet(emb_path)
        out_dir = args.output_dir or os.path.join(work, "out")
        num_workers = min(args.num_workers, 2)
        storage_path = os.path.join(work, "ray_results")
    else:
        if not args.base_dir:
            p.error("--base-dir is required without --smoke")
        emb_path = args.embeddings_path or f"{args.base_dir}/embeddings/{args.run}"
        out_dir = args.output_dir or f"{args.base_dir}/downstream/{args.run}_distxgb"
        num_workers = args.num_workers
        storage_path = os.path.join(args.base_dir, "ray_results")

    ds = ray.data.read_parquet(emb_path)
    missing = [c for c in REQUIRED_COLS if c not in ds.schema().names]
    if missing:
        raise RuntimeError(
            f"{emb_path} is missing columns {missing} — expected tokenized-eval "
            f"embeddings (schema: {ds.schema().names})"
        )
    ds = ds.select_columns(list(REQUIRED_COLS))
    splits = {s: ds.filter(expr=col("split") == s).drop_columns(["split"])
              for s in ("train", "val", "test")}

    params, num_boost_round, early_stopping_rounds = native_params(XGB_PARAMS_EMBED)
    params["nthread"] = args.cpus_per_worker
    run_name = f"distxgb_{args.run}_{time.strftime('%Y%m%d-%H%M%S')}"
    trainer = XGBoostTrainer(
        train_func,
        train_loop_config={
            "params": params,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            resources_per_worker={"CPU": args.cpus_per_worker},
        ),
        datasets={"train": splits["train"], "val": splits["val"]},
        run_config=RunConfig(name=run_name, storage_path=storage_path),
    )
    result = trainer.fit()

    # Booster lands on shared storage via the checkpoint — readable by every
    # scoring replica. best_iteration is persisted as a model attribute when
    # early stopping fires; honor it exactly like the sklearn API does.
    import xgboost as xgb

    with result.checkpoint.as_directory() as ckpt_dir:
        model_path = os.path.join(ckpt_dir, RayTrainReportCallback.CHECKPOINT_NAME)
        bst = xgb.Booster()
        bst.load_model(model_path)
        best_attr = bst.attr("best_iteration")
        best_iteration = int(best_attr) if best_attr is not None else None
        os.makedirs(out_dir, exist_ok=True)
        final_model = os.path.join(out_dir, "model.ubj")
        bst.save_model(final_model)
    print(f"[dist-xgb] trained; best_iteration={best_iteration} -> {final_model}")

    scored = splits["test"].map_batches(
        Scorer,
        fn_constructor_kwargs={"model_path": final_model,
                               "best_iteration": best_iteration},
        batch_size=32_768,
        compute=ray.data.ActorPoolStrategy(size=num_workers),
        num_cpus=1,
    ).to_pandas()  # (y, proba) only — the driver never holds a feature matrix

    from sklearn.metrics import average_precision_score, roc_auc_score

    y, proba = scored["y"].to_numpy(), scored["proba"].to_numpy()
    metrics = {
        "auc_roc": float(roc_auc_score(y, proba)),
        "ap": float(average_precision_score(y, proba)),
        "n_test": int(len(y)),
        "n_test_fraud": int(y.sum()),
        "best_iteration": best_iteration,
        "num_workers": num_workers,
        "xgb_params": params,
        "embeddings_path": emb_path,
    }
    print(f"[dist-xgb] TEST: AP {metrics['ap']:.4f}  ROC-AUC {metrics['auc_roc']:.4f}  "
          f"({metrics['n_test']:,} rows, {metrics['n_test_fraud']:,} fraud)")

    if args.smoke:
        assert metrics["ap"] > 0.3, f"smoke AP {metrics['ap']:.4f} below planted signal"
        print(f"SMOKE_OK ap={metrics['ap']:.4f} auc={metrics['auc_roc']:.4f}")
        return

    # Side-by-side vs the single-node run of record on the same rows.
    single = {}
    bm_path = f"{args.base_dir}/downstream/{args.run}/benchmark_metrics.json"
    ci_path = f"{args.base_dir}/downstream/{args.run}/bootstrap_ci.json"
    if os.path.exists(bm_path):
        single = json.load(open(bm_path))["results"].get("embed_xgb", {})
        print(f"[dist-xgb] single-node embed_xgb: AP {single.get('ap'):.4f}  "
              f"ROC-AUC {single.get('auc_roc'):.4f}")
    if os.path.exists(ci_path):
        ci = json.load(open(ci_path))["results"]["embed_xgb"]["ap_ci95"]
        inside = ci[0] <= metrics["ap"] <= ci[1]
        metrics["single_node_ap_ci95"] = ci
        metrics["inside_single_node_ci"] = bool(inside)
        print(f"[dist-xgb] vs single-node 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]: "
              f"{'INSIDE — measures properly' if inside else 'OUTSIDE — investigate'}")
    metrics["single_node_embed_xgb"] = single

    with open(os.path.join(out_dir, "distributed_xgb_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # bootstrap_ci.py-compatible predictions (run with --scale <run>_distxgb).
    import pandas as pd

    pd.DataFrame({"y": y, "embed_xgb_dist": proba}).to_parquet(
        os.path.join(out_dir, "test_predictions.parquet"), index=False
    )
    print(f"[dist-xgb] metrics + test predictions -> {out_dir}")


if __name__ == "__main__":
    main()
