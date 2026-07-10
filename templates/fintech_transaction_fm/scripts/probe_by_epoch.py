"""Probe fraud + reco quality as a function of pretraining epoch.

Motivation (TensorBoard, small-scale runs): the fraud-relevant heads plateau
by mid-training while the merchant InfoNCE head keeps improving — so the whole
second half of pretraining optimizes the shared encoder for merchant identity
alone, plausibly rotating the embedding away from what fraud needs (fm-AUC
0.83 -> 0.78 vs the pre-InfoNCE architecture). This script turns that
hypothesis into a curve: for every saved epoch checkpoint, re-extract
embeddings and score fraud (raw / fm / fusion) plus — on the learned-vocab
path — next-merchant recommendation. If the story is right, fm-AUC peaks at
an early/mid epoch while reco keeps climbing; the fix is then two extraction
points from one run (early checkpoint for fraud, final for reco), not a
different model.

Run after run_pipeline.py at the same scale (needs its raw data + vocab and
the per-epoch checkpoints under <base>/ray_results/<run>/):

    python scripts/probe_by_epoch.py --scale small            # every epoch
    python scripts/probe_by_epoch.py --scale small --every 2  # faster pass

Writes per-epoch metrics under <base>/probe/<scale>/, a summary JSON, and
probe/* scalars into the training run's TensorBoard dir (x-axis = epoch) so
the trade shows up next to that run's training curves.

Caveat at `full`: the eval set (~5M windows at holdout_keep 1.0) is cached in
the object store once and reused across epochs; pass --no-cache-eval if it
doesn't fit, at the cost of re-running the tokenize shuffle per epoch.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray  # noqa: E402
import torch  # noqa: E402

from src.paths import artifact_paths, get_demo_base_dir  # noqa: E402
from src.scale_config import add_scale_args, load_scale  # noqa: E402


def find_run(base: str, run_name: str | None) -> str:
    """Resolve the pretraining run dir (latest timestamped one by default)."""
    root = os.path.join(base, "ray_results")
    if run_name:
        return os.path.join(root, run_name)
    runs = sorted(
        d for d in os.listdir(root) if d.startswith("transaction_fm_pretrain")
    )
    if not runs:
        raise SystemExit(f"no pretraining runs under {root}")
    return os.path.join(root, runs[-1])


def epoch_checkpoints(run_dir: str) -> list:
    """(epoch, checkpoint_dir) for every saved checkpoint, sorted by epoch."""
    out = []
    for d in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, d)
        state_path = os.path.join(path, "train_state.pt")
        if not (d.startswith("checkpoint") and os.path.isfile(state_path)):
            continue
        state = torch.load(state_path, map_location="cpu")
        out.append((int(state["epoch"]), path))
    if not out:
        raise SystemExit(
            f"no epoch checkpoints in {run_dir} — the probe needs "
            "CheckpointConfig(num_to_keep=None) (the current default)"
        )
    return sorted(out)


def main():
    p = argparse.ArgumentParser()
    add_scale_args(p)
    p.add_argument("--run-name", default=None, help="ray_results run dir (default: latest)")
    p.add_argument("--every", type=int, default=1, help="probe every Nth epoch (last always included)")
    p.add_argument("--no-cache-eval", action="store_true",
                   help="re-tokenize per epoch instead of caching the eval set in the object store")
    args = p.parse_args()

    cfg = load_scale(args.scale, args.scale_config)
    base = get_demo_base_dir()
    paths = artifact_paths(base, args.scale)
    run_dir = find_run(base, args.run_name)
    ckpts = epoch_checkpoints(run_dir)
    picked = ckpts[:: args.every]
    if picked[-1] != ckpts[-1]:
        picked.append(ckpts[-1])
    print(f"[probe] {os.path.basename(run_dir)}: probing epochs {[e for e, _ in picked]}")

    with open(paths["vocab"]) as f:
        vocab = json.load(f)
    do_reco = bool(vocab.get("infonce_fields"))
    with open(paths["splits"]) as f:
        splits = json.load(f)

    ray.init(ignore_reinit_error=True)
    from ray.data.expressions import col

    from src.tokenizer import eval_normal_keep, tokenize_dataset

    tk = cfg["tokenize"]

    def eval_windows():
        return (
            tokenize_dataset(
                ray.data.read_parquet(paths["raw"]),
                tk["seq_len"],
                train_end=splits["train_end"],
                val_end=splits["val_end"],
                normal_keep=eval_normal_keep(splits, tk["target_eval_samples"]),
                holdout_keep=tk["holdout_keep"],
                max_pretrain_windows=tk["max_pretrain_windows"],
                num_partitions=tk["shuffle_partitions"],
                emit="eval",
                merchant_vocab=vocab.get("merchant_vocab"),
            )
            .filter(expr=col("kind") == "eval")
            .drop_columns(["kind"])
        )

    ev = None if args.no_cache_eval else eval_windows().materialize()

    from src.downstream import run_downstream
    from src.embed import extract_embeddings

    e = cfg["embed"]
    rows = []
    for epoch, ck in picked:
        out_dir = os.path.join(base, "probe", args.scale, f"epoch_{epoch:03d}")
        emb_dir = os.path.join(out_dir, "embeddings")
        shutil.rmtree(emb_dir, ignore_errors=True)  # write_parquet appends
        print(f"=== [probe] epoch {epoch}: embed + score ===", flush=True)
        extract_embeddings(
            ds=ev if ev is not None else eval_windows(),
            checkpoint_dir=ck,
            output_path=emb_dir,
            num_workers=e["num_workers"],
            use_gpu=e["use_gpu"],
            batch_size=e["batch_size"],
            gpus_per_worker=e.get("gpus_per_worker"),
        )
        summary = run_downstream(emb_dir, out_dir, raw_path=paths["raw"])
        row = {"epoch": epoch}
        for name, r in summary["results"].items():
            row[f"{name}_auc_roc"] = r["auc_roc"]
            row[f"{name}_pr_auc"] = r["pr_auc"]

        if do_reco:
            from src.recommend import run_recommend

            rec = run_recommend(
                checkpoint_dir=ck,
                output_dir=out_dir,
                ds=ev if ev is not None else eval_windows(),
                num_workers=e["num_workers"],
                use_gpu=e["use_gpu"],
                batch_size=e["batch_size"],
                gpus_per_worker=e.get("gpus_per_worker"),
            )
            row["reco_hr10"] = rec["model"]["hr@10"]
            novel = rec["by_target"]["novel"]
            row["reco_novel_hr10"] = novel["model"]["hr@10"] if novel["n"] else float("nan")
        rows.append(row)

    # Persist + mirror into the run's TensorBoard dir (same run name, so the
    # probe curves land next to the training curves, x-axis = epoch).
    probe_json = os.path.join(base, "probe", args.scale, "probe_by_epoch.json")
    with open(probe_json, "w") as f:
        json.dump({"run": os.path.basename(run_dir), "rows": rows}, f, indent=2)
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(os.path.join(base, "tensorboard", os.path.basename(run_dir)))
        for row in rows:
            for k, v in row.items():
                if k != "epoch" and v == v:  # skip NaN
                    writer.add_scalar(f"probe/{k}", v, row["epoch"])
        writer.close()
    except ImportError:
        pass

    cols = [k for k in rows[0] if k != "epoch"]
    print(f"\n{'epoch':>5} " + " ".join(f"{c:>15}" for c in cols))
    for row in rows:
        print(f"{row['epoch']:>5} " + " ".join(f"{row[c]:>15.4f}" for c in cols))
    print(f"\n[probe] per-epoch metrics -> {probe_json}")
    print("[probe] raw_* should be flat across epochs (same features + deterministic "
          "stage 5) — drift there means nondeterminism, not model change.")


if __name__ == "__main__":
    main()
