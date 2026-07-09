"""Run the full transaction-FM pipeline end to end, headless — the Anyscale Job
entrypoint (Part 8, see job_config.yaml).

Composes the SAME stage functions the notebooks (Parts 2-6) show inline, in the same
order, with the same skip-guards — so the walkthrough and the job cannot drift:

    ensure_download + ensure_parquet_shards          (one-time CSV -> seq-tagged shards)
    build_temporal_split_distributed                 (Part 2 — Ray Data, CPU workers)
    build_corpus_distributed                         (Part 3 — per-card map_groups, CPU)
    TorchTrainer(train_func) + save/export           (Part 4 — Ray Train, N x GPU)
    embed_splits_distributed                         (Part 5 — CPU tokenize -> GPU actors)
    run_downstream                                   (Part 6 — NVIDIA NB05 recipe, GPU)

Each stage is skipped when its output already exists (delete a stage's output dir, or
pass --force to rebuild everything). Nothing is wiped implicitly.

Usage:
    python scripts/run_pipeline.py                    # mini (CPU-only, minutes)
    python scripts/run_pipeline.py --scale full       # the real thing (GPUs, ~3h)
    python scripts/run_pipeline.py --force            # ignore cached stage outputs
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import ray


def main(scale: str, force: bool) -> None:
    from src.paths import artifact_paths, get_demo_base_dir
    from src.scale_config import load_scale

    cfg = load_scale(scale)
    base = get_demo_base_dir()
    paths = artifact_paths(base, scale)
    ray.init(ignore_reinit_error=True,
             runtime_env={"working_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))})
    timings = {}

    def stage(name, output_probe, fn):
        if not force and os.path.exists(output_probe):
            print(f"[pipeline] {name}: cached ({output_probe})", flush=True)
            return
        t0 = time.time()
        fn()
        timings[name] = round(time.time() - t0, 1)
        print(f"[pipeline] {name}: done in {timings[name]}s", flush=True)

    # ── Part 2: temporal split (Ray Data, CPU workers) ──────────────────────
    def _split():
        from src.nvsplit import build_temporal_split_distributed
        from src.tabformer import ensure_download
        csv_path = ensure_download(paths["source"])
        meta = build_temporal_split_distributed(
            csv_path, paths["nvsplit"],
            eval_samples=cfg["data"]["eval_samples"],
            max_users=cfg["data"]["max_users"])
        print(json.dumps(meta, indent=2), flush=True)

    stage("split", os.path.join(paths["nvsplit"], "split_meta.json"), _split)

    # ── Part 3: pretrain corpus (per-card map_groups, CPU workers) ──────────
    def _corpus():
        from src.nvcorpus import build_corpus_distributed
        tk = cfg["tokenize"]
        meta = build_corpus_distributed(
            paths["nvsplit"], paths["nvcorpus"], seq_len=tk["seq_len"],
            chunk=max(1, tk["seq_len"] // 13), max_seq=tk.get("max_pretrain_windows"))
        print(json.dumps(meta, indent=2), flush=True)

    stage("corpus", os.path.join(paths["nvcorpus"], "ids.npy"), _corpus)

    # ── Part 4: pretrain (Ray Train) + HF export ────────────────────────────
    def _pretrain():
        import torch
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        from src.model import build_model
        from src.pretrain import save_checkpoint, train_func

        pt = cfg["pretrain"]
        seq_len = cfg["tokenize"]["seq_len"]
        ids = np.load(os.path.join(paths["nvcorpus"], "ids.npy"))
        attn = np.load(os.path.join(paths["nvcorpus"], "attn.npy"))
        train_ds = (ray.data.from_numpy(ids).rename_columns({"data": "input_ids"})
                    .zip(ray.data.from_numpy(attn).rename_columns({"data": "attention_mask"})))
        steps_per_epoch = max(1, math.ceil(len(ids) / pt["num_workers"] / pt["batch_size"]))
        total_steps = steps_per_epoch * pt["epochs"]
        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "vocab_path": os.path.join(paths["nvcorpus"], "vocab.json"),
                "arch": cfg["model"], "size": scale, "max_len": seq_len,
                "epochs": pt["epochs"], "batch_size": pt["batch_size"], "lr": pt["lr"],
                "use_fsdp": pt["use_fsdp"], "seed": 0,
                "weight_decay": pt.get("weight_decay", 0.0),
                "betas": tuple(pt.get("betas", (0.9, 0.999))),
                "lr_schedule": pt.get("lr_schedule"), "total_steps": total_steps,
                "warmup_steps": int(pt.get("warmup_ratio", 0.0) * total_steps),
                "min_lr_ratio": pt.get("min_lr_ratio", 0.0),
            },
            scaling_config=ScalingConfig(num_workers=pt["num_workers"], use_gpu=pt["use_gpu"]),
            datasets={"train": train_ds},
            run_config=RunConfig(name=f"transaction_fm_pretrain_{scale}",
                                 storage_path=os.path.join(base, "ray_results")),
        )
        result = trainer.fit()
        save_checkpoint(result, paths["checkpoint"])
        print(f"[pipeline] pretrain: lm_loss {result.metrics['lm_loss']:.3f} "
              f"perplexity {result.metrics['perplexity']:.1f}", flush=True)

        # export the inner LlamaForCausalLM as a HF dir for the embedder (Part 4's last cell)
        with open(os.path.join(paths["checkpoint"], "model_config.json")) as f:
            mc = json.load(f)
        m = build_model(vocab_path=os.path.join(paths["checkpoint"], "vocab.json"),
                        arch=mc["arch"], max_len=mc["max_len"])
        sd = torch.load(os.path.join(paths["checkpoint"], "model.pt"), map_location="cpu")
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        m.load_state_dict(sd, strict=False)
        os.makedirs(paths["hf"], exist_ok=True)
        m.lm.save_pretrained(paths["hf"])

    stage("pretrain", os.path.join(paths["hf"], "config.json"), _pretrain)

    # ── Part 5: embeddings (streaming CPU -> GPU actors) ────────────────────
    def _embed():
        from src.nvembed import embed_splits_distributed
        eb = cfg["embed"]
        meta = embed_splits_distributed(
            paths["hf"], paths["nvsplit"], paths["embeddings"],
            balanced_train=eb["balanced_train"],
            max_length=eb.get("embed_max_length", 128),
            batch_size=eb["batch_size"],
            num_gpu_workers=eb["num_workers"], use_gpu=eb["use_gpu"],
            embed_dim=cfg["model"]["d_model"])
        print(json.dumps(meta, indent=2), flush=True)

    stage("embed", os.path.join(paths["embeddings"], "embed_test.npy"), _embed)

    # ── Part 6: downstream raw vs fm vs fusion (NVIDIA NB05 recipe) ─────────
    def _downstream():
        from src.nvscore import print_summary, run_downstream
        ds_cfg = cfg["downstream"]
        summary = run_downstream(paths["embeddings"], paths["downstream"],
                                 pca_dim=ds_cfg["pca_dim"], use_gpu=ds_cfg["use_gpu"])
        print_summary(summary)

    stage("downstream", os.path.join(paths["downstream"], "downstream_metrics.json"),
          _downstream)

    print(f"[pipeline] complete — stage timings: {json.dumps(timings)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="mini", choices=["mini", "small", "full"])
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage even if its output exists")
    args = ap.parse_args()
    main(args.scale, args.force)
