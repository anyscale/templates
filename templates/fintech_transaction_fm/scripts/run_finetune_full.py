"""Full-scale fine-tune experiment (extension candidate #5 / Part 7), headless.

Runs both variants end-to-end at full scale and writes every result to
/mnt/cluster_storage/transaction-fm/finetune/full/RESULTS.json as it goes:

    single-txn tokens -> finetune -> score (alone + fused with raw)
    history windows   -> finetune -> score (alone + fused with raw)

Stage outputs are cached the same way the notebook caches them (delete a stage's
output to re-run it). Comparison targets from Part 6 full: raw 0.1238,
embedding 0.04-0.06, fusion typical 0.136.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray


def main(scale: str):
    from src.finetune import (build_history_windows, build_single_txn_tokens,
                              finetune, score_finetuned)
    from src.paths import artifact_paths, get_demo_base_dir
    from src.scale_config import load_scale

    cfg = load_scale(scale)
    base = get_demo_base_dir()
    paths = artifact_paths(base, scale)
    ft = cfg["finetune"]
    FT = paths["finetune"]
    os.makedirs(FT, exist_ok=True)
    results_path = os.path.join(FT, "RESULTS.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    ray.init(ignore_reinit_error=True,
             runtime_env={"working_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          # torch's native JIT needs a C compiler workers don't have, and the
                          # decision is frozen at torch import — must be set process-level.
                          "env_vars": {"TORCH_DISABLE_NATIVE_JIT": "1"}})

    def save():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print("[ft] RESULTS:", json.dumps(results), flush=True)

    def stage(name, probe, fn):
        if os.path.exists(probe):
            print(f"[ft] {name}: cached", flush=True)
            return
        t0 = time.time()
        fn()
        results.setdefault("timings_s", {})[name] = round(time.time() - t0, 1)
        save()
        print(f"[ft] {name}: done in {results['timings_s'][name]}s", flush=True)

    # ---- variant 1: single transaction --------------------------------------
    stage("tokens_single", os.path.join(FT, "ft_ids_test.npy"),
          lambda: build_single_txn_tokens(paths["nvsplit"], FT,
                                          balanced_train=cfg["embed"]["balanced_train"],
                                          max_length=ft["max_length"]))

    sft = os.path.join(FT, "model_single")
    stage("finetune_single", os.path.join(sft, "model.pt"),
          lambda: results.update(single_meta=finetune(
              hf_dir=paths["hf"], tokens_dir=FT, out_dir=sft, variant="single",
              epochs=ft["epochs"], batch_size=ft["batch_size"], lr=ft["lr"],
              num_workers=ft["num_workers"], use_gpu=ft["use_gpu"],
              storage_base=os.path.join(base, "transaction-fm"))))

    if "single_scores" not in results:
        results["single_scores"] = score_finetuned(paths["hf"], sft, FT, paths["embeddings"],
                                                   variant="single", use_gpu=ft["use_gpu"])
        save()

    # ---- variant 2: history window ------------------------------------------
    stage("tokens_history", os.path.join(FT, "ft_hist_ids_test.npy"),
          lambda: build_history_windows(paths["nvsplit"], paths["source_parquet"], FT,
                                        balanced_train=cfg["embed"]["balanced_train"],
                                        window_txns=ft["window_txns"],
                                        window_tokens=ft["window_tokens"]))

    hist = os.path.join(FT, "model_history")
    stage("finetune_history", os.path.join(hist, "model.pt"),
          lambda: results.update(history_meta=finetune(
              hf_dir=paths["hf"], tokens_dir=FT, out_dir=hist, variant="history",
              epochs=ft["epochs"], batch_size=ft["batch_size"], lr=ft["lr"],
              num_workers=ft["num_workers"], use_gpu=ft["use_gpu"],
              storage_base=os.path.join(base, "transaction-fm"))))

    if "history_scores" not in results:
        results["history_scores"] = score_finetuned(paths["hf"], hist, FT, paths["embeddings"],
                                                    variant="history", use_gpu=ft["use_gpu"])
        save()

    print("[ft] COMPLETE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="full", choices=["mini", "small", "full"])
    main(ap.parse_args().scale)
