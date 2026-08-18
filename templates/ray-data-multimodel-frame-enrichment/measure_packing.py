#!/usr/bin/env python3
"""Measure whether co-residency actually beats running the stages serially.

    HF_TOKEN=hf_... python measure_packing.py --frames 96 --runs 3
    HF_TOKEN=hf_... python measure_packing.py --out arms.jsonl   # every run, for re-scoring

`packing.py` computes whether four models FIT on one card. This measures whether packing
them was worth doing. It exists because the README quotes a number -- co-residency ahead by
>=26.5% on one L4 -- and a quoted number with no runnable path back to it is how a claim
outlives the thing it was measured on. Re-run this after any change to the model set, the
image, the batch sizes or the card.

NEEDS: one GPU, the gated weights, and about 20 minutes. This is the opposite of rung 1.

TWO ARMS

  coresident  the DAG `pipeline.build()` produces: four stages, fractional GPU reservations
              summing to <1, all resident on one card together.
  serial      the same four stages, each given the WHOLE GPU and run to completion before
              the next starts. This is what you do if you do not pack.

The `materialize()` in the serial arm is load-bearing, not stylistic. Without it Ray Data
pipelines the stages and both arms are the same arm with different numbers.

WHAT THIS HARNESS IS MOSTLY MADE OF, AND WHY

Three of the first four attempts at this measurement measured the wrong thing, and none of
them errored. Each guard below is one of them:

  1. NO WARMUP. The HuggingFace download is paid once, by whichever arm runs first, and it
     cost ~175 s against a ~75 s run. The first arm looked 3x worse than itself.
  2. WARMUP IN THE DRIVER. The driver is the head node; the actors run on a worker, and the
     HuggingFace cache is per-node. The "warmup" finished in 24 s -- too fast for ~5 GB, the
     only tell -- and the first arm was still penalised. `warmup()` is a Ray task holding a
     GPU so it lands where the work does, and it PRINTS THE HOSTNAME AND DURATION so the
     next person can see that it went to the right place.
  3. FIRST TIMED RUN OF THE SESSION IS SLOW ANYWAY, by ~2x, reproducible to within 1% across
     runs, cause never isolated (fixture page cache, per-process CUDA setup, processor files
     the model warmup does not touch -- unknown). So each arm gets one THROWAWAY pass that
     is not timed and not recorded.

The throwaway is inside the harness on purpose. Discarding a run after seeing it is a
decision about the data; discarding it before is a protocol. Same arithmetic, and only one
of them is honest. Rounds that dropped a run post-hoc read >=27.7% and >=27.9%; this harness
read >=26.5% with nothing excluded, and only the last one was claimable.

ARMS ARE INTERLEAVED (A,B,A,B,...). Running one arm to completion and then the other
confounds arm with time: anything that drifts lands entirely on one of them.

THE VERDICT IS TWO BLUNT RULES, NOT A RATIO OF MEANS. `separable()` below applies them:

  SINGLE   an arm with one timed run has no observed spread, so any delta from it is
           unfalsifiable and is refused rather than reported.
  OVERLAP  two arms whose observed ranges overlap are NOT separable at this sample size,
           however far apart their averages sit.

Separable means the WORST run of the better arm beats the BEST run of the worse one, and the
margin printed is that gap -- a lower bound. It is a weaker statement than a significance
test and a much stronger one than comparing averages, and it is the one that would have
caught every wrong margin this harness was written for. The >=26.5% in the README is this
rule applied to the run recorded below, so it can be re-derived from what ships rather than
taken on trust.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback


def _warm_this_node(repos: list[tuple[str, str]]) -> str:
    """Runs inside a Ray task, so it warms the node that will do the work."""
    import gc

    import torch
    import transformers

    for repo, cls in repos:
        model = getattr(transformers, cls).from_pretrained(repo, dtype=torch.bfloat16)
        del model
    gc.collect()
    torch.cuda.empty_cache()
    return socket.gethostname()


def separable(better: list[float], worse: list[float],
              better_name: str = "coresident", worse_name: str = "serial") -> str:
    """The verdict line, by the two rules in the module docstring. Returns text, not a bool.

    A bool would be read as "packing wins" or "packing loses", and UNSUPPORTED is neither:
    it says the runs cannot answer the question. Collapsing those two into one flag is the
    mistake the rules exist to prevent, so the outcome is a sentence you have to read.

    The margin is `(min(better) - max(worse)) / max(worse)`, which is a LOWER bound: the
    worst run of the better arm against the best run of the worse one. A ratio of the two
    averages would read higher and would not be supported by three runs.
    """
    if len(better) < 2 or len(worse) < 2:
        return (f"UNSUPPORTED  need >=2 timed runs per arm, have {len(better)} {better_name} "
                f"and {len(worse)} {worse_name}. Nothing is separable from a single run.")
    lo, hi = (worse, better) if min(better) > min(worse) else (better, worse)
    lo_name, hi_name = ((worse_name, better_name) if min(better) > min(worse)
                        else (better_name, worse_name))
    if min(hi) > max(lo):
        margin = (min(hi) - max(lo)) / max(lo) * 100
        return (f"SEPARABLE    {hi_name} > {lo_name} by >= {margin:.1f}% "
                f"(worst {hi_name} {min(hi):.4f} > best {lo_name} {max(lo):.4f} rows/s)")
    return (f"OVERLAP      not separable at this sample size: {better_name} "
            f"[{min(better):.4f}, {max(better):.4f}] vs {worse_name} "
            f"[{min(worse):.4f}, {max(worse):.4f}] rows/s. More runs, or a real difference.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", default="/mnt/cluster_storage/pack-fixture")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3, help="timed runs per arm (>=2, or nothing "
                                                       "is separable by construction)")
    ap.add_argument("--out", default="packing-arms.jsonl")
    args = ap.parse_args(argv)

    if args.runs < 2:
        raise SystemExit("--runs must be >= 2: an arm with one run has no observed spread, "
                         "so any delta from it is unfalsifiable. `separable()` refuses it.")
    if not os.environ.get("HF_TOKEN", "").strip():
        raise SystemExit("Set HF_TOKEN: two of the model repos are gated.")

    import ray

    import pipeline

    # Same lock the driver installed, handed to the actors. Both arms build actors on a GPU
    # worker that never ran the README's install; see `pipeline.runtime_env`.
    ray.init(ignore_reinit_error=True, runtime_env=pipeline.runtime_env())
    if not ray.cluster_resources().get("GPU"):
        raise SystemExit("no GPU in this cluster. Ray does not fail on an unsatisfiable "
                         "resource, it waits -- so this refuses now rather than hanging.")
    print(f"cluster: {ray.cluster_resources().get('GPU')} GPU, "
          f"LABELS={pipeline.LABELS}", flush=True)

    def coresident() -> tuple[int, float]:
        t0 = time.perf_counter()
        ds = pipeline.build(args.input, stub=False)
        rows = sum(len(b["frame_id"]) for b in ds.iter_batches(batch_format="numpy"))
        return rows, time.perf_counter() - t0

    def serial() -> tuple[int, float]:
        from ray.data import ActorPoolStrategy

        t0 = time.perf_counter()
        ds = ray.data.read_parquet(args.input,
                                  ray_remote_args={"num_cpus": pipeline.DOWNLOAD_NUM_CPUS})
        for cls, batch in [(pipeline.Detector, pipeline.DETECTOR_BATCH),
                           (pipeline.ObjectEmbedder, pipeline.EMB_BATCH),
                           (pipeline.ImageEmbedder, pipeline.EMB_BATCH),
                           (pipeline.Metrics, pipeline.METRICS_BATCH)]:
            ds = ds.map_batches(cls, batch_size=batch, num_cpus=0, num_gpus=1,
                                compute=ActorPoolStrategy(size=1)).materialize()
        return ds.count(), time.perf_counter() - t0

    arms = [("coresident", coresident), ("serial", serial)]

    t0 = time.perf_counter()
    host = ray.get(ray.remote(num_gpus=1, num_cpus=2)(_warm_this_node).remote(
        [(pipeline.DETECTOR_MODEL, "Sam3Model"),
         (pipeline.OBJECT_EMBED_MODEL, "AutoModel"),
         (pipeline.IMAGE_EMBED_MODEL, "AutoModel")]))
    print(f"warmup: weights cached on {host} in {time.perf_counter() - t0:.1f}s. Well under "
          f"a minute means already cached, or warmed the wrong node.", flush=True)

    print("throwaway pass of each arm (not timed, not recorded)", flush=True)
    for name, fn in arms:
        try:
            rows, secs = fn()
            print(f"  discarded {name}: {rows} rows in {secs:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  discarded {name}: FAIL {type(exc).__name__}: {exc}", flush=True)

    records = []
    for i in range(args.runs):
        for name, fn in arms:
            try:
                rows, secs = fn()
                rec = {"label": name, "group": "e2e", "run": i, "rows": rows,
                       "seconds": round(secs, 2), "rows_per_s": round(rows / secs, 4)}
                print(f"  {name:11s} run {i}: {rows} rows in {secs:.1f}s = "
                      f"{rows / secs:.3f} rows/s", flush=True)
            except Exception as exc:  # noqa: BLE001
                rec = {"label": name, "group": "e2e", "run": i,
                       "error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:11s} run {i}: FAIL {rec['error']}", flush=True)
                traceback.print_exc()
            records.append(rec)

    with open(args.out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    ok = [r for r in records if "rows_per_s" in r]
    by_arm: dict[str, list[float]] = {}
    for r in ok:
        by_arm.setdefault(r["label"], []).append(r["rows_per_s"])
    for name, vals in by_arm.items():
        print(f"  {name:11s} n={len(vals)} range=[{min(vals):.3f}, {max(vals):.3f}]", flush=True)

    print(f"\n{len(ok)}/{len(records)} runs usable, written to {args.out}")
    print(separable(by_arm.get("coresident", []), by_arm.get("serial", [])))
    print("Every run is in the JSONL, including the failures, so the verdict can be "
          "re-derived or re-scored by a stricter test without re-running the GPU.")
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
