#!/usr/bin/env python3
"""Measure whether co-residency beats running the stages serially.

    HF_TOKEN=hf_... python measure_packing.py --runs 3
    HF_TOKEN=hf_... python measure_packing.py --out arms.jsonl   # every run, for re-scoring

`packing.py` computes whether the four stages fit on one card; this measures whether packing
them paid. Re-run it after any change to the model set, the image, the batch sizes or the card.

NEEDS: one GPU, the gated weights (HF_TOKEN on every node, since the warmup task and the actors
run on the GPU worker), and about 20 minutes. It times the fixture at --input as it is:
--frames and --files do not change it. Actor counts and batch sizes come from pipeline.py's
environment variables. The README's figure used one actor per stage; pipeline.py's defaults
need more than one GPU.

TWO ARMS

  coresident  the DAG `pipeline.build()` produces: four stages with fractional GPU
              reservations, all resident on one card at one actor per stage.
  serial      the same four stages, each given the whole GPU and run to completion before the
              next starts.

Keep the `materialize()` in the serial arm. Without it Ray Data pipelines the stages and both
arms measure the same thing.

THREE GUARDS

Each one fixes a failure that produced a wrong number and no error.

  1. No warmup. The Hugging Face download is paid once, by whichever arm runs first, and cost
     ~175 s against a ~75 s run. The first arm looked 3x worse than itself.
  2. Warmup in the driver. The driver is the head node, the actors run on a worker, and the
     Hugging Face cache is per node. That warmup finished in 24 s, too fast for ~5 GB, and the
     first arm was still penalised. `_warm_this_node` is a Ray task holding a GPU, and it
     prints the hostname and duration so you can see where it ran.
  3. The first timed run of a session is ~2x slow anyway, reproducible to within 1% across
     runs, cause never isolated. Candidates: fixture page cache, per-process CUDA setup,
     processor files the model warmup does not touch. So each arm gets one throwaway pass,
     untimed and unrecorded.

Keep the throwaway inside the harness. Rounds that dropped a run after seeing it read >=27.7%
and >=27.9%; this harness read >=26.5% with nothing excluded.

The arms interleave (A,B,A,B,...). Running one arm to completion and then the other confounds
arm with time.

VERDICT

`separable()` applies two rules:

  SINGLE   an arm with one timed run has no observed spread, so any delta from it is
           unfalsifiable. Refused.
  OVERLAP  two arms whose observed ranges overlap are not separable at this sample size,
           however far apart their averages sit.

Separable means the worst run of the faster arm beats the best run of the slower one, and the
margin printed is that gap: a lower bound, which a ratio of means would overstate. The README's
>=26.5% is this rule applied to the runs in tests/test_packing.py (class Separability).
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
    """Runs inside a Ray task, so it warms the node that does the work."""
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
    """The verdict line, by the two rules in the module docstring.

    Returns text because UNSUPPORTED is neither "packing wins" nor "packing loses": it says the
    runs cannot answer the question, and a bool would fold it into one of the two.

    The margin is (worst run of the faster arm - best run of the slower arm) / best run of the
    slower arm, a lower bound. The arms' order comes from the data, not the argument order. A
    ratio of the two averages reads higher, and three runs do not support it.
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
                         "resource, it waits. Refusing now beats hanging.")
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
    print(f"warmup: weights cached on {host} in {time.perf_counter() - t0:.1f}s. Under a "
          f"minute means cached, or the wrong node.", flush=True)

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
    print("Every run is in the JSONL, failures included, so the verdict can be re-scored "
          "without re-running the GPU.")
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
