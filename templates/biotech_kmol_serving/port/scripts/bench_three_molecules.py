"""Per-molecule latency on the three compounds Takeda uses as routine test inputs.

Compares the two serving designs on the *same* single-request path:

  * `serve_app.py`          — one deployment; featurize + forward in one replica.
  * `serve_pipeline_app.py` — three deployments; Ingress -> CPU Featurizer -> GPU forward.

Both get one whole L4 and one worker, so what's measured is the per-request path, not
scale. Expect the two-stage design to be *slower* here: a single molecule crosses one
extra deployment hop. Its advantage is throughput under load (see
`scripts/serve_pipeline_bulk.py`), not single-request latency — this script exists to
quantify that trade, not to pick a winner.

Both apps are sent a **one-element list**, not a bare string. `serve_app.py`'s single-
SMILES path goes through `@serve.batch` with `batch_wait_timeout_s=0.01`, which would add
up to 10 ms of batch-wait to every sequential request and measure the batching config
rather than the model. The list path bypasses batching in both apps.

Usage (from the bundle root):
    python scripts/bench_three_molecules.py                 # both apps
    python scripts/bench_three_molecules.py --reps 500
"""
import argparse
import json
import os
import statistics
import sys
import time

import ray
from ray import serve

_BUNDLE = os.environ.get("KMOL_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BUNDLE not in sys.path:
    sys.path.insert(0, _BUNDLE)
SHIP_DIR = os.environ.get("KMOL_SHIP_DIR", _BUNDLE)
CONFIG = os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
CKPT_DIR = os.environ.get("KMOL_CKPT_DIR", "checkpoints")
OUT = os.environ.get("KMOL_OUT", "three_molecule_results.json")

TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2"]}

# The three compounds named on the 2026-07-23 call as routine sanity-check inputs.
# Canonical SMILES; heavy-atom counts are recomputed at run time, not trusted from here.
MOLECULES = [
    # minoxidil is the pyrimidine 3-N-oxide — without the [n+]([O-]) it's a different
    # compound and comes out at 14 heavy atoms instead of 15.
    ("minoxidil", "Nc1cc(N2CCCCC2)[n+]([O-])c(N)n1"),
    ("sildenafil (Viagra)",
     "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"),
    ("atorvastatin (Lipitor)",
     "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O"),
]


def latencies(call, reps, warmup=20):
    for _ in range(warmup):
        call()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        call()
        out.append((time.perf_counter() - t) * 1e3)
    out.sort()
    return {
        "reps": reps,
        "mean_ms": statistics.fmean(out),
        "p50_ms": out[len(out) // 2],
        "p99_ms": out[min(int(len(out) * 0.99), len(out) - 1)],
        "min_ms": out[0],
        "max_ms": out[-1],
    }


def run_monolith(reps):
    import serve_app

    app = serve_app.KmolEnsemble.options(
        ray_actor_options={"num_gpus": 1, "num_cpus": 1, "runtime_env": TASK_PIP},
        autoscaling_config={"min_replicas": 1, "max_replicas": 1},
    ).bind(CONFIG, CKPT_DIR)
    handle = serve.run(app, name="monolith")
    rows = {}
    for name, smi in MOLECULES:
        first = handle.infer.remote([smi]).result()
        rows[name] = {**latencies(lambda: handle.infer.remote([smi]).result(), reps),
                      "n_logits": len(first[0]["logits"])}
    serve.delete("monolith")
    return rows


def run_two_stage(reps):
    import serve_pipeline_app as spa

    gpu = spa.GpuForward.options(
        ray_actor_options={"num_gpus": 1, "num_cpus": 1, "runtime_env": TASK_PIP},
        autoscaling_config={"min_replicas": 1, "max_replicas": 1},
    )
    feat = spa.Featurizer.options(
        ray_actor_options={"num_cpus": 1, "runtime_env": TASK_PIP},
        autoscaling_config={"min_replicas": 1, "max_replicas": 1},
    )
    ingress = spa.Ingress.options(
        ray_actor_options={"num_cpus": 1},
        autoscaling_config={"min_replicas": 1, "max_replicas": 1},
    )
    handle = serve.run(ingress.bind(feat.bind(gpu.bind(CONFIG, CKPT_DIR)), CONFIG),
                       name="twostage")
    rows = {}
    for name, smi in MOLECULES:
        first = handle.predict.remote([smi]).result()
        rows[name] = {**latencies(lambda: handle.predict.remote([smi]).result(), reps),
                      "n_logits": len(first[0]["logits"])}
    serve.delete("twostage")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})

    # Molecule sizes, measured here so the report is self-contained.
    @ray.remote(num_cpus=1, runtime_env=TASK_PIP)
    def sizes():
        from rdkit import Chem
        return {n: Chem.MolFromSmiles(s).GetNumHeavyAtoms() for n, s in MOLECULES}

    heavy = ray.get(sizes.remote())
    print("heavy atoms:", heavy, flush=True)

    results = {}
    for label, fn in (("monolith (serve_app.py)", run_monolith),
                      ("two_stage (serve_pipeline_app.py)", run_two_stage)):
        print(f"\n=== {label} ===", flush=True)
        results[label] = fn(args.reps)
        for name, r in results[label].items():
            print(f"  {name:<24} mean={r['mean_ms']:6.1f} ms  p50={r['p50_ms']:6.1f}  "
                  f"p99={r['p99_ms']:6.1f}  (n_logits={r['n_logits']})", flush=True)

    out = {
        "what": "single-molecule request latency, one whole L4 and one worker per design",
        "note": "one-element list per request, so @serve.batch wait time is not included",
        "reps_per_molecule": args.reps,
        "heavy_atoms": heavy,
        "molecules": {n: s for n, s in MOLECULES},
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    serve.shutdown()


if __name__ == "__main__":
    main()
