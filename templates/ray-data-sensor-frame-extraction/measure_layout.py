#!/usr/bin/env python3
"""Measure a lever this template teaches, with enough runs that the answer is falsifiable.

Why this is a separate file from pipeline.py and from tests.sh. CI runs each arm once, and a
single run cannot tell a real difference from a warm cache or a noisy neighbour. This
template's own write-side ratio came out 6.96x, 5.86x, 4.51x, 2.14x and 4.61x across five
single runs during development -- twice at the SAME scale, differing by more than 2x. So a
number quoted off one run per arm is not a measurement, and CI asserts only the direction.

This harness is where a magnitude may be quoted from. Two rules, both borrowed from the
neighbouring ray-data-multimodel-frame-enrichment template's measure_packing.py:

  1. At least two timed runs per arm. An arm with one run has no observed spread, so any
     delta from it is unfalsifiable.
  2. The verdict is SEPARABLE only when the arms' observed ranges do not overlap, and the
     margin quoted is the LOWER bound -- worst run of the better arm against the best run of
     the worse one. A ratio of the two medians reads higher and is not supported at n=3.

The verdict is a sentence, not a boolean, because UNSUPPORTED and OVERLAP are not "the lever
does not work" -- they say these runs cannot answer the question.

One confound is handled by default rather than left to the reader. The first pipeline run on a
fresh cluster also pays for Ray building the runtime_env virtualenv from python_depset.lock on
the worker: measured on a g6.4xlarge, 104.4s against 8.3s for the next run of the same arm. That
lands entirely in whichever arm happens to go first, and on the first attempt it made the
recommended layout look 17x SLOWER than the naive one and turned a real result into OVERLAP. So
--warmup defaults to 1 untimed run per arm. It is declared here, not chosen after the fact.

    python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

RATE_RE = re.compile(r"=\s*([0-9.]+) rows/s")

# An arm is (name, fixture subdirectory, environment overrides for pipeline.py). pipeline.py
# reads every lever from the environment at import, so a subprocess per run is what makes a
# sweep possible at all -- and it is also how the notebook and tests.sh invoke it, so the
# harness measures the shipped path rather than a private one.
SWEEPS: dict[str, list[tuple[str, str, dict]]] = {
    # The headline claim: the Parquet physical type of the blob column.
    "layout": [
        ("fixed_binary", "fixed_binary", {}),
        ("list_uint8", "list_uint8", {}),
    ],
    # The trap: num_cpus < 1.0 on a read task gives it a ONE-THREAD decoder, because Ray sets
    # OMP_NUM_THREADS = max(1, floor(num_cpus)) and pyarrow sizes its thread pool from that
    # once per worker process.
    "read-cpus": [
        ("num_cpus=1.0", "fixed_binary", {"READ_NUM_CPUS": "1.0"}),
        ("num_cpus=0.25", "fixed_binary", {"READ_NUM_CPUS": "0.25"}),
    ],
    # Decoder threads, scoped to the read operator. Past the crossover this is a cost.
    "decode-threads": [
        ("omp=1", "fixed_binary", {"READ_OMP_THREADS": "1"}),
        ("omp=4", "fixed_binary", {"READ_OMP_THREADS": "4"}),
        ("omp=8", "fixed_binary", {"READ_OMP_THREADS": "8"}),
    ],
}


def one_run(input_dir: Path, env_overrides: dict) -> tuple[float, float, str]:
    """One pipeline.py run. Returns (rows_per_s, wall_seconds, stdout)."""
    env = dict(os.environ)
    env.update(env_overrides)
    here = Path(__file__).resolve().parent
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(here / "pipeline.py"), "--input", str(input_dir)],
        capture_output=True, text=True, env=env, cwd=here,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"pipeline.py failed (rc={proc.returncode}) for {env_overrides or 'shipped defaults'}\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )
    match = RATE_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"pipeline.py printed no rate:\n{proc.stdout[-3000:]}")
    return float(match.group(1)), wall, proc.stdout


def separable(better: list[float], worse: list[float],
              better_name: str, worse_name: str) -> str:
    """The verdict line. Text, not a bool -- see the module docstring."""
    if len(better) < 2 or len(worse) < 2:
        return (f"UNSUPPORTED  need >=2 timed runs per arm, have {len(better)} {better_name} "
                f"and {len(worse)} {worse_name}. Nothing is separable from a single run.")
    lo, hi = (worse, better) if min(better) > min(worse) else (better, worse)
    lo_name, hi_name = ((worse_name, better_name) if min(better) > min(worse)
                        else (better_name, worse_name))
    if min(hi) > max(lo):
        margin = (min(hi) - max(lo)) / max(lo) * 100
        return (f"SEPARABLE    {hi_name} > {lo_name} by >= {margin:.1f}% "
                f"(worst {hi_name} {min(hi):.2f} > best {lo_name} {max(lo):.2f} rows/s)")
    return (f"OVERLAP      not separable at this sample size: {better_name} "
            f"[{min(better):.2f}, {max(better):.2f}] vs {worse_name} "
            f"[{min(worse):.2f}, {max(worse):.2f}] rows/s. More runs, or no real difference.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="fixture root (contains fixed_binary/ and list_uint8/)")
    ap.add_argument("--sweep", default="layout", choices=sorted(SWEEPS), help="which lever to measure")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per arm; >=2 or nothing is separable")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed runs per arm before timing starts. Default 1 and you want it: "
                         "the FIRST run of a sweep also pays for Ray building the runtime_env "
                         "virtualenv from python_depset.lock on the worker, measured at ~95s "
                         "against ~8s for a warm run. Discarding it is declared here, up front, "
                         "rather than chosen after seeing which arm it landed in.")
    ap.add_argument("--out", default=None, help="write the per-run records here as JSON")
    args = ap.parse_args(argv)

    if args.runs < 2:
        raise SystemExit(
            "--runs must be >= 2. An arm with one run has no observed spread, so any delta "
            "from it is unfalsifiable and separable() will refuse to score it."
        )

    root = Path(args.input)
    arms = SWEEPS[args.sweep]
    records: list[dict] = []
    rates: dict[str, list[float]] = {}

    for name, subdir, env_overrides in arms:
        rates[name] = []
        for w in range(1, args.warmup + 1):
            rate, wall, _ = one_run(root / subdir, env_overrides)
            print(f"  {args.sweep}/{name} warmup {w}/{args.warmup}: {rate:.2f} rows/s "
                  f"({wall:.1f}s wall) -- DISCARDED, not timed")
        for run in range(1, args.runs + 1):
            rate, wall, _ = one_run(root / subdir, env_overrides)
            rates[name].append(rate)
            records.append({"sweep": args.sweep, "arm": name, "run": run,
                            "rows_per_s": rate, "wall_s": wall, "env": env_overrides})
            print(f"  {args.sweep}/{name} run {run}/{args.runs}: {rate:.2f} rows/s ({wall:.1f}s wall)")

    print(f"\n=== {args.sweep}: {args.runs} runs per arm ===")
    for name in rates:
        vals = rates[name]
        print(f"  {name:16s} min {min(vals):8.2f}  median {statistics.median(vals):8.2f}  "
              f"max {max(vals):8.2f}  rows/s   (n={len(vals)})")

    # Score every adjacent pair, in the order the arms are declared: for `layout` that is the
    # recommended layout against the naive one; for the thread sweeps it walks the crossover.
    print()
    for (a_name, _, _), (b_name, _, _) in zip(arms, arms[1:]):
        print(f"  {a_name} vs {b_name}: {separable(rates[a_name], rates[b_name], a_name, b_name)}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"sweep": args.sweep, "runs_per_arm": args.runs, "warmup_per_arm": args.warmup,
             "records": records,
             "summary": {n: {"min": min(v), "median": statistics.median(v), "max": max(v),
                             "n": len(v)} for n, v in rates.items()}}, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
