#!/usr/bin/env python3
"""Measure one of this template's levers with enough runs to tell a difference from noise.

CI runs each arm once, and one run cannot tell a real difference from a warm cache or a noisy
neighbour: five single runs of the write-side ratio on a laptop gave 6.96x, 5.86x, 4.51x, 2.14x
and 4.61x, two of them at the same scale differing by more than 2x. So CI asserts only
directions, and magnitudes come from here, under two rules taken from the neighbouring
ray-data-multimodel-frame-enrichment template's measure_packing.py:

  1. At least two timed runs per arm. An arm with one run has no observed spread, so any
     delta from it is unfalsifiable.
  2. The verdict is SEPARABLE only when the arms' observed ranges do not overlap, and the
     margin quoted is the lower bound: worst run of the better arm against the best run of
     the worse one. A ratio of the two medians reads higher and is not supported at n=3.

UNSUPPORTED and OVERLAP mean these runs cannot answer the question, which is different from
the lever doing nothing, so the verdict is a sentence rather than a boolean.

The first pipeline run on a fresh cluster also pays for Ray building the runtime_env
virtualenv from python_depset.lock on the worker: 104.4s against 8.3s for the next run of the
same arm (g6.4xlarge, Ray 2.57.0). That cost lands in whichever arm goes first; unwarmed, it
made the recommended layout look 17x slower. --warmup therefore defaults to 1 untimed run per
arm, fixed before any result is seen.

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
# reads every lever from the environment at import, so each run is a subprocess, which is also
# how the notebook runs it: the harness measures the shipped path.
SWEEPS: dict[str, list[tuple[str, str, dict]]] = {
    # The Parquet physical type of the blob column. Results are in the README.
    "layout": [
        ("fixed_binary", "fixed_binary", {}),
        ("list_uint8", "list_uint8", {}),
    ],
    # Read task CPU. num_cpus < 1.0 gives a read task a one-thread decoder, because Ray sets
    # OMP_NUM_THREADS = max(1, floor(num_cpus)) and pyarrow sizes its thread pool from that
    # once per worker process. On this template's cluster, Ray 2.57.0, this sweep returned
    # OVERLAP.
    "read-cpus": [
        ("num_cpus=1.0", "fixed_binary", {"READ_NUM_CPUS": "1.0"}),
        ("num_cpus=0.25", "fixed_binary", {"READ_NUM_CPUS": "0.25"}),
    ],
    # Decoder threads for the read operator. The source workload saw a cost once
    # threads_per_task x concurrent_tasks passed the core count. On this template's cluster,
    # Ray 2.57.0, 1 to 4 was SEPARABLE by >= 1.2% and 4 to 8 did not separate at 2 runs per
    # arm.
    "decode-threads": [
        ("omp=1", "fixed_binary", {"READ_OMP_THREADS": "1"}),
        ("omp=4", "fixed_binary", {"READ_OMP_THREADS": "4"}),
        ("omp=8", "fixed_binary", {"READ_OMP_THREADS": "8"}),
    ],
}


# ------------------------------------------------------------------------------------------
# On-disk grid: bytes at rest, with no GPU, cluster or Ray, so it runs anywhere pyarrow does.
# Sizes come from the Parquet footer rather than `du`: each column chunk records
# total_uncompressed_size (after encoding, before the codec) and total_compressed_size, which
# separates encoding from compression. Units: MB = 1e6, MiB = 2**20.
# ------------------------------------------------------------------------------------------
# pyarrow 23.0.1 rejects two plausible names. "uncompressed" is not one: use "none", which the
# footer reports as UNCOMPRESSED. "lz4_raw" is rejected too, but "lz4" writes the LZ4_RAW codec
# (Parquet codec id 7), which pyarrow's metadata labels LZ4.
CODECS = ["none", "snappy", "gzip", "brotli", "lz4", "zstd"]
PAYLOADS = ["sensor", "quantized", "random"]
ZSTD_LEVELS = [1, 3, 9, 22]
MB = 1e6
MiB = 2 ** 20


def _frame(width, height, bpp, payload, seed=0) -> bytes:
    import numpy as np

    rng = np.random.default_rng(seed)
    n = width * height * bpp
    if payload == "random":
        return os.urandom(n)
    if payload == "quantized":
        return (rng.integers(0, 16, size=n, dtype=np.uint16) * 256).astype(np.uint16).tobytes()[:n]
    base = np.linspace(0, 4095, n, dtype=np.uint16)
    noise = rng.integers(0, 64, size=n, dtype=np.uint16)
    return ((base + noise) % 4096).astype(np.uint16).tobytes()[:n]


def _blob_chunk(path: Path):
    """(uncompressed_bytes, compressed_bytes, encodings, codec) for the blob column, bytes
    summed over row groups. total_uncompressed_size is after encoding and before compression,
    so it shows whether list<uint8>'s INT32 expansion reached disk."""
    import pyarrow.parquet as pq

    md = pq.ParquetFile(str(path)).metadata
    unc = comp = 0
    enc = ()
    codec_used = ""
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for c in range(g.num_columns):
            col = g.column(c)
            if col.path_in_schema.split(".")[0] != "data":
                continue
            unc += col.total_uncompressed_size
            comp += col.total_compressed_size
            enc = col.encodings
            codec_used = col.compression
    return unc, comp, enc, codec_used


def _write_pair(tmp: Path, payload_bytes: bytes, codec, level, rows, rgs, list_dict):
    """Write both layouts of the same payload and return their footer decompositions."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    width = len(payload_bytes)
    frames = [payload_bytes] * rows
    kw = dict(compression=codec, compression_level=level, row_group_size=rgs)

    fix = tmp / "fix.parquet"
    pq.write_table(pa.table({"data": pa.array(frames, type=pa.binary(width))}),
                   str(fix), use_dictionary=False, **kw)
    lst = tmp / "lst.parquet"
    pq.write_table(pa.table({"data": pa.array([list(f) for f in frames],
                                              type=pa.list_(pa.uint8()))}),
                   str(lst), use_dictionary=list_dict, **kw)
    out = (_blob_chunk(fix), _blob_chunk(lst))
    fix.unlink(); lst.unlink()
    return out


def on_disk_grid(out_root: Path, width: int, height: int, bpp: int) -> list:
    import tempfile

    rows = []
    payload_cache = {p: _frame(width, height, bpp, p) for p in PAYLOADS}
    raw = width * height * bpp
    print(f"payload {raw:,} bytes ({raw / MB:.1f} MB = {raw / MiB:.1f} MiB) per frame, 1 frame, "
          f"1 row group\n")

    def row(tag, codec, level, payload, list_dict, rgs=1):
        with tempfile.TemporaryDirectory() as td:
            (fu, fc, fe, fcod), (lu, lc, le, lcod) = _write_pair(
                Path(td), payload_cache[payload], codec, level, 1, rgs, list_dict)
        rows.append({"axis": tag, "codec": codec, "level": level, "payload": payload,
                     "list_dictionary": list_dict, "row_group_size": rgs,
                     "binary_uncompressed": fu, "binary_compressed": fc,
                     "list_uncompressed": lu, "list_compressed": lc,
                     "gap_uncompressed": lu / fu, "gap_compressed": lc / fc,
                     "binary_encodings": list(fe), "list_encodings": list(le),
                     "footer_codec": lcod})
        print(f"{codec:12s} {str(level):>5s} {payload:9s} dict={str(list_dict):5s} "
              f"| binary {fu/MB:8.1f}/{fc/MB:8.1f} | list {lu/MB:8.1f}/{lc/MB:8.1f} "
              f"| gap unc {lu/fu:5.2f}x comp {lc/fc:5.2f}x")

    hdr = (f"{'codec':12s} {'lvl':>5s} {'payload':9s} {'dict':10s} "
           f"| {'binary unc/comp MB':>19s} | {'list unc/comp MB':>19s} | gap")
    print("=== A. codec x payload x dictionary (row group = 1 row) ===")
    print(hdr); print("-" * len(hdr))
    for codec in CODECS:
        for payload in PAYLOADS:
            for list_dict in (True, False):
                row("codec-payload-dict", codec, None, payload, list_dict)

    print("\n=== B. zstd level, default payload, both dictionary settings ===")
    print(hdr); print("-" * len(hdr))
    for lvl in ZSTD_LEVELS:
        for list_dict in (True, False):
            row("zstd-level", "zstd", lvl, "sensor", list_dict)

    print("\n=== C. rows per row group, zstd default payload ===")
    print(hdr); print("-" * len(hdr))
    for rgs in (1, 4, 16):
        for list_dict in (True, False):
            with tempfile.TemporaryDirectory() as td:
                (fu, fc, _, _), (lu, lc, _, _) = _write_pair(
                    Path(td), payload_cache["sensor"], "zstd", None, 16, rgs, list_dict)
            rows.append({"axis": "row-group", "codec": "zstd", "level": None,
                         "payload": "sensor", "list_dictionary": list_dict,
                         "row_group_size": rgs, "rows": 16,
                         "binary_uncompressed": fu, "binary_compressed": fc,
                         "list_uncompressed": lu, "list_compressed": lc,
                         "gap_uncompressed": lu / fu, "gap_compressed": lc / fc})
            print(f"{'zstd':12s} {'-':>5s} {'sensor':9s} dict={str(list_dict):5s} "
                  f"| binary {fu/MB:8.1f}/{fc/MB:8.1f} | list {lu/MB:8.1f}/{lc/MB:8.1f} "
                  f"| gap unc {lu/fu:5.2f}x comp {lc/fc:5.2f}x   (rows/group={rgs}, 16 rows)")

    print("""
`gap unc` is list<uint8> / binary(N) on total_uncompressed_size -- AFTER Parquet encoding,
BEFORE the codec. That is the number that says whether the per-value bookkeeping ever reached
disk. `gap comp` is the same ratio after the codec.

BYTES AT REST ONLY. This grid does not measure read throughput. Where a gap reopens, whether
the read ratio follows is a separate measurement: run `--sweep layout` against that fixture
instead of assuming it does.""")
    return rows


# ------------------------------------------------------------------------------------------
# Storage sweep: layout x dictionary x mount, cold and warm. Needs a cluster, because
# /mnt/cluster_storage only exists on one. Codec `none` throughout: with the dictionary off it
# keeps list<uint8>'s full 4x byte gap on disk, where an I/O component has the most room to
# show.
#
# Cold: fsync, then posix_fadvise(POSIX_FADV_DONTNEED) drops the file's clean pages, with no
# root and no fixture larger than RAM. Linux only.
#
# Runs inside a Ray task requesting a CPU so it lands on a worker. A head pinned to CPU: 0,
# this repo's policy, cannot take it, and the head's local disk and cgroup CPU limit are not
# what the pipeline's read tasks see.
# ------------------------------------------------------------------------------------------
STORAGE_MOUNTS = ["/mnt/local_storage", "/mnt/cluster_storage"]


def _fadvise_dontneed(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def storage_sweep(frames: int, runs: int) -> list:
    """Measure both layouts from each mount, cold and warm. Returns per-arm records."""
    import ray

    @ray.remote(num_cpus=1)
    def _arm(frames: int, runs: int, mounts: list) -> dict:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        W, H, BPP = 3848, 2168, 2
        n = W * H * BPP
        rng = np.random.default_rng(0)
        base = np.linspace(0, 4095, n, dtype=np.uint16)
        noise = rng.integers(0, 64, size=n, dtype=np.uint16)
        payload = ((base + noise) % 4096).astype(np.uint16).tobytes()[:n]
        logical = frames * n
        res = {"host": os.uname().nodename, "nproc": os.cpu_count(), "arms": []}

        for mnt in mounts:
            if not Path(mnt).is_dir():
                continue
            for use_dict in (True, False):
                for layout in ("fixed_binary", "list_uint8"):
                    d = Path(mnt) / f"sweep-{'dict' if use_dict else 'nodict'}"
                    d.mkdir(parents=True, exist_ok=True)
                    f = d / f"{layout}.parquet"
                    if layout == "fixed_binary":
                        tbl = pa.table({"data": pa.array([payload] * frames,
                                                         type=pa.binary(n))})
                        flag = False
                    else:
                        tbl = pa.table({"data": pa.array([list(payload)] * frames,
                                                         type=pa.list_(pa.uint8()))})
                        flag = use_dict
                    pq.write_table(tbl, str(f), compression="none",
                                   use_dictionary=flag, row_group_size=1)
                    del tbl
                    on_disk = f.stat().st_size

                    def timed():
                        t0 = time.perf_counter()
                        tb = pq.read_table(str(f))
                        dt = time.perf_counter() - t0
                        del tb
                        return dt

                    cold = []
                    for _ in range(runs):
                        _fadvise_dontneed(f)
                        cold.append(timed())
                    timed()                      # declared warmup for the warm series
                    warm = [timed() for _ in range(runs)]
                    res["arms"].append({
                        "mount": mnt, "list_dictionary": use_dict, "layout": layout,
                        "on_disk_bytes": on_disk, "logical_bytes": logical,
                        "cold_MBps": [logical / s / 1e6 for s in cold],
                        "warm_MBps": [logical / s / 1e6 for s in warm],
                    })
                    f.unlink()
        return res

    ray.init(ignore_reinit_error=True)
    res = ray.get(_arm.remote(frames, runs, STORAGE_MOUNTS))
    print(f"worker {res['host']}, nproc {res['nproc']}, {frames} frames\n")
    hdr = (f"{'mount':22s} {'dict':5s} {'layout':13s} {'on disk MB':>11s} "
           f"{'cold MB/s':>19s} {'warm MB/s':>19s}")
    print(hdr); print("-" * len(hdr))
    for a in res["arms"]:
        print(f"{a['mount']:22s} {str(a['list_dictionary']):5s} {a['layout']:13s} "
              f"{a['on_disk_bytes']/1e6:11.1f} "
              f"{min(a['cold_MBps']):8.1f}-{max(a['cold_MBps']):<10.1f} "
              f"{min(a['warm_MBps']):8.1f}-{max(a['warm_MBps']):<10.1f}")
    print("\nratio fixed_binary over list<uint8>, on logical bytes:")
    for mnt in STORAGE_MOUNTS:
        for ud in (True, False):
            sel = {a["layout"]: a for a in res["arms"]
                   if a["mount"] == mnt and a["list_dictionary"] == ud}
            if len(sel) != 2:
                continue
            fb, lu = sel["fixed_binary"], sel["list_uint8"]
            for label in ("cold_MBps", "warm_MBps"):
                lo = min(fb[label]) / max(lu[label])
                hi = max(fb[label]) / min(lu[label])
                sep = "SEPARABLE" if min(fb[label]) > max(lu[label]) else "OVERLAP"
                print(f"  {mnt:22s} dict={str(ud):5s} bytes "
                      f"{lu['on_disk_bytes']/fb['on_disk_bytes']:4.2f}x "
                      f"{label[:4]:5s} {lo:6.2f}-{hi:<6.2f}x {sep}")
    return res["arms"]


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
    """The verdict line, as text; see the module docstring."""
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
    ap.add_argument("--input", required=True,
                    help="fixture root containing fixed_binary/ and list_uint8/, for the "
                         "layout, read-cpus and decode-threads sweeps. Required but unused by "
                         "--sweep on-disk and storage, which generate their own payloads.")
    ap.add_argument("--frames", type=int, default=12,
                    help="--sweep storage: frames per layout per arm")
    ap.add_argument("--width", type=int, default=3848, help="--sweep on-disk: frame width")
    ap.add_argument("--height", type=int, default=2168, help="--sweep on-disk: frame height")
    ap.add_argument("--bpp", type=int, default=2, help="--sweep on-disk: bytes per pixel")
    ap.add_argument("--sweep", default="layout",
                    choices=sorted(SWEEPS) + ["on-disk", "storage"],
                    help="which lever to measure. `on-disk`: the codec x payload x dictionary "
                         "grid of bytes at rest, no GPU or cluster. `storage`: layout x "
                         "dictionary x mount, cold and warm; needs a cluster.")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per arm; >=2 or nothing is separable")
    ap.add_argument("--warmup", type=int, default=1,
                    help="untimed runs per arm before timing starts. Keep the default of 1: the "
                         "first run of a sweep also pays for Ray building the runtime_env "
                         "virtualenv from python_depset.lock on the worker, 104.4s against 8.3s "
                         "for the next run on a g6.4xlarge, Ray 2.57.0.")
    ap.add_argument("--out", default=None, help="write the per-run records here as JSON")
    args = ap.parse_args(argv)

    if args.sweep == "storage":
        if args.runs < 2:
            raise SystemExit(
                "--runs must be >= 2 for --sweep storage. One run per arm has no observed "
                "spread, so a cold-against-warm or layout-against-layout delta from it is "
                "unfalsifiable."
            )
        rows = storage_sweep(args.frames, args.runs)
        if args.out:
            Path(args.out).write_text(json.dumps({"sweep": "storage", "arms": rows}, indent=2) + "\n")
            print(f"\nwrote {args.out}")
        return 0

    if args.sweep == "on-disk":
        rows = on_disk_grid(Path(args.input), args.width, args.height, args.bpp)
        if args.out:
            Path(args.out).write_text(json.dumps({"sweep": "on-disk", "grid": rows}, indent=2) + "\n")
            print(f"\nwrote {args.out}")
        return 0

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
    # recommended layout against the naive one; for the thread sweeps it steps through the counts.
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
