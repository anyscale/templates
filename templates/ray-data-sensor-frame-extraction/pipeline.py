#!/usr/bin/env python3
"""Read multi-megabyte sensor frames from Parquet, normalise and downsample them on a GPU
actor pool, then write the result or count the rows.

Every lever is an environment variable read at import. Each lever's comment says what the
default assumes about the data, when to change it, and what was measured: on this template's
cluster (one g6.4xlarge L4 worker) or, where labelled, on the source workload at production
scale. The two can disagree.

    python pipeline.py --input /mnt/cluster_storage/fixture/fixed_binary
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import ray

# --------------------------------------------------------------------------------------
# Levers
# --------------------------------------------------------------------------------------

# Read concurrency. Cap it only when a CPU stage downstream of the read binds. On the source
# workload the cap gave 1.66-1.91x there and 0.56x on a read-bound pipeline: same knob,
# opposite sign. Not measured on this template's cluster. Find the binding operator in
# ds.stats() first. 0 leaves the read uncapped.
READ_CONCURRENCY = int(os.environ.get("READ_CONCURRENCY", "0")) or None

# Read task CPU, default 1.0. The decoder gets one thread either way: Ray sets
# OMP_NUM_THREADS = max(1, floor(num_cpus)) when it is unset, and pyarrow sizes its CPU
# thread pool from that once per worker process. What num_cpus changes is how many read
# tasks share a core. On multi-megabyte blob columns decode is most of the read's work, so
# the default gives each read task a core. Measured on this template's cluster, Ray 2.57.0:
# 16.67-17.00 rows/s at 1.0 against 16.24-16.87 at 0.25, ranges overlapping; 4 files make
# only 4 read tasks on 16 vCPUs, so the fixture can't separate the two. Go below 1.0 only
# when rows are thin, files are many, and read tasks wait on storage rather than decode.
READ_NUM_CPUS = float(os.environ.get("READ_NUM_CPUS", "1.0"))

# Decoder threads, set for the read operator only so other actors keep their thread pools.
# Size them so threads_per_task x concurrent_tasks ~= cores. On the source workload: up to
# 2.15x on local disk while the read was under-parallel, about 1.6x from object storage (the
# network part of the task does not speed up), and +52% wall clock at 8 threads, past that
# point, on a large object-storage read. On this template's cluster, Ray 2.57.0, 1 to 4
# threads gained 1.2% or better and 4 to 8 did not separate at 2 runs per arm.
# 0 leaves OMP_NUM_THREADS alone, the right default until the sizing rule says otherwise.
READ_OMP_THREADS = int(os.environ.get("READ_OMP_THREADS", "0"))

# GPU actor pool. Fractional GPUs and num_cpus=0, so the pool does not take cores from the
# read. The source workload's best configuration ran 8 actors x 0.5 GPU on 4 of 8 available
# GPUs; its GPU was never the constraint, and packing mattered more than actor count. On this
# template's cluster, one L4 and 2 actors, the GPU stage was closer to the constraint than the
# read. See "Which operator binds" in the README.
GPU_ACTORS = int(os.environ.get("GPU_ACTORS", "2"))
GPU_PER_ACTOR = float(os.environ.get("GPU_PER_ACTOR", "0.5"))
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "8"))

# Object store, not set in this file. Set the fraction
# (RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION) in the image or the compute config; in a job
# config's env_vars it is a silent no-op. The compute config's object_store_memory field takes
# precedence over the variable. On the source workload 0.6 held a 69.4 GiB peak with zero
# spill. Lower it if workers are killed rather than spilling: prefetched batches live in the
# heap, and an oversized object store starves them.

OUT_HEIGHT = int(os.environ.get("OUT_HEIGHT", "720"))

# Dependency delivery. The README notebook installs the lock on the driver. The map_batches
# actors get it only through ray.init(runtime_env=...) in main(); without that they run the
# image's packages, which include no torch. That passes in a workspace, which propagates a
# plain pip install, and fails as a standalone Job or Service.
# Don't copy the notebook's install line into this file: check-dep-delivery's lock-installed
# check searches source files for it, and a copy in a comment passes the check without
# installing anything.
LOCK_PATH = Path(
    os.environ.get("PYTHON_DEPSET_LOCK")
    or Path(__file__).resolve().parent / "python_depset.lock"
)


def read_geometry(input_path: str) -> tuple[int, int, int]:
    """Return (width, height, bytes_per_pixel) from the fixture's Parquet schema metadata.

    make_fixture.py writes the geometry there, so the pipeline follows any --width/--height.
    Files without it need FRAME_WIDTH and FRAME_HEIGHT, and optionally BYTES_PER_PIXEL.
    """
    import glob

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if files:
        meta = pq.ParquetFile(files[0]).schema_arrow.metadata or {}
        if b"frame_width" in meta and b"frame_height" in meta:
            return (
                int(meta[b"frame_width"]),
                int(meta[b"frame_height"]),
                int(meta.get(b"bytes_per_pixel", b"2")),
            )
    # A fixture this template did not write carries no geometry, so it has to be named.
    try:
        return (
            int(os.environ["FRAME_WIDTH"]),
            int(os.environ["FRAME_HEIGHT"]),
            int(os.environ.get("BYTES_PER_PIXEL", "2")),
        )
    except KeyError:
        raise SystemExit(
            f"{input_path} carries no frame geometry in its Parquet schema metadata, and "
            "FRAME_WIDTH / FRAME_HEIGHT are not set. Either regenerate the fixture with "
            "make_fixture.py, which records the geometry, or export both."
        ) from None


class ISP:
    """The GPU stage: normalise each 12-bit frame and downsample it bilinearly to OUT_HEIGHT.

    Light arithmetic. On one L4, Ray 2.57.0, it ran at about 0.85 saturation (3.4s of UDF
    across 2 actors in a 1.99s span), closer to the constraint than the read at this
    template's scale. See "Which operator binds" in the README.
    """

    def __init__(self, width: int, height: int, out_height: int = OUT_HEIGHT):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.width = width
        self.height = height
        self.out_height = out_height

    def __call__(self, batch: dict) -> dict:
        torch = self.torch
        width, height = self.width, self.height
        frames = []
        for raw in batch["data"]:
            arr = np.frombuffer(raw, dtype=np.uint16)
            if arr.size != width * height:
                raise ValueError(
                    f"frame is {arr.size} uint16 elements, geometry says "
                    f"{width}x{height} = {width * height}. Refusing to crop: a silent crop "
                    "here produces plausible wrong output."
                )
            frames.append(arr.reshape(height, width))
        # One host-to-device transfer per batch, not per frame. Per-image device round-trips
        # inside a UDF are a common and costly mistake on GPU image stages; see the sibling
        # pattern on device syncs.
        t = torch.from_numpy(np.stack(frames).astype(np.float32)).to(self.device)
        t = t.unsqueeze(1) / 4095.0
        scale = self.out_height / height
        t = torch.nn.functional.interpolate(
            t, scale_factor=scale, mode="bilinear", align_corners=False
        )
        out = (t.squeeze(1) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return {"frame_id": batch["frame_id"], "image": list(out)}


def build(input_path: str):
    read_args: dict = {"num_cpus": READ_NUM_CPUS}
    if READ_OMP_THREADS:
        read_args["runtime_env"] = {
            "env_vars": {"OMP_NUM_THREADS": str(READ_OMP_THREADS)}
        }

    kwargs = {"ray_remote_args": read_args}
    if READ_CONCURRENCY:
        kwargs["concurrency"] = READ_CONCURRENCY

    width, height, _bpp = read_geometry(input_path)

    ds = ray.data.read_parquet(input_path, **kwargs)
    return ds.map_batches(
        ISP,
        fn_constructor_args=(width, height),
        batch_size=GPU_BATCH_SIZE,
        num_gpus=GPU_PER_ACTOR,
        num_cpus=0,
        # Ray 2.51 deprecated map_batches' `concurrency=` in favour of `compute=`.
        # read_parquet's `concurrency=` above is a different parameter and is not deprecated.
        compute=ray.data.ActorPoolStrategy(size=GPU_ACTORS),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="directory of Parquet files, such as a fixture's fixed_binary/")
    ap.add_argument("--output", default=None,
                    help="write the transformed frames here as Parquet; without it, count rows")
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True,
                    help="print ds.stats() after the run")
    args = ap.parse_args()

    # The actors get their deps here or nowhere. See LOCK_PATH above.
    runtime_env = None
    if LOCK_PATH.is_file():
        runtime_env = {"pip": str(LOCK_PATH)}
    elif not os.environ.get("ALLOW_IMAGE_DEPS"):
        raise SystemExit(
            f"{LOCK_PATH} not found, so the map_batches actors would run whatever the image "
            "ships -- and this template's image ships no torch. Compile the lock with "
            "./scripts/depsets/update_deps.sh, or set ALLOW_IMAGE_DEPS=1 if you have "
            "provisioned torch another way."
        )
    ray.init(ignore_reinit_error=True, runtime_env=runtime_env)
    ds = build(args.input)

    t0 = time.perf_counter()
    if args.output:
        ds.write_parquet(args.output)
        rows = ds.count()
    else:
        rows = sum(b["frame_id"].size for b in ds.iter_batches(batch_format="numpy"))
    elapsed = time.perf_counter() - t0

    print(f"\n{rows} rows in {elapsed:.1f}s = {rows / elapsed:.2f} rows/s")
    print(
        "\nState the unit. On a multi-camera rig, rows/s, frames/s and camera-frames/s "
        "differ by the camera count, and a real-time bar is quoted in one of them."
    )
    if args.stats:
        # Rank map stages by UDF time; spans overlap under the streaming executor. A map stage
        # at udf_total / (span x parallelism) ~ 1.0 is saturated. Reads have no user function
        # and report 0us of UDF time, so judge the read by its span and output bytes/s
        # against your storage.
        print("\n" + ds.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
