#!/usr/bin/env python3
"""The template's pipeline: read multi-megabyte sensor frames from Parquet, transform
them on a GPU actor pool, downsample, write.

This is the OPTIMIZATION CONTROL PANEL for the template. Every knob below is one the
source engagement measured, each carries its measured effect inline, and the defaults
are workload-shaped -- each default's comment says which data property justifies it and
what different property should flip it.

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

# READ CONCURRENCY. Cap it only when a CPU stage DOWNSTREAM of the read is what binds
# (measured 1.66-1.91x there). On a read-bound pipeline the same cap measured 0.56x --
# same knob, opposite sign. Establish the binding operator from ds.stats() first.
READ_CONCURRENCY = int(os.environ.get("READ_CONCURRENCY", "0")) or None

# READ TASK CPU. Why the default is 1.0:
#   Ray sets OMP_NUM_THREADS = max(1, floor(num_cpus)) when it is not already set, and
#   pyarrow sizes its Arrow CPU thread pool from that ONCE PER WORKER PROCESS.
#   So num_cpus < 1.0 does not just make read tasks "cheap" -- it gives each one a
#   ONE-THREAD DECODER. On thin rows and many small files that is fine and buys
#   concurrency. On multi-megabyte blob columns, decode is the work.
# Flip to <1.0 only if: rows are thin, files are many, and ds.stats() shows the read
# stage is running wider than the core count.
READ_NUM_CPUS = float(os.environ.get("READ_NUM_CPUS", "1.0"))

# DECODE THREADS, scoped to the read operator so it does not resize every actor in the
# job. Governing equation: threads_per_task x concurrent_tasks ~= cores.
# Measured worth up to 2.15x on local disk while the read stage is under-parallel, ~1.6x
# from object storage (Amdahl: the network half of the task is untouched), and a COST
# past the crossover -- +52% wall clock at 8 threads on a large object-storage read.
# 0 means "leave it alone", which is the right default until the equation says otherwise.
READ_OMP_THREADS = int(os.environ.get("READ_OMP_THREADS", "0"))

# GPU ACTOR POOL. Fractional GPU with num_cpus=0 so the pool does not compete with the
# read stage for cores. The source engagement's champion ran 8 actors x 0.5 GPU and
# occupied 4 of 8 available GPUs -- the GPU was never the constraint, and packing
# mattered more than count.
GPU_ACTORS = int(os.environ.get("GPU_ACTORS", "2"))
GPU_PER_ACTOR = float(os.environ.get("GPU_PER_ACTOR", "0.5"))
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "8"))

# OBJECT STORE. Set the fraction where it takes effect -- image or compute config, NOT a
# job config's env_vars, where it is a silent no-op. The compute config's own
# object-store-memory field takes precedence over the environment variable.
# 0.6 held peak 69.4 GiB with zero spill on the source fleet. Lower it if workers are
# being KILLED and not spilling. Prefetched batches live in the heap, and an oversized
# object store starves them.

OUT_HEIGHT = int(os.environ.get("OUT_HEIGHT", "720"))

# DEPENDENCY DELIVERY. The lock reaches the driver through the install line in the README
# notebook. Do not repeat that line here: check-dep-delivery's lock-installed check greps for
# it, and a copy in a comment satisfies the check without installing anything. It reaches the
# map_batches ACTORS only
# through ray.init(runtime_env=...) below. Install on the driver alone and the actors run
# whatever the image shipped -- which is no torch -- and that passes in a workspace, because a
# workspace tracks a plain pip install and propagates it, then fails as a standalone Job or
# Service, which has no propagation.
LOCK_PATH = Path(
    os.environ.get("PYTHON_DEPSET_LOCK")
    or Path(__file__).resolve().parent / "python_depset.lock"
)


def read_geometry(input_path: str) -> tuple[int, int, int]:
    """Ask the fixture for its frame geometry instead of hardcoding it.

    make_fixture.py records width/height/bytes-per-pixel in the Parquet schema metadata, so a
    reader who raises --width/--height -- which its docstring tells them to do -- gets a
    pipeline that follows. The previous module constant raised ValueError on a smaller fixture
    and SILENTLY CROPPED a larger one.
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
    """The GPU transform stage: demosaic-shaped work plus a downsample.

    Arithmetic-light. Measured on one L4, the GPU stage span was 1.99s against the read's
    2.44s, so it is closer to the constraint at CI scale.
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
        # One host-to-device transfer per BATCH, not per frame. Per-image device
        # round-trips inside a UDF are the single most expensive habit on GPU image
        # stages -- see the sibling pattern on device syncs.
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
        # `concurrency=` is DEPRECATED on map_batches as of Ray 2.51 in favour of `compute=`.
        # Note this does NOT apply to read_parquet's own `concurrency=` above, which is a live
        # parameter with no deprecation -- same word, two different APIs.
        compute=ray.data.ActorPoolStrategy(size=GPU_ACTORS),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    # A flag that defaulted to True could never be turned off; --no-stats is the off switch.
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction, default=True)
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
        # Rank map stages by UDF time, not by span: under the streaming executor spans
        # overlap and on one measured run summed to 2.1x the pipeline's wall clock. A map
        # stage at udf_total / (span x parallelism) ~ 1.0 is saturated. That ratio cannot
        # rank a read: Ray reports UDF time 0us for read operators, which have no user
        # function. Judge the read by its span and output bytes/s against your storage.
        print("\n" + ds.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
