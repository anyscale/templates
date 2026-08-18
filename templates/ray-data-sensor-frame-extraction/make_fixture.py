#!/usr/bin/env python3
"""Generate the template's own fixture: fixed-geometry sensor frames in Parquet,
written in two physical layouts from byte-identical payloads.

Why this exists: the workload this template teaches reads multi-megabyte opaque blobs
out of Parquet, and the single largest lever on it is the *physical type* of that blob
column. A template cannot demonstrate that with someone else's dataset, and the
customer data it was derived from is private. So the template makes its own, and the
reader measures the lever on their own hardware.

Defaults are the CI-affordable knob. Raise --frames (and --width/--height) to see the
production regime; the ratio holds, the wall clock does not.

    python make_fixture.py --out /mnt/cluster_storage/fixture --frames 256
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# A single-plane 12-bit Bayer frame at this geometry is ~16.7 MB, which is the regime
# where per-value Parquet bookkeeping dominates. Smaller frames blunt the lesson.
DEFAULT_WIDTH = 3848
DEFAULT_HEIGHT = 2168
DEFAULT_BYTES_PER_PIXEL = 2  # 12-bit packed into uint16


def make_frames(n: int, width: int, height: int, bpp: int, seed: int = 0):
    """Deterministic pseudo-sensor payloads. Content is irrelevant; size and
    uniformity are the point."""
    rng = np.random.default_rng(seed)
    nbytes = width * height * bpp
    for i in range(n):
        # A gradient plus noise compresses like sensor data does -- neither
        # incompressible (which would flatter the fast layouts) nor constant
        # (which would flatter the slow ones).
        base = np.linspace(0, 4095, nbytes, dtype=np.uint16)
        noise = rng.integers(0, 64, size=nbytes, dtype=np.uint16)
        yield ((base + noise + i) % 4096).astype(np.uint16).tobytes()[:nbytes]


def write_list_uint8(frames, path: Path, row_group_size: int) -> float:
    """The layout that turns up by default. One Parquet INT32 value, one definition
    level and one repetition level PER BYTE."""
    tbl = pa.table(
        {
            "frame_id": pa.array(range(len(frames)), type=pa.int64()),
            "data": pa.array([list(f) for f in frames], type=pa.list_(pa.uint8())),
        }
    )
    t0 = time.perf_counter()
    pq.write_table(tbl, path, compression="zstd", row_group_size=row_group_size)
    return time.perf_counter() - t0


def write_fixed_binary(frames, path: Path, row_group_size: int) -> float:
    """The recommended layout: FIXED_LEN_BYTE_ARRAY, required, no dictionary.
    One Parquet value per frame; the reader preallocates from the footer."""
    width = len(frames[0])
    schema = pa.schema(
        [
            pa.field("frame_id", pa.int64(), nullable=False),
            pa.field("data", pa.binary(width), nullable=False),
        ]
    )
    tbl = pa.table(
        {
            "frame_id": pa.array(range(len(frames)), type=pa.int64()),
            "data": pa.array(frames, type=pa.binary(width)),
        },
        schema=schema,
    )
    t0 = time.perf_counter()
    pq.write_table(
        tbl,
        path,
        compression="zstd",
        use_dictionary=False,
        row_group_size=row_group_size,
    )
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=64, help="frames per file")
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--bytes-per-pixel", type=int, default=DEFAULT_BYTES_PER_PIXEL)
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=1,
        help="rows per row group. A read task cannot parallelise below one row group, "
        "and on multi-MB rows small is right -- unlike narrow tabular data.",
    )
    args = ap.parse_args()

    out = Path(args.out)
    (out / "list_uint8").mkdir(parents=True, exist_ok=True)
    (out / "fixed_binary").mkdir(parents=True, exist_ok=True)

    slow_total = fast_total = 0.0
    for f in range(args.files):
        frames = list(
            make_frames(
                args.frames, args.width, args.height, args.bytes_per_pixel, seed=f
            )
        )
        slow_total += write_list_uint8(
            frames, out / "list_uint8" / f"part-{f:05d}.parquet", args.row_group_size
        )
        fast_total += write_fixed_binary(
            frames, out / "fixed_binary" / f"part-{f:05d}.parquet", args.row_group_size
        )

    rows = args.files * args.frames
    print(f"wrote {rows} rows x 2 layouts to {out}")
    print(f"  list<uint8>          write: {slow_total:7.2f}s")
    print(f"  binary(N) required   write: {fast_total:7.2f}s")
    if fast_total > 0:
        print(f"  write-side ratio: {slow_total / fast_total:.2f}x")
    print(
        "\nNote: the write-side gap is the first half of the lesson. The producing team "
        "is usually told they must change their writer for the reader's benefit; "
        "measured, the recommended layout is also cheaper to WRITE."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
