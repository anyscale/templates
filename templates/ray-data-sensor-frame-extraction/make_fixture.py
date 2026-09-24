#!/usr/bin/env python3
"""Generate the template's own fixture: fixed-geometry sensor frames in Parquet,
written in two physical layouts from byte-identical payloads.

Why this exists: the workload this template teaches reads multi-megabyte opaque blobs
out of Parquet, and the lever this template measures on it is the *physical type* of
that blob column. A template cannot demonstrate that with someone else's dataset, and
the customer data it was derived from is private. So the template makes its own, and the
reader measures the lever on their own hardware.

Defaults are the CI-affordable knob. Raise --frames (and --width/--height) toward the
production regime; the wall clock grows with it, and whether the ratios hold at that scale is
unmeasured.

    python make_fixture.py --out /mnt/cluster_storage/fixture --frames 256
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# A single-plane 12-bit Bayer frame at this geometry is ~16.7 MB, the frame size every
# fleet figure in the README was measured at. Smaller frames are unmeasured.
DEFAULT_WIDTH = 3848
DEFAULT_HEIGHT = 2168
DEFAULT_BYTES_PER_PIXEL = 2  # 12-bit packed into uint16


def make_frames(n: int, width: int, height: int, bpp: int, seed: int = 0,
                payload: str = "sensor"):
    """Deterministic payloads. Size and uniformity are the point, not content.

    `payload` selects COMPRESSIBILITY, which turns out to matter more than anything else
    about the content:

    sensor  a gradient plus noise. Compresses the way sensor data does -- neither
            incompressible (which would flatter the fast layout) nor constant (which would
            flatter the slow one).
    random  os.urandom per row. The incompressible FLOOR, not a realistic sensor payload.
            Real data sits between the two, and closer to `sensor` for anything with
            spatial structure.
    """
    rng = np.random.default_rng(seed)
    nbytes = width * height * bpp
    for i in range(n):
        if payload == "random":
            yield os.urandom(nbytes)
            continue
        if payload == "quantized":
            # A middle entropy point: 4-bit noise widened to the full range, so it compresses
            # some but far less than the gradient. Real sensor data sits between the ends;
            # NONE of these three is real sensor data.
            q = rng.integers(0, 16, size=nbytes, dtype=np.uint16) * 256
            yield q.astype(np.uint16).tobytes()[:nbytes]
        base = np.linspace(0, 4095, nbytes, dtype=np.uint16)
        noise = rng.integers(0, 64, size=nbytes, dtype=np.uint16)
        yield ((base + noise + i) % 4096).astype(np.uint16).tobytes()[:nbytes]


def geometry_metadata(width: int, height: int, bpp: int) -> dict:
    """Frame geometry, written into the Parquet schema so it travels with the data.

    pipeline.py used to carry the geometry as a module constant while this script took
    --width/--height and its own docstring told you to raise them. A mismatch either raised
    ValueError (fixture smaller than the constant) or SILENTLY CROPPED every frame (fixture
    larger), and the silent half is the one that produces plausible wrong output. A reader
    cannot be expected to keep two files in sync, so the file answers the question.
    """
    return {
        b"frame_width": str(width).encode(),
        b"frame_height": str(height).encode(),
        b"bytes_per_pixel": str(bpp).encode(),
    }


def write_list_uint8(frames, path: Path, row_group_size: int, meta: dict,
                     compression: str = "zstd", level=None, use_dictionary: bool = True) -> float:
    """The layout that turns up by default. One Parquet INT32 value, one definition
    level and one repetition level PER BYTE."""
    tbl = pa.table(
        {
            "frame_id": pa.array(range(len(frames)), type=pa.int64()),
            "data": pa.array([list(f) for f in frames], type=pa.list_(pa.uint8())),
        }
    )
    tbl = tbl.replace_schema_metadata(meta)
    t0 = time.perf_counter()
    pq.write_table(tbl, path, compression=compression, compression_level=level,
                   use_dictionary=use_dictionary, row_group_size=row_group_size)
    return time.perf_counter() - t0


def write_fixed_binary(frames, path: Path, row_group_size: int, meta: dict,
                       compression: str = "zstd", level=None) -> float:
    """The recommended layout: FIXED_LEN_BYTE_ARRAY, required, no dictionary.
    One Parquet value per frame; the reader preallocates from the footer."""
    width = len(frames[0])
    schema = pa.schema(
        [
            pa.field("frame_id", pa.int64(), nullable=False),
            pa.field("data", pa.binary(width), nullable=False),
        ],
        metadata=meta,
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
        compression=compression,
        compression_level=level,
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
        "--compression",
        default="zstd",
        choices=["none", "snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Parquet codec. It is not what brings the two layouts to nearly the same size on "
        "disk for the default payload: dictionary encoding on the list column does that before "
        "any codec runs, and does it with `none` too. See --no-dictionary for the raw gap.",
    )
    ap.add_argument(
        "--payload",
        default="sensor",
        choices=["sensor", "quantized", "random"],
        help="`sensor` is a gradient plus noise, compressible like real frames. `random` is "
        "os.urandom, the incompressible floor -- not realistic, but it bounds the question.",
    )
    ap.add_argument(
        "--compression-level", type=int, default=None,
        help="codec level, for zstd/gzip/brotli. None uses pyarrow's default (zstd 1 in "
        "pyarrow 23; the level a fixture was written at is not recorded in the footer, so "
        "record it yourself).",
    )
    ap.add_argument(
        "--no-dictionary", action="store_true",
        help="disable dictionary encoding on the list<uint8> column. This is the single "
        "biggest lever on the ON-DISK gap and it is ON by pyarrow default: with it on, the "
        "256 possible byte values become a dictionary and indices bit-pack to 1 byte each, so "
        "the per-value bookkeeping never reaches disk. Turn it off with --compression none to see "
        "the textbook 4x; under zstd the 4x list column compresses back to 0.86x of binary(N) "
        "on pyarrow 23.0.1.",
    )
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

    meta = geometry_metadata(args.width, args.height, args.bytes_per_pixel)

    slow_total = fast_total = 0.0
    for f in range(args.files):
        frames = list(
            make_frames(
                args.frames, args.width, args.height, args.bytes_per_pixel, seed=f,
                payload=args.payload,
            )
        )
        slow_total += write_list_uint8(
            frames, out / "list_uint8" / f"part-{f:05d}.parquet", args.row_group_size, meta,
            args.compression, args.compression_level, not args.no_dictionary
        )
        fast_total += write_fixed_binary(
            frames, out / "fixed_binary" / f"part-{f:05d}.parquet", args.row_group_size, meta,
            args.compression, args.compression_level
        )

    rows = args.files * args.frames
    print(f"wrote {rows} rows x 2 layouts to {out}")
    print(f"  codec {args.compression} level {args.compression_level}, "
          f"payload {args.payload}, list dictionary {not args.no_dictionary}")
    print(f"  geometry {args.width}x{args.height} x {args.bytes_per_pixel}B "
          f"= {args.width * args.height * args.bytes_per_pixel / 1e6:.1f} MB/row, "
          f"recorded in the Parquet schema metadata")
    print(f"  list<uint8>          write: {slow_total:7.2f}s")
    print(f"  binary(N) required   write: {fast_total:7.2f}s")
    if fast_total > 0:
        print(f"  write-side ratio: {slow_total / fast_total:.2f}x")
    print(
        "\nNote: the producing team is usually told they must change their writer for the "
        "reader's benefit; measured, the recommended layout is also cheaper to WRITE."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
