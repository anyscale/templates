#!/usr/bin/env python3
"""Write the template's fixture: fixed-geometry sensor frames in two Parquet layouts,
list<uint8> and binary(N), from byte-identical payloads.

The source data is private, so the template generates its own frames and you measure the
layout on your own hardware, with no dataset to stage.

The notebook writes 24 frames per file; this script's default is 64. Raising --frames (or
--width/--height) toward production scale grows the wall clock and the driver's memory: the
list<uint8> writer holds each file's frames as Python lists, about 8 bytes of RAM per payload
byte. Whether the ratios hold at larger scale is unmeasured.

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

# A single-plane 12-bit Bayer frame at this geometry is about 16.7 MB. Every cluster figure in
# the README uses it; other sizes are unmeasured.
DEFAULT_WIDTH = 3848
DEFAULT_HEIGHT = 2168
DEFAULT_BYTES_PER_PIXEL = 2  # 12-bit packed into uint16


def make_frames(n: int, width: int, height: int, bpp: int, seed: int = 0,
                payload: str = "sensor"):
    """Payloads for both layouts; size and uniformity matter here, not content. Seeded,
    except `random`. `payload` sets how varied and how compressible the bytes are:

    sensor     a gradient plus noise, 256 distinct byte values. Neither constant nor
               incompressible.
    quantized  4-bit noise scaled to the 12-bit range: 16 distinct byte values. zstd
               compresses it further than `sensor`.
    random     os.urandom per row, incompressible.

    None of the three is real sensor data.
    """
    rng = np.random.default_rng(seed)
    nbytes = width * height * bpp
    for i in range(n):
        if payload == "random":
            yield os.urandom(nbytes)
            continue
        if payload == "quantized":
            # 4-bit noise in the high byte; the low byte is always 0.
            q = rng.integers(0, 16, size=nbytes, dtype=np.uint16) * 256
            yield q.astype(np.uint16).tobytes()[:nbytes]
        base = np.linspace(0, 4095, nbytes, dtype=np.uint16)
        noise = rng.integers(0, 64, size=nbytes, dtype=np.uint16)
        yield ((base + noise + i) % 4096).astype(np.uint16).tobytes()[:nbytes]


def geometry_metadata(width: int, height: int, bpp: int) -> dict:
    """Frame geometry for the Parquet schema metadata, so it travels with the data.

    pipeline.py reads it back, so changing --width/--height here needs no change there.
    """
    return {
        b"frame_width": str(width).encode(),
        b"frame_height": str(height).encode(),
        b"bytes_per_pixel": str(bpp).encode(),
    }


def write_list_uint8(frames, path: Path, row_group_size: int, meta: dict,
                     compression: str = "zstd", level=None, use_dictionary: bool = True) -> float:
    """The common default layout: one Parquet INT32 value per byte, each with a definition
    and a repetition level. Returns the pq.write_table time in seconds."""
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
    """The recommended layout: FIXED_LEN_BYTE_ARRAY, required, no dictionary. One Parquet
    value per frame; the reader preallocates from the footer. Returns the pq.write_table time
    in seconds."""
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
    ap.add_argument("--out", required=True,
                    help="fixture root; the layouts go to list_uint8/ and fixed_binary/ under it")
    ap.add_argument("--frames", type=int, default=64, help="frames per file")
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--bytes-per-pixel", type=int, default=DEFAULT_BYTES_PER_PIXEL)
    ap.add_argument(
        "--compression",
        default="zstd",
        choices=["none", "snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Parquet codec. For the default payload the layouts end up nearly the same size on "
        "disk because of dictionary encoding on the list column, with any codec including "
        "`none`; see --no-dictionary for the raw gap.",
    )
    ap.add_argument(
        "--payload",
        default="sensor",
        choices=["sensor", "quantized", "random"],
        help="`sensor`: a gradient plus noise. `quantized`: 16 distinct byte values. "
        "`random`: os.urandom, incompressible. None of them is real sensor data.",
    )
    ap.add_argument(
        "--compression-level", type=int, default=None,
        help="codec level for zstd, gzip or brotli. The default is pyarrow's (zstd level 1 in "
        "pyarrow 23). The footer does not record the level, so note it yourself.",
    )
    ap.add_argument(
        "--no-dictionary", action="store_true",
        help="turn off dictionary encoding on the list<uint8> column too. pyarrow turns it on "
        "by default, and it is what closes the on-disk gap: the 256 possible byte values become "
        "dictionary indices of about 1 byte each, so the INT32 expansion never reaches disk. "
        "With --compression none this shows the textbook 4x; under zstd the 4x list column "
        "compresses to 0.86x of binary(N) on pyarrow 23.0.1.",
    )
    ap.add_argument(
        "--row-group-size",
        type=int,
        default=1,
        help="rows per row group. A read task cannot split below one row group, so keep it "
        "small for multi-MB rows, unlike narrow tabular data.",
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
