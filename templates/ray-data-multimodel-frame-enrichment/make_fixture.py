#!/usr/bin/env python3
"""Generate the frame fixture. No dataset licence, no download, no customer data.

    python make_fixture.py --out /mnt/cluster_storage/frames --frames 48 --files 4
    python make_fixture.py --out ... --width 3848 --height 2168   # production geometry

WHY SYNTHETIC

Public image sets are available and several are permissively licensed (Open Images V7 is
CC-BY-4.0 both ways), but each adds a download, an attribution obligation and a licence to
track. Generating the frames removes all three.

WHAT MAY SHRINK

Frame count and frame geometry, for CI. Neither carries the lesson.

Do not trim the resident-model count to fit a budget. CI does run two of the four, because
two are gated on Hugging Face and licensed per account, so CI has no right to their weights;
`pipeline.py --ungated-only` names the pair it ran. That is a licence, not a budget.

Each frame carries a few high-contrast shapes on a textured background, so a promptable
detector has something to find. A pure-noise fixture returns zero detections and the
downstream stages then measure nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Fixed so the fixture is reproducible: the same --seed and geometry produce the same bytes.
DEFAULT_SEED = 20260817


def one_frame(rng: np.random.Generator, w: int, h: int, n_shapes: int) -> tuple[np.ndarray, list]:
    """A textured RGB frame plus the ground-truth boxes of the shapes drawn into it."""
    # Low-frequency background, so the detector is not scoring against flat colour.
    # Round the tile count up and crop. Rounding down leaves a black strip along any edge
    # whose geometry is not a multiple of the tile (240 is not a multiple of 64), and a
    # detector will fire on the band.
    tile = 64
    small = rng.integers(
        40, 90,
        size=(-(-h // tile), -(-w // tile), 3),
        dtype=np.uint8,
    )
    bg = np.repeat(np.repeat(small, tile, axis=0), tile, axis=1)[:h, :w]
    frame = np.ascontiguousarray(bg)
    assert frame.shape == (h, w, 3), f"background is {frame.shape}, wanted {(h, w, 3)}"

    boxes = []
    for _ in range(n_shapes):
        bw = int(rng.integers(w // 12, max(w // 12 + 1, w // 5)))
        bh = int(rng.integers(h // 12, max(h // 12 + 1, h // 5)))
        x = int(rng.integers(0, max(1, w - bw)))
        y = int(rng.integers(0, max(1, h - bh)))
        colour = rng.integers(180, 256, size=3, dtype=np.uint8)
        if rng.random() < 0.5:
            frame[y : y + bh, x : x + bw] = colour
        else:  # a filled ellipse, so not every object is axis-aligned and rectangular
            yy, xx = np.ogrid[:bh, :bw]
            cy, cx = bh / 2, bw / 2
            mask = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2 <= 1.0
            region = frame[y : y + bh, x : x + bw]
            region[mask] = colour
        boxes.append([x, y, x + bw, y + bh])
    return frame, boxes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--files", type=int, default=4,
                    help="one file per read task. 1 file of 2 row groups yields 2 blocks and "
                         "2 usable workers, whatever the cluster size")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--shapes", type=int, default=4)
    ap.add_argument("--row-group-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_file = max(1, args.frames // args.files)
    frame_bytes = args.width * args.height * 3
    written = 0
    truth: dict[str, list] = {}

    for f in range(args.files):
        count = per_file if f < args.files - 1 else args.frames - per_file * (args.files - 1)
        if count <= 0:
            continue
        ids, blobs = [], []
        for i in range(count):
            frame, boxes = one_frame(rng, args.width, args.height, args.shapes)
            fid = f"frame-{written + i:06d}"
            ids.append(fid)
            blobs.append(frame.tobytes())
            truth[fid] = boxes
        table = pa.table({
            "frame_id": pa.array(ids, pa.string()),
            # A fixed-size binary column: every row is the same multi-hundred-KB blob, so the
            # Parquet file is a catalogue around a blob store.
            "image": pa.array(blobs, pa.binary(frame_bytes)),
            "width": pa.array([args.width] * count, pa.int32()),
            "height": pa.array([args.height] * count, pa.int32()),
        })
        pq.write_table(table, out / f"part-{f:04d}.parquet", row_group_size=args.row_group_size)
        written += count

    (out / "ground_truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{written} frames, {args.files} file(s), {frame_bytes / 1e6:.2f} MB per frame, "
          f"{written * frame_bytes / 1e9:.3f} GB total -> {out}")
    print(f"ground truth for {len(truth)} frames -> {out / 'ground_truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
