# Sensor Frame Extraction: Parquet Layout for Multi-Megabyte Blobs

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/ray-data-sensor-frame-extraction"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/ray-data-sensor-frame-extraction" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: about 9 min, including cluster start (8 min 58 s to 9 min 26 s over 3 runs
on Ray 2.58.0).

Camera frames stored in Parquet as `list<uint8>` make every byte a separate Parquet value, and
decoding pays for each one. This template writes the same synthetic 16.7 MB frames as `list<uint8>`
and as `binary(N)`, reads both through a Ray Data pipeline with a GPU stage, and measures the
difference on your cluster.

**Write the frame column as `binary(N)`, `required`, with dictionary encoding off.** `binary(N)` is
Arrow's fixed-size binary type, stored as Parquet `FIXED_LEN_BYTE_ARRAY`, so each frame is one
value. Keep row groups small, since a read task can't parallelise below one row group; the fixture
uses one row per group.

The gain is in decode and write time. With pyarrow's default dictionary encoding both layouts take
the same space on disk, so the change saves no I/O, and the slower your storage, the smaller the
gain.

The notebook checks a Parquet footer for the problem, times both layouts' writes, runs the
end-to-end A/B, and shows how to find the binding stage in `ds.stats()`. `measure_layout.py`
repeats each comparison with replicates, for numbers you can quote.

## Measured results

`binary(N)` against `list<uint8>`, 16.7 MB frames. "Cluster" means the shipped compute config: one
`g6.4xlarge` worker (L4 GPU, 16 vCPUs) and an `m5.2xlarge` head, with 96 frames per layout in 4
files. Per-run figures are under Run detail, near the end.

| | `binary(N)` against `list<uint8>` | Measured on |
|---|---|---|
| Write time | 4.07x to 4.19x faster | cluster, Ray 2.57.0, 8 runs |
| | 4.01x to 4.18x faster | cluster, Ray 2.58.0, 3 runs |
| Decode, `pq.read_table`, warm cache | 10.94x faster with no codec, 4.51x with zstd | macOS arm64 laptop, pyarrow 23.0.1, 3 runs per arm |
| End to end, this pipeline | 1.56x to 1.65x faster | cluster, Ray 2.57.0, 5 runs |
| | 1.69x to 1.78x faster | cluster, Ray 2.58.0, 3 runs |
| Size on disk, zstd, dictionary on | within 1%: 831.7 MB against 836.0 MB | cluster, every run |

The rows measure different work, so the ratios differ. End to end, your gain depends on how much of
your pipeline's time goes to blob decode.

On this cluster the GPU stage was closer to its limit than the read; see Which operator binds.
Slower storage shrinks the gain. On the `g6.4xlarge` worker with codec `none`, `pq.read_table` ran
about 32x faster for `binary(N)` warm from local disk and 3.2x cold from `/mnt/cluster_storage`;
see Storage speed.

## What changes

| | Naive | This template |
|---|---|---|
| Blob column | `list<uint8>`, Parquet INT32 | `binary(N)` `required`, dictionary off |
| Row groups | writer default, often one per file | one row each, so a read task can parallelise |
| Read task CPU | `num_cpus=0.25` | `num_cpus=1.0`, Ray Data's default. No measurable difference here |
| GPU stage | one actor per GPU, `num_gpus=1` | 2 actors at 0.5 GPU each, `num_cpus=0` |
| Diagnosis | operator span | UDF time for map stages; span and bytes per second for reads |

Every lever is an environment variable in `pipeline.py`. Its comment says when to change it and
what was measured.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/ray-data-sensor-frame-extraction
```

## Read the footer first

Three fields in one Parquet footer show whether a blob column has this problem: the leaf's physical
type, its repetition level and the rows per row group. On object storage, reading them costs one
range request.

In a `list<uint8>` column every byte is a Parquet value with its own definition and repetition
level. In a `binary(N)` `required` column each frame is one value.

The next cell defines two helpers and prints nothing. `describe_blob_column` reads those fields
from the first Parquet file under a path, and `verdict` says whether the layout change applies.
Run them on your own files too. The footer can't tell you how much you'd gain; `measure_layout.py`
measures that.


```python
import glob
import os

import pyarrow.parquet as pq


def describe_blob_column(path: str, column: str = "data") -> dict:
    """Read one Parquet footer and report the fields that decide read cost."""
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no .parquet files under {path}")
    md = pq.ParquetFile(files[0]).metadata
    schema = md.schema
    leaf = next(
        schema.column(i)
        for i in range(len(schema))
        if schema.column(i).path.split(".")[0] == column
    )
    return {
        "files": len(files),
        "leaf_path": leaf.path,
        "physical_type": leaf.physical_type,
        "type_length": leaf.length or None,
        "max_definition_level": leaf.max_definition_level,
        "max_repetition_level": leaf.max_repetition_level,
        "row_groups": md.num_row_groups,
        "rows": md.num_rows,
    }


def verdict(d: dict) -> str:
    """Whether the lever applies. Not what it is worth."""
    if d["physical_type"] == "FIXED_LEN_BYTE_ARRAY" and d["max_repetition_level"] == 0:
        return "one Parquet value per frame; the reader preallocates from the footer"
    if d["max_repetition_level"] > 0:
        return ("one Parquet value plus a definition level plus a repetition level per byte; "
                "the lever applies, measure it before assuming a size")
    return "neither shape this template compares; read the levels above before tuning"
```

## Generate the fixture

`make_fixture.py` writes the same payload in both layouts, with zstd and one row per row group.
Each frame is a single-plane 12-bit Bayer frame at 3848x2168, about 16.7 MB. The geometry goes into
the Parquet schema metadata, and `pipeline.py` reads it from there.

`FRAMES` (per file) and `FILES` set the size: 24 x 4 by default, 96 frames and about 1.6 GB of
payload per layout. `FIXTURE_DIR` defaults to `/mnt/cluster_storage/sensor-fixture`. To run
smaller, lower `FRAMES` and keep the geometry: every figure in this README uses 16.7 MB frames.

The cell prints both write times and their ratio, and asserts only that `binary(N)` wrote faster.
Compare the ratio with the write rows in Measured results.


```python
import os
import re
import subprocess
import sys

FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/sensor-fixture")
FRAMES = int(os.environ.get("FRAMES", "24"))   # frames per file
FILES = int(os.environ.get("FILES", "4"))

# A read task cannot parallelise below one row group. On multi-megabyte rows small is right.
ROW_GROUP_SIZE = int(os.environ.get("ROW_GROUP_SIZE", "1"))

proc = subprocess.run(
    [sys.executable, "make_fixture.py",
     "--out", FIXTURE,
     "--frames", str(FRAMES),
     "--files", str(FILES),
     "--row-group-size", str(ROW_GROUP_SIZE)],
    capture_output=True, text=True, check=True,
)
print(proc.stdout)

# Assert the sign, never a threshold. Measured ranges are in the results table above.
slow_s = float(re.search(r"list<uint8>\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
fast_s = float(re.search(r"binary\(N\) required\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
print(f"write side: binary(N) {fast_s:.2f}s vs list<uint8> {slow_s:.2f}s "
      f"-> {slow_s / fast_s:.2f}x cheaper to write")

assert fast_s < slow_s, (
    "the recommended layout was not cheaper to write. Check the fixture, the codec and the writer "
    "before quoting the write-side figures above."
)
```

The next cell prints each layout's footer fields, the verdict and the size on disk. Expect
`FIXED_LEN_BYTE_ARRAY` with repetition level 0 for `fixed_binary`, `INT32` with repetition level 1
for `list_uint8`, and sizes within 1% of each other.


```python
for layout in ["fixed_binary", "list_uint8"]:
    d = describe_blob_column(os.path.join(FIXTURE, layout))
    print(f"{layout}:")
    for key, value in d.items():
        print(f"    {key:24s} {value}")
    print(f"    -> {verdict(d)}")

print()
for layout in ["fixed_binary", "list_uint8"]:
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(FIXTURE, layout, "*.parquet")))
    print(f"{layout:14s} {total / 1e6:8.1f} MB on disk")
print("\nClose numbers mean the layout buys no I/O and any win comes from decode. Dictionary "
      "encoding closes this gap, not the codec. On this fleet: 831.7 MB against 836.0 MB. "
      "Run `--sweep on-disk` for the grid.")
```

## Why the sizes match

`list<uint8>` has Parquet physical type INT32, so each payload byte takes 4 bytes before encoding.
That is the 4x in the usual advice. Dictionary encoding (`RLE_DICTIONARY`) removes it before any
codec runs: a byte has 256 possible values, so each one becomes a dictionary index of about 1 byte.

Footer `total_uncompressed_size`, which is after encoding and before the codec, for one 16.7 MB
frame, pyarrow 23.0.1:

| | encoded bytes | against `binary(N)` |
|---|---|---|
| `binary(N)` `required`, no dictionary | 16,684,961 | |
| `list<uint8>`, dictionary on, pyarrow's default | 16,719,140 | 1.00x |
| `list<uint8>`, dictionary off | 66,739,776 | 4.00x |

Encodings: `('PLAIN', 'RLE', 'RLE_DICTIONARY')` with the dictionary on, `('RLE', 'PLAIN')` with it
off.

The grid behind these figures, `measure_layout.py --sweep on-disk`, covers 6 codecs (`none`,
`snappy`, `gzip`, `brotli`, `lz4`, `zstd`), both dictionary settings and three payloads: `sensor`
(a gradient plus noise, the default), `quantized` (16 distinct byte values) and `random`
(`os.urandom`). None of them is real sensor data.

The flag is `use_dictionary` on the list column. People turn it off for high-cardinality columns;
on a `list<uint8>` blob column that costs up to 4x the space. The encoded gap was 4.00x in every
dictionary-off configuration. With the dictionary on it was 1.00x for `sensor` and `random`, and
0.50x, list smaller, for `quantized`, whose 16 byte values pack into 4-bit indices. Row groups of
1, 4 and 16 rows did not change it.

The codec plays no part: the gap closes with codec `none` and on incompressible `random` data. With
the dictionary off, zstd shrinks the 4x `list<uint8>` column to 0.86x of `binary(N)`, since INT32
values with 3 zero bytes compress well. On-disk size doesn't predict the decode cost in either
direction.

Only the list column's flag matters: setting it on `binary(N)` too moved that column's encoded size
by 17 bytes out of 16.7 MB, across two payloads and two codecs. `make_fixture.py` writes
`binary(N)` with the dictionary off and `list<uint8>` at pyarrow's default, on; `--no-dictionary`
turns it off for both.

These figures are pyarrow 23.0.1, the base image's version. On pyarrow 25.0.1 the dictionary-off
compressed gaps differ (zstd, `sensor`: 1.28x against 0.86x); the uncompressed 4.00x and 1.00x held
on both.

### Write side

The writer still pays per byte: it encodes every byte of a `list<uint8>` column as a separate value
before any codec runs, so the write-side gap stays while the on-disk gap closes. The recommended
layout saves the producing team write time as well.

## Dependencies

The base image ships numpy, pyarrow and Ray, and no torch; `cu129` in the image tag is the CUDA
runtime. The GPU stage needs torch, so the template pins it in `python_depset.lock`, compiled
against a `pip freeze` of the image.

The lock has to reach two places. The first cell below checks that it exists, and raises with the
command that compiles it if not; the second installs it on the driver with
`uv pip install --system`. The `map_batches` actors get it from
`ray.init(runtime_env={"pip": ...})` in `pipeline.py`. Install it on the driver alone and the actors
run what the image shipped: that works in a workspace, which propagates a plain `pip install`, and
fails as a standalone Job or Service.


```python
import pathlib

LOCK = pathlib.Path("python_depset.lock")
if not LOCK.is_file():
    raise FileNotFoundError(
        """python_depset.lock is not in this template directory.

It is generated. Add this template's requirements.txt, register a compile entry in
dependencies/template.depsets.yaml with include_setuptools: true and a seed-image-freeze.sh
pre_hook naming the image the BUILD.yaml entry runs on, then run

    ./scripts/depsets/update_deps.sh --name <that entry>

from the repository root and commit the result. The cells below need torch, and the base image
does not ship it."""
    )
print(f"{LOCK} present ({LOCK.stat().st_size} bytes)")
```


```python
import subprocess

# subprocess with check=True, because a `!command` cell cannot fail the notebook: papermill runs
# the next cell and exits 0 when the command exits non-zero.
#
# One shell string, because the dependency-delivery pre-commit check looks for the requirements
# flag followed by the lock filename, and an argv list separates them. Don't quote this command in
# a markdown cell: the check reads the whole notebook, and a quoted copy passes it without
# installing anything.
subprocess.run(
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match",
    shell=True, check=True,
)
```

## The end-to-end A/B

Both arms, one per layout, run the same `pipeline.py` on the same payload bytes; only the input path
changes. After a warmup the cell times one run per layout, prints each run's rate and `ds.stats()`,
then the ratio, and asserts only that `binary(N)` was faster. One run per arm gives the direction;
`measure_layout.py` gives a magnitude. On this cluster the ratio was 1.56x to 1.78x over 8 runs on
Ray 2.57.0 and 2.58.0.

Keep the warmup. The first pipeline run on a fresh cluster also pays for Ray building the
`runtime_env` virtualenv from the lock on the worker: 85.1 s to 96.8 s, against 5.7 s to 5.9 s for
the next run, over 3 runs on each Ray version. That cost lands in whichever arm runs first. Without
the warmup this cell measured 1.07 rows/s against 10.07, a 0.11x inversion, on Ray 2.57.0.


```python
import re
import subprocess
import sys


def run_arm(layout: str) -> float:
    """Run pipeline.py over one layout and return the rows/s it reports."""
    proc = subprocess.run(
        [sys.executable, "pipeline.py", "--input", os.path.join(FIXTURE, layout)],
        capture_output=True, text=True, check=True,
    )
    print(proc.stdout)
    match = re.search(r"=\s*([0-9.]+) rows/s", proc.stdout)
    if not match:
        raise AssertionError(f"pipeline.py printed no rate for {layout}")
    return float(match.group(1))


print("warmup run, discarded; it pays for the runtime_env build on the worker")
run_arm("fixed_binary")

fast = run_arm("fixed_binary")
slow = run_arm("list_uint8")

print(f"fixed_binary {fast:.2f} rows/s vs list<uint8> {slow:.2f} rows/s -> {fast / slow:.2f}x")
print("End-to-end channel. Decode alone is much larger from byte-identical inputs; the GPU stage "
      "takes most of the difference out of the wall clock here. See the results table in this "
      "README for both figures.")

assert fast > slow, (
    "the recommended layout was not faster to read. The expected end-to-end margin is small, "
    "see the results table in this README, so an inversion is plausible on other hardware. Run "
    "measure_layout.py with replicates and read the binding-operator section below."
)
```

## Which operator binds

`pipeline.py` prints `ds.stats()`. Rank map stages by UDF time. Operator spans overlap under the
streaming executor, so they don't rank stages; on the source workload they summed to 2.1x the
pipeline's wall clock.

A map stage's saturation is its UDF time divided by span x actors, and near 1.0 the stage is the
limit. The read has no user function, so Ray reports `UDF time: 0us` for it and the ratio is 0
however hard the read works. Judge the read by its span and its output bytes per second against
what your storage delivers.

On this cluster the read's span was 2.39 s to 2.6 s and the GPU stage ran at 0.81 to 0.96
saturation, over 4 runs on Ray 2.57.0 and 2.58.0. At this scale the GPU stage is closer to the
limit than the read. Per-run spans are in Run detail.

## Where the decode gain goes

Decode rates, warm cache, `pq.read_table`, 3 timed runs after a warmup per arm, on a macOS arm64
laptop with pyarrow 23.0.1. Ranges don't overlap.

| configuration | bytes on disk | `binary(N)` | `list<uint8>` | ratio |
|---|---|---|---|---|
| no codec, dictionary on | identical | 12,936-13,783 MB/s | 1,065-1,384 | 10.94x |
| zstd, dictionary on, the shipped default | 1.00x | 6,355-9,356 MB/s | 1,102-1,503 | 4.51x |
| no codec, dictionary off | list moves 4x | 10,459-12,259 MB/s | 1,039-1,378 | 8.30x |

The first two rows hold I/O constant, so they isolate decode. The third doesn't.

A codec lowers the ratio without changing which layout wins. Decompression costs both arms and
dominates the fast one: `binary(N)` fell from about 13,100 to about 6,400 MB/s under zstd, while
`list<uint8>` stayed near 1,200. A ratio measured under compression understates the decode gap.

In this pipeline the GPU stage is closer to the limit, so most of a 4.51x decode gain doesn't reach
the wall clock. `measure_layout.py --sweep storage` times the no-codec case on a cluster, in the
next section; no shipped sweep times the zstd row.

## Storage speed

With dictionary encoding on, both layouts occupy the same bytes, so a bigger fixture can't create
an I/O difference. Slow storage or dictionary encoding off could still add an I/O component, so
both were measured on the `g6.4xlarge` worker, where the read tasks run. Codec `none` keeps the full
4x byte gap when the dictionary is off. Cold means `fsync`, then `posix_fadvise(POSIX_FADV_DONTNEED)`
to drop the file's cached pages. 12 frames of 16.7 MB, 2 runs per arm, every comparison
non-overlapping.

Cold mount throughput on the worker: `/mnt/local_storage` 1906.6 MB/s, `/mnt/cluster_storage`
138.2 MB/s. The head measured 141.5 MB/s on the shared mount.

| mount | list dictionary | bytes on disk | cold ratio | warm ratio |
|---|---|---|---|---|
| `/mnt/local_storage` | on | 1.00x | 12.16-22.11x | 31.92-32.33x |
| `/mnt/local_storage` | off | 4.00x | 21.61-21.74x | 31.30-31.55x |
| `/mnt/cluster_storage` | on | 1.00x | 3.17-3.23x | 32.40-32.73x |
| `/mnt/cluster_storage` | off | 4.00x | 4.48-4.72x | 31.16-31.67x |

Slow storage shrinks the advantage, from 32x warm on local disk to 3.2x cold on the shared mount.
The mount limits `binary(N)`, which fell from 3,980 MB/s warm to 296 MB/s cold there, while
`list<uint8>` is CPU-bound and barely moves. Across all 8 arms `list<uint8>` decoded 65.3 to
127.5 MB/s of payload, whatever the mount or dictionary setting, against 292.8 to 4,094.2 MB/s for
`binary(N)`.

Dictionary off adds a small I/O component. Cold on the shared mount the ratio rises from 3.17x-3.23x
to 4.48x-4.72x once `list<uint8>` moves 4x the bytes. Cold on local disk it goes from 12.16x-22.11x
to 21.61x-21.74x.

Not timed: zstd, where the byte gap closes; the full Ray Data pipeline on these mounts, since these
are `pq.read_table` figures; and anything above 12 frames.

## Measure your own data

`measure_layout.py` runs each comparison with replicates.

```bash
python measure_layout.py --input /tmp/grid --sweep on-disk                 # bytes at rest, no cluster
python measure_layout.py --input /tmp/grid --sweep storage --runs 2        # mounts, cold and warm
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep layout --runs 3
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep read-cpus --runs 3
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep decode-threads --runs 3
```

`--sweep on-disk` needs only pyarrow. It prints every codec x payload x dictionary configuration
with both footer sizes, a zstd level sweep at 1, 3, 9 and 22 (pyarrow 23 defaults to 1), and a
row-group sweep. `--sweep storage` needs a cluster, since `/mnt/cluster_storage` only exists on one;
it runs as a Ray task requesting a CPU, so it lands on a worker and not on a head pinned to
`CPU: 0`. Both generate their own payloads, and `--input` is required but unused.

The other sweeps run `pipeline.py` over `fixed_binary/` and `list_uint8/` under `--input`. For your
own data, write it both ways there: `pipeline.py` expects `frame_id` and `data` columns of 16-bit
frames, and needs `FRAME_WIDTH` and `FRAME_HEIGHT` if the files carry no geometry metadata. To test
another lever, add an arm to `SWEEPS`.

Each sweep needs at least 2 timed runs per arm and discards 1 warmup run per arm by default.
`SEPARABLE` quotes the margin between the worst run of the better arm and the best run of the worse
one. `OVERLAP` means more runs are needed, or there is no real difference.

Codec names: pyarrow rejects `"uncompressed"`; use `"none"`, which the footer reports as
`UNCOMPRESSED`. pyarrow 23.0.1 also rejects `"lz4_raw"`, but its `"lz4"` writes the LZ4_RAW codec,
which pyarrow's metadata labels `LZ4`.

## Read task CPU and decoder threads

Ray sets `OMP_NUM_THREADS = max(1, floor(num_cpus))` for a worker when the variable isn't already
set, and pyarrow sizes its CPU thread pool from it once per worker process. A read task at
`num_cpus=0.25` therefore decodes with one thread. On thin rows and many small files that buys
concurrency. On a multi-megabyte blob column, decode is most of the read's work.

Size decoder threads so that threads per task x concurrent read tasks is about the core count.
Beyond that, more threads cost wall clock: +52% at 8 threads on the source workload. `pipeline.py`
exposes `READ_OMP_THREADS`, scoped to the read operator through a `runtime_env` so it doesn't
resize every actor in the job. The default, 0, leaves it alone.

On this cluster neither lever moved much: `num_cpus=1.0` against `0.25` overlapped, and 1 to 4
decoder threads gained 1.2% or better. With 4 files there are 4 read tasks on 16 vCPUs, so the read
isn't CPU-bound and a one-thread decoder isn't the constraint.

## Scale and units

Throughput that stays flat as you add concurrency but grows linearly with nodes has hit a per-node
limit. Add nodes or change the layout rather than raising a knob, and quote throughput with the
node count that produced it.

State the unit. On a multi-camera rig, rows/s, frames/s and camera-frames/s differ by the camera
count, and a real-time target is quoted in one of them. `pipeline.py` reports rows/s.

## Run detail

All cluster runs used the shipped compute config and torch 2.9.1+cu129. Ray 2.57.0 runs were
`tests.sh` (papermill over this notebook) or `measure_layout.py`; Ray 2.58.0 runs were 3 test runs
of this notebook on 2026-09-24, on `anyscale/ray:2.58.0-py312-cu129`. End-to-end rates are the
notebook's A/B, 1 timed run per arm after the warmup; spans are from `ds.stats()` of the timed
`binary(N)` run.

| | Ray 2.57.0 | Ray 2.58.0 |
|---|---|---|
| Write, `binary(N)` against `list<uint8>` | 8 runs, spread 2.9% | 12.65-13.39 s against 52.94-54.14 s, 3 runs |
| End to end, rows/s | 15.92-16.81 against 9.94-10.21, 5 runs | 16.19-16.72 against 9.20-9.61, 3 runs |
| Read span, read UDF time | 2.44 s, 0us | 2.39-2.6 s, 0us on all 3 |
| GPU stage span, UDF across 2 actors, saturation | 1.99 s, 3.4 s, about 0.85 | 1.9-2.06 s, 3.33-3.63 s, 0.81-0.96 |
| Pipeline wall clock, timed `binary(N)` run | 5.8 s | 5.7-5.9 s |
| First run on a fresh cluster, then the next | 85.1-94.0 s, then 5.8 s and 5.9 s, 3 runs | 89.1-96.8 s, then 5.7-5.9 s, 3 runs |
| Notebook wall clock, `tests.sh` | 335-354 s, 6 runs | not measured |
| Whole test run, workspace start and teardown included | not measured | 8 min 58 s to 9 min 26 s, 3 runs |
| Fixture on disk, `binary(N)` against `list<uint8>` | 831.7 MB against 836.0 MB | the same, all 3 runs |
| Read `num_cpus` 1.0 against 0.25, `measure_layout.py` | 16.67-17.00 against 16.24-16.87 rows/s, overlapping | not run |
| Decoder threads, `measure_layout.py` | 1 to 4: 1.2% or better; 4 to 8: not separable at 2 runs per arm | not run |

On Ray 2.58.0 the end-to-end ratio sat above the 2.57.0 range on all 3 runs: `list<uint8>` ran below
its 2.57.0 range while `binary(N)` stayed inside its own. That is 1 timed run per arm, and the cause
is unmeasured.

## Provenance

The pipeline shape and levers come from a customer workload whose data is private; the fixture is
synthetic. That workload reported these figures at production scale, on different hardware and
data. None is reproduced here.

| Lever | Reported |
|---|---|
| Layout, read side | 7x to 23x, depending on the pair compared |
| Capping read concurrency | 1.66x to 1.91x when a CPU stage downstream of the read was binding; 0.56x on a read-bound pipeline |
| Decoder threads | up to 2.15x on local disk, about 1.6x from object storage, +52% wall clock at 8 threads |
| GPU packing | 8 actors at 0.5 GPU each, on 4 of 8 GPUs |
| Object-store fraction | 0.6, peak 69.4 GiB, no spill |

Not measured here: whether the read binds at production scale. At 96 frames the fixture's 1.6 GB of
payload per layout fits in page cache, and 4 files give 4 read tasks on a 16-vCPU worker, so the
read doesn't become CPU-bound. A larger fixture might move these results toward the source
workload's; that is untested.

## Where to go next

- [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html)
- [Reading Parquet](https://docs.ray.io/en/latest/data/loading-data.html#reading-files) and [`map_batches`](https://docs.ray.io/en/latest/data/transforming-data.html)
- [ComputeConfig reference](https://docs.anyscale.com/reference/compute-config-api#computeconfig). This template's compute config is in `configs/ray-data-sensor-frame-extraction/`.
