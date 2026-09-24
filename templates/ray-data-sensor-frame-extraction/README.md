# Sensor Frame Extraction: Parquet Layout for Multi-Megabyte Blobs

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/ray-data-sensor-frame-extraction"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/ray-data-sensor-frame-extraction" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: about 9½ min, measured. Two `rayapp test` runs on Ray 2.58.0 took
9 min 26 s on prod and 9 min 17 s on staging, workspace start and teardown included; see
Provenance. Notebook execution alone measured 5.6 to 5.9 min on Ray 2.57.0.

This template reads multi-megabyte sensor frames from Parquet, transforms them on a GPU actor pool,
and writes them back. It ships the same payload in two Parquet layouts and measures the difference.

The layout choice costs decode CPU, not I/O. On disk the two layouts are within 1% of each other
with dictionary encoding on, and 4.00x apart with it off. Arrow decodes `binary(N)` 10.94x faster
than `list<uint8>` from byte-identical inputs, 4.51x under zstd, and 1.56x to 1.65x end to end
once the GPU stage is attached, on Ray 2.57.0. Two runs on Ray 2.58.0 measured 1.76x and 1.78x
end to end.

The advantage is largest on fast storage and smallest on slow storage, which is what a decode
lever looks like. On `/mnt/cluster_storage` cold it falls to 3.2x.

`measure_layout.py` measures bytes at rest and decode separately. Point it at your own Parquet.

## Measured results

One `g6.4xlarge` L4 worker, `m5.2xlarge` head, Ray 2.57.0, `torch 2.9.1+cu129`. 96 frames of
16.7 MB in 4 files per layout, from `configs/ray-data-sensor-frame-extraction/`. Local decode
figures are macOS arm64, pyarrow 23.0.1, warm page cache.

Rows marked Ray 2.58.0 are two `rayapp test` runs of this notebook on 2026-09-24, one on the
Anyscale prod console and one on staging. Same fleet, `anyscale/ray:2.58.0-py312-cu129`,
`torch 2.9.1+cu129`, one fixture write and one timed run per layout after the warmup, each. Where
a row gives two 2.58.0 values, prod is first.

| lever | result | instrument |
|---|---|---|
| Layout, bytes at rest, dictionary on | 1.00x. 0.50x on the quantized payload, list smaller | footer `total_uncompressed_size`, 6 codecs x 3 payloads |
| Layout, bytes at rest, dictionary off | 4.00x | same grid, every dictionary-off cell |
| Layout, bytes at rest, shipped fixture, Ray 2.58.0 | `binary(N)` 831.7 MB against 836.0 MB on both runs | file sizes, zstd, dictionary on |
| Layout, decode CPU, no codec | 10.94x. 12,936-13,783 MB/s against 1,065-1,384 | `pq.read_table`, 3 timed runs + warmup per arm |
| Layout, decode CPU, zstd | 4.51x. 6,355-9,356 MB/s against 1,102-1,503 | same |
| Layout, write side | 4.07x to 4.19x on 8 fleet runs, spread 2.9% | `make_fixture.py` timings |
| Layout, write side, Ray 2.58.0 | 4.18x and 4.04x on 2 runs. 12.65 s and 13.39 s against 52.94 s and 54.14 s | `make_fixture.py` timings, 1 per run |
| Layout, end to end with the GPU stage | 1.56x to 1.65x on 5 fleet runs. 15.92-16.81 rows/s against 9.94-10.21 | fleet, 3 timed runs per arm |
| Layout, end to end, Ray 2.58.0 | 1.76x and 1.78x on 2 runs, above the 2.57.0 range on both. 16.19 and 16.72 rows/s against 9.20 and 9.41 | the notebook's A/B, 1 timed run per arm |
| Read task `num_cpus` 1.0 against 0.25 | no measurable difference. 16.67-17.00 against 16.24-16.87 rows/s, ranges overlap | fleet |
| Decoder threads 1 to 4 | 1.2% or better. 4 to 8 not separable at 2 runs per arm | fleet |
| Row-group size 1, 4, 16 rows | no effect on the on-disk gap | footer, zstd |
| Binding operator | read span 2.44 s, GPU stage span 1.99 s with 3.4 s UDF across 2 actors | `ds.stats()` |
| Binding operator, Ray 2.58.0 | read span 2.6 s and 2.39 s, read UDF time 0us on both. GPU stage span 1.9 s and 2.05 s with 3.63 s and 3.33 s UDF across 2 actors | `ds.stats()`, timed `binary(N)` run |
| Layout on local disk, cold | 12.16x to 22.11x. Warm 31.92x to 32.33x | `pq.read_table` on the worker, 2 runs per arm |
| Layout on `/mnt/cluster_storage`, cold | 3.17x to 3.23x with equal bytes, 4.48x to 4.72x with dictionary off | same |
| Mount cold throughput | local 1906.6 MB/s, `/mnt/cluster_storage` 138.2 MB/s | sequential read after `posix_fadvise` |

The first nine rows are one lever in four channels. They disagree. Which one you get depends on how
much of your pipeline is blob decode.

End to end on Ray 2.58.0 is above the 2.57.0 range on both runs, 1.76x and 1.78x against 1.56x to
1.65x. `binary(N)` at 16.19 and 16.72 rows/s is inside its 2.57.0 range, and `list<uint8>` at 9.20
and 9.41 is below its own. That is 2 runs at 1 timed run per arm. Why it moved is unmeasured.

## Why the on-disk gap closes

`list<uint8>` gets Parquet physical type INT32, so a payload byte occupies 4 bytes. That is the 4x
in the usual advice.

`RLE_DICTIONARY` removes it before any codec runs. There are 256 distinct byte values, so the
dictionary holds 256 entries and each value becomes about 1 index byte. Footer
`total_uncompressed_size` on a 16.7 MB frame, pyarrow 23.0.1:

| | encoded bytes | against `binary(N)` |
|---|---|---|
| `binary(N)` `required`, no dictionary | 16,684,961 | |
| `list<uint8>`, dictionary on, pyarrow's default | 16,719,140 | 1.00x |
| `list<uint8>`, dictionary off | 66,739,776 | 4.00x |

Encodings: `('PLAIN', 'RLE', 'RLE_DICTIONARY')` with the dictionary on, `('RLE', 'PLAIN')` with it
off.

The flag is `use_dictionary` on the list column. Off, the layout is worth up to 4x at rest, and
people do turn it off for high-cardinality columns. On, it is worth nothing at rest. The encoded gap
was 4.00x in every dictionary-off cell, across `none`, `snappy`, `gzip`, `brotli`, `lz4` and `zstd`
and across all three payloads including `os.urandom`. With the dictionary on it was 1.00x for the
gradient and `os.urandom` payloads and 0.50x for the quantized one, list smaller: its bytes take 16
distinct values, so each index packs into 4 bits. Row-group size at 1, 4 and 16 rows did not
change it.

Compression is not the mechanism. The effect is complete with codec `none` and survives an
incompressible payload. With the dictionary off, zstd compresses the 4x-larger `list<uint8>` to
0.86x of `binary(N)`, below parity, because INT32 with 3 zero bytes per value compresses well.
On-disk size does not predict this lever in either direction.

Only the list column's flag matters. Varying `use_dictionary` on `binary(N)` too, across both
payloads and both codecs, moved its encoded size by 17 bytes out of 16.7 MB. `make_fixture.py`
writes `binary(N)` with `use_dictionary=False` and `list<uint8>` at pyarrow's default of `True`.
`--no-dictionary` writes the symmetric cell.

Version dependence: these figures are pyarrow 23.0.1, which the base image ships. On pyarrow 25.0.1
the dictionary-off compressed gaps differ, 1.28x against 0.86x for zstd on the gradient payload. The
uncompressed 4.00x and 1.00x split held on both.

### Where the win is: decode

Same codec, same cell, warm cache, `pq.read_table`, 3 timed runs plus a warmup per arm. Ranges do
not overlap.

| cell | I/O held constant | `binary(N)` | `list<uint8>` | ratio |
|---|---|---|---|---|
| no codec, dictionary on | yes, identical bytes | 12,936-13,783 MB/s | 1,065-1,384 | 10.94x |
| zstd, dictionary on, the shipped default | yes, 1.00x on disk | 6,355-9,356 MB/s | 1,102-1,503 | 4.51x |
| no codec, dictionary off | no, list moves 4x the bytes | 10,459-12,259 MB/s | 1,039-1,378 | 8.30x |

Row 3 does not separate decode from I/O. Rows 1 and 2 do.

A codec lowers the measured ratio without changing the layout choice. Decompression costs both arms
and dominates the fast one: `binary(N)` fell from about 13,100 to about 6,400 MB/s under zstd while
`list<uint8>` stayed near 1,200. A ratio measured under compression understates the decode gap.

With this template's GPU stage attached, the same lever measured 1.56x to 1.65x end to end across 5
fleet runs on Ray 2.57.0, and 1.76x and 1.78x across 2 on Ray 2.58.0. Measure the channel you can
spend.

### Storage speed and the byte asymmetry, measured

An earlier version of this template said, as an inference, that a fixture large enough to leave
page cache would restore an I/O component. That claim was wrong in its framing. With dictionary
encoding on the two layouts occupy the same bytes, so there is no I/O asymmetry to restore at any
scale. Two variables can produce one and both were measured, on the `g6.4xlarge` worker where the
read tasks run, codec `none` throughout because that is the only configuration where the byte gap
exists. Cold means `fsync` then `posix_fadvise(POSIX_FADV_DONTNEED)` on the file, so the pages are
genuinely gone. 12 frames of 16.7 MB, 2 runs per arm, every comparison non-overlapping.

Mount throughput on the worker: `/mnt/local_storage` 1906.6 MB/s cold, `/mnt/cluster_storage`
138.2 MB/s cold, 13.8x apart. The head node measured 141.5 MB/s for the shared mount, so it is
consistent across nodes.

| mount | list dictionary | bytes on disk | cold ratio | warm ratio |
|---|---|---|---|---|
| `/mnt/local_storage` | on | 1.00x | 12.16-22.11x | 31.92-32.33x |
| `/mnt/local_storage` | off | 4.00x | 21.61-21.74x | 31.30-31.55x |
| `/mnt/cluster_storage` | on | 1.00x | 3.17-3.23x | 32.40-32.73x |
| `/mnt/cluster_storage` | off | 4.00x | 4.48-4.72x | 31.16-31.67x |

**Slow storage shrinks the layout advantage, it does not grow it.** The ratio falls from 32x warm
on local disk to 3.2x cold on the shared mount. The mount caps the fast layout, `binary(N)` going
from 3,980 MB/s warm to 296 MB/s cold there, while `list<uint8>` is already CPU-bound and barely
moves. Across all 8 arms `list<uint8>` decoded between 65.3 and 127.5 MB/s of logical payload
whatever the mount and whatever the dictionary setting, against 292.8 to 4,094.2 MB/s for
`binary(N)`.

**Dictionary off does add an I/O component, and it is small.** Cold on the shared mount the ratio
rises from 3.17-3.23x to 4.48-4.72x when the list layout starts moving 4x the bytes. Cold on local
disk it goes from 12.16-22.11x to 21.61-21.74x. Real, measurable, and far short of an order of
magnitude.

So the conclusion holds across storage as well, and more strongly than the inference it replaces:
this is a decode-CPU lever. It is largest where storage is fastest and smallest where storage is
slowest, which is the opposite of an I/O effect.

Timed: all 8 arms, cold and warm. Not timed: the zstd cells, because the byte gap closes there;
the full Ray Data pipeline on these mounts, since these figures are `pq.read_table`; and any scale
above 12 frames.

The three payloads are a gradient plus noise, a quantized 4-bit-noise frame, and `os.urandom`. None
of them is real sensor data. Real data sits between the gradient and the random case.

### Reproduce the grid

The on-disk half needs no GPU, no cluster and no Ray.

```bash
python measure_layout.py --input /tmp/grid --sweep on-disk
```

That prints every codec x payload x dictionary cell with both footer numbers, a zstd level sweep at
1, 3, 9 and 22, and the row-group sweep. pyarrow 23's default zstd level is 1.

Two codec names to know: pyarrow rejects `"uncompressed"`, and the name is `"none"`, which the footer
reports as `UNCOMPRESSED`. pyarrow 23.0.1 also rejects `"lz4_raw"`, so the grid measures `LZ4`.

## What changes

| | Naive | This template |
|---|---|---|
| Blob column | `list<uint8>`, physical INT32 | `binary(N)` `required`, no dictionary |
| Row groups | writer default, often one per file | sized so a read task can parallelise |
| Read task CPU | `num_cpus=0.25` | `num_cpus=1.0`. No measurable difference at this scale |
| GPU stage | one actor per GPU, `num_gpus=1` | fractional GPU with `num_cpus=0` |
| Diagnosis | operator span | UDF time for map stages, span and bytes/second for reads |

Every lever is an environment variable in `pipeline.py`, with its measured effect in a comment.

## Provenance

This template comes from a customer engagement whose data is private. The pipeline shape and the
levers are the engagement's. The fixture is synthetic.

**Measured on the fleet this template ships.** The table above, plus `tests.sh` wall times of 335 s
to 354 s across 6 runs on Ray 2.57.0, papermill over this notebook, all 6 code cells, no errors.
On Ray 2.58.0, two `rayapp test` runs on 2026-09-24, one on the prod console and one on staging,
exited 0 in 9 min 26 s and 9 min 17 s, start to finish with workspace start and teardown included.

**Measured on a developer laptop.** The decode table, pyarrow 23.0.1, macOS arm64, warm cache. Also
the write-side ratio during development, which ranged 2.1x to 7.0x across 5 runs, twice at the same
scale differing by more than 2x. `measure_layout.py` refuses to score an arm with fewer than 2 timed
runs.

**From the source engagement, on a different fleet and a different dataset.** Not reproduced here.
Field numbers at production scale:

- read-side layout ratio 7x to 23x, depending on the pair compared
- capping read concurrency, 1.66x to 1.91x where a CPU stage downstream of the read was binding, and
  0.56x on a read-bound pipeline
- decoder threads, up to 2.15x on local disk, about 1.6x from object storage, and +52% wall clock at
  8 threads past the crossover
- GPU packing, 8 actors at 0.5 GPU each on 4 of 8 available GPUs
- object-store fraction 0.6 holding a peak of 69.4 GiB with no spill

**Unmeasured.** Whether the read binds at production scale. The CI knob here is small: 1.6 GB per
layout fits in page cache on a 32 GiB node, and 4 files gives 4 read tasks on a 16-vCPU worker, so
the read does not become CPU-bound. Raising the scale may move these results toward the engagement's.
That is a hypothesis.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/ray-data-sensor-frame-extraction
```

## Read the footer first

Three fields from one Parquet footer say whether the blob column is a candidate: the physical type of
the leaf, its repetition level, and the rows per row group. On object storage that is one range
request.

In a `list<uint8>` column every byte is a Parquet value with a definition level and a repetition
level. In a `binary(N)` `required` column the frame is one value.

The footer says whether the lever applies. It does not say what the lever is worth. Use
`measure_layout.py` for that.


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

`make_fixture.py` writes byte-identical payloads in both layouts at a scale knob. The default
geometry is a single-plane 12-bit Bayer frame at 3848x2168, about 16.7 MB a row.

Shrink the frame count, not the frame size. The geometry travels in the Parquet schema metadata, so
`pipeline.py` follows whatever you set.


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

# Assert the sign, never a threshold. The fleet range is in the results table in this README.
slow_s = float(re.search(r"list<uint8>\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
fast_s = float(re.search(r"binary\(N\) required\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
print(f"write side: binary(N) {fast_s:.2f}s vs list<uint8> {slow_s:.2f}s "
      f"-> {slow_s / fast_s:.2f}x cheaper to write")

assert fast_s < slow_s, (
    "the recommended layout was not cheaper to write. Check the fixture, the codec and the writer "
    "before quoting the write-side figures above."
)
```


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

### Write side

Compression does not help the writer. It encodes every per-byte value before the codec runs, so the
write-side gap holds where the read-side on-disk gap does not. Measured 4.07x to 4.19x across 8
fleet runs on Ray 2.57.0, spread 2.9%, and 4.18x and 4.04x across 2 on Ray 2.58.0.

The recommended layout is cheaper for the producing team too.

## Dependencies

The base image ships numpy, pyarrow and Ray, and no torch. `cu128` and `cu129` in an image tag are
the CUDA runtime, not PyTorch. The GPU stage needs torch, so this template pins it in a
`python_depset.lock` compiled against a `pip freeze` of the image it runs on.

The lock reaches two places:

- `uv pip install --system`, reading the lock as a requirements file, covers the driver
- `ray.init(runtime_env={"pip": ...})` covers Ray Data's `map_batches` actors

Install on the driver alone and the actors run what the image shipped. That passes in a workspace,
which tracks a plain `pip install` and propagates it, and fails as a standalone Job or Service.
`pipeline.py` does the `ray.init` half.

The install command is the code cell below. Do not copy it into this prose: the repo's
dependency-delivery gate finds installs by pattern-matching source files, and a copy in a markdown
cell satisfies the gate without installing anything.


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

# Keep subprocess with check=True. A `!command` cell cannot fail a notebook: papermill runs the
# next cell and exits 0 when the command exits non-zero.
#
# Keep it as one shell string. The dependency-delivery gate looks for the requirements flag
# immediately followed by the lock filename, and an argv list separates them.
subprocess.run(
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match",
    shell=True, check=True,
)
```

## The end-to-end A/B

Same `pipeline.py`, same payload bytes, two layouts. Only the input path changes.

One run per arm gives the sign, not a magnitude. On this fleet the direction held every run, at
1.56x to 1.65x on Ray 2.57.0 and 1.76x and 1.78x on Ray 2.58.0. The decode gap from byte-identical
inputs is 10.94x; this pipeline has a GPU stage, so most of it does not reach the wall clock. Use
`measure_layout.py` for a magnitude.

Keep the warmup run. The first pipeline run on a fresh cluster pays for Ray building the
`runtime_env` virtualenv from the lock on the worker, measured at 85.1 s to 94.0 s against 5.8 s
and 5.9 s for the next run on Ray 2.57.0, and 96.8 s and 89.1 s against 5.9 s and 5.7 s on Ray
2.58.0. It lands in whichever arm runs first. Without it this cell reported 1.07 rows/s against
10.07, or 0.11x.


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

`pipeline.py` prints `ds.stats()`. Rank map stages by UDF time, not by span. Operator spans overlap
under the streaming executor, and on one measured run they summed to 2.1x the pipeline's wall clock.

UDF time cannot rank a read. Ray reports `UDF time: 0us` for read operators because a read has no
user function, so `udf_total / (span x parallelism)` is 0 for the read whatever it is doing.
Measured: `ReadFiles` span 2.44 s, `UDF time: 0us min, 0us max, 0us total` on Ray 2.57.0, and 0us
again on both Ray 2.58.0 runs. Judge the read by its span and its output bytes per second against
what your storage delivers.

On this fleet the read's span was 2.44 s and the GPU stage's was 1.99 s with 3.4 s of UDF across 2
actors, about 0.85 of saturation, against a 5.8 s pipeline, on Ray 2.57.0. On Ray 2.58.0 the read's
was 2.6 s and 2.39 s and the GPU stage's 1.9 s and 2.05 s with 3.63 s and 3.33 s of UDF, about 0.96
and 0.81, against 5.9 s and 5.7 s pipelines. At this scale the GPU stage is closer to the constraint
than the read. The engagement saw the read bind at production scale.

## The harness

`measure_layout.py` answers what a lever is worth here.

```bash
python measure_layout.py --input /tmp/grid --sweep on-disk                 # bytes at rest, no cluster
python measure_layout.py --input /tmp/grid --sweep storage --runs 2        # mounts, cold and warm
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep layout --runs 3
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep read-cpus --runs 3
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep decode-threads --runs 3
```

`--sweep storage` needs a cluster, since `/mnt/cluster_storage` only exists on one, and it runs
inside a Ray task requesting CPU so it lands on a worker instead of a head pinned to `CPU: 0`.

It exits on fewer than 2 timed runs per arm. `--warmup` defaults to 1 untimed run per arm. The
verdict is a sentence, because `OVERLAP` and `UNSUPPORTED` mean the runs cannot answer the question.
The margin it quotes is the worst run of the better arm against the best run of the worse one.

On this fleet it returned `OVERLAP` for the `num_cpus` sweep and `SEPARABLE ... by >= 1.2%` for 1 to
4 decoder threads.

Point `--input` at your own Parquet and add an arm to `SWEEPS`.

## Read task CPU and decoder threads

Ray sets `OMP_NUM_THREADS = max(1, floor(num_cpus))` when it is not already set, and pyarrow sizes
its Arrow CPU thread pool from that once per worker process. `num_cpus=0.25` on a read task gives
that task a one-thread decoder. On thin rows and many small files that buys concurrency. On a
multi-megabyte blob column, decode is the work.

The equation is `threads_per_task x concurrent_tasks ~= cores`. Past the crossover more threads
costs wall clock. `pipeline.py` exposes `READ_OMP_THREADS`, scoped to the read operator through a
`runtime_env` so it does not resize every actor in the job. The default is 0, meaning leave it alone.

Measured here: `num_cpus=1.0` against `0.25` gave overlapping ranges, and 1 to 4 decoder threads
gave 1.2% or better. 4 files gives 4 read tasks on a 16-vCPU worker, so the read is not CPU-bound
and a one-thread decoder has nothing to bite on. Diagnose by CPU-seconds per byte, not by
utilization or memory.

## Scale and units

Throughput that is flat in concurrency and linear in nodes is a ceiling. The fix is a node or a
layout change, not a larger knob.

"Faster than real time, with a minimum of twice the nodes and expense" is a result. Throughput
without the node count is not.

State the unit. On a multi-camera rig, rows/s, frames/s and camera-frames/s differ by the camera
count, and a real-time bar is quoted in one of them. `pipeline.py` reports rows/s.

State the scale. Everything measured here is 96 frames, 1.6 GB per layout, in page cache, on one L4.
Whether the read binds at production scale on this fleet is unmeasured.

## Where to go next

- [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html)
- [Reading Parquet](https://docs.ray.io/en/latest/data/loading-data.html#reading-files) and [`map_batches`](https://docs.ray.io/en/latest/data/transforming-data.html)
- [ComputeConfig reference](https://docs.anyscale.com/reference/compute-config-api#computeconfig). This template's fleet is in `configs/ray-data-sensor-frame-extraction/`.
