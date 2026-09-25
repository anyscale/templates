# Sensor Frame Extraction: Parquet Layout for Multi-Megabyte Blobs

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/ray-data-sensor-frame-extraction"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/ray-data-sensor-frame-extraction" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: about 9 min, including cluster start.

Camera frames stored in Parquet as `list<uint8>` make every byte a separate Parquet value, and
decoding pays for each one. This template writes the same synthetic 16.7 MB frames, single-plane
12-bit Bayer at 3848x2168, as `list<uint8>` and as `binary(N)`, reads both through a Ray Data
pipeline with a GPU stage, and measures the difference on your cluster.

**Write the frame column as `binary(N)`, `required`, with dictionary encoding off.** `binary(N)` is
Arrow's fixed-size binary type, stored as Parquet `FIXED_LEN_BYTE_ARRAY`, so each frame is one
value. Keep row groups small, since a read task can't parallelise below one row group; the fixture
uses one row per group.

The gain is in decode and write time. With pyarrow's default dictionary encoding both layouts take
the same space on disk, so the change saves no I/O, and the slower your storage, the smaller the
gain.

## Measured results

`binary(N)` against `list<uint8>`, 16.7 MB frames, with pyarrow's default dictionary encoding on the
list column. "Cluster" is the shipped compute config: one `g6.4xlarge` worker (L4 GPU, 16 vCPUs) and
an `m5.2xlarge` head, with 96 frames per layout.

| | `binary(N)` against `list<uint8>` | Measured on |
|---|---|---|
| Write time | 4.01x to 4.19x faster | cluster, Ray 2.57.0 and 2.58.0, 11 runs |
| Decode, `pq.read_table`, warm cache | 9.35x to 12.94x faster with no codec, 4.23x to 8.49x with zstd | macOS arm64 laptop, pyarrow 23.0.1, 3 runs per arm, ratio range over the runs |
| Decode, `pq.read_table`, no codec | about 32x faster warm from local disk, 3.2x cold from `/mnt/cluster_storage` | cluster worker, 12 frames, 2 runs per arm |
| End to end, this pipeline | 1.56x to 1.78x faster | cluster, Ray 2.57.0 and 2.58.0, 8 runs |
| Size on disk, zstd | within 1%: 831.7 MB against 836.0 MB | cluster, every run |

End to end, only the decode share of your pipeline's time speeds up. Here the GPU stage was closer
to its limit than the read, so most of the decode gain didn't reach the wall clock. Per-run figures,
the size and storage grids and the source workload's numbers are in
[NOTES.md](https://github.com/anyscale/templates/blob/main/templates/ray-data-sensor-frame-extraction/NOTES.md).

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/ray-data-sensor-frame-extraction
```

## Read the footer first

The next cell defines two helpers and prints nothing: `describe_blob_column` reads the blob column's
physical type, levels and row groups from the first Parquet file under a path, and `verdict` says
whether the layout change applies.


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

The next cell writes `FRAMES` x `FILES` frames, 96 by default, in both layouts from the same bytes
and prints both write times and their ratio, which should land near the write row above; it asserts
only that `binary(N)` wrote faster.


```python
import os
import re
import subprocess
import sys

FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/sensor-fixture")
FRAMES = int(os.environ.get("FRAMES", "24"))   # frames per file; lower it to run smaller
FILES = int(os.environ.get("FILES", "4"))

# Rows per row group. A read task can't parallelise below one.
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

# Sign only: a threshold would depend on the hardware.
slow_s = float(re.search(r"list<uint8>\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
fast_s = float(re.search(r"binary\(N\) required\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
print(f"write side: binary(N) {fast_s:.2f}s vs list<uint8> {slow_s:.2f}s "
      f"-> {slow_s / fast_s:.2f}x cheaper to write")

assert fast_s < slow_s, (
    "the recommended layout was not cheaper to write. Check the fixture, the codec and the writer "
    "before quoting the write-side figures above."
)
```

The next cell prints each layout's footer fields, verdict and size on disk; expect
`FIXED_LEN_BYTE_ARRAY` at repetition level 0 for `fixed_binary`, `INT32` at level 1 for
`list_uint8`, and sizes within 1%.


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

To check your own data, run the two helpers on your files: the footer says whether the change
applies, and `measure_layout.py`, below, measures what it's worth.

## Install dependencies

The base image ships no torch, so the next two cells check that `python_depset.lock` is present,
raising with the compile command if not, and install it on the driver.


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

# check=True: a failed `!` cell wouldn't fail the notebook.
subprocess.run(
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match",
    shell=True, check=True,
)
```

`pipeline.py` passes the same lock to its GPU actors through `ray.init(runtime_env={"pip": ...})`;
without that, an install on the driver alone works in a workspace and fails as a Job or Service.

## Run the A/B

The next cell runs `pipeline.py` once per layout after a discarded warmup, prints each run's rows/s
and `ds.stats()`, and asserts only that `binary(N)` was faster; expect a ratio near the end-to-end
row above.


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

One run per arm gives the direction; `measure_layout.py` gives a magnitude. Keep the warmup: the
first run on a fresh cluster also builds the lock's virtualenv on the worker, 85.1 s to 96.8 s
against 5.7 s to 5.9 s for the next run, and that cost lands in whichever arm runs first.

## Which operator binds

`pipeline.py` prints `ds.stats()`. Rank map stages by UDF time: operator spans overlap under the
streaming executor, so they don't rank stages. A map stage's saturation is its UDF time divided by
span x actors, and near 1.0 the stage is the limit. The read has no user function, so it reports
`UDF time: 0us` however hard it works; judge it by its span and output bytes per second against what
your storage delivers.

On this cluster the GPU stage, 2 actors at 0.5 GPU each, ran at 0.81 to 0.96 saturation and the
read's span was 2.39 s to 2.6 s, over 4 runs on Ray 2.57.0 and 2.58.0. At 96 frames the fixture's
1.6 GB per layout fits in page cache and 4 files give 4 read tasks on 16 vCPUs, so the read doesn't
become CPU-bound; whether it binds at production scale is untested.

Every lever in `pipeline.py` is an environment variable whose comment says when to change it and
what was measured.

## Measure your own data

`measure_layout.py` repeats a comparison with replicates: at least 2 timed runs per arm, after
1 discarded warmup run per arm by default.

```bash
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep layout --runs 3
python measure_layout.py --input /tmp/unused --sweep on-disk           # bytes at rest, pyarrow only
python measure_layout.py --input /tmp/unused --sweep storage --runs 2  # mounts, cold and warm; needs a cluster
```

`SEPARABLE` quotes the margin between the worst run of the better arm and the best run of the worse
one; `OVERLAP` means more runs are needed, or there is no real difference. `--sweep read-cpus` and
`--sweep decode-threads` time the read levers the same way.

For your own data, write it both ways as `fixed_binary/` and `list_uint8/` under `--input`, with
`frame_id` and `data` columns of 16-bit frames, and set `FRAME_WIDTH` and `FRAME_HEIGHT` if the
files carry no geometry metadata. To test another lever, add an arm to `SWEEPS`.

## Where to go next

- [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html)
- [Reading Parquet](https://docs.ray.io/en/latest/data/loading-data.html#reading-files) and [`map_batches`](https://docs.ray.io/en/latest/data/transforming-data.html)
- [ComputeConfig reference](https://docs.anyscale.com/reference/compute-config-api#computeconfig). This template's compute config is in `configs/ray-data-sensor-frame-extraction/`.
