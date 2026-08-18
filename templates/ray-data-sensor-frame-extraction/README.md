# Read-bound Sensor Frame Extraction with Ray Data

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/ray-data-sensor-frame-extraction"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/ray-data-sensor-frame-extraction" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: ~25 min — an estimate, not yet a measurement. See "Which numbers here are measured" below.

Batch pipelines over multi-megabyte rows — camera frames, point clouds, fixed-shape tensors —
usually stall somewhere that is not the GPU. This template builds one end to end, then shows you
how to find out which stage is actually holding it, using two levers most people never reach for:
the **Parquet physical type of the blob column**, and the **number of decode threads each read task
gets**.

You will generate your own fixture, run the same pipeline over two byte-identical layouts, and read
the difference off your own hardware.

**This is a read-path template.** If your GPU stage is already at full SM occupancy, this is the
wrong template — the levers here move the read, and moving the read will not help you.

## What changes

|  | Naive | This template |
|---|---|---|
| Blob column | `list<uint8>` — one Parquet value + a definition level + a repetition level **per byte** | `binary(N)` `required`, no dictionary — one value per frame |
| Row groups | writer default (often one per file) | sized so a read task has real work and can parallelise |
| Read task CPU | `num_cpus=0.25`, "reads are I/O-bound" | `num_cpus=1.0` unless the equation says otherwise — below 1.0 pins the decoder to one thread |
| GPU stage | one actor per GPU, `num_gpus=1` | fractional GPU packing with `num_cpus=0`, sized to leave cores for the read |
| Diagnosis | operator span | operator **UDF time** — spans overlap under the streaming executor and can sum past wall clock |

Every lever lives in `pipeline.py`, which is the control panel for this template: each knob is an
environment variable, each carries its measured effect inline, and each default's comment says which
data property justifies it and what different property should flip it.

## Which numbers here are measured

This template is derived from a customer engagement whose data is private. The pipeline shape and
the levers are the engagement's; the fixture is synthetic and the measurements below are separated
by where they came from, because they are not equally strong.

**Measured by this notebook, on your hardware, when you run it.** The write-side ratio the fixture
prints, the read-side rows/s for both layouts and the ratio between them, and the operator timings
in `ds.stats()`. These are the numbers to trust and the only ones you should quote about your own
fleet.

**Measured while building this template**, on a developer laptop (macOS arm64, local SSD, Python
3.12, numpy 2.2.6, pyarrow 23.0.1) — **not** on the compute configs this template ships:

- the recommended layout was cheaper to write in **every** run — four out of four — but the
  multiplier ranged from **2.1x to 7.0x** across scales, at one run per point. The direction is
  the finding; the multiplier is noisy at small frame counts and is a property of your fixture,
  which is exactly why the notebook prints yours rather than asserting one;
- the fixture at the default knob below cost about **11 s and 4.8 GB of peak RSS per file**.

**From the source engagement, on a different fleet and a different dataset. Not reproduced here,
and nothing in this template re-measures them.** Treat them as direction, not magnitude:

- the read-side layout ratio was **roughly 7x to 23x**, depending on which pair was compared;
- capping read concurrency was worth **1.66–1.91x** when a CPU stage *downstream* of the read was
  binding, and **0.56x** — a loss — on a read-bound pipeline. Same knob, opposite sign;
- raising decoder threads was worth up to **2.15x** on local disk, about **1.6x** from object
  storage, and cost **+52% wall clock** at 8 threads past the crossover;
- the winning GPU configuration ran **8 actors at 0.5 GPU each and occupied 4 of 8 available
  GPUs** — the GPU was never the constraint, and packing mattered more than count;
- an object-store fraction of **0.6** held a peak of **69.4 GiB with zero spill**.

**Not yet measured at all.** Whether the read is the binding operator on the compute configs in
`configs/ray-data-sensor-frame-extraction/` — this template has not been run there. That is the
template's central claim, so the cell that reports the binding operator is the one to read first,
and the time-to-complete estimate above should be replaced with a real figure on the first
end-to-end run.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/ray-data-sensor-frame-extraction
```

## Read the footer before you read the code

Before touching a single reader knob, ask what the file is. Three fields out of one Parquet footer
settle whether the blob column is the problem: the **physical type** of the leaf, its
**repetition level**, and the **rows per row group**. That is one HTTP range request against object
storage — no cluster, no scan, no cost worth measuring.

The two levels are the part people miss. In a `list<uint8>` column every single byte becomes a
Parquet value carrying a definition level and a repetition level, so a 16 MB frame arrives as
16 million bookkeeping entries. In a `binary(N)` `required` column the same frame is one value and
the reader preallocates straight from the footer.


```python
import glob
import os

import pyarrow.parquet as pq


def describe_blob_column(path: str, column: str = "data") -> dict:
    """Read one Parquet FOOTER and report the fields that decide read cost.

    No row data and no cluster: on object storage this is a single range request.
    """
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
    """The rule, stated as a rule rather than as a speedup number."""
    if d["physical_type"] == "FIXED_LEN_BYTE_ARRAY" and d["max_repetition_level"] == 0:
        return "one Parquet value per frame -- the reader preallocates from the footer"
    if d["max_repetition_level"] > 0:
        return (
            "one Parquet value plus a definition level plus a repetition level PER BYTE"
            " -- this is the layout to change"
        )
    return "neither shape this template compares -- read the levels above before tuning anything"
```

## Generate your own fixture

The workload this template teaches reads multi-megabyte opaque blobs out of Parquet, and the
customer data it came from is private. So the template makes its own: `make_fixture.py` writes
**byte-identical payloads in both physical layouts**, at a scale knob, and you measure the lever on
your own hardware.

The geometry defaults to a single-plane 12-bit Bayer frame at 3848x2168 — about 16.7 MB a row, which
is the regime where per-value Parquet bookkeeping dominates. Smaller frames blunt the lesson, so
shrink the frame *count* rather than the frame *size* if you need this to run faster.

`FRAMES` and `FILES` are read from the environment so the same notebook covers an interactive run and
a CI run. Raising them, or raising the geometry, moves you toward the production regime; the
direction of the result holds, the wall clock does not.


```python
import os
import subprocess
import sys

FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/sensor-fixture")
FRAMES = int(os.environ.get("FRAMES", "24"))   # frames per file
FILES = int(os.environ.get("FILES", "4"))

# A read task cannot parallelise below one row group, and on multi-megabyte rows small is
# right -- the opposite of the advice for narrow tabular data.
ROW_GROUP_SIZE = int(os.environ.get("ROW_GROUP_SIZE", "1"))

subprocess.run(
    [sys.executable, "make_fixture.py",
     "--out", FIXTURE,
     "--frames", str(FRAMES),
     "--files", str(FILES),
     "--row-group-size", str(ROW_GROUP_SIZE)],
    check=True,
)
```


```python
for layout in ["fixed_binary", "list_uint8"]:
    d = describe_blob_column(os.path.join(FIXTURE, layout))
    print(f"{layout}:")
    for key, value in d.items():
        print(f"    {key:24s} {value}")
    print(f"    -> {verdict(d)}\n")
```

### The half of the lesson that moves the producing team

Note the write-side ratio the fixture just printed.

The usual version of this conversation asks a producing team to change their writer for the
*reader's* benefit, which is a favour and gets scheduled accordingly. Measured, the recommended
layout is also **cheaper to write** — same payload, fewer Parquet values, less bookkeeping on the
way out. That reframes the request, and it is the argument worth leading with.

The exact multiplier is fixture-dependent and scale-dependent; yours is the one above. In
template development the recommended layout won every run, while the multiplier moved by more than
3x between scales — so lead with the direction and let the reader measure their own magnitude.

## Dependencies

The base image ships numpy, pyarrow and Ray, and **no torch** — `cu128`/`cu129` in an image tag is
the CUDA runtime, not PyTorch. The GPU stage needs torch, so this template supplies it through a
fully-pinned `python_depset.lock`, compiled against a `pip freeze` of the exact image the template
runs on.

The lock has to be installed in **two** places, and the second one is the one people omit:

- `uv pip install -r python_depset.lock --system` covers **the driver only**;
- `ray.init(runtime_env={"pip": ...})` is what reaches **Ray Data's `map_batches` actors**.

Install only on the driver and the actors silently run whatever the image shipped. That passes in a
workspace — a workspace tracks a plain `pip install` and propagates it — and then fails the moment
the same code runs as a standalone Job or Service, which has no propagation.


```python
import pathlib

LOCK = pathlib.Path("python_depset.lock")
if not LOCK.is_file():
    raise FileNotFoundError(
        """python_depset.lock is not in this template directory yet.

It is generated, never hand-written: add this template's requirements.txt, register a
compile entry in dependencies/template.depsets.yaml (with include_setuptools: true and a
seed-image-freeze.sh pre_hook naming the image the BUILD.yaml entry runs on), then run
    ./scripts/depsets/update_deps.sh --name <that entry>
from the repository root and commit the result.

Until it exists the cells below cannot run: the GPU stage imports torch, and the base
image does not ship torch."""
    )
print(f"{LOCK} present ({LOCK.stat().st_size} bytes)")
```


```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```

## The A/B

Same `pipeline.py`, same payload bytes, two physical layouts. `pipeline.py` reads its levers from the
environment, so nothing below changes the code between arms — only the input path.

**Report the ratio you got, not a number this template hardcodes.** The direction reproduces; the
multiplier depends on your fixture, your storage and your instance family.


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


fast = run_arm("fixed_binary")
slow = run_arm("list_uint8")

print(f"fixed_binary {fast:.2f} rows/s vs list<uint8> {slow:.2f} rows/s -> {fast / slow:.2f}x")

# Direction only, deliberately. A wall-clock threshold here would be fleet-dependent and would
# go stale; the shape is what this template actually claims, and if the shape inverts on your
# hardware then the lesson does not apply to it and you want to know that loudly.
assert fast > slow, (
    "the recommended layout was not faster here -- the thesis does not hold on this fleet, "
    "and the operator ranking below is where to look first"
)
```

## Which operator is actually binding

`pipeline.py` prints `ds.stats()`. Read it by **UDF time**, not by span: under the streaming executor
operator spans overlap, and on one measured run they summed to 2.1x the pipeline's wall clock, so
ranking by span will hand you the wrong answer with total confidence.

Rank by `udf_total / (span x parallelism)`. The operator sitting near 1.0 is the one holding the
pipeline. For this workload shape it should be the read — that is the whole claim, and it is the
claim not yet checked on this template's own compute configs.

Then look at the executor's `Active & requested resources` line. Requested exceeding cluster size
means the read and a downstream CPU stage are already fighting each other for cores, and no reader
knob fixes contention.

## The decoder thread-pool trap

This is the trap the template exists to teach, and it is invisible in any dashboard.

Ray sets `OMP_NUM_THREADS = max(1, floor(num_cpus))` when it is not already set, and pyarrow sizes
its Arrow CPU thread pool from that **once per worker process**. So `num_cpus=0.25` on a read task
does not merely make the task cheap — it gives that task a **one-thread decoder**. On thin rows and
many small files that is a fine trade and buys concurrency. On a multi-megabyte blob column, decode
*is* the work, and you have serialised it.

The governing equation is `threads_per_task x concurrent_tasks ≈ cores`. Past that crossover more
threads is a **cost**, not a gain — the source engagement measured +52% wall clock at 8 threads on a
large object-storage read. Diagnose it by CPU-seconds per byte, not by utilization and not by memory,
neither of which moves in a way that tells you.

`pipeline.py` exposes this as `READ_OMP_THREADS`, scoped to the read operator through a
`runtime_env` so it does not resize every actor in the job. The default is 0, meaning "leave it
alone", which is the right default until the equation says otherwise.

## Scaling honestly

Two things worth saying plainly, because most write-ups of this workload say neither.

**Flat in concurrency and linear in nodes is a ceiling, not a tuning opportunity.** If throughput
stops responding to actor counts and only responds to node counts, you are not under-tuned — you
have found the read-path limit. The fix is a node or a layout change, not a bigger knob.

**"Faster than real time, with a minimum of twice the nodes and expense" is a complete result.** It
is an answer, and quoting the throughput without the node count is not.

And state your unit. On a multi-camera rig, rows/s, frames/s and camera-frames/s differ by the camera
count, and a real-time bar is quoted in exactly one of them. `pipeline.py` reports rows/s and says so.

## Where to go next

- **Deployment mechanics** — the Ray Data guide for Anyscale: [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html).
- **The read-path levers in detail** — [Parquet reading in Ray Data](https://docs.ray.io/en/latest/data/loading-data.html#reading-parquet-files) and [`map_batches`](https://docs.ray.io/en/latest/data/transforming-data.html).
- **Compute configuration** — [ComputeConfig reference](https://docs.anyscale.com/reference/compute-config-api#computeconfig); this template's own fleet is in `configs/ray-data-sensor-frame-extraction/`.
