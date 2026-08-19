# Sensor Frame Extraction: Measuring a Parquet Layout Win on Your Own Fleet

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/ray-data-sensor-frame-extraction"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/ray-data-sensor-frame-extraction" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: ~15 min, of which **5.6 min is measured execution**.

Batch pipelines over multi-megabyte rows — camera frames, point clouds, fixed-shape tensors — are
usually stored as Parquet, and the *physical type* of the blob column is the lever people reach
for first. The advice you will find says one layout beats the other by an order of magnitude.

**On the fleet this template ships, it does not.** The write side wins by ~4.1x, reproducibly. The
read side wins by ~1.6x, not 7x–23x, because `zstd` compresses away the per-value bookkeeping the
layout is supposed to be about — the two layouts come out **the same size on disk**. And the read
was not the binding stage at all.

So this template does two things. It builds the pipeline end to end over two byte-identical
layouts, and it hands you `measure_layout.py`, the harness that tells you which way the lever
points **on your hardware, with your data, at your scale** — because that is the only place the
answer is valid. The measured numbers below are this fleet's, not yours.

## What this template measured

One `g6.4xlarge` L4 worker, `m5.2xlarge` head, Ray 2.57.0, `torch 2.9.1+cu129`, 96 frames of
16.7 MB in 4 files per layout — the configuration in
`configs/ray-data-sensor-frame-extraction/`.

| lever | result | strength |
|---|---|---|
| **Blob column physical type, write side** | **~4.1x** cheaper to write as `binary(N)` `required` | **4.08x, 4.13x, 4.19x** — three independent runs, within 3%. The strongest result here |
| **Blob column physical type, read side** | **~1.6x** faster | 16.49–16.81 vs 10.39–11.41 rows/s, 3 timed runs per arm, ranges do not overlap; 1.63x on a separate run |
| **On-disk size** | **no difference** — 831.7 MB vs 836.0 MB, 0.5% apart | this is *why* the read-side win is small, and it is the most useful thing on this page |
| **Read task `num_cpus` 1.0 vs 0.25** | **no measurable effect** | 16.67–17.00 vs 16.24–16.87 rows/s — ranges overlap |
| **Decoder threads 1 → 4** | **≥ 1.2%** | and 4 → 8 not separable at 2 runs per arm |
| **Which operator binds** | **not the read** | read span 2.44 s against the GPU stage's 1.99 s with 3.4 s of UDF across two actors |

Every one of those is a measurement on *one* fleet at *one* scale. The transferable part is the
method, not the multipliers.

## The finding worth taking away: compression eats the layout win

A `list<uint8>` blob column stores **one Parquet value, one definition level and one repetition
level per byte**. A `binary(N)` `required` column stores one value per frame and the reader
preallocates from the footer. On the face of it that is an enormous difference in work, and it is
— *before compression*.

Then `zstd` runs over it. The per-value bookkeeping in the `list<uint8>` encoding is extremely
regular, which is exactly what a general-purpose compressor is good at, and it very nearly
vanishes:

```
831.7 MB   fixed_binary/     <- binary(N) required, no dictionary
836.0 MB   list_uint8/       <- one value + def level + rep level PER BYTE
```

(`du -sh` reports those as 794M and 798M, because `du` counts MiB. Same files, same 0.5% gap —
the cell below prints the exact byte totals so the two never disagree in front of you.)

**Half a percent apart** — and that holds across scales: 4 rows on a laptop came out 34.7 MB
against 34.8 MB, 0.3% apart, so it is a property of the encoding-plus-codec pair rather than an
artifact of this fixture size. So the two layouts move the same bytes off storage, the I/O is
identical, and the only thing left to differ is **decode CPU** — which is where the ~1.6x comes from, and why
it is 1.6x and not an order of magnitude. The published order-of-magnitude figures come from a
regime where the naive layout also costs far more I/O; compressed, at this scale, it does not.

Two consequences worth carrying to your own data:

- **A layout or encoding win measured uncompressed does not survive compression unchanged.** Check
  the on-disk sizes before you believe a read-side ratio. It costs one `du`.
- **The write side is a different question and it kept its win.** Compression does not help the
  *writer*: it still has to emit and encode every one of those values before the compressor sees
  them. That is why ~4.1x on the write is the sturdiest number here, and it is usually the easier
  side to change — the producing team is being asked for something that makes their own job
  cheaper, not for a favour.

## What changes

| | Naive | This template |
|---|---|---|
| Blob column | `list<uint8>` — one Parquet value + a definition level + a repetition level **per byte** | `binary(N)` `required`, no dictionary — one value per frame |
| Row groups | writer default (often one per file) | sized so a read task has real work and can parallelise |
| Read task CPU | `num_cpus=0.25`, "reads are I/O-bound" | `num_cpus=1.0` — but see below: the mechanism is real and the effect was **not measurable here** |
| GPU stage | one actor per GPU, `num_gpus=1` | fractional GPU packing with `num_cpus=0`, sized to leave cores for the read |
| Diagnosis | operator span, or a ratio someone else measured | operator **UDF time** for map stages, **span and bytes/second** for reads, and a harness with replicates |

Every lever lives in `pipeline.py`, which is the control panel: each knob is an environment
variable and each carries its measured effect inline.

## Which numbers here are measured

This template is derived from a customer engagement whose data is private. The pipeline shape and
the levers are the engagement's; the fixture is synthetic. The numbers are separated by where they
came from, because they are not equally strong **and because they disagree**.

**Measured on the fleet this template ships** — the table in "What this template measured" above,
plus: the notebook's own execution took **338 s** end to end through papermill, all six code cells,
no errors. Trust these for this fleet and no further.

**Measured while building the template**, on a developer laptop — weaker, kept because it shows
the spread: the recommended layout was cheaper to write in every one of five runs, but the
multiplier ranged **2.1x to 7.0x**, twice at the *same* scale differing by more than 2x. A
magnitude off one run per arm is not a measurement, which is why `measure_layout.py` refuses to
score an arm with fewer than two timed runs.

**From the source engagement, on a different fleet and a different dataset.** Not reproduced here.
These are field numbers at production scale and they are kept because the contrast *is* the
lesson — same code, different fleet, different data, different answer. Nothing here says the
engagement was wrong; it says its numbers are not yours:

- read-side layout ratio **7x to 23x**, depending on which pair was compared;
- capping read concurrency was worth **1.66–1.91x** where a CPU stage *downstream* of the read was
  binding, and **0.56x** — a loss — on a read-bound pipeline. Same knob, opposite sign;
- raising decoder threads was worth up to **2.15x** on local disk, about **1.6x** from object
  storage, and cost **+52% wall clock** at 8 threads past the crossover;
- the winning GPU configuration ran **8 actors at 0.5 GPU each on 4 of 8 available GPUs** — the GPU
  was never the constraint there, and packing mattered more than count;
- an object-store fraction of **0.6** held a peak of **69.4 GiB with zero spill**.

**Still unmeasured, and the template will not pretend otherwise: whether the read binds at
production scale.** The CI knob here is deliberately small — 1.6 GB per layout fits in page cache
on a 32 GiB node, and 4 files gives 4 read tasks on a 16-vCPU worker, so the read never becomes
CPU-bound and has nothing to bite on. Closing that needs a fixture large enough to leave cache.
Raising the scale may well move these results toward the engagement's; that is a hypothesis, and
it is stated as one.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/ray-data-sensor-frame-extraction
```

## Read the footer before you read the code

Before touching a reader knob, ask what the file *is*. Three fields out of one Parquet footer
settle whether the blob column is even a candidate: the **physical type** of the leaf, its
**repetition level**, and the **rows per row group**. That is one HTTP range request against object
storage — no cluster, no scan.

The two levels are the part people miss. In a `list<uint8>` column every byte becomes a Parquet
value carrying a definition level and a repetition level, so a 16 MB frame arrives as 16 million
bookkeeping entries. In a `binary(N)` `required` column the same frame is one value.

This is the cheapest diagnostic here and the only one that needs nothing at all. It tells you the
lever *applies*. It does not tell you what it is worth — that is what the harness at the end is
for.


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
    """Whether the lever APPLIES. Not what it is worth -- that needs measuring."""
    if d["physical_type"] == "FIXED_LEN_BYTE_ARRAY" and d["max_repetition_level"] == 0:
        return "one Parquet value per frame -- the reader preallocates from the footer"
    if d["max_repetition_level"] > 0:
        return (
            "one Parquet value plus a definition level plus a repetition level PER BYTE"
            " -- the lever applies here; measure it before assuming what it is worth"
        )
    return "neither shape this template compares -- read the levels above before tuning anything"
```

## Generate your own fixture

The workload this template teaches reads multi-megabyte opaque blobs out of Parquet, and the
customer data it came from is private. So the template makes its own: `make_fixture.py` writes
**byte-identical payloads in both physical layouts**, at a scale knob.

The geometry defaults to a single-plane 12-bit Bayer frame at 3848x2168 — about 16.7 MB a row.
Shrink the frame *count* rather than the frame *size* if you need this faster; the geometry is
where the regime lives, and it travels with the fixture in the Parquet schema metadata so
`pipeline.py` follows whatever you choose.


```python
import os
import re
import subprocess
import sys

FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/sensor-fixture")
FRAMES = int(os.environ.get("FRAMES", "24"))   # frames per file
FILES = int(os.environ.get("FILES", "4"))

# A read task cannot parallelise below one row group, and on multi-megabyte rows small is right
# -- the opposite of the advice for narrow tabular data.
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

# THE HEADLINE RESULT, asserted as a direction and never as a threshold. Measured 4.08x, 4.13x
# and 4.19x on three independent runs of this template's own fleet -- but a number here would be
# fleet-dependent and would go stale, so what CI guards is the SIGN.
slow_s = float(re.search(r"list<uint8>\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
fast_s = float(re.search(r"binary\(N\) required\s+write:\s+([0-9.]+)s", proc.stdout).group(1))
print(f"write side: binary(N) {fast_s:.2f}s vs list<uint8> {slow_s:.2f}s "
      f"-> {slow_s / fast_s:.2f}x cheaper to write")

assert fast_s < slow_s, (
    "the recommended layout was NOT cheaper to write here. That is this template's most "
    "reproducible result (4.08x / 4.13x / 4.19x on its own fleet), so an inversion means the "
    "fixture, the compression codec or the writer has changed and the lesson needs re-measuring."
)
```


```python
for layout in ["fixed_binary", "list_uint8"]:
    d = describe_blob_column(os.path.join(FIXTURE, layout))
    print(f"{layout}:")
    for key, value in d.items():
        print(f"    {key:24s} {value}")
    print(f"    -> {verdict(d)}")

# The on-disk sizes are the finding, so print them rather than trusting the encoding argument.
print()
for layout in ["fixed_binary", "list_uint8"]:
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(FIXTURE, layout, "*.parquet")))
    print(f"{layout:14s} {total / 1e6:8.1f} MB on disk")
print("\nIf those two numbers are close, compression has already removed most of what the "
      "layout was going to buy you on the READ path. Measured on this template's own fleet: "
      "831.7 MB against 836.0 MB, 0.5% apart.")
```

### The write side is the sturdy result

Note the ratio the cell above printed, and note which side of the pipeline it is on.

Compression does not help the *writer*: it still has to emit and encode every per-byte value
before the compressor ever sees them. So the write-side gap survives where the read-side gap
largely does not — measured **4.08x, 4.13x and 4.19x** across three independent runs of this
fleet, within 3% of each other.

That also makes it the easier argument to win. Asking a producing team to change their writer
usually sounds like a favour for the reader's benefit; measured, the recommended layout is
**cheaper for them too**.

## Dependencies

The base image ships numpy, pyarrow and Ray, and **no torch** — `cu128`/`cu129` in an image tag is
the CUDA runtime, not PyTorch. The GPU stage needs torch, so this template supplies it through a
fully-pinned `python_depset.lock`, compiled against a `pip freeze` of the exact image it runs on.

The lock has to reach **two** places, and the second is the one people omit:

- `uv pip install --system`, reading the lock as a requirements file, covers **the driver only**;
- `ray.init(runtime_env={"pip": ...})` is what reaches **Ray Data's `map_batches` actors**.

Install only on the driver and the actors silently run whatever the image shipped. That passes in a
workspace — a workspace tracks a plain `pip install` and propagates it — then fails the moment the
same code runs as a standalone Job or Service, which has no propagation. `pipeline.py` does the
`ray.init` half; the cell below does the driver half.

(The exact install command is the code cell below and is deliberately not repeated in this prose:
the repo's dependency-delivery gate finds the install by pattern-matching source files, and a copy
in a markdown cell satisfies the gate without installing anything.)


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
import subprocess

# A `!command` cell CANNOT fail a notebook. Papermill runs the next cell and exits 0 even when the
# command exited non-zero -- verified -- so an install written that way is unasserted, and a silent
# failure here surfaces much later as a torch ImportError inside a Ray actor. Hence check=True.
#
# One shell string rather than an argv list, deliberately: the repo's dependency-delivery gate
# looks for the requirements flag immediately followed by the lock filename, and an argv list puts
# a quote and a comma between them.
subprocess.run(
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match",
    shell=True, check=True,
)
```

## The read-side A/B

Same `pipeline.py`, same payload bytes, two physical layouts. Only the input path changes.

**What this cell establishes is a direction, not a magnitude.** One run per arm is enough to show
the sign and nothing more. On this fleet the direction held every time and came out at **~1.6x**;
for a number you can quote, use `measure_layout.py` further down, which takes several timed runs
per arm and refuses to score an arm with fewer than two.

**Note the warmup, and do not remove it.** The first pipeline run on a fresh cluster also pays for
Ray building the `runtime_env` virtualenv from the lock on the worker — measured at 85.1 s against
5.9 s for the next run. That cost lands entirely in whichever arm runs first. Without the warmup
this cell measured the recommended layout at 1.07 rows/s against 10.07 and reported **0.11x** — the
template confidently disproving itself, on every fresh cluster. It was a real CI failure, not a
hypothetical, and it is the cheapest possible illustration of why a single timed run is not a
measurement.


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


print("warmup run (discarded -- it pays for the runtime_env build on the worker)")
run_arm("fixed_binary")

fast = run_arm("fixed_binary")
slow = run_arm("list_uint8")

print(f"fixed_binary {fast:.2f} rows/s vs list<uint8> {slow:.2f} rows/s -> {fast / slow:.2f}x")
print("Direction only. Measured ~1.6x on this fleet, NOT the 7x-23x in the field notes -- "
      "compression closes most of the gap. See 'compression eats the layout win' above.")

assert fast > slow, (
    "the recommended layout was not faster to read here. At this scale the expected margin is "
    "small (~1.6x measured) because compression removes most of the difference, so an inversion "
    "is plausible on other hardware -- run measure_layout.py with replicates before concluding "
    "anything, and read the binding-operator section below."
)
```

## Which operator is actually binding

`pipeline.py` prints `ds.stats()`. Rank map stages by **UDF time**, not by span: under the
streaming executor operator spans overlap, and on one measured run they summed to 2.1x the
pipeline's wall clock, so ranking by span hands you the wrong answer with total confidence.

**But that metric cannot rank a read.** Ray reports `UDF time: 0us` for read operators, because a
read has no user function — so `udf_total / (span x parallelism)` is 0 for the read *by
construction*, however hard it is working. Measured here: `ReadFiles` span **2.44 s**, `UDF time:
0us min, 0us max, 0us total`. Judge the read by its **span** and its **output bytes per second**
against what your storage actually delivers.

Doing that on this fleet is how the template's original framing came apart. The read's span was
2.44 s; the GPU stage's was 1.99 s with **3.4 s of UDF time across two actors**, i.e. roughly 0.85
of saturation, against a 5.8 s pipeline. **At this scale the GPU stage is nearer the constraint
than the read.** The engagement saw the read bind in a production regime; 96 cached frames is not
that regime, and the template says so rather than inheriting the conclusion.

## The harness: measure it on your own data

This is the part to take with you. `measure_layout.py` is the template's answer to "is this lever
worth anything *here*", and it is built to refuse the two mistakes that make performance claims
worthless:

- **fewer than two timed runs per arm** — an arm with one run has no observed spread, so any delta
  from it is unfalsifiable. It exits rather than scoring;
- **a cold first run** — `--warmup` defaults to 1 untimed run per arm, declared up front, because
  the runtime_env build lands in whichever arm goes first.

Its verdict is a **sentence, not a boolean**, because `OVERLAP` and `UNSUPPORTED` are not "the
lever does not work" — they say these runs cannot answer the question. The margin it quotes is a
lower bound: worst run of the better arm against the best run of the worse one, never a ratio of
means.

```bash
# the headline lever, three timed runs per arm plus a warmup
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep layout --runs 3

# the read-task CPU trap, and the decoder-thread crossover
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep read-cpus --runs 3
python measure_layout.py --input /mnt/cluster_storage/sensor-fixture --sweep decode-threads --runs 3
```

Point `--input` at your own Parquet, add an arm to `SWEEPS`, and the same discipline applies to
whatever you are actually tuning. On this fleet it returned `OVERLAP` for the `num_cpus` sweep and
`SEPARABLE ... by >= 1.2%` for 1 → 4 decoder threads — both of which are answers, and neither of
which is the answer the field notes would have led you to write down.

## The decoder thread-pool trap

The mechanism is real and worth knowing even though it did not move on this fleet.

Ray sets `OMP_NUM_THREADS = max(1, floor(num_cpus))` when it is not already set, and pyarrow sizes
its Arrow CPU thread pool from that **once per worker process**. So `num_cpus=0.25` on a read task
does not merely make the task cheap — it gives that task a **one-thread decoder**. On thin rows and
many small files that is a fine trade and buys concurrency. On a multi-megabyte blob column, decode
*is* the work.

The governing equation is `threads_per_task x concurrent_tasks ≈ cores`, and past that crossover
more threads is a cost rather than a gain. `pipeline.py` exposes this as `READ_OMP_THREADS`, scoped
to the read operator through a `runtime_env` so it does not resize every actor in the job. The
default is 0, meaning "leave it alone".

**Measured here: `num_cpus=1.0` against `0.25` produced overlapping ranges — no effect — and 1 → 4
decoder threads bought ≥1.2%.** The reason is in the setup rather than in the mechanism: 4 files
means 4 read tasks on a 16-vCPU worker, so the read is never CPU-bound and a one-thread decoder has
nothing to bite on. Diagnose this by CPU-seconds per byte, not by utilization or memory, neither of
which moves in a way that tells you.

## Scaling honestly

**Flat in concurrency and linear in nodes is a ceiling, not a tuning opportunity.** If throughput
stops responding to actor counts and only responds to node counts, you are not under-tuned — you
have found the read-path limit. The fix is a node or a layout change, not a bigger knob.

**"Faster than real time, with a minimum of twice the nodes and expense" is a complete result.**
Quoting throughput without the node count is not.

**State your unit.** On a multi-camera rig, rows/s, frames/s and camera-frames/s differ by the
camera count, and a real-time bar is quoted in exactly one of them. `pipeline.py` reports rows/s
and says so.

**And state your scale.** Everything measured here is at 96 frames, 1.6 GB per layout, in page
cache, on one L4. Whether the read binds at production scale on this fleet is unmeasured — see the
provenance section. The method below is what settles it for your data; the multipliers above are
not.

## Where to go next

- **Ray Data mechanics** — [Ray Data documentation](https://docs.ray.io/en/latest/data/data.html).
- **The read path** — [reading Parquet](https://docs.ray.io/en/latest/data/loading-data.html#reading-parquet-files) and [`map_batches`](https://docs.ray.io/en/latest/data/transforming-data.html).
- **Compute configuration** — [ComputeConfig reference](https://docs.anyscale.com/reference/compute-config-api#computeconfig); this template's fleet is in `configs/ray-data-sensor-frame-extraction/`.
