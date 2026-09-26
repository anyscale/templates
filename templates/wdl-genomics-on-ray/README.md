# Run WDL genomics workflows on Ray

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/wdl-genomics-on-ray"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/wdl-genomics-on-ray" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

**⏱️ Time to complete**: ~20 min (~10 min at `quick` scale; see Step 1)

This template runs an existing WDL workflow over a cohort on one autoscaling Ray cluster. miniwdl
runs it on the head node and still does the language work, call caching and localization included;
the `wdl_on_ray` backend (`[scheduler] container_backend = ray`) runs each task as a
[Ray task](https://docs.ray.io/en/latest/ray-core/tasks.html), so
`runtime { cpu: 30  memory: "32 GiB" }` becomes a 30-CPU, 32 GiB request on a worker that is already
up. Cromwell on Terra boots a VM per task; here tasks pack onto running workers, as under Cromwell's
HPC backends or miniwdl-slurm, and the pool scales with the queue.

The workflow is a port of
[Broad's ONT assembly pipeline](https://github.com/broadinstitute/long-read-pipelines): Flye
assembly, QUAST against GRCh38, and minimap2 and paftools variant calls. The notebook runs it on the
GIAB Ashkenazi trio over part of chr20, leaving an assembly, QUAST report and VCF per sample and a
per-task node timeline. `job.yaml` runs all of chr20 in about 2h04m for $10 on demand (one measured
run, Ray 2.56.0 image, AWS `m5.8xlarge` workers).

![WDL tasks as Ray tasks on an autoscaling cluster](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/architecture.png)

Seven of the ten `.wdl` files come from long-read-pipelines 4.0.68 (`02089d9`, BSD-3-Clause, see
`wdl/LICENSE`) and the other three are new (Apache-2.0);
[`PIPELINE.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl/pipelines/ONT/Assembly/PIPELINE.md)
tabulates the changes. Two alter a default run's output:

- `medaka_rounds` is 0, not upstream's 3, because medaka is not in the cluster image. The assemblies
  carry only Flye's own polishing round, which is not equivalent.
- The read mode follows the declared chemistry, so R10.4.1 gets `--nano-hq`, as Flye's docs
  prescribe, instead of upstream's hardcoded `--nano-raw`, which took 7.6x longer in Flye on the
  same full-chr20 reads (HG002, `m5.8xlarge`) for the same N50. `--asm-coverage` is off, since
  capping at Flye's documented 40x cut that N50 to a third. `flye_impute_params = false` restores
  upstream's command line.

## Step 1: Check the cluster

`WDL_DEMO_SCALE`, or `SCALE` below, sets the region per sample: `standard`, the default, is 10 Mbp
of chr20 and takes about 17 minutes; `quick`, which CI runs, is 2 Mbp and takes about 10 (Anyscale
Jobs with provisioning, AWS `m5.8xlarge` workers, Ray 2.56.0 image). The cell runs
`wdl-on-ray doctor`, a dry report of the backend's decisions; check that `none` reads
`always available` and the run dir is on shared storage.


```python
import json
import os
import pathlib
import subprocess

# `standard` by default; CI sets `quick`. Only the region size differs.
SCALE = os.getenv("WDL_DEMO_SCALE", "standard")
SCALES = {
    "quick":    {"region": "chr20:1,000,000-3,000,000",  "span": "2 Mbp",  "expect": "~10 min"},
    "standard": {"region": "chr20:1,000,000-11,000,000", "span": "10 Mbp", "expect": "~17 min"},
}
if SCALE not in SCALES:
    raise ValueError(f"WDL_DEMO_SCALE must be one of {sorted(SCALES)}, got {SCALE!r}")
CFG = SCALES[SCALE]

# The GIAB Ashkenazi trio (son, father, mother), assembled independently. One sample's task
# graph is a chain, which leaves the scheduler nothing to pack.
SAMPLES = ["HG002", "HG003", "HG004"]

TEMPLATE_DIR = pathlib.Path.cwd()
DATA_URI = f"s3://anyscale-public-materials/genomics/giab-trio-chr20/{SCALE}"

# Must be shared storage: a task reads the previous task's outputs by path, from any node.
# /mnt/local_storage fails with "file not found" on the second task.
WORK = pathlib.Path("/mnt/cluster_storage/wdl-on-ray")
DATA_DIR, RUN_DIR, SMOKE_DIR = WORK / "data", WORK / "runs", WORK / "smoke"
for d in (DATA_DIR, RUN_DIR, SMOKE_DIR):
    d.mkdir(parents=True, exist_ok=True)
# Required, and before TMPDIR moves. Ray finds the running cluster under <TMPDIR>/ray, and a
# workspace sets no RAY_ADDRESS (a job does), so moving TMPDIR alone makes `wdl-on-ray run`
# start a second, empty Ray on the head (CPU: 0), where every task waits forever.
os.environ.setdefault("RAY_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
os.environ["TMPDIR"] = str(WORK / "tmp")
pathlib.Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)


def run(cmd, **kwargs):
    """Run a command, streaming output, and raise if it fails, which `!` does not."""
    printable = " ".join(str(c) for c in cmd)
    print(f"$ {printable}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


print(f"scale       {SCALE}  ({CFG['span']} of {CFG['region']}, expect {CFG['expect']})")
print(f"samples     {', '.join(SAMPLES)}")
print(f"data        {DATA_URI}")
print(f"run dir     {RUN_DIR}\n")
run(["wdl-on-ray", "doctor"])
```

## Step 2: Read the workflow

Both files are plain WDL 1.0: `ONTAssembleWithFlye.wdl` is the per-sample pipeline, and
`ONTAssembleCohort.wdl` scatters it over the samples.

![The pipeline's task graph and resource requests](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/pipeline-dag.png)

The cell type-checks the cohort, lists the calls and prints `Flye.Assemble`'s `runtime {}` block;
expect exit 0 and two `NameCollision` lint warnings about inherited call names.


```python
wdl_dir = TEMPLATE_DIR / "wdl/pipelines/ONT/Assembly"
cohort_wdl = wdl_dir / "ONTAssembleCohort.wdl"
sample_wdl = wdl_dir / "ONTAssembleWithFlye.wdl"

# miniwdl's own type checker; nothing Ray-specific.
run(["wdl-on-ray", "check", str(cohort_wdl)])

# The calls, in the order each file declares them.
for path in (cohort_wdl, sample_wdl):
    print(f"\ncalls in {path.stem}:")
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("call "):
            print(f"  {stripped.split('{')[0].strip()}")

# One task's runtime block, as written upstream.
flye_task = (TEMPLATE_DIR / "wdl/tasks/Assembly/Flye.wdl").read_text()
block = flye_task[flye_task.index("runtime {"):]
print("\nFlye.Assemble runtime block:")
print("\n".join(block[: block.index("}") + 1].splitlines()))
```

Every task below runs under `--container-runtime none`, with the cluster image's tools.

## Step 3: Run a smoke test

`smoke.wdl` scatters eight 2-CPU shards with no tools or data, and the cell counts the nodes that
ran them from each task's `ray_placement.json`: expect one until a second worker is up.


```python
# SPREAD is best-effort, over the workers up at dispatch. Ray's default packs first, which
# suits the real pipeline, so the override is on this command only.
run([
    "wdl-on-ray", "run", str(TEMPLATE_DIR / "wdl/smoke/smoke.wdl"),
    "shards=8",
    "--container-runtime", "none",
    "--dir", str(SMOKE_DIR),
], env={**os.environ, "MINIWDL__RAY__SCHEDULING_STRATEGY": "SPREAD"})

# Which node ran each shard, in the latest run only; miniwdl gives each run its own directory.
smoke_run = max((p.parent for p in SMOKE_DIR.glob("*/outputs.json")), key=lambda p: p.stat().st_mtime)
placements = sorted(smoke_run.glob("**/ray_placement.json"))
nodes = {}
for path in placements:
    record = json.loads(path.read_text())
    node = record.get("node_id", record.get("node_ip", "unknown"))
    nodes.setdefault(node, []).append(path.parent.name)

print(f"\n{len(placements)} tasks ran across {len(nodes)} node(s):")
for node, tasks in nodes.items():
    print(f"  {node}: {len(tasks)} task(s)")
```

## Step 4: Stage the reads

The reads are ONT's public [`s3://ont-open-data`](https://registry.opendata.aws/ont-open-data/)
`giab_2023.05` release (R10.4.1, dorado sup v4.1.0), sliced from their GRCh38 alignments by
`tools/stage-demo-data.sh`. The reference slice is a contig named `chr20:<start>-<end>`, numbered
from 1, so outputs use slice-local coordinates. The cell stages the files, prints the manifest and
checks each FASTQ's sha256.


```python
manifest_path = DATA_DIR / f"MANIFEST.{SCALE}.json"
reference = DATA_DIR / f"reference.{SCALE}.fa"
reads = {s: DATA_DIR / f"{s}.reads.{SCALE}.fastq.gz" for s in SAMPLES}

wanted = [(f"{DATA_URI}/MANIFEST.json", manifest_path), (f"{DATA_URI}/reference.fa", reference)]
wanted += [(f"{DATA_URI}/{s}.reads.fastq.gz", path) for s, path in reads.items()]

for uri, dest in wanted:
    if dest.exists():
        print(f"already staged: {dest.name}")
        continue
    # Public bucket. A signed request is checked against the node role's policy and can fail.
    run(["aws", "s3", "cp", "--no-sign-request", "--only-show-errors", uri, str(dest)])

manifest = json.loads(manifest_path.read_text())
print(f"\n{manifest['region']}   {manifest['chemistry']}, {manifest['basecaller']}")
print(f"{'sample':<8}{'reads':>10}{'bases':>14}{'read N50':>10}{'coverage':>10}")
for entry in manifest["samples"]:
    print(f"{entry['sample']:<8}{entry['reads']:>10,}{entry['bases']:>14,}"
          f"{entry['read_n50']:>10,}{entry['coverage']:>9}x")

# Check each FASTQ against the manifest's sha256 before anything runs on it.
import hashlib

for entry in manifest["samples"]:
    path = reads[entry["sample"]]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == entry["sha256"], f"{path.name} does not match the manifest checksum"
print("\nall checksums match the manifest")
```

## Step 5: Assemble the cohort

One workflow runs the ten-task pipeline for all three samples. Each `Assemble` takes a worker to
itself (30 CPUs, 32 GiB), and tasks of 2 CPUs or fewer fit beside it. The 32 GiB covers the 17.6 GiB
peak of one full-chr20 run, against 106 GiB from upstream's formula; change `runtime_attr_flye` and
the 32-vCPU instance type together.


```python
inputs = {
    "ONTAssembleCohort.samples": [
        {"name": s, "fastqs": [str(reads[s])], "ref_fasta": str(reference)} for s in SAMPLES
    ],
    # From the manifest; selects Flye's read mode.
    "ONTAssembleCohort.read_chemistry": manifest["chemistry"],
    "ONTAssembleCohort.flye_num_threads": 30,
    "ONTAssembleCohort.quast_num_threads": 8,
    "ONTAssembleCohort.align_num_threads": 8,
    # medaka is not in the image. MedakaPolish still runs and passes the draft through, so
    # Flye's own round is the only polish.
    "ONTAssembleCohort.medaka_rounds": 0,
    "ONTAssembleCohort.medaka_use_gpu": False,
    "ONTAssembleCohort.runtime_attr_fastq_stats":     {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 50, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_genome_length":   {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_read_divergence": {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_merge_fastqs":    {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_flye":            {"cpu_cores": 30, "mem_gb": 32, "disk_gb": 500, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_medaka":          {"cpu_cores": 4,  "mem_gb": 16, "disk_gb": 200, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_quast":           {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_quast_summary":   {"cpu_cores": 1,  "mem_gb": 4,  "disk_gb": 20, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_align_paf":       {"cpu_cores": 8,  "mem_gb": 32, "disk_gb": 100, "preemptible_tries": 3},
    "ONTAssembleCohort.runtime_attr_paftools":        {"cpu_cores": 2,  "mem_gb": 8,  "disk_gb": 50, "preemptible_tries": 3},
}
inputs_path = WORK / f"inputs.cohort.{SCALE}.json"
inputs_path.write_text(json.dumps(inputs, indent=2))

print(f"assembling {len(SAMPLES)} samples x {CFG['span']} of {CFG['region']}")
print(f"expect about {CFG['expect']}, roughly one sample's wall clock, because they run"
      f" concurrently once the cluster has scaled up\n")
run([
    "wdl-on-ray", "run", str(cohort_wdl),
    "-i", str(inputs_path),
    "--container-runtime", "none",
    "--dir", str(RUN_DIR),
    "--verbose",
])
```

The next cell draws each task as a bar coloured by its node; look for small tasks sharing a colour
with another sample's assembly, and a total span close to one sample's.


```python
import matplotlib.pyplot as plt

run_dir = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime).parent

# One row per task, from ray_placement.json's mtime to task.log's. Each attempt rewrites the
# placement file, so a retried task shows its last attempt. The scatter shard in the path
# (.../call-assemble/shard-1/...) gives the sample.
rows = []
for placement in run_dir.glob("**/ray_placement.json"):
    task_dir = placement.parent
    parts = placement.relative_to(run_dir).parts
    shard = next((p for p in parts if p.startswith("shard-")), None)
    sample = SAMPLES[int(shard.split("-")[1])] if shard else "cohort"
    rows.append((sample, task_dir.name.removeprefix("call-"),
                 json.loads(placement.read_text()).get("node_id", "?")[-6:],
                 placement.stat().st_mtime, (task_dir / "task.log").stat().st_mtime))

# By sample, then by start time.
rows.sort(key=lambda r: (SAMPLES.index(r[0]) if r[0] in SAMPLES else -1, r[3]))

t0 = min(r[3] for r in rows)
span_min = (max(r[4] for r in rows) - t0) / 60
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#a855c7", "#d4a017"]
nodes = list(dict.fromkeys(r[2] for r in rows))
node_color = {n: (palette[i] if i < len(palette) else "#c3c2b7") for i, n in enumerate(nodes)}

fig, ax = plt.subplots(figsize=(10, 0.30 * len(rows) + 1.6), dpi=110)
for i, (sample, name, node, started, finished) in enumerate(rows):
    left, dur = (started - t0) / 60, (finished - started) / 60
    # Keep sub-second pass-through tasks visible.
    ax.barh(i, max(dur, span_min * 0.004), left=left, height=0.6, color=node_color[node])
    if dur > span_min * 0.05:
        label = f"{dur:.1f} min" if dur >= 1 else f"{dur * 60:.0f} s"
        ax.text(left + max(dur, span_min * 0.004) + span_min * 0.01, i,
                label, va="center", fontsize=8, color=MUTED)

ax.set_yticks(range(len(rows)), [f"{s}  {n}" for s, n, *_ in rows], fontsize=7.5)
ax.invert_yaxis()
# A rule between samples.
for i in range(1, len(rows)):
    if rows[i][0] != rows[i - 1][0]:
        ax.axhline(i - 0.5, color=GRID, linewidth=1)
ax.set_xlabel("minutes since the first task started", fontsize=9, color=MUTED)
ax.set_title(f"{len(rows)} WDL tasks as Ray tasks: {len(SAMPLES)} samples on {len(nodes)} node(s),"
             f" {span_min:.0f} min wall clock", fontsize=11, color=INK, loc="left")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis="x", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
if len(nodes) > 1:  # one node needs no legend; the title already counts it
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=node_color[n]) for n in nodes],
              labels=[f"node \u2026{n}" for n in nodes], loc="lower right",
              fontsize=8.5, frameon=False, title="colour = worker node",
              title_fontsize=8.5)
plt.tight_layout()
plt.show()
```

## Step 6: Read the QC

The cell prints QUAST's metrics, Flye flags and read statistics per sample. One `quick` run (Ray
2.56.0 image, `--nano-hq --iterations 1` for all three):

| | HG002 | HG003 | HG004 |
|---|---|---|---|
| # contigs | 1 | 1 | 1 |
| N50 | 2,036,301 | 2,081,008 | 2,155,765 |
| NGA50 | 586,366 | 1,122,380 | 583,361 |
| Genome fraction (%) | 98.104 | 99.984 | 98.230 |
| # mismatches per 100 kbp | 112.63 | 104.41 | 113.65 |
| # indels per 100 kbp | 33.08 | 30.37 | 32.03 |
| # misassemblies | 6 | 3 | 6 |

Flye's result depends on thread order, so the alignment-based columns move a percent or two between
runs; a rerun on the Ray 2.58.0 image (2026-09-24) matched genome fraction exactly and N50 to within
0.06%.

QUAST against GRCh38 measures distance from the reference, not correctness:

- Rearrangements over 1 kbp count as misassemblies and break NGA50's aligned blocks, so the sample's
  own structural variants count as errors. On full chr20, polished and unpolished runs alike reach
  an NGA50 near 2 Mbp against a 33 Mbp N50.
- A collapsed human assembly differs from GRCh38 at roughly 85-95 real SNVs per 100 kbp, so use the
  mismatch rate to compare a draft with its polished version, not as an error rate. Indels, ONT's
  main error class, have their own row.
- The reads were selected by where their primary alignment to GRCh38 fell. Reads too divergent to
  align are missing, and so are most reads from paralogs elsewhere in the genome, the repeats that
  make whole-genome assembly hard; the few mismapped into the window are present. Genome fraction
  near 100% is expected, and contiguity is optimistic.

So don't rank samples on these columns. HG003 has the best NGA50 and fewest misassemblies at
middling coverage (75x, against 89x and 59x) and the lowest read N50. HG002's agreement with its
mother, HG004, within 1% on NGA50, genome fraction, mismatches and misassemblies is consistent
with sequence the two share that differs from GRCh38 in this window.


```python
# Run roots only: nested sub-workflow directories hold their own, possibly newer, outputs.json.
outputs_path = max(RUN_DIR.glob("*/outputs.json"), key=lambda p: p.stat().st_mtime)
# A bare name -> value map; the {"dir", "outputs"} envelope is CLI stdout only. Accept both.
report = json.loads(outputs_path.read_text())
outputs = report.get("outputs", report)


def quast_key(name):
    """Map a QUAST display name onto its `quast_summary` key.

    SummarizeQuastReport's sed turns spaces into underscores and `>=` into `gt`, so
    "Genome fraction (%)" arrives as "Genome_fraction_(%)".
    """
    return name.replace(" ", "_").replace(">=", "gt")


METRICS = (
    "# contigs", "Total length", "N50", "NG50", "NGA50",
    "Genome fraction (%)", "# misassemblies",
    "# mismatches per 100 kbp", "# indels per 100 kbp",
)

names = outputs["ONTAssembleCohort.sample_names"]
summaries = outputs["ONTAssembleCohort.quast_summaries"]

# Without the minimap2 it builds at install, QUAST exits 0 with contiguity only. Quast.wdl fails
# the task then; this is a second check.
for name, summary in zip(names, summaries):
    assert quast_key("Genome fraction (%)") in summary, (
        f"{name}: QUAST produced no reference-based metrics; its minimap2 is missing. "
        "See tools/manifest.toml, [tools.quast]."
    )

print(f"{'metric':<28}" + "".join(f"{n:>16}" for n in names))
for metric in METRICS:
    key = quast_key(metric)
    if not any(key in s for s in summaries):
        continue
    print(f"{metric:<28}" + "".join(f"{s.get(key, '-'):>16}" for s in summaries))

print("\nFlye parameters each sample derived from its own reads:")
for name, params in zip(names, outputs["ONTAssembleCohort.flye_params"]):
    print(f"  {name}  {params['read_mode']}  {params['extra_args'] or '(no extra args)'}")

print("\nWhat each read set measured:")
print(f"  {'sample':<8}{'reads':>10}{'read N50':>10}{'coverage':>10}{'divergence':>12}")
for name, stats in zip(names, outputs["ONTAssembleCohort.read_stats"]):
    print(f"  {name:<8}{int(stats['num_reads']):>10,}{int(stats['read_n50']):>10,}"
          f"{float(stats['coverage']):>9.1f}x{float(stats['pairwise_divergence']):>12.4f}")

print("\nassemblies:")
for name, path in zip(names, outputs["ONTAssembleCohort.assemblies"]):
    print(f"  {name}  {path}")
```

The next cell plots Nx (solid) and NGx (dashed) with QUAST's 500 bp floor, so x = 50 matches the N50
and NG50 above. NGx sits on or above Nx because whole reads overhang the slice.


```python
import bisect

import matplotlib.pyplot as plt

# QUAST's default contig floor, so the curve at x = 50 is QUAST's N50.
MIN_CONTIG = 500


def contig_lengths(fasta_path, min_length=MIN_CONTIG):
    lengths, current = [], 0
    with open(fasta_path) as fasta:
        for line in fasta:
            if line.startswith(">"):
                if current:
                    lengths.append(current)
                current = 0
            else:
                current += len(line.strip())
    if current:
        lengths.append(current)
    return sorted((n for n in lengths if n >= min_length), reverse=True)


def nx_points(lengths, denominator):
    """(x, Nx) for x in 1..100: the contig length at which the largest-first running
    total first covers x% of `denominator`. Stops where the assembly stops covering."""
    cums, total = [], 0
    for length in lengths:
        total += length
        cums.append(total)
    pts = []
    for x in range(1, 101):
        idx = bisect.bisect_left(cums, denominator * x / 100)
        if idx >= len(lengths):
            break
        pts.append((x, lengths[idx]))
    return pts


ref_len = sum(contig_lengths(reference, min_length=0))  # the reference slice staged in Step 4
assemblies = {n: contig_lengths(p)
              for n, p in zip(names, outputs["ONTAssembleCohort.assemblies"])}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
palette = ["#2a78d6", "#eb6834", "#1baf7a"]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
for (name, lengths), color in zip(assemblies.items(), palette):
    # Solid: Nx, against the assembly's length. Dashed: NGx, against the reference's.
    for pts, style, label in (
        (nx_points(lengths, sum(lengths)), "-", f"{name}  Nx"),
        (nx_points(lengths, ref_len), "--", f"{name}  NGx"),
    ):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.step(xs, [y / 1e6 for y in ys], where="post", color=color,
                linestyle=style, linewidth=1.8, label=label)

ax.set_xlabel("x (%)", fontsize=9, color=MUTED)
ax.set_ylabel("contig length (Mbp)", fontsize=9, color=MUTED)
ax.set_title(f"{len(assemblies)} assemblies of {ref_len / 1e6:.1f} Mbp"
             "   (solid: Nx, assembly length   dashed: NGx, reference length)",
             fontsize=10.5, color=INK, loc="left")
ax.set_xlim(0, 100)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=len(assemblies))
plt.tight_layout()
plt.show()
```

### Step 6b: Compare the trio's calls

The cell, plain Ray code on the same cluster, normalizes each paftools VCF with `bcftools norm`,
parses them with Ray Data and prints the share of HG002's calls in either parent. Allele dropout
alone leaves roughly 25-40% in neither, since each collapsed assembly carries one mosaic haplotype,
and indel placement and differing callable blocks add more. This is a consistency check with no
unrelated control, not a Mendelian error rate: much variation against GRCh38 is common, so an
unrelated sample would share many of these calls.


```python
import os

import ray

# The cluster the WDL tasks ran on; nothing new is provisioned here.
vcfs = dict(zip(names, outputs["ONTAssembleCohort.vcfs"]))

# Normalize first: in a homopolymer, two assemblies can place one indel differently.
# `-m -any` splits multiallelic records, so sites compare allele by allele.
NORM_DIR = WORK / "vcf-normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)
if not pathlib.Path(f"{reference}.fai").exists():
    run(["samtools", "faidx", str(reference)])

normalized = {}
for sample, path in vcfs.items():
    dest = NORM_DIR / f"{sample}.norm.vcf"
    run(["bcftools", "norm", "-f", str(reference), "-m", "-any", "-o", str(dest), str(path)])
    normalized[sample] = dest

# Ray Data reports each row's source path; map basenames back to samples.
SAMPLE_BY_FILE = {path.name: sample for sample, path in normalized.items()}
assert len(SAMPLE_BY_FILE) == len(normalized), "VCF basenames are not unique across samples"


def parse_vcf_lines(batch):
    """VCF text -> (sample, chrom, pos, ref, alt), skipping headers. Runs on the workers."""
    out = {"sample": [], "chrom": [], "pos": [], "ref": [], "alt": []}
    for text, path in zip(batch["text"], batch["path"]):
        text = str(text)
        if text.startswith("#") or not text.strip():
            continue
        fields = text.split("\t")
        if len(fields) < 5:
            continue
        sample = SAMPLE_BY_FILE.get(os.path.basename(str(path)))
        if sample is None:
            continue
        out["sample"].append(sample)
        out["chrom"].append(fields[0])
        out["pos"].append(int(fields[1]))
        out["ref"].append(fields[3])
        out["alt"].append(fields[4])
    return out


variants = (
    ray.data.read_text([str(p) for p in normalized.values()], include_paths=True)
    .map_batches(parse_vcf_lines, batch_format="numpy")
    .to_pandas()
)

if variants.empty:
    # 2 Mbp of collapsed human assembly carries ~1,500 SNVs against GRCh38. Empty means no
    # alignment, or every block fell under `paftools call -L` (50 kb default), which exits 0.
    raise AssertionError(
        "no variant calls in any sample. Check the QUAST genome fraction above, then "
        "min_alignment_length_call in CallAssemblyVariants.wdl against this run's NGA50."
    )

print(f"{len(variants):,} normalized calls across {variants['sample'].nunique()} samples")
print(variants.groupby("sample").size().to_string(header=False))

# A call is identified by (chrom, pos, ref, alt).
def keyset(sample):
    rows = variants[variants["sample"] == sample]
    return set(zip(rows["chrom"], rows["pos"], rows["ref"], rows["alt"]))


child, father, mother = keyset("HG002"), keyset("HG003"), keyset("HG004")
inherited = child & (father | mother)
unshared = child - father - mother

print(f"\nHG002 calls:                    {len(child):,}")
print(f"  also in HG003 or HG004:       {len(inherited):,}"
      f"  ({100 * len(inherited) / max(len(child), 1):.1f}%)")
print(f"  in neither parent:            {len(unshared):,}"
      f"  ({100 * len(unshared) / max(len(child), 1):.1f}%)")
print("\nAllele dropout from haplotype collapse alone puts the expected 'in neither'")
print("figure around 25-40%, so read the second number against that rather than")
print("against zero. It is a consistency check, not a violation rate.")
```

## Step 7: Persist the outputs

A job terminates its cluster on success, taking `/mnt/cluster_storage` with it.
[`persist_outputs.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/persist_outputs.py)
copies the declared outputs, not the intermediates, to `WDL_ON_RAY_RESULTS` or else the first
writable durable mount (`/mnt/user_storage`, then `/mnt/shared_storage`).


```python
run(["python", str(TEMPLATE_DIR / "persist_outputs.py"), "--outputs", str(outputs_path)])
```

## Run the whole chromosome as a job

[`job.yaml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/job.yaml)
runs the trio over all 64 Mbp of chr20 at full coverage. Submit it from the template directory,
because `working_dir: .` resolves against the shell:

```bash
cd templates/wdl-genomics-on-ray
anyscale job submit --config-file job.yaml
```

### Time and cost

One run (Ray 2.56.0 image, three on-demand `m5.8xlarge` workers, 97x, 80x and 63x coverage,
uncapped):

| | HG002 | HG003 | HG004 |
|---|---|---|---|
| Flye | 1h55m33s | 1h44m12s | 1h24m04s |
| Sample, end to end | 2h01m15s | 1h48m46s | 1h37m45s |
| # contigs | 62 | 71 | 52 |
| N50 | 33,263,886 | 16,721,982 | 33,233,489 |
| L50 / L90 | 1 / 2 | 2 / 3 | 1 / 2 |
| NGA50 | 2,082,054 | 2,188,745 | 1,960,991 |
| Genome fraction (%) | 95.798 | 95.854 | 95.820 |
| # mismatches per 100 kbp | 154.02 | 142.61 | 149.30 |
| # misassemblies | 121 | 111 | 129 |

The workflow took 2h01m58s against 2h01m15s for its slowest sample, and the job 2h03m56s: the
samples used about a serial run's node-hours and finished in the slowest one's time. That is about 6
worker node-hours, or $10 with the head at us-east-1 on-demand list prices. Node-hours are the
durable unit, since prices vary by region and commitment.

Workers ask for spot, falling back to on-demand; the head stays on demand. With
`preemptible_tries: 3` on every task, a reclaim restarts that node's tasks from zero, not the run;
no real reclaim has tested it. At the Spot Advisor's -69% for `m5.8xlarge` (us-west-2, 2026-09-24),
the run would cost about $3.65, an estimate; at 10% reclaims per node-hour, restarts eat the saving
on assemblies past about ten hours.
[`NOTES.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/NOTES.md)
has the read-mode history, queue times, contigs and spot model.

## Bring your own WDL and inputs

Start with `wdl-on-ray check <your.wdl>`.
[`wdl_on_ray/backend.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/backend.py)
holds the miniwdl contract, and
[`resources.py`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/wdl_on_ray/resources.py)
maps `runtime {}` onto Ray:

- `cpu`, `memory`: Ray `num_cpus` and `memory`.
- `gpuCount`, `gpuType`, `acceleratorType`: Ray GPU and accelerator-type requests.
- `docker`: depends on `--container-runtime`, below.
- `preemptible`: spent on Ray node or worker loss, raised as miniwdl's `Interrupted`.
- `maxRetries`: left to miniwdl; Ray-level retries default to 0.
- `disks`: logged and ignored, unless you set `[ray] disk_resource_name`.

For your own reads, copy `wdl/pipelines/ONT/Assembly/inputs.chr20.cohort.json`, which carries the
`runtime_attr_*` sizing, and replace the samples and chemistry:

```json
{
  "ONTAssembleCohort.samples": [
    {"name": "NA12878",
     "fastqs": ["s3://my-bucket/NA12878/flowcell1.fastq.gz",
                "s3://my-bucket/NA12878/flowcell2.fastq.gz"],
     "ref_fasta": "s3://my-bucket/ref/GRCh38.fa"}
  ],
  "ONTAssembleCohort.read_chemistry": "R10.4.1"
}
```

`fastqs` takes one file per flow cell, local or `gs://`, `s3://` or `https://`. Then check:

- `ComputeGenomeLength` sums every sequence in `ref_fasta` into Flye's genome size and, with
  `runtime_attr_flye` unset, its memory request; QUAST and paftools use the whole file too. For
  reads from one region, pass the matching slice or set
  `ONTAssembleCohort.assemble.flye_genome_size`.
- `read_chemistry` picks the read mode (R10 gets `--nano-hq`, unset keeps `--nano-raw`); take it
  from run metadata, as a FASTQ may not record it.
- `runtime_attr_flye` must fit a node you have; upstream's whole-genome sizing is 16 cores and ~410
  GiB. Above ~100 Mbp, also set `ONTAssembleCohort.assemble.quast_is_large`.

## Containers and tools

Under `none` the declared tag is a label: `Flye.wdl` declares `lr-flye:2.8.3`, but the Flye that
runs is 2.9.5, pinned in
[`tools/manifest.toml`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/manifest.toml);
2.8.3 has no `--nano-hq`. To add a tool, give it a `[tools.<name>]` block there (version, URL,
sha256) and rebuild from the template directory:

```bash
anyscale image build -n my-wdl-tools --containerfile Dockerfile
```

A GATK Best Practices pipeline outgrows one image; a validated clinical pipeline needs each task in
its declared image. Nested `podman run` fails on Anyscale, where Ray itself runs in a container, so
use `--container-runtime ray`, which starts each task's Ray worker in its own image. Build those
images from the cluster's base, since Ray and Python must match exactly, map each declared tag in
`[ray] task_image_map`, set `task_image_fallback = cluster` if unmapped tasks may use the cluster
image, and test each with `wdl-on-ray probe-image <uri>`.
[`tools/BUILDING.md`](https://github.com/anyscale/templates/blob/main/templates/wdl-genomics-on-ray/tools/BUILDING.md)
has the config, `native` mode, the privileged container that Kubernetes-backed clouds need, and a
GPU medaka example that turns polishing back on (its image is not yet built or run). For human-scale
polishing, ONT now points to `dorado polish` rather than medaka.

## Limits

- miniwdl clamps `runtime.cpu` and `runtime.memory` to the largest worker up when the run starts,
  with a warning, and clamps nothing if no worker is up yet; pass `--max-cpu` and set
  `[ray] max_memory_bytes` to the worker shape. A request no node can serve, such as a GPU with no
  GPU worker group, waits forever without an error.
- Call caching is off by default and miniwdl's default cache directory is node-local;
  `--call-cache DIR` turns it on, and DIR must be on shared storage.
- On a cluster, miniwdl submits up to 200 tasks at once, its guideline for one driver process; Ray
  queues those that do not fit, and the autoscaler adds workers for them, up to `max_nodes`.
  `--task-concurrency` changes the 200.
- A task with `preemptible` and `maxRetries` both at 0 fails the workflow on its first reclaim.
