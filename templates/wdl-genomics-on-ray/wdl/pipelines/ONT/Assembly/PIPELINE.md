# ONT assembly with Flye

A single-sample genome assembly pipeline for Oxford Nanopore reads, adapted from
[broadinstitute/long-read-pipelines](https://github.com/broadinstitute/long-read-pipelines/tree/main/wdl/pipelines/ONT/Assembly)'s
[`ONTAssembleWithFlye.wdl`](https://github.com/broadinstitute/long-read-pipelines/blob/main/wdl/pipelines/ONT/Assembly/ONTAssembleWithFlye.wdl).
The call graph, the tools and their flags are upstream's; what changed is portability,
and every file records its own deviations in its header.

Adapted from long-read-pipelines **4.0.68** (`02089d9`). The source file itself last
changed in `475331a`, 2023-04-11, so the divergences below have a fixed baseline
rather than a moving one. Upstream's sibling Canu pipeline is not shipped: Flye needs
no separate correction pass, so it typically finishes in a fraction of Canu's wall
time on the same input, at some cost in per-base accuracy before polishing.

```
                      ┌──► FastqStats ─────────┐
fastqs ──► MergeFastqs ┤                       ├──► (flye parameters) ──► Flye.Assemble
                │      └──► MeasureDivergence ─┘                               │
                └──────────────────► MedakaPolish ◄─────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
          Quast         AlignAsPAF ──► Paftools
            │                                 │
   SummarizeQuastReport                  paftools_vcf
ref_fasta ──► ComputeGenomeLength ──► (genome size, for --genome-size and memory sizing)
```

`FastqStats` and `MeasureDivergence` are additions rather than ports: they measure the
read set so that Flye's parameters can be derived from evidence instead of hardcoded.
The rules and the reasoning behind each live in
[the pipeline's own header](ONTAssembleWithFlye.wdl); the measurements and the
resolved parameters are emitted as the `read_stats` and `flye_params` outputs, so
every run records what it saw and what it chose. The two tasks are independent, so
they run concurrently.

## What differs from upstream

| | upstream | here |
|---|---|---|
| reads | `String gcs_fastq_dir`, expanded by `gsutil ls` | `Array[File]+ fastqs` (miniwdl localizes `gs://`, `s3://`, `https://`) |
| outputs | six `FinalizeToFile` calls (`gsutil cp`) | declared outputs; miniwdl collects them |
| reference | `File ref_map_file` read with `read_map()` | `File ref_fasta` (only the `fasta` key was used) |
| Medaka GPU | one `nvidia-tesla-t4`, unconditionally | `medaka_use_gpu`, default false |
| Medaka rounds | 3 | 0, because medaka is not in this template's image; Flye's own round is the polish |
| `--threads` | `num_core=$(cat /proc/cpuinfo ...)`, in Flye and QUAST | explicit `flye_num_threads` / `quast_num_threads`, each also its task's default `cpu_cores` |
| memory request | `{ 'mem_gb': 100 + genome_size/1e7 }` fixed at the call site | `RuntimeAttr? runtime_attr_flye`, with upstream's formula as the fallback |
| resources | fixed per task | `RuntimeAttr?` overrides per task, from the inputs file |
| Flye tuning | not reachable | `flye_extra_args` |
| read type | `--nano-raw`, hardcoded | selected from `read_chemistry`, as Flye's docs prescribe |
| `--genome-size` | never passed | derived, and required by `--asm-coverage` |
| `--asm-coverage` | never passed | derived from measured coverage |
| `--iterations` | Flye's default | 0 when Medaka will re-polish anyway |
| variant prefix | `prefix + ".canu"` | `prefix + ".flye"` |
| polished name | `basename(Flye.fa, ".fasta")`, which strips nothing | `basename(Flye.fa, ".fa")` |

Two of those rows are upstream bugs rather than deployment differences, so each row
gets its own explanation below.

`--threads` from `/proc/cpuinfo` reads the *host's* CPU list, not the cgroup's
allowance. Under `--cpus N` (a CFS quota, not a CPU mask) and under no-container mode
alike, the tool plans for every core on the machine regardless of what `runtime.cpu`
requested, oversubscribing the node and inflating peak memory with it. Upstream's
Flye and QUAST tasks both do this; both now take an explicit thread count that also
feeds the task's default `cpu_cores`, so the Ray reservation and the flag agree
unless an override disagrees with both. [`Medaka.wdl`](../../../tasks/Preprocessing/Medaka.wdl)
states `medaka_consensus -t` the same way, and the minimap2 tasks state `-t` likewise.

The memory request was unreachable. Upstream's Flye sub-workflow hardcodes
`runtime_attr_override = { 'mem_gb': 100.0 + (genome_size/10000000.0) }` at the call
site, so it has no `runtime_attr_override` input of its own and a caller can't set
`cpu_cores` at all. On Cromwell that formula sizes a VM to order. On Ray it's a
request against nodes that already exist, and ~100 GiB is more than a demo cluster
has in total, which Ray answers by waiting indefinitely instead of failing. The port
takes a `RuntimeAttr?` and falls back to upstream's formula when none is given, so
upstream behaviour is still the default. The fallback is all-or-nothing, though: an
override replaces it wholesale and unset fields come from the task's own flat 100 GiB
default, so state `mem_gb` even when you only mean to change `cpu_cores`.

Medaka's polishing rounds default to 0 here; upstream defaults to 3. Nothing
scientific drives that change. Medaka is not in this template's cluster image,
so under `--container-runtime none` a default of 3 would fail with exit 127 *after*
the hours-long assembly had succeeded. The workflow's invariant is that total
polishing passes never reach zero: with `medaka_rounds = 0`, Flye's own polishing
round is imputed back on. Those two are not interchangeable. Flye's polisher fixes
what its own read alignments support; medaka is a neural consensus trained on Flye
output for a specific chemistry and basecaller, and it is what removes the homopolymer
indels Flye leaves. "At least one pass" is an invariant against a silent no-op, not a
quality guarantee.

To polish with medaka, get it onto the workers first (`tools/BUILDING.md` covers
adding a tool) and pick a model matching your chemistry. `medaka_model` defaults to
`r1041_e82_400bps_sup_v4.1.0` here, matching the reads this template ships; upstream's
default is `r941_prom_high_g360`, from the R9.4.1 / Guppy era, and a mismatched model
degrades the polish without failing. Run `medaka tools list_models` and match the
sampling rate as well as the chemistry: 4 kHz and 5 kHz R10.4.1 runs take different
model lines (`v4.1.0` against `v4.2.0` and later).

`prefix + ".canu"` in a Flye pipeline is a copy-paste from upstream's Canu sibling.
It labelled the PAF and VCF with the wrong assembler, and is `".flye"` here.

`basename(Flye.fa, ".fasta")` stripped nothing, because the Flye task emits
`*.flye.fa`. The polished assembly came out as `<prefix>.flye.fa.consensus.fasta`.
Stripping `".fa"` gives `<prefix>.flye.consensus.fasta`.

`genome_size` is deliberately left alone: it stays a `Float` in base pairs, matching
`flye --genome-size`'s own units. Upstream's Canu sibling takes an `Int` in megabases
for the same reason, each pipeline matching its assembler.

## Running it

The template's README walks the pipeline end to end on the GIAB Ashkenazi trio
(HG002, HG003, HG004) over a chromosome 20 region, one copy of this workflow per
sample under [`ONTAssembleCohort.wdl`](ONTAssembleCohort.wdl):
the notebook builds its inputs at demo scale in Step 5, and
[`inputs.chr20.json`](inputs.chr20.json) is the full-chromosome inputs file that
[`job.yaml`](../../../../job.yaml) submits as an Anyscale Job. The `runtime_attr_*`
entries in both are sized to the template's 32-vCPU / 128 GiB workers. On Ray these
decide schedulability, not just speed: a WDL task is one process on one node, so a
request larger than the biggest node never runs. It waits, indefinitely, because Ray
cannot distinguish "no node this big exists" from "the autoscaler hasn't caught up
yet".

## Real data

```json
{
  "ONTAssembleWithFlye.fastqs": [
    "s3://my-bucket/sample/flowcell1.fastq.gz",
    "s3://my-bucket/sample/flowcell2.fastq.gz"
  ],
  "ONTAssembleWithFlye.ref_fasta": "s3://my-bucket/ref/GRCh38.fa",
  "ONTAssembleWithFlye.participant_name": "NA12878",
  "ONTAssembleWithFlye.prefix": "NA12878",
  "ONTAssembleWithFlye.quast_is_large": true,
  "ONTAssembleWithFlye.flye_num_threads": 32,
  "ONTAssembleWithFlye.runtime_attr_flye": { "cpu_cores": 32, "mem_gb": 420 }
}
```

(The file has no comments because miniwdl rejects a `"//"` pseudo-key outright, with
"unknown input/output: //", instead of ignoring it the way some engines do.)

Omit `flye_num_threads` and `runtime_attr_flye` to get upstream's sizing, which for a
3.1 Gbp genome is 16 cores and ~410 GiB. That's reasonable on the Broad's fleet, and
a node you have to actually possess on Ray, so size the compute config and the request
together.

The `Flye.wdl` task itself still defaults to `--nano-raw`, as upstream wrote it. The
workflow above it selects the read mode from `read_chemistry`: R10 gets `--nano-hq`, as
Flye's docs prescribe, as does R9 basecalled by Guppy5+ or in sup mode. An unset
chemistry keeps `--nano-raw` rather than guessing. `flye_read_mode` overrides it, and
`flye_impute_params = false` restores upstream's command line exactly.

An earlier version gated this on measured read-to-read divergence instead, and that rule
does not survive a full chromosome. The same sample measured 0.0721 over 2 Mbp, 0.0986
over 10 Mbp and 0.1532 over all of chr20, because the estimator tracks repeat content and
the whole chromosome includes the centromere. Against a 0.10 threshold that sends
R10.4.1 sup reads to `--nano-raw`. The measurement is still taken and still reported, as
`divergence_flag` in `flye_params` and `pairwise_divergence` in `read_stats`; nothing
branches on it. [The workflow header](ONTAssembleWithFlye.wdl) has the numbers and
[`ReadStats.wdl`](../../../tasks/QC/ReadStats.wdl) the estimator's known biases.
