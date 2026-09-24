# Pipeline provenance and divergences

The pipeline is GATK Best Practices germline short-variant discovery, shaped the
way [nf-core/sarek](https://nf-co.re/sarek) shapes it. The processes are adapted
from [nf-core/modules](https://github.com/nf-core/modules) (MIT). This file records
what came from where and — more usefully — every place this pipeline does something
*different*, and why.

Divergences are listed because an undeclared one is indistinguishable from a bug.
If a number here disagrees with a number from sarek, the reason should be on this
page.

## Upstream

| Process | Adapted from | Notes |
|---|---|---|
| `FASTP` | `nf-core/modules/fastp` | `--detect_adapter_for_pe`; no UMI handling |
| `BWAMEM2_INDEX`, `BWAMEM2_MEM` | `nf-core/modules/bwamem2/{index,mem}` | read group built inline |
| `GATK4_MARKDUPLICATES` | `nf-core/modules/gatk4/markduplicates` | indexes with samtools rather than `--CREATE_INDEX` |
| `GATK4_BASERECALIBRATOR`, `GATK4_APPLYBQSR` | `nf-core/modules/gatk4/{baserecalibrator,applybqsr}` | one known-sites resource, not three |
| `MAKE_INTERVALS` | `nf-core/modules/gatk4/intervallisttools` | `SplitIntervals` with `INTERVAL_SUBDIVISION` |
| `GATK4_HAPLOTYPECALLER` | `nf-core/modules/gatk4/haplotypecaller` | `-ERC GVCF`, scattered by interval |
| `GATK4_GENOMICSDBIMPORT`, `GATK4_GENOTYPEGVCFS` | `nf-core/modules/gatk4/*` | unchanged in shape |
| `GATK4_MERGEVCFS` | `nf-core/modules/gatk4/mergevcfs` | sorted input list — see below |
| `GATK4_VARIANTFILTRATION` | `nf-core/modules/gatk4/variantfiltration` | hard filters, not VQSR — see below |
| `DEEPVARIANT` | `nf-core/modules/deepvariant` | off by default, no binaries to run — see below |
| `RTG_FORMAT`, `RTG_VCFEVAL` | `nf-core/modules/rtgtools/*` | `--evaluation-regions`, split by variant type |
| `MULTIQC` | `nf-core/modules/multiqc` | unchanged |

No upstream counterpart, written for this template:
`SPLIT_SAMPLE`, `SUBSET_VARIANT_TYPE`, `COLLECT_BENCHMARK`, `SHARD_VCF`,
`ANNOTATE_VARIANTS`, `COLLECT_SCORES`, `COLLECT_PLACEMENT`, and `smoke.nf`.

## Divergences

### No `container` directives

Every nf-core module carries `container 'quay.io/biocontainers/...'`. They are
dropped here rather than kept as decoration.

A Ray worker is already inside a container and cannot nest another, so a container
directive would be a promise the executor cannot keep — Nextflow would ignore it
and run the command against whatever is on `PATH`, which is exactly what happens
now, only without the misleading declaration. The toolchain is baked into the
image instead (`tools/env.main.yml`).

**This is the largest behavioural delta from upstream**, because it means tool
versions come from the image rather than from the pipeline. `-profile conda` is the
escape hatch: Nextflow builds each process its declared environment, cached on
shared storage so it is built once per cluster.

### Hard filters, not VQSR

sarek runs VariantRecalibrator genome-wide. This pipeline uses GATK's published
hard-filter thresholds instead.

VQSR needs far more variants than one chromosome provides — it would either refuse
to build a model or build a bad one — and hard filtering is GATK's own documented
fallback for small callsets. **It costs precision**, and it is part of why the
numbers here should not be compared to a published genome-wide sarek benchmark.

### One known-sites resource

sarek passes dbSNP, Mills and 1000G indels to BQSR. This passes one, subset to
chr20, to keep the staged demo data small. The recalibration is slightly less well
informed as a result.

### DeepVariant is off by default

The plan had DeepVariant as a second caller, from bioconda, in its own python
3.10 environment (it pins `python <3.11`, and a Ray driver and its workers must
agree on the interpreter to the patch). That cannot work with the package as
published. bioconda's `deepvariant` 1.10.0 (`pyh697b589_0`) is a 235 KB noarch
package: three `dv_*.py` wrappers and `tf_slim`. The wrappers call
`$PREFIX/BINARYSUB/make_examples.zip` and a `WGSMODELSUB` checkpoint, literal
placeholders, because the recipe's `build.sh` has the steps that install
Google's binaries and models commented out. There is no `run_deepvariant` and no
model in it. (Read from the package and its bundled recipe, 2026-09-24.)

So `params.deepvariant` defaults to `false`, the image builds no DeepVariant
environment, and `--deepvariant true` stops at startup with that explanation
rather than after alignment. The `DEEPVARIANT` process, its benchmark wiring and
`tools/run_deepvariant.sh` are kept: installing Google's own DeepVariant 1.10.0
into a 3.10 environment so that `bin/run_deepvariant` exists is the one change
that turns it on, and the wrapper still keeps that environment off `PATH`.

GPU DeepVariant was considered and rejected: GPU support exists only in Google's
own `-gpu` Docker image and covers only the `call_variants` stage, so using it
would mean rebuilding DeepVariant on a Ray-matched base. The sibling WDL template
lost the equivalent fight with medaka and shipped it disabled; this template's GPU
work is a separate process instead.

Not scattered by interval, unlike HaplotypeCaller: `run_deepvariant` is a
three-stage pipeline that shards internally, and an outer scatter would mean
re-merging partial callsets whose records disagree at shard boundaries.

### `MergeVcfs` input is sorted

`path vcfs` arrives in whatever order the channel emitted, which depends on which
interval finished first. Without `LC_ALL=C sort`, the merged header's contig order
can vary between runs of the same pipeline, and two runs stop being
byte-comparable. Same reasoning in `COLLECT_SCORES`.

### `MAKE_INTERVALS` fails when the scatter collapses

`SplitIntervals` on a single-contig reference with the wrong subdivision mode
produces exactly one interval. The pipeline would still succeed — just serially,
hours later, having demonstrated nothing. So it asserts it got more than one, and
fails if not.

### Benchmarking is split by variant type

vcfeval is run separately for SNPs and indels. A combined F1 hides which moved,
and the two fail for different reasons: SNP F1 is dominated by sequencing error,
indel F1 by alignment and local reassembly.

`--evaluation-regions` rather than `--bed-regions`: the former restricts *scoring*
to the high-confidence set while still allowing calls just outside it to
participate in haplotype matching. The latter truncates haplotypes and invents
mismatches at every boundary.

## Bounds on the numbers

State these wherever the results are shown, not in a footnote.

- **Reference-selected reads.** The demo FASTQs are derived by slicing an existing
  chr20 alignment, so reads that would mismap *into* chr20 from elsewhere in the
  genome are absent by construction. Precision therefore reads slightly high
  relative to a real whole-genome run.
- **Scored on an intersection.** chr20 ∩ the GIAB high-confidence BED ∩ the demo
  slice. Outside it the truth set makes no claim.
- **One chromosome, hard filters, one known-sites resource.** Not comparable to a
  published genome-wide sarek or DeepVariant benchmark.
- **The annotation score is a proxy.** Embedding distance from a nucleotide
  language model. It correlates with "this changes the sequence in a way the model
  noticed" — it is not a pathogenicity score and is not comparable to SpliceAI or
  CADD.

## The DAG

```
samplesheet ──> FASTP ──> BWAMEM2_MEM ──> MARKDUPLICATES ──> BQSR ──> analysis-ready BAM
                                                                          │
                          ┌───────────────────────────────────────────────┤
                          │  x N intervals                                │  per sample
                          v                                               v
                 HAPLOTYPECALLER (3 x N tasks)                      DEEPVARIANT
                          │  group by interval                            │
                          v                                               │
                 GENOMICSDBIMPORT ──> GENOTYPEGVCFS                       │
                          │  gather                                       │
                          v                                               │
                 MERGEVCFS ──> VARIANTFILTRATION ─────┬───────────────────┤
                                                      │                   │
                                     SHARD_VCF        │  split per sample, per type
                                          │           v                   v
                                   ANNOTATE_VARIANTS  RTG_VCFEVAL  (samples x callers x types)
                                     (GPU, x shards)          │
                                          │                   v
                                   COLLECT_SCORES      COLLECT_BENCHMARK ──> benchmark.tsv
```

At `standard` scale: 3 samples, 24 intervals → 72 concurrent HaplotypeCaller
tasks, 24 joint-genotyping tasks, 12 vcfeval comparisons, 8 GPU shards.
