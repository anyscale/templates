# Pipeline provenance and divergences

The pipeline is GATK germline short-variant discovery in the shape of the joint-germline mode of
[nf-core/sarek](https://nf-co.re/sarek). The processes are adapted from
[nf-core/modules](https://github.com/nf-core/modules) (MIT). This page records where each process
came from and every place the pipeline differs, so that a number that disagrees with sarek's can be
traced to a cause here.

## Upstream

| Process | Adapted from | Notes |
|---|---|---|
| `FASTP` | `nf-core/modules/fastp` | `--detect_adapter_for_pe`; no UMI handling |
| `BWAMEM2_INDEX`, `BWAMEM2_MEM` | `nf-core/modules/bwamem2/{index,mem}` | read group built inline |
| `GATK4_MARKDUPLICATES` | `nf-core/modules/gatk4/markduplicates` | indexes with samtools rather than `--CREATE_INDEX` |
| `GATK4_BASERECALIBRATOR`, `GATK4_APPLYBQSR` | `nf-core/modules/gatk4/{baserecalibrator,applybqsr}` | one known-sites resource, not three |
| `SAMTOOLS_STATS` | `nf-core/modules/samtools/stats` | unchanged in shape |
| `MAKE_INTERVALS` | `nf-core/modules/gatk4/intervallisttools` | `SplitIntervals` with `INTERVAL_SUBDIVISION` |
| `GATK4_HAPLOTYPECALLER` | `nf-core/modules/gatk4/haplotypecaller` | `-ERC GVCF`, scattered by interval |
| `GATK4_GENOMICSDBIMPORT`, `GATK4_GENOTYPEGVCFS` | `nf-core/modules/gatk4/*` | unchanged in shape |
| `GATK4_MERGEVCFS` | `nf-core/modules/gatk4/mergevcfs` | sorted input list; see below |
| `GATK4_VARIANTFILTRATION` | `nf-core/modules/gatk4/variantfiltration` | hard filters, SNP thresholds on every record; see below |
| `RTG_FORMAT`, `RTG_VCFEVAL` | `nf-core/modules/rtgtools/*` | `--evaluation-regions`, split by variant type |
| `MULTIQC` | `nf-core/modules/multiqc` | unchanged |

Written for this template, with no upstream counterpart: `SPLIT_SAMPLE`, `SUBSET_VARIANT_TYPE`,
`COLLECT_BENCHMARK`, `SHARD_VCF`, `ANNOTATE_VARIANTS`, `COLLECT_SCORES`, `COLLECT_PLACEMENT` and
`smoke.nf`.

## Divergences

### No `container` directives

Every nf-core module declares `container 'quay.io/biocontainers/...'`, and this pipeline drops
them. A Ray worker runs inside a container and cannot start another, so Nextflow would run the
command against whatever is on `PATH` anyway, and the directive would only mislead. The tools are
baked into the image instead (`tools/env.main.yml`).

This is the largest behavioural difference from upstream, because tool versions come from the image
rather than from the pipeline. `-profile conda` is the alternative: Nextflow builds each process's
declared environment, cached on shared storage so that each is built once per cluster.

### Hard filters, with SNP thresholds on every record

sarek's joint-germline mode filters with VQSR. GATK recommends VQSR for callsets of at least one
whole genome or about 30 exomes; this pipeline's callsets cover one region of one chromosome, so it
hard-filters, which is GATK's documented alternative for small callsets.

`GATK4_VARIANTFILTRATION` applies one set of thresholds to every record: QD < 2, QUAL < 30, SOR > 3,
FS > 60, MQ < 40, MQRankSum < -12.5 and ReadPosRankSum < -8. Those are GATK's recommendations for
SNPs. For indels GATK recommends looser ones, FS > 200 and ReadPosRankSum < -20 with no MQ or
MQRankSum filter, so every indel GATK's indel recipe would remove is removed here too, and indel
recall can only be lower than that recipe gives. Selecting SNPs and indels and filtering each with
its own thresholds would restore GATK's recipe.

Hard filters are also less discriminating than VQSR, which is part of why these numbers are not
comparable to a published genome-wide sarek benchmark.

### One known-sites resource

sarek gives BQSR dbSNP, Mills and a known-indels set. This pipeline gives it dbSNP 138 alone, subset
to chr20, to keep the staged data small, so the recalibration is slightly less well informed.

### `MergeVcfs` input is sorted

`path vcfs` arrives in the order the intervals finished. The list is sorted with `LC_ALL=C sort` so
that `MergeVcfs` sees its inputs in the same order on every run. `COLLECT_SCORES` sorts its shards
for the same reason.

### `MAKE_INTERVALS` fails when the scatter collapses

On a single-contig reference, `SplitIntervals` with the wrong subdivision mode produces one
interval, and the pipeline would still succeed, serially. `MAKE_INTERVALS` checks that it got at
least two intervals and fails otherwise.

### Benchmarking is split by variant type

vcfeval runs separately for SNPs and indels, because their error profiles differ and a combined F1
hides which one moved.

Both sides are split: the calls in `SUBSET_VARIANT_TYPE` and the truth set in `RTG_VCFEVAL`, each
after splitting multi-allelic records. With only the calls split, every truth indel would count as
a missed SNP and every truth SNP as a missed indel. Splitting before matching rather than
stratifying after it is an approximation at complex sites, where one representation is an MNP and
the other a SNP plus an indel; hap.py's stratified counts would be exact there.

`--evaluation-regions` rather than `--bed-regions`: the former restricts scoring to the benchmark
regions while still letting calls just outside them take part in haplotype matching. The latter
truncates haplotypes and invents mismatches at the boundaries.

vcfeval requires genotypes to match, its default, and the reported row is the unthresholded `None`
row: a threshold chosen to maximise F-measure against the truth set being scored is fitted to the
answer.

## Bounds on the numbers

- Reference-selected reads. The demo FASTQs are read pairs whose primary alignment in GIAB's
  whole-genome GRCh38 BAM falls in the region, realigned here against chr20 alone. Reads from the
  rest of the genome that a whole-genome run would misplace into chr20 are mostly absent, so
  precision reads higher than it would in a whole-genome run.
- Scored on an intersection: the region and each sample's GIAB v4.2.1 benchmark regions, from the
  `_noinconsistent` BED, which also excludes regions around Mendelian inconsistencies in GIAB's
  trio benchmark. Outside it the truth set makes no claim.
- One region of one chromosome, hard filters with SNP thresholds on indels, and one known-sites
  resource. Not comparable to a published genome-wide sarek benchmark.
- No pedigree. The three samples are joint-genotyped without a pedigree file, and nothing checks
  Mendelian consistency.
- The GPU annotation is not a variant-effect score. It is the L2 distance between a DNA language
  model's mean-pooled embeddings of the 1 kb reference window with and without the alternate
  allele: zero-shot, uncalibrated, and not validated against pathogenicity, function or call
  quality. The model reads 6-mer tokens, so an indel whose length is not a multiple of six scores
  well above SNPs from the re-tokenization alone.

## The DAG

```
samplesheet ──> FASTP ──> BWAMEM2_MEM ──> MARKDUPLICATES ──> BQSR ──> analysis-ready BAM
                                                                          │
                          ┌───────────────────────────────────────────────┘
                          │  x N intervals
                          v
                 HAPLOTYPECALLER (3 x N tasks)
                          │  group by interval
                          v
                 GENOMICSDBIMPORT ──> GENOTYPEGVCFS
                          │  gather
                          v
                 MERGEVCFS ──> VARIANTFILTRATION
                                      │
                          ┌───────────┴───────────┐
                          │                       │  split per sample, per type
                          v                       v
                      SHARD_VCF              RTG_VCFEVAL  (samples x types)
                          │                       │
                          v                       v
                  ANNOTATE_VARIANTS        COLLECT_BENCHMARK ──> benchmark.tsv
                   (GPU, x shards)
                          │
                          v
                   COLLECT_SCORES
```

At `standard`, 3 samples and 24 intervals give 72 independent HaplotypeCaller tasks, 24
GenomicsDBImport and 24 GenotypeGVCFs tasks, 6 vcfeval comparisons and 8 GPU shards.
