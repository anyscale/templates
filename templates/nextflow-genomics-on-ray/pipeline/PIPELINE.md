# Pipeline provenance and divergences

The pipeline is GATK germline short-variant discovery in the shape of the joint-germline mode of
[nf-core/sarek](https://nf-co.re/sarek), with processes adapted from
[nf-core/modules](https://github.com/nf-core/modules) (MIT). This page records where each process
came from and where the pipeline differs, so that a number that disagrees with sarek's can be
traced to a cause.

## Upstream

| Process | Adapted from | Notes |
|---|---|---|
| `FASTP` | `nf-core/modules/fastp` | `--detect_adapter_for_pe`; no UMI handling |
| `BWAMEM2_INDEX`, `BWAMEM2_MEM` | `nf-core/modules/bwamem2/{index,mem}` | read group built inline |
| `GATK4_MARKDUPLICATES` | `nf-core/modules/gatk4/markduplicates` | indexes with samtools rather than `--CREATE_INDEX` |
| `GATK4_BASERECALIBRATOR`, `GATK4_APPLYBQSR` | `nf-core/modules/gatk4/{baserecalibrator,applybqsr}` | one known-sites resource, not three |
| `SAMTOOLS_STATS` | `nf-core/modules/samtools/stats` | unchanged in shape |
| `MAKE_INTERVALS` | `nf-core/modules/gatk4/intervallisttools` | `SplitIntervals` with `INTERVAL_SUBDIVISION`; fails on fewer than two intervals, where a collapsed scatter would otherwise run serially and succeed |
| `GATK4_HAPLOTYPECALLER` | `nf-core/modules/gatk4/haplotypecaller` | `-ERC GVCF`, scattered by interval |
| `GATK4_GENOMICSDBIMPORT`, `GATK4_GENOTYPEGVCFS` | `nf-core/modules/gatk4/*` | unchanged in shape |
| `GATK4_MERGEVCFS` | `nf-core/modules/gatk4/mergevcfs` | input list sorted (`LC_ALL=C sort`), so every run merges in the same order |
| `GATK4_VARIANTFILTRATION` | `nf-core/modules/gatk4/variantfiltration` | hard filters, SNP thresholds on every record; see below |
| `RTG_FORMAT`, `RTG_VCFEVAL` | `nf-core/modules/rtgtools/*` | `--evaluation-regions`, split by variant type; see below |
| `MULTIQC` | `nf-core/modules/multiqc` | unchanged |

Written for this template, with no upstream counterpart: `SPLIT_SAMPLE`, `SUBSET_VARIANT_TYPE`,
`COLLECT_BENCHMARK`, `SHARD_VCF`, `ANNOTATE_VARIANTS`, `COLLECT_SCORES`, `COLLECT_PLACEMENT` and
`smoke.nf`.

## Divergences

### No `container` directives

Every nf-core module declares `container 'quay.io/biocontainers/...'`, and this pipeline drops
them: a Ray worker has no runtime to start one (the README's "Containers" section has the
alternatives). The tools are baked into the image instead, pinned in `tools/env.main.yml`, so tool
versions come from the image rather than from the pipeline. This is the largest behavioural
difference from upstream.

### Hard filters, with SNP thresholds on every record

In place of sarek's VQSR, `GATK4_VARIANTFILTRATION` hard-filters, GATK's documented alternative
for small callsets, with one set of thresholds on every record: QD < 2, QUAL < 30, SOR > 3,
FS > 60, MQ < 40, MQRankSum < -12.5 and ReadPosRankSum < -8. Those are GATK's recommendations for
SNPs. For indels GATK recommends QD < 2, QUAL < 30, FS > 200 and ReadPosRankSum < -20, with no MQ
or MQRankSum filter, so every indel that recipe removes is removed here too. Selecting SNPs and
indels and filtering each with its own thresholds would restore GATK's recipe.

## Bounds on the numbers

The README's "Measured runs" section lists what biases the scores: reference-selected reads, the
SNP thresholds on indels, one known-sites resource and one region. How they are computed:

- vcfeval runs separately for SNPs and indels, because a combined F1 hides which one moved. Both
  sides are split, the calls in `SUBSET_VARIANT_TYPE` and the truth set in `RTG_VCFEVAL`, each
  after splitting multi-allelic records; with only the calls split, every truth indel would count
  as a missed SNP. Splitting before matching rather than stratifying after it is an approximation
  at complex sites, where one representation is an MNP and the other a SNP plus an indel; hap.py's
  stratified counts would be exact there.
- `--evaluation-regions` rather than `--bed-regions`: the former restricts scoring to each sample's
  GIAB v4.2.1 benchmark regions within the region while still letting calls just outside them take
  part in haplotype matching. The latter truncates haplotypes and invents mismatches at the
  boundaries. Outside the benchmark regions the truth set makes no claim.
- vcfeval requires genotypes to match, its default, and the reported row is the unthresholded
  `None` row: a threshold chosen to maximise F-measure against the truth set being scored is
  fitted to the answer.

The GPU score has no benchmark. A local check on CPU, with the pinned model revision at 40
positions in chr20:1.00-1.03 Mb, gave median scores of 0.133 for SNPs (maximum 0.263), 0.921 and
0.927 for 1 bp insertions and deletions, 1.450 for 2 bp insertions, 0.904 for 3 bp deletions, and
0.199 and 0.168 for 6 bp insertions and deletions. Every 1 bp indel scored above the highest SNP;
indels whose length is a multiple of six scored like SNPs.

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
