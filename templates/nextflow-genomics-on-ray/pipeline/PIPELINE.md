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
| `GATK4_MERGEVCFS` | `nf-core/modules/gatk4/mergevcfs` | used twice: to gather the joint callset and to rejoin its filtered halves |
| `GATK4_SELECTVARIANTS` | `nf-core/modules/gatk4/selectvariants` | SNPs, and everything else; see below |
| `GATK4_VARIANTFILTRATION` | `nf-core/modules/gatk4/variantfiltration` | GATK's hard filters for the record's type; see below |
| `GATK4_GENOTYPEGVCFS_SAMPLE` | `nf-core/modules/gatk4/{mergevcfs,genotypegvcfs}` | one sample's GVCFs, gathered and genotyped alone, for the CNN arm |
| `NVSCOREVARIANTS` | `nf-core/modules/gatk4/cnnscorevariants` | GATK 4.6's PyTorch port of CNNScoreVariants, 1D model, on an L4 |
| `GATK4_FILTERVARIANTTRANCHES` | `nf-core/modules/gatk4/filtervarianttranches` | `--info-key CNN_1D`, GATK's default tranches; see below |
| `RTG_FORMAT`, `RTG_VCFEVAL` | `nf-core/modules/rtgtools/*` | `--evaluation-regions`, split by variant type; see below |
| `MULTIQC` | `nf-core/modules/multiqc` | unchanged |

Written for this template, with no upstream counterpart: `SPLIT_SAMPLE`, `SUBSET_VARIANT_TYPE`,
`COLLECT_BENCHMARK`, `COLLECT_PLACEMENT` and `smoke.nf`.

## Divergences

### No `container` directives

Every nf-core module declares `container 'quay.io/biocontainers/...'`, and this pipeline drops
them: a Ray worker has no runtime to start one (the README's "Containers" section has the
alternatives). The tools are baked into the image instead, pinned in `tools/env.main.yml`, so tool
versions come from the image rather than from the pipeline. This is the largest behavioural
difference from upstream.

### Hard filters instead of VQSR, per variant type

sarek filters a joint callset with VQSR, which fits a Gaussian mixture to the callset's own
annotations and needs the variant count of a whole genome, or about 30 exomes, to do it; a region of
chr20 has a small fraction of that. So the joint callset is hard-filtered, GATK's documented
alternative for small callsets, the way GATK's article
["(How to) Filter variants either with VQSR or by hard-filtering"](https://gatk.broadinstitute.org/hc/en-us/articles/360035531112--How-to-Filter-variants-either-with-VQSR-or-by-hard-filtering)
(section 2) lays it out: `SelectVariants` splits the callset, `VariantFiltration` filters each half
with its type's thresholds, and `MergeVcfs` rejoins them.

| Type | Filtered when |
|---|---|
| SNP | QD < 2, QUAL < 30, SOR > 3, FS > 60, MQ < 40, MQRankSum < -12.5, ReadPosRankSum < -8 |
| everything else | QD < 2, QUAL < 30, FS > 200, ReadPosRankSum < -20 |

The article lists no SOR, MQ or MQRankSum filter for indels. "Everything else" is indels plus mixed
records, a SNP and an indel allele at one site: `-select-type INDEL` drops those, and the article
suggests filtering them with the indel thresholds, as VQSR's indel mode does.
`--select-type-to-exclude SNP` does that and leaves no record type out of the merge.

### The CNN arm: single-sample calls, NVScoreVariants 1D, tranche-filtered

With `--cnn`, each sample's GVCFs are also genotyped alone and filtered the way sarek filters a
single-sample HaplotypeCaller run: GATK's CNN scores every call, and `FilterVariantTranches` cuts
at a score chosen from known sites. GATK 4.6.1 replaced CNNScoreVariants, which sarek calls, with
NVScoreVariants, NVIDIA's PyTorch port of the same models.

- **Single-sample calls, not the joint callset.** GATK's CNNScoreVariants documentation says the
  models were trained on single-sample VCFs and should not be used on annotations from a joint
  callset. The seven annotations the 1D model reads (MQ, DP, SOR, FS, QD, MQRankSum,
  ReadPosRankSum) are site-level, computed over every sample, and DP in a three-sample callset is
  about three times one sample's. The price is that `gatk_hard` and `gatk_cnn` differ in calling
  mode as well as filter: joint genotyping emits sites a single-sample run leaves below QUAL 30.
- **The 1D model.** It reads the 128 bp of reference around the variant and those annotations.
  The 2D model also reads the pileup, but GATK trains and runs it on HaplotypeCaller's realigned
  reads (`-bamout`, in gatk-workflows/gatk4-cnn-variant-filter), which this pipeline does not
  write, and NVScoreVariants builds its read tensors one variant at a time in single-threaded
  Python, where the GPU waits on them. The 1D model is also sarek's (`--info-key CNN_1D`).
- **Tranches.** GATK's defaults, 99.95 for SNPs and 99.4 for indels, which it tuned for F1 on
  whole human genomes, against the resources GATK's CNN workflow uses: HapMap 3.3 and 1000G phase
  1 high-confidence SNPs, and Mills and 1000G gold-standard indels (sarek uses Mills, GATK's
  known-indels set and 1000G Omni 2.5). The resources are chr20 slices of the Broad hg38 bundle;
  `tools/stage-demo-data.sh` pins each file's MD5.
- **A small region sets the cutoffs from few sites.** A cutoff is the CNN score of one resource
  site: sorted best first, the one at position (n - 1) x tranche / 100, rounded down, where n is
  the resource sites in this sample's calls, and everything scoring at or below it is filtered. At
  `quick` n is a few hundred indels per sample, so one site is several tenths of a percent of
  sensitivity, and filtering at or below the cutoff site leaves the realized sensitivity to the
  resource indels below the nominal 99.4%. Mixed records are neither SNPs nor indels to
  `FilterVariantTranches` and always pass.

The model runs in the image's Python 3.12 with torch 2.9.1, where GATK's own environment pins
Python 3.10 and torch 2.1; `requirements.txt` has the pins, and `tools/smoke-nvscorevariants.sh`,
which the image build runs, checks them. Two things differ from a stock GATK install:
`nvscorevariants.py` loads the model, a pickled `nn.Module` inside the GATK jar, with a bare
`torch.load()`, which torch 2.6 and later refuse without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; and
NVScoreVariants exits 0 when its Python fails, so `NVSCOREVARIANTS` checks that every record came
back scored.

## Bounds on the numbers

The README's "Measured runs" section lists what biases the scores: reference-selected reads, hard
filters in place of VQSR, the CNN arm's single-sample calls and region-sized tranches, one
known-sites resource for BQSR, and one region. How they are computed, for both callsets:

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

## The DAG

```
samplesheet ──> FASTP ──> BWAMEM2_MEM ──> MARKDUPLICATES ──> BQSR ──> analysis-ready BAM
                                                                          │
                          ┌───────────────────────────────────────────────┘
                          │  x N intervals
                          v
                 HAPLOTYPECALLER (3 x N tasks, GVCF)
                          │
            ┌─────────────┴────────────────────┐
            │ group by interval                │ group by sample (--cnn)
            v                                  v
   GENOMICSDBIMPORT ──> GENOTYPEGVCFS    GENOTYPEGVCFS_SAMPLE (x 3)
            │ gather                           │
            v                                  v
        MERGEVCFS                        NVSCOREVARIANTS (x 3, GPU)
            │ SNPs | the rest                  │
            v                                  v
   SELECTVARIANTS ──> VARIANTFILTRATION  FILTERVARIANTTRANCHES (x 3)
            │ x 2                              │
            v                                  │
        MERGEVCFS                              │
            │ gatk_hard                        │ gatk_cnn
            └────────────────┬─────────────────┘
                             │  split per sample, per type
                             v
                  RTG_VCFEVAL ──> COLLECT_BENCHMARK ──> benchmark.tsv
```

At `standard`, 3 samples and 24 intervals give 72 independent HaplotypeCaller tasks, 24
GenomicsDBImport and 24 GenotypeGVCFs tasks, and 6 vcfeval comparisons per callset. The CNN arm
adds three tasks of each of its steps, one per sample, the three on the GPU included.
