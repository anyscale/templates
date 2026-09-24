#!/usr/bin/env nextflow

/*
 * GIAB Ashkenazi trio germline short-variant calling, benchmarked against the
 * GIAB v4.2.1 truth set.
 *
 *   nextflow run pipeline/main.nf -profile ray
 *
 * The pipeline is deliberately ordinary -- GATK Best Practices as nf-core/sarek
 * implements it, rtg vcfeval for scoring, and DeepVariant as an optional second
 * caller (off by default; see PIPELINE.md for why).
 * Nothing in this file, in modules/, or in conf/base.config knows it is running
 * on Ray. That is the claim: `-profile ray` is the diff.
 *
 * The shape that matters is the scatter. Three samples over N intervals is
 * 3 x N independent calling tasks, converging per interval for joint genotyping
 * and then once more for the callset. A single-sample pipeline is a chain, and a
 * chain gives an autoscaler nothing to do -- which is why the demonstration is a
 * cohort.
 */

nextflow.enable.dsl = 2

include { BWAMEM2_INDEX; FASTP; BWAMEM2_MEM; GATK4_MARKDUPLICATES;
          GATK4_BASERECALIBRATOR; GATK4_APPLYBQSR; SAMTOOLS_STATS } from './modules/local/preprocessing.nf'
include { MAKE_INTERVALS; GATK4_HAPLOTYPECALLER; GATK4_GENOMICSDBIMPORT;
          GATK4_GENOTYPEGVCFS; GATK4_MERGEVCFS; GATK4_VARIANTFILTRATION } from './modules/local/calling_gatk.nf'
include { DEEPVARIANT } from './modules/local/calling_deepvariant.nf'
include { RTG_FORMAT; SPLIT_SAMPLE; SUBSET_VARIANT_TYPE; RTG_VCFEVAL;
          COLLECT_BENCHMARK } from './modules/local/benchmark.nf'
include { SHARD_VCF; ANNOTATE_VARIANTS; COLLECT_SCORES } from './modules/local/annotate.nf'
include { MULTIQC; COLLECT_PLACEMENT } from './modules/local/reporting.nf'

/*
 * Scale presets. Same tools, same DAG, same resource requests -- only the number
 * of bases differs, so a green CI run at `quick` exercises the code path a reader
 * gets at `standard`. The regions match what tools/stage-demo-data.sh published;
 * tests/ asserts the two agree, because a silent mismatch would have the pipeline
 * scoring a region the data does not cover.
 *
 * A function, not a top-level `def`: the strict parser does not allow statements
 * to be mixed with script declarations, so that a script included as a module
 * cannot execute anything at import time.
 */
def scales() {
    return [
        quick   : [ region: 'chr20:1000000-3000000',  intervals: 8,  annotate_shards: 4,
                    note: 'chr20 2 Mbp, all three samples -- what CI runs' ],
        standard: [ region: 'chr20:1000000-11000000', intervals: 24, annotate_shards: 8,
                    note: 'chr20 10 Mbp, all three samples -- the notebook default' ],
        full    : [ region: 'chr20',                  intervals: 48, annotate_shards: 16,
                    note: 'the whole of chr20, all three samples -- what job.yaml runs' ],
    ]
}

workflow {

    def SCALES = scales()

    // -- inputs ---------------------------------------------------------------

    if( !SCALES.containsKey(params.scale) )
        error "Unknown --scale '${params.scale}'. One of: ${SCALES.keySet().join(', ')}"

    def preset  = SCALES[params.scale]
    def region  = params.region ?: preset.region

    // `as Integer`, always. A param supplied on the command line arrives as a
    // String, and Groovy's arithmetic on Strings is silently something else
    // entirely -- `"4" - 1` is string subtraction yielding "4", and a Range built
    // from it compares by character code. The smoke pipeline turned `--shards 4`
    // into 53 shards this way, and reported success.
    def n_intervals = (params.intervals ?: preset.intervals) as Integer
    def n_shards    = (params.annotate_shards ?: preset.annotate_shards) as Integer
    if( n_intervals < 2 )
        error "--intervals must be >= 2 (got ${n_intervals}); the scatter is the point"

    log.info """
    ${workflow.manifest.name} ${workflow.manifest.version}
      scale        ${params.scale}  (${preset.note})
      region       ${region}
      intervals    ${n_intervals}   -> ${n_intervals} x samples calling tasks
      callers      GATK HaplotypeCaller${params.deepvariant ? ' + DeepVariant' : ''}
      annotate     ${params.annotate ? "yes (${n_shards} GPU shards)" : 'no'}
      workDir      ${workflow.workDir}
      outdir       ${params.outdir}
    """.stripIndent()

    // Checked here, on the head node, because every node runs the same image: a
    // missing DeepVariant is missing everywhere, and finding out at the first
    // DEEPVARIANT task means finding out after alignment and BQSR.
    if( params.deepvariant ) {
        def dv_prefix = System.getenv('NF_RAY_DEEPVARIANT_PREFIX') ?: '/opt/nf-tools/envs/deepvariant'
        if( !file("${dv_prefix}/bin/run_deepvariant").exists() )
            error "--deepvariant needs ${dv_prefix}/bin/run_deepvariant, which this image does not have. " +
                  "bioconda's deepvariant 1.10.0 ships no binaries; see pipeline/PIPELINE.md."
    }

    if( !params.samplesheet ) error "--samplesheet is required"
    if( !params.reference )   error "--reference is required"
    if( !params.outdir )      error "--outdir is required"

    ch_samples = channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true)
        .map { row ->
            if( !row.sample || !row.fastq_1 || !row.fastq_2 )
                error "samplesheet needs columns: sample,fastq_1,fastq_2 (got ${row.keySet()})"
            tuple([id: row.sample], [file(row.fastq_1, checkIfExists: true),
                                     file(row.fastq_2, checkIfExists: true)])
        }

    // The reference travels as one tuple everywhere. GATK needs all three files
    // present next to each other and fails late and unhelpfully when the .dict is
    // missing, so they are never separated.
    ch_reference = channel.value(tuple(
        file(params.reference,                            checkIfExists: true),
        file("${params.reference}.fai",                   checkIfExists: true),
        file(params.reference.replaceAll(/\.(fa|fasta|fna)$/, '.dict'), checkIfExists: true),
    ))

    ch_known_sites = channel.value(tuple(
        file(params.known_sites,          checkIfExists: true),
        file("${params.known_sites}.tbi", checkIfExists: true),
    ))

    ch_truth = channel.value(tuple(
        file(params.truth_vcf,          checkIfExists: true),
        file("${params.truth_vcf}.tbi", checkIfExists: true),
        file(params.truth_bed,          checkIfExists: true),
    ))

    // -- preprocessing: FASTQ -> analysis-ready BAM ---------------------------

    BWAMEM2_INDEX(ch_reference.map { fasta, _fai, _dict -> fasta })
    FASTP(ch_samples)
    BWAMEM2_MEM(FASTP.out.reads, BWAMEM2_INDEX.out.index)
    GATK4_MARKDUPLICATES(BWAMEM2_MEM.out.bam)
    GATK4_BASERECALIBRATOR(GATK4_MARKDUPLICATES.out.bam, ch_reference, ch_known_sites)

    ch_bqsr_input = GATK4_MARKDUPLICATES.out.bam
        .join(GATK4_BASERECALIBRATOR.out.table)
    GATK4_APPLYBQSR(ch_bqsr_input, ch_reference)

    ch_bam = GATK4_APPLYBQSR.out.bam
    SAMTOOLS_STATS(ch_bam)

    // -- the scatter ----------------------------------------------------------

    MAKE_INTERVALS(ch_reference, n_intervals, region)
    ch_intervals = MAKE_INTERVALS.out.intervals.flatten()

    // The cross product IS the fan-out: every sample against every interval, all
    // independent, all submitted at once. This is the line the template's claim
    // rests on -- 3 samples x 24 intervals = 72 concurrent HaplotypeCaller tasks
    // with nothing serialising them but the cluster's size.
    ch_calling_units = ch_bam.combine(ch_intervals)
    GATK4_HAPLOTYPECALLER(ch_calling_units, ch_reference)

    // Regroup by interval: joint genotyping needs every sample's GVCF for one
    // interval together, which is the one place the fan-out has to converge
    // before it can fan out again.
    ch_by_interval = GATK4_HAPLOTYPECALLER.out.gvcf
        .map { interval_id, _meta, gvcf, tbi -> tuple(interval_id, gvcf, tbi) }
        .groupTuple()
        .join(ch_intervals.map { interval -> tuple(interval.baseName, interval) })

    GATK4_GENOMICSDBIMPORT(ch_by_interval)
    GATK4_GENOTYPEGVCFS(GATK4_GENOMICSDBIMPORT.out.db, ch_reference)

    GATK4_MERGEVCFS(
        GATK4_GENOTYPEGVCFS.out.vcf.map { vcf, _tbi -> vcf }.collect(),
        GATK4_GENOTYPEGVCFS.out.vcf.map { _vcf, tbi -> tbi }.collect(),
    )
    GATK4_VARIANTFILTRATION(GATK4_MERGEVCFS.out.vcf, ch_reference)

    // -- second caller --------------------------------------------------------

    ch_dv_calls = channel.empty()
    if( params.deepvariant ) {
        DEEPVARIANT(ch_bam, ch_reference, region)
        ch_dv_calls = DEEPVARIANT.out.vcf.map { meta, vcf, tbi ->
            tuple(meta, 'deepvariant', vcf, tbi)
        }
    }

    // -- benchmarking ---------------------------------------------------------

    // The joint callset carries every sample, so it is fanned back out per sample
    // to be scored; DeepVariant already produced one VCF each.
    ch_gatk_calls = ch_bam
        .map { meta, _bam, _bai -> meta }
        .combine(GATK4_VARIANTFILTRATION.out.vcf)
        .map { meta, vcf, tbi -> tuple(meta, 'gatk', vcf, tbi) }

    ch_to_score = ch_gatk_calls.mix(ch_dv_calls)

    RTG_FORMAT(ch_reference)
    SPLIT_SAMPLE(ch_to_score, ch_reference)

    // samples x callers x {snp, indel}. SNPs and indels fail for different
    // reasons, and a combined F1 would hide which one moved.
    ch_typed = SPLIT_SAMPLE.out.vcf.combine(channel.of('snp', 'indel'))
    SUBSET_VARIANT_TYPE(ch_typed)

    RTG_VCFEVAL(SUBSET_VARIANT_TYPE.out.vcf, RTG_FORMAT.out.sdf, ch_truth, region)
    // The vcfeval directories, not their summary.txt files: see COLLECT_BENCHMARK.
    COLLECT_BENCHMARK(RTG_VCFEVAL.out.dir.collect())

    // -- GPU annotation -------------------------------------------------------

    ch_scores = channel.empty()
    if( params.annotate ) {
        SHARD_VCF(GATK4_VARIANTFILTRATION.out.vcf, n_shards)
        ANNOTATE_VARIANTS(SHARD_VCF.out.shards.flatten(), ch_reference)
        COLLECT_SCORES(ANNOTATE_VARIANTS.out.scores.collect())
        ch_scores = COLLECT_SCORES.out.table
    }

    // -- reporting ------------------------------------------------------------

    ch_reports = FASTP.out.json
        .mix(GATK4_MARKDUPLICATES.out.metrics)
        .mix(SAMTOOLS_STATS.out.stats)
        .collect()
    MULTIQC(ch_reports)

    // Ordering only: the placement record is complete once the work is.
    COLLECT_PLACEMENT(COLLECT_BENCHMARK.out.table.map { _table -> 'done' })

    // Inside the entry workflow, and with an equals sign. The strict parser does
    // not allow statements at script level, so the familiar top-level
    // `workflow.onComplete { ... }` no longer parses.
    workflow.onComplete = {
        log.info """
        ${workflow.success ? 'SUCCEEDED' : 'FAILED'}  duration=${workflow.duration}
          benchmark  ${params.outdir}/benchmark/benchmark.tsv
          placement  ${params.outdir}/pipeline_info/nf_ray_placement.tsv
          multiqc    ${params.outdir}/multiqc/multiqc_report.html
        """.stripIndent()
    }
}
