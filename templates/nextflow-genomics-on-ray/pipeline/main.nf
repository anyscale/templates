#!/usr/bin/env nextflow

/*
 * Germline short-variant calling for the GIAB Ashkenazi trio, each sample scored
 * against its own GIAB v4.2.1 truth set. Nothing here refers to Ray; `-profile ray`
 * adds the executor.
 */

nextflow.enable.dsl = 2

include { BWAMEM2_INDEX; FASTP; BWAMEM2_MEM; GATK4_MARKDUPLICATES;
          GATK4_BASERECALIBRATOR; GATK4_APPLYBQSR; SAMTOOLS_STATS } from './modules/local/preprocessing.nf'
include { MAKE_INTERVALS; GATK4_HAPLOTYPECALLER; GATK4_GENOMICSDBIMPORT; GATK4_GENOTYPEGVCFS;
          GATK4_MERGEVCFS; GATK4_MERGEVCFS as GATK4_MERGEVCFS_FILTERED;
          GATK4_SELECTVARIANTS; GATK4_VARIANTFILTRATION } from './modules/local/calling_gatk.nf'
include { CNN_FILTER } from './modules/local/cnn_filter.nf'
include { BENCHMARK } from './modules/local/benchmark.nf'
include { MULTIQC; COLLECT_PLACEMENT } from './modules/local/reporting.nf'

// Regions must match tools/stage-demo-data.sh. A function because the strict
// parser forbids mixing top-level statements with declarations.
def scales() {
    return [
        quick   : [ region: 'chr20:1000000-3000000',  intervals: 8,
                    note: 'chr20 2 Mbp, all three samples -- what CI runs' ],
        standard: [ region: 'chr20:1000000-11000000', intervals: 24,
                    note: 'chr20 10 Mbp, all three samples -- the notebook default' ],
        full    : [ region: 'chr20',                  intervals: 48,
                    note: 'the whole of chr20, all three samples -- what job.yaml runs' ],
    ]
}

// CLI params arrive as Strings, and a non-empty String is truthy: `--cnn false`.
def flag(value) {
    return value instanceof Boolean ? value : value.toString().trim().toLowerCase() in ['true', 'yes', '1']
}

workflow {

    def SCALES = scales()

    if( !SCALES.containsKey(params.scale) )
        error "Unknown --scale '${params.scale}'. One of: ${SCALES.keySet().join(', ')}"

    def preset  = SCALES[params.scale]
    def region  = params.region ?: preset.region

    // A CLI String would make `"4" - 1` string subtraction.
    def n_intervals = (params.intervals ?: preset.intervals) as Integer
    if( n_intervals < 2 )
        error "--intervals must be >= 2 (got ${n_intervals}); the scatter is the point"
    def cnn = flag(params.cnn)

    def resource_paths = (params.tranche_resources ?: '').toString().tokenize(',')
        .collect { p -> p.trim() }
        .findAll { p -> p }

    log.info """
    ${workflow.manifest.name} ${workflow.manifest.version}
      scale        ${params.scale}  (${preset.note})
      region       ${region}
      intervals    ${n_intervals}   -> ${n_intervals} x samples calling tasks
      joint        GATK HaplotypeCaller, GenotypeGVCFs, hard filters per type
      cnn          ${cnn ? "yes: single-sample calls, NVScoreVariants 1D on a GPU, ${resource_paths.size()} tranche resources" : 'no'}
      workDir      ${workflow.workDir}
      outdir       ${params.outdir}
    """.stripIndent()

    if( !params.samplesheet ) error "--samplesheet is required"
    if( !params.reference )   error "--reference is required"
    if( !params.outdir )      error "--outdir is required"
    if( cnn && !resource_paths )
        error "--tranche_resources is required with --cnn (comma-separated VCFs); --cnn false skips the CNN arm"

    ch_rows = channel
        .fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true)

    ch_samples = ch_rows.map { row ->
        if( !row.sample || !row.fastq_1 || !row.fastq_2 )
            error "samplesheet needs columns: sample,fastq_1,fastq_2 (got ${row.keySet()})"
        tuple([id: row.sample], [file(row.fastq_1, checkIfExists: true),
                                 file(row.fastq_2, checkIfExists: true)])
    }

    // Per-sample truth: scoring HG002 against HG003's truth would count the
    // difference between two people as caller error.
    ch_truth = ch_rows
        .filter { row -> row.truth_vcf && row.truth_bed }
        .map { row ->
            tuple([id: row.sample],
                  file(row.truth_vcf,          checkIfExists: true),
                  file("${row.truth_vcf}.tbi", checkIfExists: true),
                  file(row.truth_bed,          checkIfExists: true))
        }

    // Kept together: GATK fails late and unhelpfully when the .dict is missing.
    ch_reference = channel.value(tuple(
        file(params.reference,                            checkIfExists: true),
        file("${params.reference}.fai",                   checkIfExists: true),
        file(params.reference.replaceAll(/\.(fa|fasta|fna)$/, '.dict'), checkIfExists: true),
    ))

    ch_known_sites = channel.value(tuple(
        file(params.known_sites,          checkIfExists: true),
        file("${params.known_sites}.tbi", checkIfExists: true),
    ))

    ch_resources     = channel.value(resource_paths.collect { p -> file(p, checkIfExists: true) })
    ch_resource_tbis = channel.value(resource_paths.collect { p -> file("${p}.tbi", checkIfExists: true) })

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

    MAKE_INTERVALS(ch_reference, n_intervals, region)
    ch_intervals = MAKE_INTERVALS.out.intervals.flatten()

    ch_calling_units = ch_bam.combine(ch_intervals)
    GATK4_HAPLOTYPECALLER(ch_calling_units, ch_reference)

    ch_by_interval = GATK4_HAPLOTYPECALLER.out.gvcf
        .map { interval_id, _meta, gvcf, tbi -> tuple(interval_id, gvcf, tbi) }
        .groupTuple()
        .join(ch_intervals.map { interval -> tuple(interval.baseName, interval) })

    GATK4_GENOMICSDBIMPORT(ch_by_interval)
    GATK4_GENOTYPEGVCFS(GATK4_GENOMICSDBIMPORT.out.db, ch_reference)

    // Sorted, so the -resume hash does not depend on which interval finished first.
    GATK4_MERGEVCFS(
        GATK4_GENOTYPEGVCFS.out.vcf.map { vcf, _tbi -> vcf }.collect(sort: true),
        GATK4_GENOTYPEGVCFS.out.vcf.map { _vcf, tbi -> tbi }.collect(sort: true),
        'joint',
    )

    GATK4_SELECTVARIANTS(GATK4_MERGEVCFS.out.vcf.combine(channel.of('snp', 'indel')), ch_reference)
    GATK4_VARIANTFILTRATION(GATK4_SELECTVARIANTS.out.vcf, ch_reference)
    GATK4_MERGEVCFS_FILTERED(
        GATK4_VARIANTFILTRATION.out.vcf.map { vcf, _tbi -> vcf }.collect(sort: true),
        GATK4_VARIANTFILTRATION.out.vcf.map { _vcf, tbi -> tbi }.collect(sort: true),
        'joint.filtered',
    )

    // Regrouped per sample: the CNN scores single-sample calls.
    ch_cnn = channel.empty()
    if( cnn ) {
        ch_sample_gvcfs = GATK4_HAPLOTYPECALLER.out.gvcf
            .map { _interval_id, meta, gvcf, tbi -> tuple(meta, gvcf, tbi) }
            .groupTuple(sort: true)
        CNN_FILTER(ch_sample_gvcfs, ch_reference, ch_resources, ch_resource_tbis)
        ch_cnn = CNN_FILTER.out
    }

    ch_hard_calls = ch_bam
        .map { meta, _bam, _bai -> meta }
        .combine(GATK4_MERGEVCFS_FILTERED.out.vcf)
        .map { meta, vcf, tbi -> tuple(meta, 'gatk_hard', vcf, tbi) }
    ch_cnn_calls = ch_cnn.map { meta, vcf, tbi -> tuple(meta, 'gatk_cnn', vcf, tbi) }

    BENCHMARK(ch_hard_calls.mix(ch_cnn_calls), ch_reference, ch_truth, region)

    ch_reports = FASTP.out.json
        .mix(GATK4_MARKDUPLICATES.out.metrics)
        .mix(SAMTOOLS_STATS.out.stats)
        .collect()
    MULTIQC(ch_reports)

    // Waits on every terminal output: BENCHMARK emits nothing when no sample has truth.
    ch_done = GATK4_MERGEVCFS_FILTERED.out.vcf
        .mix(BENCHMARK.out, ch_cnn, MULTIQC.out.report)
        .collect()
        .map { _outputs -> 'done' }
    COLLECT_PLACEMENT(ch_done)

    // Assigned here: the strict parser rejects a top-level `workflow.onComplete { }`.
    workflow.onComplete = {
        log.info """
        ${workflow.success ? 'SUCCEEDED' : 'FAILED'}  duration=${workflow.duration}
          benchmark  ${params.outdir}/benchmark/benchmark.tsv
          placement  ${params.outdir}/pipeline_info/nf_ray_placement.tsv
          multiqc    ${params.outdir}/multiqc/multiqc_report.html
        """.stripIndent()
    }
}
