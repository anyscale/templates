/*
 * Score the callset against each sample's GIAB v4.2.1 truth set.
 *
 * A variant count says the run finished; precision and recall against a truth
 * set say whether the calls are right.
 *
 * `rtg vcfeval` rather than hap.py: it is the comparison engine hap.py itself
 * runs under `--engine=vcfeval`, it is a single JVM tool, and it matches
 * variants by the haplotypes they imply rather than by position, so a
 * left-aligned indel and its right-aligned equivalent are the same call.
 *
 * Scored on the region intersected with GIAB's benchmark regions; outside them
 * the truth set makes no claim. Each sample is scored against its own truth
 * set; see BENCHMARK at the end of this file.
 */

process RTG_FORMAT {
    tag "${fasta.baseName}"
    label 'process_medium'

    input:
    tuple path(fasta), path(fai), path(dict)

    output:
    path "reference.sdf", emit: sdf

    script:
    """
    rtg format --output reference.sdf ${fasta}
    """
}

process SPLIT_SAMPLE {
    tag "${meta.id}:${caller}"
    label 'process_low'

    input:
    tuple val(meta), val(caller), path(vcf), path(tbi)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(meta), val(caller), path("${meta.id}.${caller}.norm.vcf.gz"),
          path("${meta.id}.${caller}.norm.vcf.gz.tbi"), emit: vcf

    script:
    """
    # Three things, in an order that matters.
    #
    # `view -s` pulls this sample out of what may be a joint callset. --min-ac 1
    # drops sites where this sample is hom-ref: they are real records in a joint
    # VCF, and counting them as calls would wreck precision.
    #
    # `norm -m -any` splits multi-allelics, because vcfeval compares alleles and a
    # packed record hides one.
    #
    # `norm -f` left-aligns against the reference. The truth set is left-aligned;
    # comparing a right-aligned indel to it produces a false positive and a false
    # negative for what is one correct call.
    bcftools view -s ${meta.id} --min-ac 1 -Ou ${vcf} \\
      | bcftools norm -m -any -Ou - \\
      | bcftools norm -f ${fasta} -Oz -o ${meta.id}.${caller}.norm.vcf.gz -
    tabix -p vcf ${meta.id}.${caller}.norm.vcf.gz
    """
}

process SUBSET_VARIANT_TYPE {
    tag "${meta.id}:${caller}:${vtype}"
    label 'process_low'

    input:
    tuple val(meta), val(caller), path(vcf), path(tbi), val(vtype)

    output:
    tuple val(meta), val(caller), val(vtype),
          path("${meta.id}.${caller}.${vtype}.vcf.gz"),
          path("${meta.id}.${caller}.${vtype}.vcf.gz.tbi"), emit: vcf

    script:
    // SNPs and indels are reported separately because they fail differently and
    // a combined F1 hides it: SNP F1 is dominated by sequencing error, indel F1
    // by alignment and by the caller's local reassembly. A single number that
    // moved would not say which.
    def selector = vtype == 'snp' ? '-v snps' : '-v indels'
    """
    bcftools view ${selector} -f PASS,. -Oz -o ${meta.id}.${caller}.${vtype}.vcf.gz ${vcf}
    tabix -p vcf ${meta.id}.${caller}.${vtype}.vcf.gz
    """
}

process RTG_VCFEVAL {
    tag "${meta.id}:${caller}:${vtype}"
    label 'process_medium'
    publishDir "${params.outdir}/benchmark", mode: params.publish_mode

    input:
    tuple val(meta), val(caller), val(vtype), path(vcf), path(tbi),
          path(truth_vcf), path(truth_tbi), path(truth_bed)
    path sdf
    val  region

    output:
    path "${meta.id}.${caller}.${vtype}/summary.txt", emit: summary
    path "${meta.id}.${caller}.${vtype}", emit: dir

    script:
    def region_arg = region ? "--region ${region}" : ""
    def selector = vtype == 'snp' ? '-v snps' : '-v indels'
    """
    # The truth set is split the way SUBSET_VARIANT_TYPE split the calls: vcfeval
    # scores one callset against one baseline, and a whole baseline counts every
    # truth indel as a missed SNP and every truth SNP as a missed indel. Measured on
    # the synthetic trio, where every planted variant was called and none was
    # false: SNP recall read 0.70 and indel recall 0.30, the two types' shares of
    # the truth set. Multi-allelic records are split first, as the calls' were.
    bcftools norm -m -any -Ou ${truth_vcf} \\
      | bcftools view ${selector} -Oz -o baseline.${vtype}.vcf.gz
    tabix -p vcf baseline.${vtype}.vcf.gz

    # --evaluation-regions, not --bed-regions: the former restricts *scoring* to
    # the high-confidence set while still letting vcfeval use calls just outside
    # it for haplotype matching. The latter would truncate the haplotypes and
    # invent mismatches at every boundary.
    rtg vcfeval \\
        --baseline baseline.${vtype}.vcf.gz \\
        --calls ${vcf} \\
        --evaluation-regions ${truth_bed} \\
        --template ${sdf} \\
        ${region_arg} \\
        --sample ${meta.id},${meta.id} \\
        --output ${meta.id}.${caller}.${vtype} \\
        --threads ${task.cpus}

    # vcfeval exits 0 having written a summary even when nothing matched, so an
    # empty result is a real possibility and it must not look like success. The
    # unthresholded `None` row is what a scored comparison always has; the earlier
    # check here matched any line starting with a digit, which rtg's "0 total
    # baseline variants" message does (captured: exit 0, that one line).
    if ! grep -qE '^ +None ' ${meta.id}.${caller}.${vtype}/summary.txt; then
        echo "vcfeval produced no scored rows for ${meta.id}/${caller}/${vtype}" >&2
        cat ${meta.id}.${caller}.${vtype}/summary.txt >&2
        exit 1
    fi
    """
}

process COLLECT_BENCHMARK {
    tag "benchmark"
    label 'process_single'
    publishDir "${params.outdir}/benchmark", mode: params.publish_mode

    input:
    path eval_dirs

    output:
    path "benchmark.tsv", emit: table

    script:
    // One tidy table, so the notebook does not have to walk a directory tree and
    // so `benchmark.tsv` is the single artifact a reader can diff between runs.
    //
    // The *directories*, not their summary.txt files. vcfeval's summary.txt says
    // nothing about what was compared -- sample, caller and type live only in the
    // directory name RTG_VCFEVAL chose -- and collecting the files instead staged
    // twelve inputs all named summary.txt into one task, which Nextflow rejects as
    // an input file name collision. A staged directory keeps its name; a staged
    // file's parent is a hash directory.
    //
    // Called bare: Nextflow puts the pipeline's bin/ on every task's PATH. Under
    // `-profile ray` that is the executor's copy on shared storage, since a worker
    // cannot see <projectDir> (RayExecutor.getBinDir).
    """
    collect_vcfeval.py --output benchmark.tsv ${eval_dirs}
    """
}

/*
 * The whole scoring stage, as one unit main.nf calls and a test can call alone.
 *
 * Every comparison is against *that sample's* truth set: the calls and the
 * truth sets meet on meta.id, one truth set per sample, with `combine(by: 0)`
 * rather than `join` because each sample has several callsets to score (one per
 * variant type) and one truth set to score them all against.
 *
 * `caller` is a label, carried into the output directory names and so into
 * benchmark.tsv. main.nf passes 'gatk_hard' for the hard-filtered joint callset
 * and 'gatk_cnn' for the CNN-filtered single-sample ones; another callset, mixed
 * into ch_calls under its own label, would be scored the same way.
 */
workflow BENCHMARK {
    take:
    ch_calls       // tuple(meta, caller, vcf, tbi); a joint VCF is fine, SPLIT_SAMPLE subsets it
    ch_reference   // value channel: tuple(fasta, fai, dict)
    ch_truth       // tuple(meta, truth_vcf, truth_tbi, truth_bed), one per sample
    region         // val: the region scored, or null for all of it

    main:
    RTG_FORMAT(ch_reference)
    SPLIT_SAMPLE(ch_calls, ch_reference)

    // samples x {snp, indel}. SNPs and indels have different error profiles,
    // and a combined F1 would hide which one moved.
    ch_typed = SPLIT_SAMPLE.out.vcf.combine(channel.of('snp', 'indel'))
    SUBSET_VARIANT_TYPE(ch_typed)

    ch_against_truth = SUBSET_VARIANT_TYPE.out.vcf
        .map { meta, caller, vtype, vcf, tbi -> tuple(meta.id, meta, caller, vtype, vcf, tbi) }
        .combine(ch_truth.map { meta, tvcf, ttbi, tbed -> tuple(meta.id, tvcf, ttbi, tbed) }, by: 0)
        .map { _id, meta, caller, vtype, vcf, tbi, tvcf, ttbi, tbed ->
            tuple(meta, caller, vtype, vcf, tbi, tvcf, ttbi, tbed)
        }

    RTG_VCFEVAL(ch_against_truth, RTG_FORMAT.out.sdf, region)
    COLLECT_BENCHMARK(RTG_VCFEVAL.out.dir.collect())

    emit:
    COLLECT_BENCHMARK.out.table   // benchmark.tsv
}
