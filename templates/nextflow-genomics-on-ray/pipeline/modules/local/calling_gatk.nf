/*
 * GATK joint germline calling, scattered by interval.
 *
 * Per sample: HaplotypeCaller in GVCF mode, once per interval.
 * Per interval: GenomicsDBImport across all samples, then GenotypeGVCFs.
 * Then gather, and hard-filter SNPs and indels separately, each with GATK's
 * thresholds for its type.
 *
 * The scatter is two-dimensional: three samples over 24 intervals is 72
 * independent calling tasks, which is what the autoscaler has to work with.
 * sarek scatters by interval for the same reason HPC users do: HaplotypeCaller
 * dominates the run's wall-clock time.
 *
 * Adapted from nf-core/modules (MIT). See PIPELINE.md.
 */

process MAKE_INTERVALS {
    tag "${n_intervals} intervals"
    label 'process_single'

    input:
    tuple path(fasta), path(fai), path(dict)
    val  n_intervals
    val  region

    output:
    path "intervals/*.interval_list", emit: intervals

    script:
    // Split by *base count*, not by contig: a single-contig reference (this
    // template slices chr20) would otherwise produce exactly one interval, and
    // the scatter this pipeline exists to demonstrate would silently collapse to
    // a chain. SplitIntervals with BALANCING_WITHOUT_INTERVAL_SUBDIVISION would
    // do the same thing.
    def region_arg = region ? "--intervals ${region}" : ""
    """
    mkdir -p intervals
    gatk SplitIntervals \\
        --reference ${fasta} \\
        ${region_arg} \\
        --scatter-count ${n_intervals} \\
        --subdivision-mode INTERVAL_SUBDIVISION \\
        --output intervals

    # Fail loudly if the scatter did not happen. A silent single interval turns a
    # 72-task fan-out into a 3-task chain, and the run still succeeds -- just
    # hours later and with nothing to show.
    n=\$(ls intervals/*.interval_list | wc -l)
    if [ "\$n" -lt 2 ]; then
        echo "SplitIntervals produced \$n interval(s); expected ${n_intervals}." >&2
        echo "The scatter this pipeline depends on would not happen." >&2
        exit 1
    fi
    echo "split into \$n intervals"
    """
}

process GATK4_HAPLOTYPECALLER {
    tag "${meta.id}:${interval.baseName}"
    label 'process_medium'

    input:
    tuple val(meta), path(bam), path(bai), path(interval)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(interval.baseName), val(meta), path("${meta.id}.${interval.baseName}.g.vcf.gz"),
          path("${meta.id}.${interval.baseName}.g.vcf.gz.tbi"), emit: gvcf

    script:
    // `x as int`, never `(int) x`: Nextflow's strict parser has no C-style
    // casts, reads `(int) (expr)` as a call to `int`, and fails the task with
    // "No signature of method: static int.call()". Every heap below does the same.
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    gatk --java-options "-Xmx${heap}g" HaplotypeCaller \\
        --input ${bam} \\
        --reference ${fasta} \\
        --intervals ${interval} \\
        --emit-ref-confidence GVCF \\
        --native-pair-hmm-threads ${task.cpus} \\
        --output ${meta.id}.${interval.baseName}.g.vcf.gz
    """
}

process GATK4_GENOMICSDBIMPORT {
    tag "${interval_id}"
    label 'process_medium'

    input:
    tuple val(interval_id), path(gvcfs), path(tbis), path(interval)

    output:
    tuple val(interval_id), path("genomicsdb_${interval_id}"), path(interval), emit: db

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    // One line, joined with spaces. The obvious multi-line join is an escaping
    // trap: inside this GString a separator of ' \\\\\n' renders as two
    // backslashes then a newline, bash reads that as a literal backslash and the
    // end of the command, and the second sample's line runs on its own as
    // `--variant: command not found`. It only bites with more than one sample.
    def variants = gvcfs.collect { gvcf -> "--variant ${gvcf}" }.join(' ')
    """
    # --genomicsdb-workspace-path must NOT exist; GenomicsDBImport creates it and
    # refuses to write into a directory Nextflow has already staged.
    gatk --java-options "-Xmx${heap}g" GenomicsDBImport \\
        ${variants} \\
        --intervals ${interval} \\
        --genomicsdb-workspace-path genomicsdb_${interval_id} \\
        --batch-size 50 \\
        --reader-threads ${task.cpus}
    """
}

process GATK4_GENOTYPEGVCFS {
    tag "${interval_id}"
    label 'process_medium'

    input:
    tuple val(interval_id), path(db), path(interval)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple path("joint.${interval_id}.vcf.gz"), path("joint.${interval_id}.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    gatk --java-options "-Xmx${heap}g" GenotypeGVCFs \\
        --reference ${fasta} \\
        --variant gendb://${db} \\
        --intervals ${interval} \\
        --output joint.${interval_id}.vcf.gz
    """
}

/*
 * Used twice by main.nf: to gather the per-interval joint callset (`joint`) and to
 * merge the two hard-filtered halves back into one (`joint.filtered`). Both are
 * published; the unfiltered one is there to filter differently.
 */
process GATK4_MERGEVCFS {
    tag "${prefix}"
    label 'process_medium'
    publishDir "${params.outdir}/variants", mode: params.publish_mode

    input:
    path vcfs
    path tbis
    val  prefix

    output:
    tuple path("${prefix}.vcf.gz"), path("${prefix}.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    # MergeVcfs sorts the records itself. The list is sorted too, as main.nf sorts
    # the collected inputs, so the command is the same on every run rather than in
    # the order the inputs finished.
    ls *.vcf.gz | LC_ALL=C sort > vcf.list
    gatk --java-options "-Xmx${heap}g" MergeVcfs \\
        --INPUT vcf.list \\
        --OUTPUT ${prefix}.vcf.gz
    """
}

/*
 * The joint callset split for hard filtering: SNPs, and everything else.
 *
 * GATK hard-filters SNPs and indels separately, and its article on it notes that
 * `-select-type INDEL` leaves out mixed records (a SNP and an indel allele at one
 * site), which it suggests filtering as indels, as VQSR does. `--select-type-to-exclude
 * SNP` is that: indels and mixed records, and anything else, so no record is dropped
 * between the split and the merge.
 */
process GATK4_SELECTVARIANTS {
    tag "${vtype}"
    label 'process_low'

    input:
    tuple path(vcf), path(tbi), val(vtype)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(vtype), path("joint.${vtype}.vcf.gz"), path("joint.${vtype}.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    def select = vtype == 'snp' ? '--select-type-to-include SNP' : '--select-type-to-exclude SNP'
    """
    gatk --java-options "-Xmx${heap}g" SelectVariants \\
        --reference ${fasta} \\
        --variant ${vcf} \\
        ${select} \\
        --output joint.${vtype}.vcf.gz
    """
}

/*
 * GATK's generic hard filters, the thresholds for the record's type, from "(How to)
 * Filter variants either with VQSR or by hard-filtering" (GATK article 360035531112).
 * Indels get no SOR, MQ or MQRankSum filter and looser FS and ReadPosRankSum
 * thresholds: an indel lowers the mapping quality of the reads that carry it, and
 * that is not evidence of an error the way it is for a SNP.
 *
 * Hard filters rather than sarek's VQSR, which needs a genome's worth of variants to
 * fit; PIPELINE.md has the detail.
 */
process GATK4_VARIANTFILTRATION {
    tag "${vtype}"
    label 'process_low'

    input:
    tuple val(vtype), path(vcf), path(tbi)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple path("joint.${vtype}.filtered.vcf.gz"), path("joint.${vtype}.filtered.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    // The filter names are the article's, so a FILTER column reads the same as GATK's.
    def filters = vtype == 'snp'
        ? [ 'QD2': 'QD < 2.0', 'QUAL30': 'QUAL < 30.0', 'SOR3': 'SOR > 3.0', 'FS60': 'FS > 60.0',
            'MQ40': 'MQ < 40.0', 'MQRankSum-12.5': 'MQRankSum < -12.5',
            'ReadPosRankSum-8': 'ReadPosRankSum < -8.0' ]
        : [ 'QD2': 'QD < 2.0', 'QUAL30': 'QUAL < 30.0', 'FS200': 'FS > 200.0',
            'ReadPosRankSum-20': 'ReadPosRankSum < -20.0' ]
    def filter_args = filters.collect { name, expression ->
        "--filter-name '${name}' --filter-expression '${expression}'"
    }.join(' ')
    """
    gatk --java-options "-Xmx${heap}g" VariantFiltration \\
        --reference ${fasta} \\
        --variant ${vcf} \\
        ${filter_args} \\
        --output joint.${vtype}.filtered.vcf.gz
    """
}
