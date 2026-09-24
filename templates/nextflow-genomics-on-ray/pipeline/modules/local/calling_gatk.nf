/*
 * GATK joint germline calling: the interval scatter that gives this pipeline its
 * shape.
 *
 * Per sample: HaplotypeCaller in GVCF mode, once per interval.
 * Per interval: GenomicsDBImport across all samples, then GenotypeGVCFs.
 * Then gather, filter, and you have a joint callset.
 *
 * The two-dimensional scatter is the point. Three samples over 24 intervals is 72
 * independent calling tasks, and they are what the autoscaler has to work with --
 * a single-sample pipeline is a chain, and a chain gives a scheduler nothing to
 * demonstrate. Upstream sarek scatters this way for the same reason HPC users do:
 * because HaplotypeCaller is the wall-clock cost of the whole run.
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

process GATK4_MERGEVCFS {
    tag "joint"
    label 'process_medium'

    input:
    path vcfs
    path tbis

    output:
    tuple path("joint.vcf.gz"), path("joint.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    # Sorted so the merge order is deterministic. `path vcfs` arrives in whatever
    # order the channel emitted, which depends on which interval finished first --
    # so without this the merged header's contig order can vary between runs of
    # the same pipeline, and two runs' outputs stop being byte-comparable.
    ls *.vcf.gz | LC_ALL=C sort > vcf.list
    gatk --java-options "-Xmx${heap}g" MergeVcfs \\
        --INPUT vcf.list \\
        --OUTPUT joint.vcf.gz
    """
}

process GATK4_VARIANTFILTRATION {
    tag "joint"
    label 'process_low'
    publishDir "${params.outdir}/variants", mode: params.publish_mode

    input:
    tuple path(vcf), path(tbi)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple path("joint.filtered.vcf.gz"), path("joint.filtered.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    // GATK's published hard-filter thresholds, not VQSR. VQSR needs far more
    // variants than a single chromosome provides -- it would either refuse to
    // build a model or build a bad one -- and this is the documented fallback for
    // small callsets. It is a real divergence from what sarek does genome-wide,
    // and it is declared in PIPELINE.md rather than left for a reader to notice
    // in the precision numbers.
    """
    gatk --java-options "-Xmx${heap}g" VariantFiltration \\
        --reference ${fasta} \\
        --variant ${vcf} \\
        --filter-name "QD2"      --filter-expression "QD < 2.0" \\
        --filter-name "QUAL30"   --filter-expression "QUAL < 30.0" \\
        --filter-name "SOR3"     --filter-expression "SOR > 3.0" \\
        --filter-name "FS60"     --filter-expression "FS > 60.0" \\
        --filter-name "MQ40"     --filter-expression "MQ < 40.0" \\
        --filter-name "MQRS-12.5" --filter-expression "MQRankSum < -12.5" \\
        --filter-name "RPRS-8"   --filter-expression "ReadPosRankSum < -8.0" \\
        --output joint.filtered.vcf.gz
    """
}
