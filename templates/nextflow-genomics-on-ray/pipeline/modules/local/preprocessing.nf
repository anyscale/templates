/*
 * FASTQ to analysis-ready BAM. Adapted from nf-core/modules (MIT). Container
 * directives dropped: a Ray worker has no container runtime; tools come from the image.
 */

process BWAMEM2_INDEX {
    tag "${fasta.baseName}"
    label 'process_high'

    input:
    path fasta

    output:
    path "bwamem2_index", emit: index

    script:
    """
    mkdir -p bwamem2_index
    cp ${fasta} bwamem2_index/
    bwa-mem2 index bwamem2_index/${fasta.name}
    """
}

process FASTP {
    tag "${meta.id}"
    label 'process_medium'
    publishDir "${params.outdir}/fastp", mode: params.publish_mode, pattern: '*.{json,html}'

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("${meta.id}.trim_{1,2}.fastq.gz"), emit: reads
    path  "${meta.id}.fastp.json"                          , emit: json
    path  "${meta.id}.fastp.html"                          , emit: html

    script:
    """
    fastp \\
        --in1 ${reads[0]} --in2 ${reads[1]} \\
        --out1 ${meta.id}.trim_1.fastq.gz \\
        --out2 ${meta.id}.trim_2.fastq.gz \\
        --json ${meta.id}.fastp.json \\
        --html ${meta.id}.fastp.html \\
        --thread ${task.cpus} \\
        --detect_adapter_for_pe
    """
}

process BWAMEM2_MEM {
    tag "${meta.id}"
    label 'process_high'

    input:
    tuple val(meta), path(reads)
    path index

    output:
    tuple val(meta), path("${meta.id}.bam"), path("${meta.id}.bam.bai"), emit: bam

    script:
    // GATK requires a read group, and SM is how GenomicsDBImport and vcfeval name the sample.
    def rg = "@RG\\\\tID:${meta.id}\\\\tSM:${meta.id}\\\\tPL:ILLUMINA\\\\tLB:${meta.id}"
    // Sorting takes memory per thread on top of the alignment itself, so it gets
    // a slice of the task's allocation rather than samtools' default, which is
    // per-thread and unaware of what the task was actually given.
    // `x as int`, never `(int) x`: Nextflow's strict parser has no C-style
    // casts, reads `(int) (expr)` as a call to `int`, and fails the task with
    // "No signature of method: static int.call()". Every heap below does the same.
    def sort_mem = Math.max(1, (task.memory.toGiga() / (task.cpus * 2)) as int)
    """
    bwa-mem2 mem \\
        -t ${task.cpus} \\
        -R "${rg}" \\
        \$(ls ${index}/*.fna ${index}/*.fa ${index}/*.fasta 2>/dev/null | head -1) \\
        ${reads[0]} ${reads[1]} \\
      | samtools sort -@ ${task.cpus} -m ${sort_mem}G -o ${meta.id}.bam -
    samtools index -@ ${task.cpus} ${meta.id}.bam
    """
}

process GATK4_MARKDUPLICATES {
    tag "${meta.id}"
    label 'process_medium'
    publishDir "${params.outdir}/markduplicates", mode: params.publish_mode, pattern: '*.metrics'

    input:
    tuple val(meta), path(bam), path(bai)

    output:
    tuple val(meta), path("${meta.id}.md.bam"), path("${meta.id}.md.bam.bai"), emit: bam
    path "${meta.id}.md.metrics", emit: metrics

    script:
    // Explicit -Xmx: otherwise the JVM sizes its heap from the node's RAM, not the task's.
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    gatk --java-options "-Xmx${heap}g" MarkDuplicates \\
        --INPUT ${bam} \\
        --OUTPUT ${meta.id}.md.bam \\
        --METRICS_FILE ${meta.id}.md.metrics \\
        --CREATE_INDEX false \\
        --VALIDATION_STRINGENCY LENIENT
    samtools index -@ ${task.cpus} ${meta.id}.md.bam
    """
}

process GATK4_BASERECALIBRATOR {
    tag "${meta.id}"
    label 'process_medium'

    input:
    tuple val(meta), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    tuple path(known_sites), path(known_sites_tbi)

    output:
    tuple val(meta), path("${meta.id}.recal.table"), emit: table

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    gatk --java-options "-Xmx${heap}g" BaseRecalibrator \\
        --input ${bam} \\
        --reference ${fasta} \\
        --known-sites ${known_sites} \\
        --output ${meta.id}.recal.table
    """
}

process GATK4_APPLYBQSR {
    tag "${meta.id}"
    label 'process_medium'
    publishDir "${params.outdir}/alignment", mode: params.publish_mode

    input:
    tuple val(meta), path(bam), path(bai), path(table)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(meta), path("${meta.id}.recal.bam"), path("${meta.id}.recal.bam.bai"), emit: bam

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    gatk --java-options "-Xmx${heap}g" ApplyBQSR \\
        --input ${bam} \\
        --reference ${fasta} \\
        --bqsr-recal-file ${table} \\
        --output ${meta.id}.recal.bam
    samtools index -@ ${task.cpus} ${meta.id}.recal.bam
    """
}

process SAMTOOLS_STATS {
    tag "${meta.id}"
    label 'process_low'
    publishDir "${params.outdir}/samtools", mode: params.publish_mode

    input:
    tuple val(meta), path(bam), path(bai)

    output:
    path "${meta.id}.stats", emit: stats

    script:
    """
    samtools stats -@ ${task.cpus} ${bam} > ${meta.id}.stats
    """
}
