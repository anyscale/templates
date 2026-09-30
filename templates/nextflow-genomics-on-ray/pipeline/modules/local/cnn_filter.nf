/*
 * The GPU arm: each sample genotyped alone, scored by NVScoreVariants (1D) on an L4,
 * then tranche-filtered. GATK's CNN models were trained on single-sample annotations.
 * Adapted from nf-core/modules gatk4/{genotypegvcfs,cnnscorevariants,filtervarianttranches} (MIT).
 */

process GATK4_GENOTYPEGVCFS_SAMPLE {
    tag "${meta.id}"
    label 'process_low'

    input:
    tuple val(meta), path(gvcfs), path(tbis)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(meta), path("${meta.id}.vcf.gz"), path("${meta.id}.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    """
    # One sample's per-interval GVCFs, gathered, then genotyped on their own: the
    # single-sample callset sarek's non-joint mode would make from these reads.
    ls *.g.vcf.gz | LC_ALL=C sort > gvcf.list
    gatk --java-options "-Xmx${heap}g" MergeVcfs \\
        --INPUT gvcf.list \\
        --OUTPUT ${meta.id}.g.vcf.gz
    gatk --java-options "-Xmx${heap}g" GenotypeGVCFs \\
        --reference ${fasta} \\
        --variant ${meta.id}.g.vcf.gz \\
        --output ${meta.id}.vcf.gz
    """
}

process NVSCOREVARIANTS {
    tag "${meta.id}"
    label 'process_gpu'

    input:
    tuple val(meta), path(vcf), path(tbi)
    tuple path(fasta), path(fai), path(dict)

    output:
    tuple val(meta), path("${meta.id}.cnn.vcf.gz"), path("${meta.id}.cnn.vcf.gz.tbi"), emit: vcf

    script:
    // Small heap: the memory is for the Python the JVM starts. java.io.tmpdir, not --tmp-dir:
    // the model is renamed into the JVM's temp dir, which fails across filesystems from scratch.
    """
    # GATK checks that `scorevariants` imports with `python`, then runs the model with
    # `python3`, each looked up on PATH. Both names are pointed at one interpreter, the
    # one the package is installed in (the image sets NF_GATK_PYTHON), so the check and
    # the run cannot see different environments.
    py="\${NF_GATK_PYTHON:-\$(command -v python3)}"
    mkdir -p gatk-python
    for name in python python3; do
        printf '#!/bin/sh\\nexec "%s" "\$@"\\n' "\$py" > gatk-python/\$name
        chmod +x gatk-python/\$name
    done
    export PATH="\$PWD/gatk-python:\$PATH"

    # The model is a pickled nn.Module inside the GATK jar, loaded by a bare
    # torch.load(). From torch 2.6 that refuses to unpickle anything but tensors unless
    # this is set, and it is set for this task only.
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

    # The GPU when Ray assigned one, so a node whose CUDA is broken fails here instead
    # of scoring on its CPU; the CPU otherwise, as under -profile local.
    accelerator=cpu
    if [ -n "\${CUDA_VISIBLE_DEVICES:-}" ]; then accelerator=gpu; fi

    gatk --java-options "-Xmx2g -Djava.io.tmpdir=\$PWD" NVScoreVariants \\
        --variant ${vcf} \\
        --reference ${fasta} \\
        --tensor-type reference \\
        --accelerator \$accelerator \\
        --output ${meta.id}.cnn.vcf

    # NVScoreVariants hands back the Python exit code as its result, and the JVM exits
    # 0 either way ("Tool returned: 1"). So the check is the output: every record in,
    # every record out, each with a score.
    n_in=\$(bcftools view -H ${vcf} | wc -l)
    n_scored=\$(bcftools query -f '%INFO/CNN_1D\\n' ${meta.id}.cnn.vcf 2>/dev/null | awk '\$1 != "." { n++ } END { print n + 0 }')
    if [ "\$n_in" -eq 0 ] || [ "\$n_scored" -ne "\$n_in" ]; then
        echo "NVScoreVariants scored \$n_scored of \$n_in records" >&2
        exit 1
    fi
    echo "scored \$n_scored records on \$accelerator"

    bgzip ${meta.id}.cnn.vcf
    tabix -p vcf ${meta.id}.cnn.vcf.gz
    """
}

// GATK's default tranches, tuned on whole genomes. Only resource sites in the called
// region count, so a small region sets the indel cutoff from a few hundred scores.
process GATK4_FILTERVARIANTTRANCHES {
    tag "${meta.id}"
    label 'process_low'
    publishDir "${params.outdir}/variants", mode: params.publish_mode

    input:
    tuple val(meta), path(vcf), path(tbi)
    path resources, arity: '1..*'
    path resource_tbis, arity: '1..*'

    output:
    tuple val(meta), path("${meta.id}.cnn.filtered.vcf.gz"), path("${meta.id}.cnn.filtered.vcf.gz.tbi"), emit: vcf

    script:
    def heap = Math.max(1, (task.memory.toGiga() * 0.8) as int)
    def resource_args = resources.collect { resource -> "--resource ${resource}" }.join(' ')
    """
    gatk --java-options "-Xmx${heap}g" FilterVariantTranches \\
        --variant ${vcf} \\
        ${resource_args} \\
        --info-key CNN_1D \\
        --snp-tranche 99.95 \\
        --indel-tranche 99.4 \\
        --output ${meta.id}.cnn.filtered.vcf.gz
    """
}

workflow CNN_FILTER {
    take:
    ch_gvcfs          // tuple(meta, [gvcf, ...], [tbi, ...]), one per sample
    ch_reference      // value channel: tuple(fasta, fai, dict)
    ch_resources      // value channel: [vcf, ...], FilterVariantTranches' resources
    ch_resource_tbis  // value channel: [tbi, ...]

    main:
    GATK4_GENOTYPEGVCFS_SAMPLE(ch_gvcfs, ch_reference)
    NVSCOREVARIANTS(GATK4_GENOTYPEGVCFS_SAMPLE.out.vcf, ch_reference)
    GATK4_FILTERVARIANTTRANCHES(NVSCOREVARIANTS.out.vcf, ch_resources, ch_resource_tbis)

    emit:
    GATK4_FILTERVARIANTTRANCHES.out.vcf   // tuple(meta, vcf, tbi), one per sample
}
