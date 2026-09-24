/*
 * Score the callset against the GIAB v4.2.1 truth set.
 *
 * This is the readout the whole pipeline exists to produce. Contiguity or a
 * variant count tells you the run finished; precision and recall against a truth
 * set tell you whether it was *right*, and that is a much harder thing to be
 * accidentally satisfied by.
 *
 * `rtg vcfeval` rather than hap.py: it is the comparison engine hap.py itself
 * calls under `--engine=vcfeval`, it is a single JVM tool with no Python-2
 * inheritance, and it does the thing that makes variant comparison hard --
 * matching variants by the haplotypes they imply rather than by position, so a
 * left-aligned indel and its right-aligned twin are correctly the same call.
 *
 * Scored on chr20 ∩ the GIAB high-confidence BED ∩ the demo slice. Outside that
 * intersection the truth set does not make a claim, so neither should we.
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
    tuple val(meta), val(caller), val(vtype), path(vcf), path(tbi)
    path sdf
    tuple path(truth_vcf), path(truth_tbi), path(truth_bed)
    val  region

    output:
    path "${meta.id}.${caller}.${vtype}/summary.txt", emit: summary
    path "${meta.id}.${caller}.${vtype}", emit: dir

    script:
    def region_arg = region ? "--region ${region}" : ""
    """
    # --evaluation-regions, not --bed-regions: the former restricts *scoring* to
    # the high-confidence set while still letting vcfeval use calls just outside
    # it for haplotype matching. The latter would truncate the haplotypes and
    # invent mismatches at every boundary.
    rtg vcfeval \\
        --baseline ${truth_vcf} \\
        --calls ${vcf} \\
        --evaluation-regions ${truth_bed} \\
        --template ${sdf} \\
        ${region_arg} \\
        --sample ${meta.id},${meta.id} \\
        --output ${meta.id}.${caller}.${vtype} \\
        --threads ${task.cpus}

    # vcfeval exits 0 having written a summary even when nothing matched, so an
    # empty result is a real possibility and it must not look like success.
    if ! grep -qE '^ *(None|[0-9])' ${meta.id}.${caller}.${vtype}/summary.txt; then
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
    path summaries

    output:
    path "benchmark.tsv", emit: table

    script:
    // One tidy table, so the notebook does not have to walk a directory tree and
    // so `benchmark.tsv` is the single artifact a reader can diff between runs.
    //
    // The filename carries sample/caller/type because vcfeval's summary.txt does
    // not: it reports the numbers and nothing about what produced them.
    //
    // Called bare: Nextflow puts <projectDir>/bin on PATH for every task, which
    // is also what makes the same file importable by the notebook's Ray Data
    // step without a second copy.
    """
    collect_vcfeval.py --output benchmark.tsv ${summaries}
    """
}
