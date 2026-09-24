/*
 * DeepVariant: the second caller.
 *
 * One truth set scored against two callers is a far more useful readout than
 * either alone -- it shows what the *method* contributes rather than only what
 * the pipeline produced, and it is the comparison GIAB exists to support.
 *
 * Runs on CPU. DeepVariant's GPU support exists only in Google's own `-gpu`
 * Docker image and covers only the `call_variants` stage; bioconda's package is
 * CPU TensorFlow, pinned to `python <3.11`. Since the cluster runs 3.12 and a Ray
 * driver and its workers must agree on the interpreter, it lives in a quarantined
 * conda environment and is reached only through tools/run_deepvariant.sh. See
 * that script for why the isolation is enforced rather than assumed.
 *
 * Not scattered by interval, unlike HaplotypeCaller. run_deepvariant is a
 * three-stage pipeline (make_examples, call_variants, postprocess_variants) that
 * shards internally across its own threads, and wrapping that in an outer scatter
 * means re-merging partial callsets whose gVCF records disagree at shard
 * boundaries. One task per sample, `--regions` bounded to the demo slice.
 */

process DEEPVARIANT {
    tag "${meta.id}"
    label 'process_high'
    publishDir "${params.outdir}/deepvariant", mode: params.publish_mode

    input:
    tuple val(meta), path(bam), path(bai)
    tuple path(fasta), path(fai), path(dict)
    val  region

    output:
    tuple val(meta), path("${meta.id}.dv.vcf.gz"), path("${meta.id}.dv.vcf.gz.tbi"), emit: vcf
    path "${meta.id}.dv.visual_report.html", optional: true, emit: report

    script:
    def region_arg = region ? "--regions ${region}" : ""
    """
    # The wrapper is on PATH; the DeepVariant *environment* is not. That
    # environment is python 3.10, and if it were on PATH something would resolve
    # `python` to it and a Ray worker would refuse to start -- a failure that
    # surfaces as a task timeout with nothing in the log naming the interpreter.
    # run_deepvariant.sh exists to cross that boundary in exactly one place.
    run_deepvariant.sh \\
        --model_type=WGS \\
        --ref=${fasta} \\
        --reads=${bam} \\
        --output_vcf=${meta.id}.dv.vcf.gz \\
        --num_shards=${task.cpus} \\
        ${region_arg} \\
        --intermediate_results_dir=\$PWD/dv_intermediate

    # DeepVariant writes a tabix index only sometimes, depending on version and
    # whether an output gVCF was requested. vcfeval requires one.
    if [ ! -f ${meta.id}.dv.vcf.gz.tbi ]; then
        tabix -p vcf ${meta.id}.dv.vcf.gz
    fi
    """
}
