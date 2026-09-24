/*
 * The GPU leg.
 *
 * One process in this DAG asks for an accelerator, and that is the whole point of
 * it being here: `accelerator 1, type: 'nvidia-l4'` becomes `num_gpus=1,
 * accelerator_type='L4'` on a Ray task, the autoscaler starts a GPU node for it,
 * and it runs alongside CPU calling tasks on the same cluster with no second
 * scheduler and no data movement.
 *
 * What it computes is a proxy, and the README says so plainly: a nucleotide
 * language model embeds the reference and alternate sequence around each variant
 * and the distance between the two embeddings is reported. That correlates with
 * "this changes the sequence in a way the model noticed". It is not a
 * pathogenicity score and is not comparable to SpliceAI or CADD.
 *
 * The scoring itself lives in bin/score_variants.py, which the notebook's Ray Data
 * step also imports. One implementation, called two ways -- which is what makes
 * "the same work, one shard here and the whole callset there" a real comparison
 * rather than two implementations that happen to agree.
 */

process SHARD_VCF {
    tag "${params.annotate_shards} shards"
    label 'process_low'

    input:
    tuple path(vcf), path(tbi)
    val  n_shards

    output:
    path "shard_*.vcf", emit: shards

    script:
    """
    # Split by record count, keeping the header on every shard so each piece is a
    # valid VCF that score_variants.py can read on its own.
    bcftools view -f PASS,. -H ${vcf} > body.txt
    bcftools view -h ${vcf} > header.txt

    total=\$(wc -l < body.txt)
    if [ "\$total" -eq 0 ]; then
        echo "no PASS variants to annotate" >&2
        cp header.txt shard_000.vcf
        exit 0
    fi

    per=\$(( (total + ${n_shards} - 1) / ${n_shards} ))
    split -l "\$per" -d -a 3 body.txt part_
    for part in part_*; do
        cat header.txt "\$part" > "shard_\${part#part_}.vcf"
    done
    ls -la shard_*.vcf
    """
}

process ANNOTATE_VARIANTS {
    tag "${shard.baseName}"
    label 'process_gpu'
    publishDir "${params.outdir}/annotation", mode: params.publish_mode

    input:
    path shard
    tuple path(fasta), path(fai), path(dict)

    output:
    path "${shard.baseName}.scores.tsv", emit: scores

    script:
    """
    # The model is baked into the image at \$HF_HOME. Downloading it per task
    # would have every GPU node in the fan-out pull the same 200 MB from Hugging
    # Face at once, and would make the pipeline need egress it otherwise does not.
    export HF_HUB_OFFLINE=1

    score_variants.py \\
        --vcf ${shard} \\
        --reference ${fasta} \\
        --output ${shard.baseName}.scores.tsv \\
        --batch-size 32
    """
}

process COLLECT_SCORES {
    tag "scores"
    label 'process_single'
    publishDir "${params.outdir}/annotation", mode: params.publish_mode

    input:
    path shards

    output:
    path "variant_scores.tsv", emit: table

    script:
    """
    # One header, then every shard's body, in a deterministic order -- `path
    # shards` arrives in completion order, so without the sort two runs of the
    # same pipeline produce different byte streams for the same result.
    set -euo pipefail
    first=\$(ls *.scores.tsv | LC_ALL=C sort | head -1)
    head -1 "\$first" > variant_scores.tsv
    for f in \$(ls *.scores.tsv | LC_ALL=C sort); do
        tail -n +2 "\$f" >> variant_scores.tsv
    done
    echo "collected \$(( \$(wc -l < variant_scores.tsv) - 1 )) scored variants"
    """
}
