#!/usr/bin/env nextflow

/*
 * Executor-only check: no tools, no downloads. Each shard's checksum depends only on
 * its index, so it must match wherever it ran. Beside main.nf to share nextflow.config.
 */

nextflow.enable.dsl = 2

params.shards = 4
params.hold   = 8      // seconds per shard; long enough to observe placement

// No outdir default: nextflow.config's would win. Pass --outdir to keep it apart from main.nf's.

// Two CPUs, so an undersized cluster fails here rather than in alignment.
process SHARD {
    tag "shard-${idx}"
    cpus 2
    memory 2.GB
    publishDir "${params.outdir}/smoke", mode: 'copy'

    input:
    val idx

    output:
    path "shard-${idx}.tsv", emit: report

    script:
    """
    # Deterministic by construction: content depends only on the shard index.
    # LC_ALL is pinned because sort order is locale-dependent and a checksum that
    # changes with the environment would make this test useless as an oracle.
    export LC_ALL=C

    seq \$(( ${idx} * 1000 + 1 )) \$(( ${idx} * 1000 + 1000 )) \\
        | awk '{ s += \$1 } END { printf "%d\\n", s }' > sum.txt

    sleep ${params.hold}

    # Where did this land? Ray sets neither of these, so they come from the OS and
    # report the node the task really ran on -- which is what the notebook's Gantt
    # chart cross-checks against the executor's own placement record.
    printf 'shard\\t%s\\n'      "${idx}"            >  shard-${idx}.tsv
    printf 'checksum\\t%s\\n'   "\$(cat sum.txt)"   >> shard-${idx}.tsv
    printf 'hostname\\t%s\\n'   "\$(hostname)"      >> shard-${idx}.tsv
    printf 'cpus_seen\\t%s\\n'  "\$(nproc 2>/dev/null || echo NA)" >> shard-${idx}.tsv
    """
}

// The join checks that one node can read another's outputs; the bin/ script, that
// bin/ reaches the workers (the executor copies it to shared storage).
process COLLECT {
    cpus 1
    memory 1.GB
    publishDir "${params.outdir}/smoke", mode: 'copy'

    input:
    path reports

    output:
    path 'smoke-summary.tsv'
    path 'checksums.txt'

    script:
    """
    smoke_collect.sh ${reports}
    """
}

workflow {
    // A CLI param is a String: without `as Integer`, `--shards 4` runs 53 shards.
    def n_shards = params.shards as Integer
    if( n_shards < 1 )
        error "--shards must be >= 1, got ${params.shards}"

    channel.of(0..<n_shards) | SHARD
    COLLECT(SHARD.out.report.collect())

    // Assigned here: the strict parser rejects a top-level `workflow.onComplete { }`.
    workflow.onComplete = {
        log.info """
        smoke ${workflow.success ? 'OK' : 'FAILED'}  duration=${workflow.duration}
        results: ${params.outdir}/smoke/checksums.txt
        """.stripIndent()
    }
}
