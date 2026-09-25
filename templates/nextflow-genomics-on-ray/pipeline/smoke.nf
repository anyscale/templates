#!/usr/bin/env nextflow

/*
 * A one-minute pipeline that exercises the executor and nothing else.
 *
 * It calls no genomics tool, reads no reference and downloads nothing, so when
 * it fails the failure is the executor's or the cluster's. The real pipeline
 * takes minutes to reach its first interesting task, and a misconfigured cluster
 * and an unhappy bwa-mem2 can look alike in a stack trace. Run this first; the
 * README does.
 *
 * Every shard's output is a pure function of its index, so the checksums must
 * be identical wherever a task ran: locally, on a Ray worker, or inside a
 * per-process image. A change in the environment (a different locale, a
 * different awk, a truncated write over NFS) shows up here as a changed number.
 *
 *   nextflow run pipeline/smoke.nf -profile ray --outdir smoke-results
 *   nextflow run pipeline/smoke.nf -profile ray --outdir smoke-results --shards 12 --hold 30
 *
 * It sits beside main.nf, not in a subdirectory, because Nextflow reads
 * nextflow.config from the script's own directory: one level down, it would run
 * without the `ray` profile it exists to test, and pass.
 */

nextflow.enable.dsl = 2

params.shards = 4
params.hold   = 8      // seconds per shard; long enough to observe placement

// No `params.outdir` default here. This script shares nextflow.config with
// main.nf, where a config param beats a script default, so a default here would
// be replaced by main.nf's 'results' and the smoke run would publish into the
// real pipeline's output directory. Outputs go under ${params.outdir}/smoke;
// pass --outdir to keep this run's trace.txt apart from main.nf's as well.

/*
 * One shard. Deliberately asks for more than one CPU so that a cluster whose
 * nodes are smaller than the request fails here, in eight seconds, rather than
 * in the alignment step.
 */
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

/*
 * Gather. Gives the pipeline a real dependency edge: a fan-out with no join
 * shows the scheduler can start tasks, not that one node can read another's
 * outputs, which is the failure that bites on a cluster.
 *
 * Through a bin/ script for the same reason. Nextflow puts the project's bin/
 * on every task's PATH, but the project sits on the node running Nextflow, and
 * a worker sees bin/ only because the executor copies it to shared storage.
 * Without that copy, main.nf's COLLECT_BENCHMARK fails on a worker with
 * `command not found`.
 */
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
    // `as Integer` is load-bearing. A param given on the command line arrives as
    // a String. Groovy's `"4" - 1` is string subtraction (remove the first "1", of
    // which there is none), so it yields "4"; then `0.."4"` compares an Integer
    // against a String by character code, and '4' is 52. Without the coercion,
    // `--shards 4` runs 53 shards and reports success. Coerce every numeric param
    // at the point of use.
    def n_shards = params.shards as Integer
    if( n_shards < 1 )
        error "--shards must be >= 1, got ${params.shards}"

    channel.of(0..<n_shards) | SHARD
    COLLECT(SHARD.out.report.collect())

    // Inside the entry workflow, and with an equals sign: the strict parser does
    // not allow statements at script level, so the familiar top-level
    // `workflow.onComplete { ... }` no longer parses.
    workflow.onComplete = {
        log.info """
        smoke ${workflow.success ? 'OK' : 'FAILED'}  duration=${workflow.duration}
        results: ${params.outdir}/smoke/checksums.txt
        """.stripIndent()
    }
}
