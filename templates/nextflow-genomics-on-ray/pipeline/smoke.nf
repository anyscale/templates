#!/usr/bin/env nextflow

/*
 * A 60-second pipeline that exercises the executor and nothing else.
 *
 * It calls no genomics tool, reads no reference, and downloads nothing, so when
 * it fails the failure is the executor's. That matters because the real pipeline
 * takes minutes to reach its first interesting task, and "my cluster is
 * misconfigured" and "bwa-mem2 is unhappy" look identical from a stack trace.
 * Run this first. The README does.
 *
 * It also serves as the cross-dispatch equivalence oracle. Every shard's output
 * is a pure function of its index, so the checksums must be byte-identical
 * whether the task ran locally, on a Ray worker, or inside a per-process image.
 * A mode that quietly changes the environment -- a different locale, a different
 * awk, a truncated write over NFS -- shows up here as a changed number rather
 * than as a subtly wrong variant call three hours in.
 *
 *   nextflow run pipeline/smoke.nf -profile ray --outdir smoke-results
 *   nextflow run pipeline/smoke.nf -profile ray --outdir smoke-results --shards 12 --hold 30
 *
 * It sits beside main.nf rather than in a subdirectory on purpose: Nextflow
 * resolves nextflow.config relative to the script's own directory, so a smoke
 * pipeline one level down would quietly run *without* the `ray` profile it
 * exists to test, and pass.
 */

nextflow.enable.dsl = 2

params.shards = 4
params.hold   = 8      // seconds per shard; long enough to observe placement

// No `params.outdir` default here. This script shares nextflow.config with
// main.nf, a config param beats a script default, and so the one this file used
// to set ('smoke-results') was silently replaced by main.nf's 'results' -- the
// smoke run published into the real pipeline's output directory (observed).
// Outputs go under ${params.outdir}/smoke instead; pass --outdir to keep this
// run's trace.txt apart from main.nf's as well.

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
 * Gather. Exists so the pipeline has a real dependency edge -- a fan-out with no
 * join proves the scheduler can start tasks but not that outputs are readable
 * from another node, which is the failure mode that actually bites on a cluster.
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
    export LC_ALL=C
    cat ${reports} | sort > smoke-summary.tsv

    # The oracle. Compare this file across dispatch modes; it must not change.
    grep '^checksum' smoke-summary.tsv | cut -f2 | sort -n > checksums.txt

    echo "shards: \$(grep -c '^shard' smoke-summary.tsv)"
    echo "distinct hosts: \$(grep '^hostname' smoke-summary.tsv | cut -f2 | sort -u | wc -l)"
    """
}

workflow {
    // `as Integer` is not defensive noise -- it is load-bearing.
    //
    // A param given on the command line arrives as a String. Groovy's `"4" - 1`
    // is *string* subtraction (remove the first "1", of which there is none), so
    // it yields "4"; then `0.."4"` compares an Integer against a String by
    // character code, and '4' is 52. `--shards 4` therefore produced 53 shards,
    // silently, with the run reporting success.
    //
    // Observed, not theorised: the first local run of this pipeline did exactly
    // that. Coerce every numeric param at the point of use.
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
