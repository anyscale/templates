/*
 * Report aggregation.
 *
 * MultiQC over the QC artifacts every stage drops, plus the placement record the
 * executor wrote. The second one is not standard and is the interesting half: it
 * is what lets the notebook draw a Gantt chart of which Ray node ran which task,
 * which is the only way to tell "the cluster scaled and the work spread out" from
 * "everything queued behind one node".
 */

process MULTIQC {
    tag "multiqc"
    label 'process_low'
    publishDir "${params.outdir}/multiqc", mode: params.publish_mode

    input:
    path report_files

    output:
    path "*multiqc_report.html", emit: report
    path "*_data", emit: data

    script:
    // Globs, as nf-core's MULTIQC module declares them. The data directory is
    // named after the report: `--filename multiqc_report.html` made multiqc 1.35
    // write multiqc_report_data/, while this process declared multiqc_data/, so
    // every run failed here on a missing output (seen on the first native run).
    """
    multiqc --force .
    """
}

process COLLECT_PLACEMENT {
    tag "placement"
    label 'process_single'
    publishDir "${params.outdir}/pipeline_info", mode: params.publish_mode

    input:
    val ready          // ordering only: makes this run after the work it describes

    output:
    path "nf_ray_placement.tsv", optional: true, emit: placement

    script:
    """
    # The daemon maintains this as tasks finish, in the work directory. Copying it
    # into the results is what makes it survive the cluster -- on a job run,
    # /mnt/cluster_storage goes away with the cluster that created it.
    #
    # Optional output, on purpose: under `-profile local` there is no Ray daemon
    # and therefore no placement record, and that is not a failure.
    src="\${NF_RAY_PLACEMENT_TSV:-${workflow.workDir}/nf_ray_placement.tsv}"
    if [ -f "\$src" ]; then
        cp "\$src" nf_ray_placement.tsv
        echo "placement rows: \$(( \$(wc -l < nf_ray_placement.tsv) - 1 ))"
    else
        echo "no placement record at \$src (not running under the ray executor?)" >&2
    fi
    """
}
