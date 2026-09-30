version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines wdl/tasks/QC/Quast.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.

# Unlike upstream: explicit num_threads, `tree ... || true`, --large as a real flag, a
# re-sizable SummarizeQuastReport on Docker Hub's ubuntu, and a hard failure when a
# reference was given but QUAST reported no alignment (its bundled minimap2 was missing).

task Quast {

    meta {
        description: "A task that runs QUAST to evaluate a given set of assemblies on a species with existing reference assembly. Entire Quast output will be tarballed"
    }
    parameter_meta {
        ref:        "reference assembly of the species"
        assemblies: "list of assemblies to evaluate"
        is_large:   "pass QUAST's --large, for genomes above ~100 Mbp"
        num_threads: "quast --threads; keep in step with runtime cpu_cores"
    }

    input {
        File? ref
        Array[File] assemblies
        Boolean is_large = false

        # Also default_attr's cpu_cores; a runtime_attr_override setting cpu_cores should match.
        Int num_threads = 16

        RuntimeAttr? runtime_attr_override
    }

    Int minimal_disk_size = 2*(ceil(size(ref, "GB") + size(assemblies, "GB")))
    Int disk_size = if minimal_disk_size > 100 then minimal_disk_size else 100

    command <<<
        set -eux

        quast --no-icarus \
              ~{true='--large' false='' is_large} \
              --threads ~{num_threads} \
              ~{true='-r' false='' defined(ref)} \
              ~{select_first([ref, ""])} \
              ~{sep=' ' assemblies}

        tree -h quast_results/ || true

        # Fail loudly when QUAST evaluated against a reference but produced no alignment.
        #
        # Everything QUAST reports splits in two: metrics computable from the contigs plus the
        # reference's *length* (N50, NG50, auN, GC) and metrics that need the contigs actually
        # *aligned* to it (misassemblies, mismatch and indel rates, genome fraction, NGA50).
        # Losing the second set is the difference between knowing an assembly is contiguous and
        # knowing it is correct, and QUAST drops it silently, because the bundled minimap2 it
        # compiles at install time is optional to the install succeeding. The `contigs_reports`
        # guard below is upstream already knowing this can happen; it just treats it as normal.
        #
        # `Genome fraction` is the check because it appears if and only if alignment ran.
        if ~{true='true' false='false' defined(ref)}; then
            grep -q 'Genome fraction' quast_results/latest/report.txt || {
                echo "QUAST ran with a reference but reported no alignment-based metrics." >&2
                echo "Its contig aligner is missing: see tools/manifest.toml [tools.quast]," >&2
                echo "which depends on the minimap2 distribution for this." >&2
                exit 1
            }
        fi

        if [[ -d quast_results/contigs_reports ]]; then
            tar -zcvf contigs_reports.tar.gz quast_results/contigs_reports
        fi
    >>>

    output {
        File report_txt = "quast_results/latest/report.txt"
        File report_html = "quast_results/latest/report.html"

        Array[File] report_in_various_formats = glob("quast_results/latest/report.*")

        Array[File] plots = glob("quast_results/latest/basic_stats/*.pdf")

        File? contigs_reports = "contigs_reports.tar.gz"
    }

    RuntimeAttr default_attr = object {
        cpu_cores:             num_threads,
        mem_gb:                80,
        disk_gb:               disk_size,
        boot_disk_gb:          25,
        preemptible_tries:     0,
        max_retries:           0,
        docker:                "us.gcr.io/broad-dsp-lrma/lr-quast:5.2.0"
    }
    RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])
    runtime {
        cpu:                   select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
        memory:                select_first([runtime_attr.mem_gb, default_attr.mem_gb]) + " GiB"
        disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
        bootDiskSizeGb:        select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
        preemptible:           select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
        maxRetries:            select_first([runtime_attr.max_retries, default_attr.max_retries])
        docker:                select_first([runtime_attr.docker, default_attr.docker])
    }
}

task SummarizeQuastReport {

    meta {
        description: "A task that summarizes the QUAST report into a single tab-delimited file"
    }
    parameter_meta {
        quast_report_txt: "the QUAST report file"
    }

    input {
        File quast_report_txt

        RuntimeAttr? runtime_attr_override
    }

    command <<<
        set -eux
        grep -v -e '^All statistics' -e '^$' ~{quast_report_txt} | \
            sed 's/ /_/g' | \
            sed 's/__\+/\t/g' | \
            sed 's/\s\+$//g' | \
            sed 's/>=/gt/g' | \
            tee report_map.txt

        for i in $(seq 2 $(awk '{print NF}' report_map.txt | sort -nu | tail -n 1))
        do
            j=$(( i - 2 ))  # to make sure the primary, assuming it's the 0-th fed in to this task and the left-most value column
            cut -d$'\t' -f1,${i} < report_map.txt > report_map_${j}.txt
        done
    >>>

    output {
        File quast_metrics_together = "report_map.txt"
        Array[File] quast_metrics = glob("report_map_*.txt")
    }

    RuntimeAttr default_attr = object {
        cpu_cores:             1,
        mem_gb:                2,
        disk_gb:               100,
        boot_disk_gb:          25,
        preemptible_tries:     0,
        max_retries:           0,
        docker:                "docker.io/library/ubuntu:20.04"
    }
    RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])
    runtime {
        cpu:                   select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
        memory:                select_first([runtime_attr.mem_gb, default_attr.mem_gb]) + " GiB"
        disks: "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
        preemptible:           select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
        maxRetries:            select_first([runtime_attr.max_retries, default_attr.max_retries])
        docker:                select_first([runtime_attr.docker, default_attr.docker])
    }
}
