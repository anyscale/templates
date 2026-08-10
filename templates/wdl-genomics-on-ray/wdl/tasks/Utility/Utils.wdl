version 1.0

import "../../structs/Structs.wdl"

# A two-task subset of broadinstitute/long-read-pipelines wdl/tasks/Utility/Utils.wdl
# (which is ~2200 lines), carrying only what ONTAssembleWithFlye calls.
# Licensed BSD-3-Clause; see wdl/LICENSE.
#
# The upstream pipeline also calls Utils.ListFilesOfType to expand a GCS
# directory with `gsutil ls`. That task is deliberately absent here: it hardcodes
# both a cloud vendor and a CLI, and the workflow takes an explicit
# `Array[File]+ fastqs` instead. miniwdl localizes gs://, s3:// and https:// URIs
# for File inputs on its own, so a caller keeps cloud input without the pipeline
# needing to know which cloud it is.
#
# Two other changes from upstream:
#
#   * ComputeGenomeLength's final awk prints with `printf "%.0f\n"` where upstream
#     uses `print`. mawk, which is /usr/bin/awk on Debian and Ubuntu and therefore
#     on the cluster image, formats any value above INT_MAX through OFMT, so
#     upstream emits GRCh38's 3,099,922,541 as "3.09992e+09" and read_float takes
#     it — quietly turning a 3.1 Gbp genome into 3.1 billionths of one. Only a
#     genome above 2^31 reaches this, which is why it survives testing on demo
#     data.
#
#     Note "%.0f" and not "%d". "%d" also fixes the OFMT problem, and was what this
#     file used first, but mawk 1.3.4-20200120 converts %d through a 32-bit int and
#     clamps: GRCh38 comes back as exactly 2147483647, read_float accepts it, and the
#     genome is then wrong by 31% with nothing failing. "%.0f" formats the double and
#     is exact to 2^53. wdl/tasks/QC/ReadStats.wdl carries the same fix, and its
#     comment has the measurement.
#   * MergeFastqs' docker moved from gcr.io/cloud-marketplace/google/ubuntu2004 to
#     docker.io/library/ubuntu:20.04: no GCP credentials needed, and multi-arch.

task ComputeGenomeLength {

    meta {
        description: "Utility to compute the length of a genome from a FASTA file"
    }

    parameter_meta {
        fasta:  "FASTA file. Every sequence is counted and gaps count as sequence; see the note below"
    }

    # What this measures, stated because callers use it as a genome size estimate.
    #
    # It sums `LN:` over every `@SQ` record, which is the FASTA's *sequence* length
    # and not the assemblable genome. Two consequences:
    #
    #   * N runs count. A reference with modelled centromeres or unclosed gaps
    #     reports more bases than an assembler can produce, so a `--genome-size`
    #     derived from it runs high and a coverage derived from it runs low. GRCh38
    #     chr20 is a few percent N; a draft reference can be far more.
    #   * Every sequence counts. A full GRCh38 with alts, decoys and unplaced contigs
    #     returns more than the primary assembly's length, and handing it a whole
    #     genome when the reads cover one chromosome is wrong by two orders of
    #     magnitude, which is why the region slices ship with a matching reference.
    #
    # Neither matters for those slices; both matter for real inputs. Callers that
    # care should pass the size explicitly (ONTAssembleWithFlye exposes
    # `flye_genome_size` for exactly this).
    input {
        File fasta

        RuntimeAttr? runtime_attr_override
    }

    Int disk_size = 2*ceil(size(fasta, "GB"))

    command <<<
        set -euxo pipefail

            samtools dict ~{fasta} | \
            grep '^@SQ' | \
            awk '{ print $3 }' | \
            sed 's/LN://' | \
            awk '{ sum += $1 } END { printf "%.0f\n", sum }' > length.txt
    >>>

    output {
        Float length = read_float("length.txt")
    }

    #########################
    RuntimeAttr default_attr = object {
        cpu_cores:          1,
        mem_gb:             1,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  0,
        max_retries:        0,
        docker:             "us.gcr.io/broad-dsp-lrma/lr-utils:0.1.8"
    }
    RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])
    runtime {
        cpu:                    select_first([runtime_attr.cpu_cores,         default_attr.cpu_cores])
        memory:                 select_first([runtime_attr.mem_gb,            default_attr.mem_gb]) + " GiB"
        disks: "local-disk " +  select_first([runtime_attr.disk_gb,           default_attr.disk_gb]) + " HDD"
        bootDiskSizeGb:         select_first([runtime_attr.boot_disk_gb,      default_attr.boot_disk_gb])
        preemptible:            select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
        maxRetries:             select_first([runtime_attr.max_retries,       default_attr.max_retries])
        docker:                 select_first([runtime_attr.docker,            default_attr.docker])
    }
}

task MergeFastqs {

    meta {
        description : "Merge fastq files."
    }

    parameter_meta {
        fastqs: "Fastq files to be merged."
        prefix: "Prefix for the output fastq file."
        runtime_attr_override: "Override the default runtime attributes."
    }

    input {
        Array[File] fastqs

        String prefix = "merged"

        RuntimeAttr? runtime_attr_override
    }

    Int disk_size = 3 * ceil(size(fastqs, "GB"))

    String disk_type = if disk_size < 375 then "LOCAL" else "HDD"

    Int memory = 8

    command <<<
        FILE="~{fastqs[0]}"
        if [[ "$FILE" =~ \.gz$ ]]; then
            cat ~{sep=' ' fastqs} > ~{prefix}.fq.gz
        else
            cat ~{sep=' ' fastqs} | gzip > ~{prefix}.fq.gz
        fi
    >>>

    output {
        File merged_fastq = "~{prefix}.fq.gz"
    }

    #########################
    # Upstream uses gcr.io/cloud-marketplace/google/ubuntu2004:latest. Swapped for
    # Docker Hub's ubuntu, which needs no GCP credentials and is multi-arch.
    RuntimeAttr default_attr = object {
        cpu_cores:          2,
        mem_gb:             memory,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  0,
        max_retries:        0,
        docker:             "docker.io/library/ubuntu:20.04"
    }
    RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])
    runtime {
        cpu:                    select_first([runtime_attr.cpu_cores,         default_attr.cpu_cores])
        memory:                 select_first([runtime_attr.mem_gb,            default_attr.mem_gb]) + " GiB"
        disks: "local-disk " +  select_first([runtime_attr.disk_gb,           default_attr.disk_gb]) + " ~{disk_type}"
        bootDiskSizeGb:         select_first([runtime_attr.boot_disk_gb,      default_attr.boot_disk_gb])
        preemptible:            select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
        maxRetries:             select_first([runtime_attr.max_retries,       default_attr.max_retries])
        docker:                 select_first([runtime_attr.docker,            default_attr.docker])
    }
}
