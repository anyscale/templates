version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines wdl/tasks/Assembly/Flye.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.

# Unlike upstream: explicit num_threads (/proc/cpuinfo lists the host's cores, not the task's),
# read_mode and extra_args inputs, a runtime_attr_override, and assembly_info/log as outputs.
# Flye does not resume across a WDL retry: each attempt gets a fresh directory.

workflow Flye {

    meta {
        description: "Assemble a genome using Flye"
    }
    parameter_meta {
        genome_size: "Estimated genome size in base pairs"
        reads: "Input reads (in fasta or fastq format, compressed or uncompressed)"
        prefix: "Prefix to apply to assembly output filenames"
        num_threads: "flye --threads; keep in step with runtime cpu_cores"
        read_mode: "flye's input read type flag, e.g. '--nano-raw' or '--nano-hq'"
        extra_args: "additional options appended to the flye invocation, e.g. '--asm-coverage 30 --genome-size 1m'"
    }

    input {
        File reads
        Float genome_size
        String prefix

        Int num_threads = 16
        String read_mode = "--nano-raw"
        String extra_args = ""

        RuntimeAttr? runtime_attr_override
    }

    # Upstream's call-site formula, as the fallback. An override replaces it wholesale, so
    # state mem_gb even when only changing cpu_cores.
    RuntimeAttr sized_attr = object {
        mem_gb: 100.0 + (genome_size/10000000.0)
    }

    call Assemble {
        input:
            reads  = reads,
            prefix = prefix,
            num_threads = num_threads,
            read_mode = read_mode,
            extra_args = extra_args,
            runtime_attr_override = select_first([runtime_attr_override, sized_attr])
    }

    output {
        File gfa = Assemble.gfa
        File fa = Assemble.fa
        File assembly_info = Assemble.assembly_info
        File log = Assemble.log
    }
}

task Assemble {
    input {
        File reads
        String prefix = "out"

        Int num_threads = 16
        String read_mode = "--nano-raw"
        String extra_args = ""

        RuntimeAttr? runtime_attr_override
    }

    parameter_meta {
        reads:    "reads (in fasta or fastq format, compressed or uncompressed)"
        prefix:   "prefix to apply to assembly output filenames"
        num_threads: "flye --threads; keep in step with runtime cpu_cores"
        read_mode:   "flye's input read type flag; upstream's --nano-raw by default"
        extra_args:  "additional options appended to the flye invocation"
    }

    Int disk_size = 10 * ceil(size(reads, "GB"))

    command <<<
        set -euxo pipefail

        flye ~{read_mode} ~{reads} --threads ~{num_threads} ~{extra_args} --out-dir asm

        mv asm/assembly.fasta ~{prefix}.flye.fa
        mv asm/assembly_graph.gfa ~{prefix}.flye.gfa

        # Upstream keeps only the fasta and the gfa and lets the run directory go.
        # assembly_info.txt is the first thing anyone asks Flye for: per-contig
        # length, coverage, circularity and the repeat flag, which is how you tell
        # a collapsed repeat from a real contig, and the only place the assembler
        # says what it thought it was doing. flye.log carries the stage timings the
        # README quotes. Both are small; the multi-GB intermediates stay behind.
        mv asm/assembly_info.txt ~{prefix}.flye.assembly_info.txt
        mv asm/flye.log ~{prefix}.flye.log
    >>>

    output {
        File gfa = "~{prefix}.flye.gfa"
        File fa = "~{prefix}.flye.fa"
        File assembly_info = "~{prefix}.flye.assembly_info.txt"
        File log = "~{prefix}.flye.log"
    }

    RuntimeAttr default_attr = object {
        cpu_cores:          num_threads,
        mem_gb:             100,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  0,
        max_retries:        0,
        docker:             "us.gcr.io/broad-dsp-lrma/lr-flye:2.8.3"
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
