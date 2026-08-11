version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines wdl/tasks/Assembly/Flye.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.
#
# Six changes from upstream, all about resources and artifacts, none about assembly:
#
#   * `--threads` comes from an explicit `Int num_threads` instead of
#     `num_core=$(cat /proc/cpuinfo | awk '/^processor/{print $3}' | wc -l)`.
#     /proc/cpuinfo is the *host's* CPU list under both `--cpus` (a CFS quota, not
#     a CPU mask) and no-container mode, so upstream's Flye plans for every core on
#     the machine no matter what `runtime.cpu` said, oversubscribing the node
#     and inflating peak memory with it. wdl/tasks/Preprocessing/Medaka.wdl states
#     its `-t` the same way and for the same reason.
#   * `cpu_cores: num_threads` in default_attr, so the Ray request and the flag Flye
#     actually receives agree by construction. A RuntimeAttr override that sets
#     cpu_cores should set num_threads to match; wdl/tasks/VariantCalling has the
#     same arrangement for minimap2's `-t`.
#   * `String extra_args` spliced into the command line. Flye's memory-reduction
#     levers live only there: `--asm-coverage N --genome-size G` caps the coverage
#     used for the initial disjointig assembly, which is the documented way to keep
#     a large genome inside a node, and `--iterations 0` skips its polishing rounds.
#     Neither has a `runtime {}` equivalent, so without a passthrough the only way
#     to run this at a size other than the Broad's is to edit the WDL. Leave it
#     empty for upstream behaviour.
#   * `String read_mode` replaces a hardcoded `--nano-raw`, defaulting to exactly
#     that so an unset caller gets upstream's command line. Flye 2.9 added
#     `--nano-hq` for Guppy 5+ / Q20 basecalls, and which one a read set deserves is
#     a property of the reads and not of the pipeline, so it cannot be answered
#     here, and smuggling it through `extra_args` would pass Flye two conflicting
#     read-type flags. ONTAssembleWithFlye.wdl derives it from a measurement; see
#     wdl/tasks/QC/ReadStats.wdl for what that measurement can and cannot settle.
#   * The sub-workflow takes a `RuntimeAttr? runtime_attr_override` of its own and
#     falls back to upstream's `100 + genome_size/1e7` GiB formula when none is
#     given. Upstream hardcodes that formula at its own call site, so the
#     sub-workflow has no runtime input at all and a caller cannot set `cpu_cores`.
#     The fallback is all-or-nothing: an override replaces it wholesale, so state
#     `mem_gb` even when you only mean to change `cpu_cores`. See `sized_attr` below.
#   * `assembly_info.txt` and `flye.log` are kept as outputs. Upstream moves only
#     the fasta and the gfa out of Flye's run directory and lets the rest go with
#     it, which discards the per-contig coverage/circularity/repeat table, the
#     one artifact that distinguishes a collapsed repeat from a real contig.
#
# On `preemptible_tries: 0` in the task's default_attr, which is upstream's value and is
# kept because this file's job is to stay upstream. Do not read it as advice.
#
# It was written for an assembly that took 14h44m, where a preemption meant redoing most
# of a run. Under this workflow's current defaults a full chr20 assembly is 1h19m on
# c6i.16xlarge or about 2h23m on m5.8xlarge, and at that length a reclaimed node costs a
# fraction of a spot node-hour to redo. inputs.chr20.json and inputs.chr20.cohort.json
# therefore set `preemptible_tries: 3` on every task, which is the right place for it: a
# spot policy is a property of the fleet you are renting, not of the assembler.
#
# Flye still does not checkpoint across a WDL retry. `--resume` reads its own `--out-dir`,
# and miniwdl gives every attempt a fresh working directory, so a retry starts from
# `configure`. Reaching the previous attempt is possible and deliberately not done; the
# README's spot note says why. Above roughly ten hours per assembly the arithmetic
# inverts and restart-from-zero starts to cost more than spot saves.
#
# So do not read the backend's Ray-node-loss -> `Interrupted` -> `runtime.preemptible`
# mapping as meaning this task retries by default. As declared here it does not; it
# retries because an inputs file gives it a budget.

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

    # Upstream instead writes, inline at the call site:
    #
    #     runtime_attr_override = { 'mem_gb': 100.0 + (genome_size/10000000.0) }
    #
    # which leaves a caller of this sub-workflow no way in at all: there is no
    # `runtime_attr_override` input to set, so cpu_cores is unreachable and the
    # memory request is whatever the formula says. On Cromwell that sizes a VM to
    # order; on Ray it decides whether the task is schedulable on the nodes that
    # exist, and 100 GiB is more than a demo cluster has in total.
    #
    # Keeping the formula as the *fallback* preserves upstream's behaviour exactly
    # when nothing is passed. Note that it is all-or-nothing: an override supplied
    # by the caller replaces it wholesale, and the task's own default_attr (a flat
    # 100 GiB) is what unset fields then fall back to, so an override should state
    # mem_gb even when it only means to change cpu_cores.
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

    #########################
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
