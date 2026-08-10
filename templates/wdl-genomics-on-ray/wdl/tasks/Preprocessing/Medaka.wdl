version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines wdl/tasks/Preprocessing/Medaka.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.
#
# Changes from upstream, both about making the GPU request optional:
#
#   * `Boolean use_gpu = false` gates gpuCount/gpuType. Upstream requests one
#     nvidia-tesla-t4 unconditionally, which on a CPU-only Ray cluster produces a
#     task Ray can never schedule: it waits forever instead of failing. medaka
#     runs correctly (just slower) on CPU, so CPU is the safer default here and
#     `use_gpu = true` restores upstream behaviour.
#   * `gpuType` is passed through as-is; wdl_on_ray.resources maps the GCE
#     accelerator names ("nvidia-tesla-t4") onto Ray's ("T4"), so either spelling
#     works.
#
# Also dropped: `zones`, `cpuPlatform` and `nvidiaDriverVersion`, which are
# Google-backend-only Cromwell keys with no meaning outside GCP.
#
# Three smaller edits: `-t 8` became `-t ~{threads}` so thread count can track the
# runtime cpu_cores; the polishing loop uses `$(seq 1 N)` instead of brace
# expansion `{1..N}`: with N=0 brace expansion counts *down* (1, 0) and would
# run a bogus round, whereas `seq 1 0` is empty, making `n_rounds = 0` a clean
# pass-through of the draft assembly; and `source /medaka/venv/bin/activate` (the
# venv inside upstream's lr-medaka image) is guarded with a file test, so the
# pass-through works quietly on a worker that has no medaka at all instead of
# logging a spurious `No such file or directory` before doing exactly the same
# thing.
#
# Three further edits are not about scheduling. `n_rounds` defaults to 1 rather than
# upstream's 3, for the reason below. `disk_size` gained a `10 +` floor, so a small
# input cannot round the request down to 0 GB. And `meta.description` plus two
# `parameter_meta` strings are rewritten, because upstream's describe the GPU-only
# arrangement this task no longer has.
#
# The model default is changed from upstream's `r941_prom_high_g360`, and that is
# the divergence in this file most likely to change results rather than scheduling.
#
# medaka does not validate the model against the data. Handed an R9.4.1 model and
# R10.4.1 reads it runs to completion, exits 0, and emits a consensus that is
# *worse* than the unpolished draft, because the error model it is correcting for is not
# the error model in the reads. Nothing downstream notices; QUAST reports a number
# and the number is bad for a reason nobody can see from the outputs. A default
# that is wrong for the data it ships with is a trap for the first person who sets
# n_rounds > 0, so the default here matches this template's reads
# (r1041_e82_400bps_sup_v4.1.0: R10.4.1, E8.2 pore, 400 bps, dorado sup v4.1.0).
# Change the model whenever you change the data, and run `medaka tools list_models`
# for the set your medaka build actually carries. Match the sampling rate too, not
# just the pore and kit: 4 kHz and 5 kHz R10.4.1 runs take different model lines
# (v4.1.0 against v4.2.0 and later), and the CRAM headers of the reads this template
# ships carry no @RG basecall_model record to settle it from the data.
#
# Two more things to know before turning this on:
#
#   * One round, not three. Upstream's n_rounds = 3 applies the model to its own
#     output, which after the first round is no longer the distribution medaka was
#     trained on — it was trained to correct draft assemblies, and the guidance that
#     did call for iteration called for iterating *racon* before a single medaka
#     pass. Rounds two and three buy wall clock and, on paper, a slightly
#     out-of-distribution input.
#   * For a *human* assembly, medaka is no longer ONT's recommendation. Since dorado
#     0.9.0 (Dec 2024) ONT points large-genome consensus polishing at `dorado
#     polish`, and keeps medaka as the recommendation for small genomes on CPU. This
#     task is retained because it is what upstream's pipeline calls; a chr20-scale
#     or larger assembly is the case ONT would send to dorado polish instead.
#
# Packaging, for anyone adding medaka to the workers: medaka 1.x (which the r941
# models ship with) requires Python <3.11; medaka >=2.1 runs on newer Pythons and
# ships a newer model generation, so moving to it means re-choosing the model too.
# tools/BUILDING.md covers getting it onto the cluster.

task MedakaPolish {

    meta {
        description: "Polish an ONT draft assembly with the basecalled reads it was assembled from. The model must match the reads' chemistry and basecaller; n_rounds = 0 passes the draft through unchanged. Upstream's timing note (a few hours for 18 GB of reads against a 23 Mbp genome) was measured on R9.4.1 data and is a rough order of magnitude, not a budget."
    }
    parameter_meta {
        basecalled_reads:   "basecalled reads to be used with polishing"
        draft_assembly:     "draft assembly to be polished"
        prefix:             "prefix for output files"
        model:              "medaka model matching the reads' pore, chemistry and basecaller. A mismatched model degrades the consensus without erroring; see the header. `medaka tools list_models`"
        n_rounds:           "number of polishing rounds to apply; 0 passes the draft through unchanged. Prefer 1; iterating medaka is no longer recommended practice"
        use_gpu:            "request a GPU for medaka; requires a GPU node group in the Ray cluster"
        gpu_type:           "accelerator to request when use_gpu is true (GCE or Ray spelling)"
        threads:            "medaka_consensus -t; keep in step with runtime cpu_cores"
    }

    input {
        File basecalled_reads
        File draft_assembly

        String prefix = "consensus"
        String model = "r1041_e82_400bps_sup_v4.1.0"
        Int n_rounds = 1

        Boolean use_gpu = false
        String gpu_type = "nvidia-tesla-t4"
        Int threads = 8

        RuntimeAttr? runtime_attr_override
    }

    # Upstream's formula, with a floor. n_rounds = 0 makes the product 0, and a
    # zero disk request is a request for nothing rather than a request for the
    # pass-through copy's worth of space. The floor costs nothing and stops a
    # backend that honours `disks` from scheduling this task onto no disk at all.
    Int disk_size = 10 + (4 * n_rounds * ceil(size([basecalled_reads, draft_assembly], "GB")))

    ###
    # Medaka models. This list is upstream's and is kept only as a record of what
    # upstream targeted: every entry is R9.4.1-or-older chemistry, and none of them
    # is correct for R10.4.1 data:
    #
    #   r103_*, r10_*, r941_*  (Guppy 3.0-3.6 era)
    #
    # The naming scheme for current models is
    # `r<pore><chemistry>_e<pore version>_<translocation speed>_<variant>_v<basecaller>`,
    # e.g. r1041_e82_400bps_sup_v4.1.0 for R10.4.1 / E8.2 / 400 bps / dorado sup 4.1.0.
    # Do not copy a name from here or from any doc, including this one: model
    # availability is a property of the installed medaka build, so run
    # `medaka tools list_models` on the workers and pick the entry whose basecaller
    # version is closest to (and not newer than) the one that produced the reads.
    ###

    command <<<
        # Present inside upstream's lr-medaka image; absent (and safely skipped) when the
        # command runs directly on a Ray worker that gets medaka some other way, or, for
        # n_rounds = 0, not at all.
        if [ -f /medaka/venv/bin/activate ]; then source /medaka/venv/bin/activate; fi

        set -euxo pipefail

        mkdir output_0_rounds
        cp ~{draft_assembly} output_0_rounds/consensus.fasta

        for i in $(seq 1 ~{n_rounds})
        do
          medaka_consensus -i ~{basecalled_reads} -d output_$((i-1))_rounds/consensus.fasta -o output_${i}_rounds -t ~{threads} -m ~{model}
        done

        cp output_~{n_rounds}_rounds/consensus.fasta ~{prefix}.fasta
    >>>

    output {
        File polished_assembly = "~{prefix}.fasta"
    }

    ###################
    RuntimeAttr default_attr = object {
        cpu_cores:              8,
        mem_gb:                 24,
        disk_gb:                disk_size,
        boot_disk_gb:           25,
        preemptible_tries:      0,
        max_retries:            0,
        docker:                 "us.gcr.io/broad-dsp-lrma/lr-medaka:0.1.0"
    }
    RuntimeAttr runtime_attr = select_first([runtime_attr_override, default_attr])
    runtime {
        cpu:                    select_first([runtime_attr.cpu_cores, default_attr.cpu_cores])
        memory:                 select_first([runtime_attr.mem_gb, default_attr.mem_gb]) + " GiB"
        disks:  "local-disk " + select_first([runtime_attr.disk_gb, default_attr.disk_gb]) + " HDD"
        bootDiskSizeGb:         select_first([runtime_attr.boot_disk_gb, default_attr.boot_disk_gb])
        preemptible:            select_first([runtime_attr.preemptible_tries, default_attr.preemptible_tries])
        maxRetries:             select_first([runtime_attr.max_retries, default_attr.max_retries])
        gpuCount:               if use_gpu then 1 else 0
        gpuType:                gpu_type
        docker:                 select_first([runtime_attr.docker, default_attr.docker])
    }
}
