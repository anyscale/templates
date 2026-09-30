version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines wdl/tasks/Preprocessing/Medaka.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.

# Unlike upstream: GPU opt-in (use_gpu), n_rounds 1, an R10.4.1 sup model, and `seq 1 N`
# rather than {1..N}, which counts down at N = 0. medaka does not check the model against
# the reads, so a mismatch silently degrades the consensus.

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

    # The floor keeps n_rounds = 0 from requesting no disk at all.
    Int disk_size = 10 + (4 * n_rounds * ceil(size([basecalled_reads, draft_assembly], "GB")))

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
