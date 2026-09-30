version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines
# wdl/tasks/VariantCalling/CallAssemblyVariants.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.

# Unlike upstream: re-sizable tasks, align_num_threads and align_preset inputs, explicit
# paftools -l/-L, and pipefail in Paftools. Flye collapses haplotypes, so the calls are
# homozygous-only: a structural check of the assembly, not a diploid callset.

workflow CallAssemblyVariants {

    meta {
        description: "Call variants from an assembly using paftools.js"
    }

    parameter_meta {
        asm_fasta:         "assembly to align; haplotype-collapsed for a diploid sample, so the calls are homozygous-only (see the header)"
        ref_fasta:         "reference to which assembly should be aligned"
        participant_name:  "participant name"
        prefix:            "prefix for output files"
        align_num_threads: "minimap2 -t; keep in step with runtime_attr_align's cpu_cores"
        align_preset:      "minimap2 -x preset for assembly-to-reference alignment; see the AlignAsPAF task note before changing it"
        min_alignment_length_cov:  "paftools call -l"
        min_alignment_length_call: "paftools call -L; blocks shorter than this are not called on at all, so raise or lower it to match your assembly's block lengths"
    }

    input {
        File asm_fasta
        File ref_fasta
        String participant_name
        String prefix

        Int align_num_threads = 4
        String align_preset = "asm20"

        Int min_alignment_length_cov = 10000
        Int min_alignment_length_call = 50000

        RuntimeAttr? runtime_attr_align
        RuntimeAttr? runtime_attr_paftools
    }

    call AlignAsPAF {
        input:
            ref_fasta = ref_fasta,
            asm_fasta = asm_fasta,
            prefix = prefix,
            num_cpus = align_num_threads,
            preset = align_preset,
            runtime_attr_override = runtime_attr_align
    }

    call Paftools {
        input:
            ref_fasta = ref_fasta,
            paf = AlignAsPAF.paf,
            participant_name = participant_name,
            prefix = prefix,
            min_alignment_length_cov = min_alignment_length_cov,
            min_alignment_length_call = min_alignment_length_call,
            runtime_attr_override = runtime_attr_paftools
    }

    output {
        File paf = AlignAsPAF.paf
        File paftools_vcf = Paftools.variants
    }
}

# asm20 is upstream's default, kept for comparability. By divergence asm5 fits (dipcall uses
# it), but its -B19 penalty fragments blocks at unpolished error clusters; use it once polished.
task AlignAsPAF {
    input {
        File ref_fasta
        File asm_fasta
        String prefix

        Int num_cpus = 4
        String preset = "asm20"

        RuntimeAttr? runtime_attr_override
    }

    parameter_meta {
        ref_fasta: "reference to align against"
        asm_fasta: "assembly to align"
        prefix:    "prefix for output files"
        num_cpus:  "minimap2 -t; keep in step with runtime cpu_cores"
        preset:    "minimap2 -x preset; see the note above this task"
    }

    Int disk_size = 4*ceil(size(ref_fasta, "GB") + size(asm_fasta, "GB"))

    command <<<
        set -euxo pipefail

        minimap2 --paf-no-hit -cx ~{preset} --cs -r 2k -t ~{num_cpus} \
            ~{ref_fasta} ~{asm_fasta} | \
            gzip -1 > ~{prefix}.paf.gz
    >>>

    output {
        File paf = "~{prefix}.paf.gz"
    }

    RuntimeAttr default_attr = object {
        cpu_cores:          num_cpus,
        mem_gb:             40,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  3,
        max_retries:        2,
        docker:             "us.gcr.io/broad-dsp-lrma/lr-asm:0.1.13"
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

task Paftools {
    input {
        File ref_fasta
        File paf
        String participant_name
        String prefix

        Int min_alignment_length_cov = 10000
        Int min_alignment_length_call = 50000

        RuntimeAttr? runtime_attr_override
    }

    parameter_meta {
        ref_fasta:        "reference the PAF was aligned against; -f, which is what makes paftools emit VCF"
        paf:              "gzipped PAF from AlignAsPAF"
        participant_name: "sample name written into the VCF"
        prefix:           "prefix for output files"
        min_alignment_length_cov:  "paftools call -l: alignment blocks shorter than this do not count towards coverage"
        min_alignment_length_call: "paftools call -L: alignment blocks shorter than this produce no variant calls. Lower it for regions whose blocks are shorter than the 50 kb default, or the callset is silently empty"
    }

    Int disk_size = 2*ceil(size(ref_fasta, "GB") + size(paf, "GB"))
    Int num_cpus = 1

    command <<<
        # `set -euo pipefail` is a divergence from upstream, and the only one in this
        # task. Without it the pipeline's exit status is `paftools.js`'s alone: a zcat
        # on a truncated PAF, or a sort that runs out of temp space, leaves a short or
        # empty VCF behind and the task still succeeds. That is the same silent
        # wrong-answer failure the Quast task guards against, and a VCF is a worse
        # thing to be quietly wrong about than a report.
        set -euo pipefail

        # LC_ALL=C because `sort -k6,6` is a lexical sort on contig names and the
        # collation order is locale-dependent. paftools.js needs the PAF grouped by
        # target and ascending by target start; which grouping you get should not
        # depend on the worker's LANG.
        # -l and -L are paftools' own defaults, stated rather than inherited. -L in
        # particular decides the callset: alignment blocks under it are aligned, counted
        # and then called on not at all, so a region whose blocks are shorter than 50 kb
        # yields an empty VCF and a zero exit. Upstream leaves both implicit, which is
        # how a legitimately empty callset and a silently filtered one look identical.
        #
        # The VCF this writes is NOT normalized. paftools places an indel wherever the
        # `cs` tag put it, and inside a homopolymer that position is arbitrary, so two
        # independently aligned assemblies can spell one variant two ways. Normalize
        # (`bcftools norm -f <ref> -m -any`) before comparing callsets across samples.
        zcat ~{paf} | \
            LC_ALL=C sort -k6,6 -k8,8n | \
            paftools.js call -f ~{ref_fasta} -s ~{participant_name} \
                -l ~{min_alignment_length_cov} -L ~{min_alignment_length_call} - \
            > ~{prefix}.paftools.vcf
    >>>

    output {
        File variants = "~{prefix}.paftools.vcf"
    }

    RuntimeAttr default_attr = object {
        cpu_cores:          num_cpus,
        mem_gb:             20,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  3,
        max_retries:        2,
        docker:             "us.gcr.io/broad-dsp-lrma/lr-asm:0.1.13"
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
