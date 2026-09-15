version 1.0

import "../../structs/Structs.wdl"

# From broadinstitute/long-read-pipelines
# wdl/tasks/VariantCalling/CallAssemblyVariants.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.
#
# Changes from upstream:
#
#   * Per-task RuntimeAttr overrides are plumbed out to the sub-workflow's
#     inputs so callers can re-size them (upstream fixes AlignAsPAF at 4 cores /
#     40 GiB, which is more memory than the demo's whole cluster).
#   * `num_cpus` (minimap2's `-t`) is exposed as `align_num_threads`. Upstream
#     leaves it at the task default of 4 with no way in, so a caller who sets
#     `runtime_attr_align.cpu_cores: 8`, as this template's inputs.chr20.json
#     did, reserves eight cores and runs minimap2 on four. Same arrangement,
#     and same reason, as flye_num_threads and quast_num_threads.
#   * The minimap2 preset is an input (`align_preset`) instead of a hardcoded
#     `asm20`. See the note on that task for why the default stays upstream's.
#   * Paftools gained `set -euo pipefail` and an `LC_ALL=C` sort; see there.
#   * `paftools call`'s `-l` and `-L` are stated on the command line and exposed as
#     `min_alignment_length_cov` / `min_alignment_length_call`, at paftools' own
#     default values. Upstream inherits them silently, and `-L 50000` is the
#     difference between an empty callset and a filtered one on a small region.
#   * Task-level `parameter_meta` blocks were added; upstream documents these
#     inputs only at the workflow level.
#
# A note on what this task is for, since `paftools.js call` on a human sample
# invites a misreading. Flye collapses haplotypes (no --keep-haplotypes, no purge
# step), so a diploid sample yields one mosaic haploid consensus and paftools
# reports homozygous calls only. That is a structural sanity check of the
# assembly against the reference, not a diploid variant callset, and the
# calls include every real sample-vs-reference difference as well as every
# assembly error. For a callset, align haplotype-resolved assemblies with dipcall
# or call from the reads.

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

# On the preset, because `asm20` on a human-vs-human alignment looks wrong at a
# glance, so the reasoning is written down here rather than defended in review.
#
# minimap2's man page states the bands directly: asm5 for an average divergence
# "not much higher than 0.1%", asm10 "around 1%", asm20 "around several percent".
# They also differ in mismatch penalty, asm5 -B19 against asm20 -B4.
#
# On divergence alone, asm5 is the indicated preset and asm20 is not. HG002 against
# GRCh38 is ~0.1% biological divergence, and this pipeline's own worst measurement,
# the unpolished chr20 arm, adds 125 mismatches and 42 indels per 100 kbp, so ~0.17%
# all told. asm5 is what dipcall and minimap2's cookbook use for exactly this job.
#
# The default stays upstream's asm20 for comparability with upstream, not because
# the divergence argument supports it. What does differ between the two on this
# input is the mismatch penalty rather than the band: -B19 breaks an alignment
# where consensus error clusters, and an ONT assembly polished only by Flye's
# single round has such clusters, so asm5 fragments alignment blocks that asm20
# carries through. That shows up as lower NGA50 and reads as an assembly problem
# when it is an aligner setting. It also changes which blocks clear Paftools'
# `min_alignment_length_call` floor, so the preset moves the callset twice over.
#
# Set `align_preset = "asm5"` once real polishing is on (medaka_rounds > 0, or
# dorado polish). The two are worth comparing on your own data; neither value is
# right for every assembly.
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

    #########################
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

    #########################
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
