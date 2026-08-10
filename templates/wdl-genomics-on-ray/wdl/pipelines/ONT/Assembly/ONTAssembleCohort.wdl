version 1.0

import "ONTAssembleWithFlye.wdl" as Single

# `RuntimeAttr` is not imported here. WDL 1.0 hoists imported structs into a
# document-global namespace, so it arrives with the sub-workflow above; importing
# Structs.wdl directly as well type-checks but trips miniwdl's UnusedImport lint,
# and this template's notebook shows `wdl-on-ray check` output to the reader.

# Assemble a cohort: one ONTAssembleWithFlye per sample, all in one workflow.
#
# Not adapted from upstream. broadinstitute/long-read-pipelines has no cohort
# wrapper because on Cromwell there is nothing to gain from one: each task is
# provisioned its own VM either way, so N samples is N independent submissions and
# a wrapper only adds a scatter the scheduler cannot exploit.
#
# On Ray the wrapper is the point. `scatter` makes every sample's task graph
# resident in one workflow, so the whole cohort's tasks compete for one autoscaling
# pool: sample B's 2-CPU ComputeGenomeLength packs onto the node already running
# sample A's 30-CPU assembly instead of waiting for a VM of its own, and the pool
# grows and shrinks against the *cohort's* demand rather than each sample's. That is
# the difference this backend exists to make, and a single-sample run cannot show it:
# one sample's graph is a chain with one long task in the middle, and a chain has
# nothing to bin-pack.
#
# The wrapper is deliberately thin. Every per-sample decision stays in
# ONTAssembleWithFlye, and everything here is either the sample list or a knob that
# has to be identical across the cohort for the results to be comparable. Anything
# that should vary per sample belongs in `Sample`, not in an input.
#
# On what a "cohort" means scientifically: these are independent single-sample
# assemblies that happen to share a cluster. Nothing here is a joint analysis --
# no trio-aware assembly, no pedigree-informed phasing, no joint variant calling.
# Assembling the GIAB Ashkenazi trio this way gives three haploid-collapsed
# assemblies, not a phased family callset. Trio-binning (as in Canu/hifiasm's trio
# mode, where parental k-mers phase the child's reads) is a different pipeline and
# would need the parents' reads as an *input* to the child's assembly rather than
# as a sibling scatter shard.

struct Sample {
    String name
    Array[File]+ fastqs
    File ref_fasta
}

workflow ONTAssembleCohort {
    meta {
        description: "Assemble every sample in a cohort, one ONTAssembleWithFlye per sample, sharing one autoscaling cluster. Samples are assembled independently; this is not a joint or pedigree-aware analysis."
    }

    parameter_meta {
        samples:             "one entry per sample: name, its FASTQs (one per flow cell), and the reference to assess it against"

        flye_num_threads:    "flye --threads for every sample; keep in step with runtime_attr_flye's cpu_cores"
        quast_num_threads:   "quast --threads for every sample"
        align_num_threads:   "minimap2 -t for every sample's assembly-to-reference alignment"

        medaka_rounds:       "medaka polishing rounds, applied to every sample. Requires medaka on the workers"
        medaka_model:        "medaka model, applied to every sample, so a cohort must share a chemistry and basecaller"
        medaka_use_gpu:      "request a GPU for medaka (needs a GPU node group in the Ray cluster)"

        flye_impute_params:  "derive Flye's parameters from each sample's own reads; false restores upstream's command line"
        flye_read_mode:      "explicit flye read type flag for every sample; overrides the per-sample measurement"
    }

    input {
        Array[Sample]+ samples

        Int flye_num_threads = 16
        Int quast_num_threads = 16
        Int align_num_threads = 4

        Int medaka_rounds = 0
        String medaka_model = "r1041_e82_400bps_sup_v4.1.0"
        Boolean medaka_use_gpu = false

        Boolean flye_impute_params = true
        String? flye_read_mode

        RuntimeAttr? runtime_attr_genome_length
        RuntimeAttr? runtime_attr_merge_fastqs
        RuntimeAttr? runtime_attr_fastq_stats
        RuntimeAttr? runtime_attr_read_divergence
        RuntimeAttr? runtime_attr_flye
        RuntimeAttr? runtime_attr_medaka
        RuntimeAttr? runtime_attr_quast
        RuntimeAttr? runtime_attr_quast_summary
        RuntimeAttr? runtime_attr_align_paf
        RuntimeAttr? runtime_attr_paftools
    }

    # One sub-workflow call per sample. The resource requests are per *task*, not
    # per sample, so this does not multiply the cluster's size requirement; it
    # multiplies how many tasks are eligible to run at once, which is what gives the
    # scheduler something to pack.
    scatter (sample in samples) {
        call Single.ONTAssembleWithFlye as assemble {
            input:
                fastqs = sample.fastqs,
                ref_fasta = sample.ref_fasta,
                participant_name = sample.name,
                prefix = sample.name,

                flye_num_threads = flye_num_threads,
                quast_num_threads = quast_num_threads,
                align_num_threads = align_num_threads,

                flye_impute_params = flye_impute_params,
                flye_read_mode = flye_read_mode,

                medaka_rounds = medaka_rounds,
                medaka_model = medaka_model,
                medaka_use_gpu = medaka_use_gpu,

                runtime_attr_genome_length = runtime_attr_genome_length,
                runtime_attr_merge_fastqs = runtime_attr_merge_fastqs,
                runtime_attr_fastq_stats = runtime_attr_fastq_stats,
                runtime_attr_read_divergence = runtime_attr_read_divergence,
                runtime_attr_flye = runtime_attr_flye,
                runtime_attr_medaka = runtime_attr_medaka,
                runtime_attr_quast = runtime_attr_quast,
                runtime_attr_quast_summary = runtime_attr_quast_summary,
                runtime_attr_align_paf = runtime_attr_align_paf,
                runtime_attr_paftools = runtime_attr_paftools
        }
    }

    output {
        # Every array is in `samples` order, so index i is samples[i] throughout.
        Array[String] sample_names = assemble.participant

        Array[File] assemblies = assemble.asm_polished
        Array[File] assemblies_unpolished = assemble.asm_unpolished
        Array[File] assembly_infos = assemble.asm_info

        Array[File] pafs = assemble.paf
        Array[File] vcfs = assemble.paftools_vcf

        Array[File] quast_reports_html = assemble.quast_report_html
        Array[Map[String, String]] quast_summaries = assemble.quast_summary

        Array[Map[String, String]] flye_params = assemble.flye_params
        Array[Map[String, String]] read_stats = assemble.read_stats
    }
}
