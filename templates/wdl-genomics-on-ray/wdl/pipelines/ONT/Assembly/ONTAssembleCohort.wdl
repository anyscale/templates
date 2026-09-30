version 1.0

import "ONTAssembleWithFlye.wdl" as Single

# RuntimeAttr arrives with the sub-workflow (WDL 1.0 hoists imported structs); importing
# Structs.wdl too trips miniwdl's UnusedImport lint.

# One ONTAssembleWithFlye per sample in one workflow, so the whole cohort draws on one pool.
# Anything that varies per sample belongs in Sample; the inputs here apply to every sample.

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
        read_chemistry:      "flow cell chemistry shared by every sample, e.g. 'R10.4.1'. This is what selects Flye's read mode. A cohort with mixed chemistry should not share one workflow: the model and the mode would be wrong for some of it"
        flye_read_mode:      "explicit flye read type flag for every sample; overrides the chemistry"
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
        String? read_chemistry
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
                read_chemistry = read_chemistry,
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
