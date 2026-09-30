version 1.0

import "../../../tasks/Utility/Utils.wdl" as Utils
import "../../../tasks/Assembly/Flye.wdl" as Flye
import "../../../tasks/Preprocessing/Medaka.wdl" as Medaka
import "../../../tasks/VariantCalling/CallAssemblyVariants.wdl" as AV
import "../../../tasks/QC/Quast.wdl" as Quast
import "../../../tasks/QC/ReadStats.wdl" as ReadStats

# Adapted from broadinstitute/long-read-pipelines
# wdl/pipelines/ONT/Assembly/ONTAssembleWithFlye.wdl.
# Licensed BSD-3-Clause; see wdl/LICENSE.

# Unlike upstream, medaka_rounds defaults to 0 (medaka is not in the image) and Flye's read
# mode, --asm-coverage and --iterations are imputed; flye_impute_params = false undoes that.

workflow ONTAssembleWithFlye {
    meta {
        description: "Single-sample de novo genome assembly from ONT reads. Merges one sample's flow cells, measures the read set, assembles with Flye, polishes with Medaka, evaluates against a reference with QUAST, and calls assembly-vs-reference variants with minimap2 and paftools."
    }
    parameter_meta {
        fastqs:              "basecalled ONT reads, one entry per flow cell; local paths or gs://, s3://, https:// URIs"

        ref_fasta:           "reference assembly for the species, used to estimate genome size and to call variants against"

        flye_num_threads:    "flye --threads; keep in step with runtime_attr_flye's cpu_cores"
        flye_extra_args:     "extra options for flye, appended after any imputed ones"

        flye_impute_params:  "derive flye's parameters from the reads; false restores upstream's command line exactly"
        read_chemistry:      "the reads' flow cell chemistry, e.g. 'R10.4.1' or 'R10.4.1 (LSK114, E8.2, 400 bps)'. This is what selects Flye's read mode, because Flye's guidance selects on chemistry. Unset keeps upstream's --nano-raw"
        flye_read_mode:      "explicit flye read type flag, e.g. '--nano-hq'; overrides the chemistry"
        flye_asm_coverage:   "explicit flye --asm-coverage; 0 disables coverage capping entirely"
        flye_iterations:     "explicit flye --iterations; overrides the rule based on medaka_rounds"
        flye_genome_size:    "explicit genome size in base pairs; overrides the estimate from ref_fasta"

        flye_asm_coverage_target:     "coverage to cap the initial disjointig stage at, and the threshold above which capping happens at all. 0, the default, means no cap: capping at Flye's documented 40x was measured to cost a third of the N50 on a 97x read set and to save fourteen minutes. See the input's comment"
        flye_nano_hq_max_divergence:  "pairwise read-read divergence above which flye_params records divergence_flag = high. Nothing branches on it; it is a QC annotation. The 0.10 default is Flye's --nano-hq band (<5% per read) in the units MeasureDivergence reports, and it is exceeded by repeat-rich regions regardless of read quality"

        medaka_model:        "Medaka polishing model name. Must match the reads' chemistry and basecaller: R10.4.1 needs an r1041_* model, and an r941_* model on R10.4.1 data degrades the consensus silently. Run `medaka tools list_models`"
        medaka_rounds:       "number of Medaka polishing rounds; 0 (the default here) passes the draft through and leaves Flye's own round as the polish. Requires medaka on the workers to raise, and it is not in this template's image; see tools/BUILDING.md"
        medaka_use_gpu:      "request a GPU for Medaka (needs a GPU node group in the Ray cluster)"

        participant_name:    "name of the participant from whom these samples were obtained"
        prefix:              "prefix for output files"

        quast_is_large:      "pass QUAST's --large, for genomes above ~100 Mbp"
        quast_num_threads:   "quast --threads; keep in step with runtime_attr_quast's cpu_cores"

        align_num_threads:   "minimap2 -t for the assembly-to-reference alignment; keep in step with runtime_attr_align_paf's cpu_cores"
        align_preset:        "minimap2 -x preset for the assembly-to-reference alignment; see the note above AlignAsPAF in CallAssemblyVariants.wdl"
    }

    input {
        Array[File]+ fastqs

        File ref_fasta

        Int flye_num_threads = 16
        String flye_extra_args = ""

        String? read_chemistry

        Boolean flye_impute_params = true
        String? flye_read_mode
        Int? flye_asm_coverage
        Int? flye_iterations
        Float? flye_genome_size

        Int flye_asm_coverage_target = 0

        Float flye_nano_hq_max_divergence = 0.10

        String medaka_model = "r1041_e82_400bps_sup_v4.1.0"
        Int medaka_rounds = 0
        Boolean medaka_use_gpu = false

        String participant_name
        String prefix

        Boolean quast_is_large = false
        Int quast_num_threads = 16

        Int align_num_threads = 4
        String align_preset = "asm20"

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

    call Utils.ComputeGenomeLength {
        input:
            fasta = ref_fasta,
            runtime_attr_override = runtime_attr_genome_length
    }

    call Utils.MergeFastqs {
        input:
            fastqs = fastqs,
            runtime_attr_override = runtime_attr_merge_fastqs
    }

    call ReadStats.FastqStats {
        input:
            fastq = MergeFastqs.merged_fastq,
            runtime_attr_override = runtime_attr_fastq_stats
    }

    call ReadStats.MeasureDivergence {
        input:
            fastq = MergeFastqs.merged_fastq,
            runtime_attr_override = runtime_attr_read_divergence
    }

    # ref_fasta's total length counts N gaps; pass flye_genome_size where that matters.
    Float genome_length = select_first([flye_genome_size, ComputeGenomeLength.length])
    Float read_coverage = FastqStats.total_bases / genome_length

    # Divergence tracks repeat content as much as read error, so it only flags flye_params.
    Boolean divergence_measured = MeasureDivergence.divergence >= 0.0
    Boolean divergence_disagrees = if divergence_measured
                                   then MeasureDivergence.divergence > flye_nano_hq_max_divergence
                                   else false

    # R10 in any spelling, or R9 basecalled by Guppy5+ or sup, gets --nano-hq, per Flye's USAGE.md.
    String chemistry_label = select_first([read_chemistry, ""])
    Boolean chemistry_is_r10 = sub(chemistry_label, "(?i).*r10.*", "HQ") == "HQ"
    Boolean chemistry_is_hq_r9 = sub(chemistry_label, "(?i).*r9.*(sup|guppy[5-9]).*", "HQ") == "HQ"
    String imputed_read_mode = if (chemistry_is_r10 || chemistry_is_hq_r9)
                               then "--nano-hq" else "--nano-raw"
    String resolved_read_mode = if defined(flye_read_mode)
                                then select_first([flye_read_mode])
                                else (if flye_impute_params then imputed_read_mode else "--nano-raw")

    Int imputed_asm_coverage = if read_coverage > flye_asm_coverage_target
                               then flye_asm_coverage_target
                               else 0
    Int resolved_asm_coverage = if defined(flye_asm_coverage)
                                then select_first([flye_asm_coverage])
                                else (if flye_impute_params then imputed_asm_coverage else 0)

    # Medaka redoes Flye's consensus, so Flye skips polishing only when Medaka runs.
    Int imputed_iterations = if medaka_rounds > 0 then 0 else 1

    # Flye rejects --asm-coverage without --genome-size, so both or neither.
    String iterations_arg = if defined(flye_iterations)
                            then " --iterations ~{select_first([flye_iterations])}"
                            else (if flye_impute_params then " --iterations ~{imputed_iterations}" else "")
    String asm_coverage_arg = if resolved_asm_coverage > 0
                              then " --asm-coverage ~{resolved_asm_coverage} --genome-size ~{ceil(genome_length)}"
                              else ""

    String resolved_flye_extra_args = sub(sub(iterations_arg + asm_coverage_arg + " " + flye_extra_args, "^ +", ""), " +$", "")

    call Flye.Flye {
        input:
            reads = MergeFastqs.merged_fastq,
            genome_size = genome_length,
            prefix = prefix,
            num_threads = flye_num_threads,
            read_mode = resolved_read_mode,
            extra_args = resolved_flye_extra_args,
            runtime_attr_override = runtime_attr_flye
    }

    call Medaka.MedakaPolish {
        input:
            basecalled_reads = MergeFastqs.merged_fastq,
            draft_assembly = Flye.fa,
            model = medaka_model,
            prefix = basename(Flye.fa, ".fa") + ".consensus",
            n_rounds = medaka_rounds,
            use_gpu = medaka_use_gpu,
            runtime_attr_override = runtime_attr_medaka
    }

    # Draft and polished in one QUAST run when medaka ran (with 0 rounds they are the same file).
    # Polishing moves bases, not contig boundaries: read NGA50 and error rates, not N50.
    Boolean quast_compares_arms = medaka_rounds > 0
    Array[File] quast_assemblies = if quast_compares_arms
                                   then [ Flye.fa, MedakaPolish.polished_assembly ]
                                   else [ MedakaPolish.polished_assembly ]

    call Quast.Quast {
        input:
            ref = ref_fasta,
            assemblies = quast_assemblies,
            is_large = quast_is_large,
            num_threads = quast_num_threads,
            runtime_attr_override = runtime_attr_quast
    }

    call AV.CallAssemblyVariants {
        input:
            asm_fasta = MedakaPolish.polished_assembly,
            ref_fasta = ref_fasta,
            participant_name = participant_name,
            prefix = prefix + ".flye",
            align_num_threads = align_num_threads,
            align_preset = align_preset,
            runtime_attr_align = runtime_attr_align_paf,
            runtime_attr_paftools = runtime_attr_paftools
    }

    call Quast.SummarizeQuastReport as summaryQ {
        input:
            quast_report_txt = Quast.report_txt,
            runtime_attr_override = runtime_attr_quast_summary
    }

    # One report_map per assembly, in quast_assemblies order, so the final assembly is last.
    Int quast_final_column = if quast_compares_arms then 1 else 0

    Map[String, String] q_metrics = read_map(summaryQ.quast_metrics[quast_final_column])
    Map[String, String] q_metrics_draft = read_map(summaryQ.quast_metrics[0])

    output {
        # Echoed so a scatter over samples yields labels aligned with the file arrays.
        String participant = participant_name

        File asm_unpolished = Flye.fa
        File asm_polished = MedakaPolish.polished_assembly

        File asm_info = Flye.assembly_info
        File flye_log = Flye.log

        File paf = CallAssemblyVariants.paf
        File paftools_vcf = CallAssemblyVariants.paftools_vcf

        File quast_report_html = Quast.report_html
        File quast_report_txt = Quast.report_txt

        Map[String, String] quast_summary = q_metrics

        # Equal to quast_summary when medaka did not run; see quast_compared_arms.
        Map[String, String] quast_summary_unpolished = q_metrics_draft

        # Setting flye_read_mode, flye_asm_coverage and flye_iterations from these reproduces
        # the run. genome_size is "" when Flye got no --genome-size.
        Map[String, String] flye_params = {
            "read_mode":    resolved_read_mode,
            "read_mode_from": if defined(flye_read_mode) then "explicit"
                              else (if !flye_impute_params then "upstream default"
                                    else (if imputed_read_mode == "--nano-hq" then "chemistry"
                                          else "chemistry unset or not high-accuracy")),
            "chemistry":    chemistry_label,
            "extra_args":   resolved_flye_extra_args,
            "asm_coverage": "~{resolved_asm_coverage}",
            "iterations":   "~{sub(iterations_arg, '^ --iterations ', '')}",
            "genome_size":  if resolved_asm_coverage > 0 then "~{ceil(genome_length)}" else "",
            "divergence_flag": if !divergence_measured then "not measured"
                               else (if divergence_disagrees then "high" else "ok"),
            "imputed":      "~{flye_impute_params}"
        }

        Boolean quast_compared_arms = quast_compares_arms

        # pairwise_divergence is -1 when MeasureDivergence found too few overlaps.
        Map[String, String] read_stats = {
            "num_reads":           "~{FastqStats.num_reads}",
            "total_bases":         "~{FastqStats.total_bases}",
            "read_n50":            "~{FastqStats.read_n50}",
            "coverage":            "~{read_coverage}",
            "pairwise_divergence": "~{MeasureDivergence.divergence}",
            "divergence_overlaps": "~{MeasureDivergence.num_overlaps}"
        }
    }
}
