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
#
# The call graph is unchanged: merge reads, estimate genome length, Flye assemble,
# Medaka polish, QUAST evaluate, minimap2 + paftools call variants. The portability
# changes below would apply equally to upstream's sibling ONTAssembleWithCanu.wdl (not
# shipped here), because they come from running outside Cromwell on GCP:
#
#   * `String gcs_fastq_dir` + `Utils.ListFilesOfType` (which shells out to
#     `gsutil ls`) became `Array[File]+ fastqs`. miniwdl localizes gs://, s3:// and
#     https:// URIs for File inputs itself, so cloud input still works and local files
#     work too. The input is also explicit, instead of whatever is in that bucket today.
#   * `String gcs_out_root_dir` and the six `FinalizeToFile` calls (each a
#     `gsutil cp`) are gone. miniwdl already collects every declared output into the
#     run directory's `out/` tree and reports absolute paths in its JSON. Point --dir at
#     cloud storage, or copy `out/` afterwards, to get the same effect.
#   * `File ref_map_file` (a TSV read with read_map) became `File ref_fasta`. Only the
#     `fasta` key was ever used; callers holding a Broad ref_map should pass its fasta
#     entry.
#   * Per-task `RuntimeAttr` overrides are exposed at the workflow level. Flye's
#     default request is 16 cores and about 100 GiB, sized for the Broad's fleet rather
#     than for a small Ray cluster, so inputs.chr20*.json (and the README's Step 5
#     inputs) carry a set sized to this template's 32-vCPU / 128 GiB workers.
#
# Two upstream bugs are fixed, both visible only in output filenames:
#
#   * `CallAssemblyVariants` was called with `prefix = prefix + ".canu"`, a copy-paste
#     from ONTAssembleWithCanu.wdl. It is ".flye" here, so the PAF and VCF name the
#     right assembler.
#   * `basename(Flye.fa, ".fasta")` stripped nothing, because Flye.wdl emits
#     `*.flye.fa`, so the polished assembly came out as
#     `<prefix>.flye.fa.consensus.fasta`. Stripping ".fa" gives
#     `<prefix>.flye.consensus.fasta`.
#
# `genome_size` stays a Float in base pairs, as upstream has it, because Flye's
# `--genome-size` takes base pairs (Canu's `genomeSize` takes megabases).
#
# It is the sum of every `@SQ` `LN:` in ref_fasta, which counts assembly gaps: a
# reference with N runs reports a genome longer than its assemblable sequence, so
# `--genome-size` runs high and `read_coverage` low. For GRCh38 chr20 that is a few
# percent and moves nothing that matters; for a reference with large modelled
# centromeres, or a whole-genome input carrying alts and decoys, it does. Pass
# `flye_genome_size` explicitly when it matters.
#
# ---------------------------------------------------------------------------------
# Scientific divergences from upstream: five, all reversible from the inputs. Only the
# first two change what a default run produces.
#
#   * `medaka_rounds` is 0 where upstream hardcodes 3, so this workflow's declared output
#     is a Flye draft carrying Flye's own single polishing round, where upstream's is a
#     medaka consensus. The reason is packaging: medaka is not in the cluster image
#     (1.2 GB even with CPU-only torch; tools/Dockerfile.medaka-gpu is its own image),
#     so under `--container-runtime none` a default of 3 would exit 127 *after* the
#     assembly had succeeded. Raise it with that image mapped under
#     `--container-runtime ray`, and pick a model matching your chemistry.
#   * Flye's read mode, --asm-coverage and --iterations are set explicitly, where
#     upstream leaves them to Flye; the rules follow. `flye_impute_params = false`
#     restores upstream's command line exactly.
#   * `medaka_model` defaults to an r1041 model rather than upstream's r941. Dormant
#     while medaka_rounds is 0; upstream's default would silently degrade R10.4.1 data.
#     See the input's comment.
#   * QUAST evaluates both the draft and the polished consensus when both exist, where
#     upstream evaluates only the polished one. Same tool, same flags, one extra column;
#     see the Quast call site.
#   * The assembly-to-reference minimap2 preset is an input rather than a hardcoded
#     `asm20`. The default is still `asm20`; see the note in CallAssemblyVariants.wdl.
#
# Upstream invokes `flye --nano-raw <reads> --threads N` and nothing more: no
# --genome-size, no --asm-coverage, default polishing. Run that way on an earlier
# 3.52 Gbp (54.6x) cut of the HG002 chr20 R10.4.1 sup reads, Flye spent 1h33m finding
# overlaps, 1h25m+ assembling disjointigs and over seven hours polishing.
#
# The rules, in the declarations below rather than hidden in a task:
#
#   read mode       Chosen by chemistry, which is what Flye's guidance selects on: its
#                   USAGE.md says "For R10 data, use --nano-hq", and separately sends
#                   R9 basecalled by Guppy5+ or in sup mode to the same mode. An unset
#                   chemistry keeps upstream's --nano-raw rather than guessing. On the
#                   full chr20 HG002 reads on m5.8xlarge, --nano-raw took 14h44m in Flye
#                   and --nano-hq 1h55m33s, for N50s within 0.05% of each other.
#
#                   It used to be gated on measured pairwise divergence against a 0.10
#                   threshold, and that rule fails on a full chromosome. Same sample,
#                   chemistry, basecaller and published FASTQ derivation, three region
#                   sizes:
#
#                     chr20:1-3 Mbp        18,697 overlaps   0.0721  -> --nano-hq
#                     chr20:1-11 Mbp      229,806 overlaps   0.0986  -> --nano-hq, by 1.4%
#                     chr20:1-64.4 Mbp  6,379,175 overlaps   0.1532  -> --nano-raw
#
#                   The estimator tracks the region's repeat content, not the reads' error
#                   rate. The first two regions are p-arm euchromatin; the third spans the
#                   centromere, whose alpha-satellite generates spurious cross-alignments
#                   between non-homologous copies that ava-ont's `dv` counts like any other
#                   overlap. The rule sent R10.4.1 dorado sup reads to --nano-raw, which
#                   Flye's table labels "ONT regular reads, pre-Guppy5 (<20% error)".
#
#                   MeasureDivergence still runs and is still emitted in read_stats, as an
#                   annotation: flye_params carries divergence_flag = ok | high |
#                   "not measured", and nothing branches on it. flye_params also records
#                   read_mode_from, so a completed run says whether its mode came from the
#                   chemistry, an explicit override, or upstream's default.
#
#   --asm-coverage  Off by default. When `flye_asm_coverage_target` is set and measured
#                   coverage exceeds it, emitted together with --genome-size (Flye
#                   hard-errors on one without the other). It caps the initial disjointig
#                   stage only.
#
#                   The default was 40x, Flye's documented number, until four full chr20
#                   runs measured what it costs, one variable at a time, all on
#                   c6i.16xlarge / 62 cores except the historical run (m5 / 30):
#
#                                    no cap                      --asm-coverage 40
#                     --nano-raw     N50 33.28 Mbp, 14h44m (m5)  N50 11.06 Mbp, 2h00m47s
#                     --nano-hq      N50 33.27 Mbp, 1h19m25s     N50 11.06 Mbp, 1h04m49s
#
#                   The cap cut the N50 to a third and saved 14.6 minutes. Both capped runs
#                   land at the same 11.06 Mbp N50 whichever read mode they use, and both
#                   uncapped runs reach 33.27-33.28 Mbp with L90 = 2, the chromosome's two
#                   arms. Uncapped also wins genome fraction, NGA50 and misassembly count.
#
#                   Flye's "typically 40x is enough" speaks to sufficiency; at 97x the
#                   discarded depth is what resolves repeats. The two levers act
#                   separately: the read mode drives wall clock (1.86x with the cap held
#                   constant), the cap drives contiguity (3x with the mode held).
#
#   --iterations    0 when Medaka will run, because Medaka redoes that consensus and
#                   Flye's rounds are the largest block of wall clock in the pipeline;
#                   otherwise 1, so the total number of polishing passes never reaches
#                   zero.
#
# Every one of these is also an explicit input. Any `flye_read_mode`,
# `flye_asm_coverage`, `flye_iterations` or `flye_genome_size` that is set wins over the
# rule, `flye_impute_params = false` restores upstream's command line exactly, and
# `flye_params` / `read_stats` are emitted as outputs, so a completed run records the
# values it chose in a form that can be pasted back into an inputs JSON.

workflow ONTAssembleWithFlye {
    meta {
        # Upstream's description reads "merges multiple samples into a single BAM prior
        # to genome assembly". Neither half is true of this workflow: it merges FASTQs,
        # never a BAM, and it merges flow cells belonging to *one* sample, and merging
        # samples before a de novo assembly would collapse different genomes into one
        # consensus. Corrected here because `miniwdl check`, the Terra UI and every doc
        # generator print this string first.
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

        # 0 means no cap, the default, because capping was measured and it lost. Flye's
        # USAGE.md says "typically, 40x longest reads is enough to produce good
        # disjointigs", which speaks to sufficiency; on a 97x read set the difference is
        # most of the assembly. Two of the four full chr20 runs in the header:
        #
        #   --nano-hq --asm-coverage 40   N50 11,060,138   L90 8   1h04m49s
        #   --nano-hq no cap              N50 33,267,366   L90 2   1h19m25s
        #
        # A third of the contiguity for fourteen and a half minutes. Uncapped also wins
        # on genome fraction (95.945 vs 95.816), NGA50 (2,082,053 vs 1,829,815) and
        # misassemblies (122 vs 133), so the cap paid on no axis.
        #
        # Set this below your read coverage to cap the *initial disjointig stage* at it;
        # nothing is capped when coverage is at or under the target. That stage is the one
        # place --asm-coverage acts. Worth doing when it is the bottleneck; it was about a
        # third of the task here.
        Int flye_asm_coverage_target = 0

        # Pairwise, not per-read; see the header. Flye documents --nano-hq for
        # reads under 5% per-read error, which is ~0.10 in these units.
        Float flye_nano_hq_max_divergence = 0.10

        # Upstream's default is `r941_prom_high_g360`, an R9.4.1 / Guppy-3.6 model.
        # This template's reads are R10.4.1 dorado sup v4.1.0, and medaka does not
        # refuse a mismatched model: it can produce a worse consensus than the draft,
        # and says nothing. Defaulting to the model that matches the shipped reads
        # means `medaka_rounds > 0` is correct for them without also changing this.
        # Change both together for other data.
        String medaka_model = "r1041_e82_400bps_sup_v4.1.0"
        # Upstream hardcodes 3. This port defaults to 0 for packaging, not science: medaka
        # is not in this template's cluster image (tools/BUILDING.md says why), so under
        # `--container-runtime none` a default of 3 would fail with exit 127 *after* the
        # assembly had already succeeded. Total polishing passes still never reach zero,
        # because Flye's own polishing round is imputed back on (see imputed_iterations
        # below). Opt in with medaka_rounds > 0 once medaka is on the workers.
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

    # ---- imputed Flye parameters; see the header for the reasoning behind each rule --

    Float genome_length = select_first([flye_genome_size, ComputeGenomeLength.length])
    Float read_coverage = FastqStats.total_bases / genome_length

    # Read mode comes from the chemistry, which is what Flye's own guidance selects on
    # ("For R10 data, use --nano-hq"). The header has the measurement behind not deriving
    # it from MeasureDivergence, which is kept as an observation: emitted in read_stats, and
    # compared against the mode's tolerance only to raise a flag in flye_params. Nothing
    # branches on it.
    Boolean divergence_measured = MeasureDivergence.divergence >= 0.0
    Boolean divergence_disagrees = if divergence_measured
                                   then MeasureDivergence.divergence > flye_nano_hq_max_divergence
                                   else false

    # R10 in any spelling (R10.4.1, r10.4.1, "R10.4.1 (LSK114, E8.2, 400 bps)") means --nano-hq,
    # as does an R9 flow cell basecalled by Guppy5+ or in sup mode, which is the other case
    # Flye's USAGE.md sends to --nano-hq. Anything else, including an unset chemistry, keeps
    # upstream's --nano-raw: a caller who has not said what the reads are should get the
    # conservative mode rather than an inference.
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

    Int imputed_iterations = if medaka_rounds > 0 then 0 else 1

    # Each fragment is empty when its flag is not being set, so with imputation off and
    # nothing explicit the whole string collapses to flye_extra_args unchanged. Flye
    # rejects --asm-coverage without --genome-size, so the two are emitted together or
    # not at all.
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

    # Evaluate the draft and the polished assembly in one QUAST run when both exist.
    # Upstream evaluates only the polished assembly, which leaves "was the polishing
    # worth it?" unanswerable from the pipeline's own output, short of a second full
    # assembly. QUAST is built for this (one reference, one set of thresholds, one
    # report, one column per assembly), so the comparison costs nothing extra and,
    # because both columns come from the same run, has no between-run variation.
    #
    # The guard matters. With medaka_rounds = 0, MedakaPolish copies the draft
    # through untouched, so passing both would hand QUAST two byte-identical files
    # and produce a report whose two columns agree by construction.
    #
    # This compares Flye's draft against the Medaka-polished consensus. It says nothing
    # about whether Flye's *own* polishing round earned its wall clock, because
    # `--iterations 0` vs 1 changes the draft itself, so that comparison needs two
    # assemblies. Polishing moves bases and not contig boundaries, so read the
    # alignment-based metrics (NGA50, genome fraction, mismatches and indels per
    # 100 kbp) and expect N50 to barely move.
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

    # SummarizeQuastReport emits one report_map_<j>.txt per assembly column, in the
    # order `quast_assemblies` was built, and glob() returns them in that order. When
    # both arms ran, column 0 is Flye's draft and column 1 the polished consensus;
    # when only one did, both indices name the same file. Keeping `quast_summary`
    # pointed at the *final* assembly means adding the draft column did not silently
    # change what that output means.
    Int quast_final_column = if quast_compares_arms then 1 else 0

    Map[String, String] q_metrics = read_map(summaryQ.quast_metrics[quast_final_column])
    Map[String, String] q_metrics_draft = read_map(summaryQ.quast_metrics[0])

    output {
        # Echoed so a scatter over samples produces an ordered label array alongside
        # the ordered file arrays; otherwise a cohort's outputs are N anonymous
        # lists whose correspondence lives only in the caller's input order.
        String participant = participant_name

        File asm_unpolished = Flye.fa
        File asm_polished = MedakaPolish.polished_assembly

        # Flye's per-contig table (length, coverage, circularity, repeat flag) and
        # its log with the per-stage timings. Neither is large and both are what you
        # actually read when an assembly looks wrong.
        File asm_info = Flye.assembly_info
        File flye_log = Flye.log

        File paf = CallAssemblyVariants.paf
        File paftools_vcf = CallAssemblyVariants.paftools_vcf

        File quast_report_html = Quast.report_html
        File quast_report_txt = Quast.report_txt

        Map[String, String] quast_summary = q_metrics

        # The draft's column of the same report. Equal to quast_summary when medaka
        # did not run, because then there was only one assembly to evaluate;
        # `quast_compared_arms` below says which case a given run was.
        Map[String, String] quast_summary_unpolished = q_metrics_draft

        # The run's own record of what it decided. Setting flye_read_mode,
        # flye_asm_coverage and flye_iterations from these three values reproduces this
        # assembly without re-measuring anything, and pins it against a later change
        # to the rules above.
        #
        # `genome_size` is the empty string when --genome-size was not on the command
        # line, which is the case whenever --asm-coverage was not emitted (Flye takes
        # the two together or not at all). It used to report ceil(genome_length)
        # unconditionally, so a run below the coverage target recorded a genome size
        # Flye never received, in the output whose entire purpose is reproducing the
        # run. `extra_args` is the authoritative copy either way.
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
            # QC only, nothing branches on it. "high" means the measured pairwise
            # divergence exceeded the --nano-hq band, which on a repeat-rich region
            # happens regardless of read quality; see the header.
            "divergence_flag": if !divergence_measured then "not measured"
                               else (if divergence_disagrees then "high" else "ok"),
            "imputed":      "~{flye_impute_params}"
        }

        # Whether quast_summary and quast_summary_unpolished are two columns (draft
        # and polished) or one assembly reported twice. Reading the pair without
        # checking this is how a "polishing changed nothing" conclusion gets drawn
        # from a run where no polishing was configured.
        Boolean quast_compared_arms = quast_compares_arms

        # The measurements the rules were applied to. `pairwise_divergence` is -1 when
        # `divergence_overlaps` fell below MeasureDivergence's minimum, in which case
        # flye_params' divergence_flag reads "not measured".
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
