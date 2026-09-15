version 1.0

import "../../structs/Structs.wdl"

# Measure the properties of a read set that an assembler's parameters should be derived
# from. Not adapted from upstream: broadinstitute/long-read-pipelines has no equivalent,
# because on Cromwell the parameters are hardcoded per pipeline.
#
# The motivation is a measured one. ONTAssembleWithFlye ran HG002 chr20 (3.52 Gbp, 54.6x)
# with Flye's parameters entirely undeclared: no --genome-size, no --asm-coverage, and
# `--nano-raw` on dorado *sup* basecalls. Flye then spent 1h33m finding overlaps, 1h25m+
# assembling disjointigs, and over seven hours polishing. Every one of those is sensitive
# to a parameter the pipeline already had the information to set; it just never
# measured it.
#
# Two tasks instead of one, because they cost very different amounts and want different
# resources: FastqStats is a single gzip pass on one core, MeasureDivergence runs an
# aligner. They take the same input and no output of one feeds the other, so the compiled
# runtime submits them concurrently.
#
# Both assume 4-line FASTQ records, which is what every ONT basecaller emits. Wrapped
# sequence lines would need a state machine to parse and nothing in this corpus produces
# them.
#
# ---------------------------------------------------------------------------------
# What MeasureDivergence's number is good for, and what it is not.
#
# It is a cheap consistency check on a read set's label, and that is the whole claim.
# Flye's read-mode flags select different algorithms, and Flye's own guidance selects
# between them by *chemistry and basecaller* ("for R10 data, use --nano-hq"), not by
# a divergence reading. Both of those are known metadata for any real read set. This
# task exists so that a read set which does not behave like its label is visible before
# a fourteen-hour assembly rather than after it, and so that the choice is recorded in
# the outputs. It is not evidence that overrules the tool's documented guidance.
#
# It is also weaker than a bare number suggests, in three ways, so read it before
# quoting a number:
#
#   * `dv:f` is minimap2's *approximate* per-base divergence, estimated from the
#     minimizer chain. Without `-c` there is no base-level alignment behind it. It is
#     the right tag for this purpose and far better than the matches/blocklen formula
#     it replaced (see the command block), but it is an estimate.
#   * `ava-ont` is tuned to *find overlaps*, not to measure them. Its divergences skew
#     high on ONT because homopolymer indels dominate the error and are not
#     gap-compressed here, and because chimeric reads and adapters survive into the
#     probe.
#   * The result is a median over overlapping read pairs in one subsample, so it
#     inherits whatever that subsample's coverage and repeat content give it.
#
# Concretely: HG002's dorado sup basecalls measure 0.061 pairwise here, which the x2
# rule reads as ~3% per read. Two things explain the gap from the ~1% "sup" suggests,
# and neither is a bad basecaller:
#
#   * The published set is quality-filtered at Q10, not Q20. The source CRAMs carry
#     `@PG ... CL:samtools view -e [qs] >= 10 --output ...pass.cram`, so `.pass` admits
#     reads down to 10% mean error, and the median overlap is drawn from all of them.
#   * The x2 conversion is an upper bound rather than a calibration. It holds when the
#     two reads' errors are independent; ONT's dominant errors are homopolymer and
#     context effects that are *correlated* between reads, so two reads often make the
#     same wrong call and agree. Real pairwise divergence therefore runs below 2x the
#     per-read rate, and dividing by 2 understates per-read error.
#
# Both biases push the same way, so treat the derived per-read figure as an upper
# bound on a lower bound. Three per cent is well inside Flye's --nano-hq band (<5% per
# read), the measurement and the chemistry agree, and the pipeline picks --nano-hq
# either way. It would take a much larger disagreement than this, corroborated by
# something with base-level alignment behind it, to conclude that a basecaller's Q
# scores are wrong.

task FastqStats {

    meta {
        description: "Read count, total bases and read N50 for a FASTQ file"
    }

    parameter_meta {
        fastq:  "FASTQ file, optionally gzipped"
        runtime_attr_override: "Override the default runtime attributes."
    }

    input {
        File fastq

        RuntimeAttr? runtime_attr_override
    }

    Int disk_size = 2 * ceil(size(fastq, "GB"))

    command <<<
        set -euxo pipefail

        # One pass for the length distribution; everything below is derived from it
        # instead of from the FASTQ, so the compressed file is read exactly once.
        gzip -dcf ~{fastq} | awk 'NR % 4 == 2 { print length($0) }' > lengths.txt

        # printf "%.0f", and neither `print` nor "%d". mawk is /usr/bin/awk on Debian and
        # Ubuntu, so on the cluster image, and it gets large integers wrong in two different
        # ways. `print` routes anything above INT_MAX through OFMT, turning 3,519,431,361 bases
        # into "3.51943e+09", which read_int() rejects and the task fails loudly. "%d" is worse:
        # mawk 1.3.4-20200120 converts through a 32-bit int, so 6,150,000,000 comes back as
        # exactly 2147483647 and the task *succeeds* with a wrong number. Verified in the
        # template's own image; a 6.1 Gbp chr20 read set is above the clamp and this demo's
        # 178 Mbp one is not, so only a full-scale run shows it. "%.0f" formats the double
        # directly and is exact to 2^53. BSD awk gets all three right, so a local test cannot
        # see any of this.
        awk 'END { printf "%.0f\n", NR }' lengths.txt > num_reads.txt
        awk '{ b += $1 } END { printf "%.0f\n", b }' lengths.txt > total_bases.txt

        # N50: the length L at which reads of at least L account for half the bases.
        # `sort -rn` puts the longest first, so the running total first crosses the
        # halfway mark at exactly that read; every line after it also satisfies the
        # condition but is shorter, hence taking the first. %.0f and not %d because half
        # a base count is not integral, and OFMT would round it to six significant figures.
        sort -rn lengths.txt > sorted_lengths.txt
        awk -v half="$(awk '{ b += $1 } END { printf "%.0f\n", b / 2 }' lengths.txt)" '{ acc += $1 } acc >= half { print $1 }' sorted_lengths.txt > n50_candidates.txt

        # An empty read set would otherwise leave read_int() with nothing to parse. A
        # trailing fallback is cheaper than branching, and is only ever reached when
        # the candidate list is empty.
        printf '0\n' >> n50_candidates.txt
        head -1 n50_candidates.txt > read_n50.txt
    >>>

    output {
        Int num_reads = read_int("num_reads.txt")
        Int total_bases = read_int("total_bases.txt")
        Int read_n50 = read_int("read_n50.txt")
    }

    #########################
    RuntimeAttr default_attr = object {
        cpu_cores:          1,
        mem_gb:             4,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  0,
        max_retries:        0,
        docker:             "us.gcr.io/broad-dsp-lrma/lr-utils:0.1.8"
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

task MeasureDivergence {

    meta {
        description: "Median pairwise sequence divergence between overlapping reads, by all-vs-all alignment of a subsample"
    }

    parameter_meta {
        fastq:            "FASTQ file, optionally gzipped"
        sample_every_nth: "keep 1 read in N for the all-vs-all probe; raise it for whole-genome inputs"
        min_overlap_bp:   "ignore alignment blocks shorter than this, which are mostly noise"
        min_overlaps:     "report -1 instead of a median derived from fewer overlaps than this"
        num_threads:      "minimap2 -t; keep in step with runtime cpu_cores"
        runtime_attr_override: "Override the default runtime attributes."
    }

    input {
        File fastq

        Int sample_every_nth = 10
        Int min_overlap_bp = 1000
        Int min_overlaps = 100

        # Feeds default_attr's cpu_cores below, so the reservation and the flag agree unless a
        # runtime_attr_override disagrees with it, in which case set both, as AlignAsPAF's
        # num_cpus has the same arrangement for the same reason.
        Int num_threads = 4

        RuntimeAttr? runtime_attr_override
    }

    Int disk_size = 3 * ceil(size(fastq, "GB"))

    command <<<
        set -euxo pipefail

        # Subsample by stride and not by taking a prefix, so the probe spans the
        # whole file: basecaller output order tracks run time, and quality drifts over
        # a run. `keep` is only reassigned on a header line, so it carries across all
        # four lines of a record.
        #
        # A stride is the right shape here because all-vs-all needs the *subsample* to
        # retain enough depth for reads to overlap each other at all: it divides the
        # original coverage by N. At 54x and N=10 that leaves 5.4x, which is ample; a
        # 10x input would leave 1x, find almost nothing, and fall through to the -1
        # path below instead of reporting a median off a handful of alignments.
        gzip -dcf ~{fastq} | awk -v stride=~{sample_every_nth} 'NR % 4 == 1 { keep = (++rec % stride == 0) } keep' > sample.fq

        minimap2 -x ava-ont -t ~{num_threads} sample.fq sample.fq > overlaps.paf

        # minimap2's own `dv:f` tag ("approximate per-base sequence divergence"), which is
        # exactly this quantity and needs no base-level alignment.
        #
        # NOT `1 - matches/blocklen`, which looks equivalent and is not: without `-c`,
        # column 10 counts matching *seed* bases and not aligned matching bases, so it
        # reads far too low. Measured, on reads simulated at a known 8% error rate: that
        # formula gave 0.78 (worse than two random sequences) where `dv:f` gave 0.1532
        # against the ~0.154 the error model implies. (`de:f` is the gap-compressed variant
        # emitted instead when `-c` is given; matching either costs one character.)
        #
        # Column 1 against 6 drops self-hits, which `ava-ont` implies -X against but which
        # cost nothing to exclude explicitly. `match` instead of a field loop because a
        # `for` header's semicolons would confuse the command scanner that decides which
        # tool wheels this task needs.
        awk -v minlen=~{min_overlap_bp} '$1 != $6 { if ($11 >= minlen) if (match($0, /d[ve]:f:[0-9.]+/)) print substr($0, RSTART + 5, RLENGTH - 5) }' overlaps.paf > divergences.txt
        # LC_ALL=C for the same reason Paftools' sort has it: these are decimal values,
        # and a locale whose decimal separator is a comma makes `sort -n` read 0.061 as
        # 0, which silently returns a median of the wrong reads.
        LC_ALL=C sort -n divergences.txt > sorted_divergences.txt

        awk 'END { print NR + 0 }' sorted_divergences.txt > num_overlaps.txt
        awk 'END { print int((NR + 1) / 2) }' sorted_divergences.txt > median_index.txt

        # -1 is the "no opinion" signal the caller branches on: too few overlaps to
        # trust a median, which is the expected outcome on a low-coverage input and
        # must not be confused with a genuinely low divergence.
        awk -v k="$(cat median_index.txt)" -v n="$(cat num_overlaps.txt)" -v need=~{min_overlaps} 'NR == k { if (n >= need) print }' sorted_divergences.txt > median_candidates.txt
        printf -- '-1\n' >> median_candidates.txt
        head -1 median_candidates.txt > divergence.txt
    >>>

    output {
        Float divergence = read_float("divergence.txt")
        Int num_overlaps = read_int("num_overlaps.txt")
    }

    #########################
    RuntimeAttr default_attr = object {
        cpu_cores:          num_threads,
        mem_gb:             16,
        disk_gb:            disk_size,
        boot_disk_gb:       25,
        preemptible_tries:  0,
        max_retries:        0,
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
