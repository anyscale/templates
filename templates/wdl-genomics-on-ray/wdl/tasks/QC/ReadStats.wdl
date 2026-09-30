version 1.0

import "../../structs/Structs.wdl"

# Read-set measurements for Flye's parameters; both tasks assume 4-line FASTQ records.
# MeasureDivergence is QC only: ava-ont's approximate dv:f also tracks repeat content.

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

        # One pass over the FASTQ; everything below derives from it.
        gzip -dcf ~{fastq} | awk 'NR % 4 == 2 { print length($0) }' > lengths.txt

        # printf "%.0f", not print or "%d": mawk (Debian's awk) mangles integers above INT_MAX.
        awk 'END { printf "%.0f\n", NR }' lengths.txt > num_reads.txt
        awk '{ b += $1 } END { printf "%.0f\n", b }' lengths.txt > total_bases.txt

        # N50: the first read, longest first, where the running total crosses half the bases.
        sort -rn lengths.txt > sorted_lengths.txt
        awk -v half="$(awk '{ b += $1 } END { printf "%.0f\n", b / 2 }' lengths.txt)" '{ acc += $1 } acc >= half { print $1 }' sorted_lengths.txt > n50_candidates.txt

        # Fallback so read_int() gets a value on an empty read set.
        printf '0\n' >> n50_candidates.txt
        head -1 n50_candidates.txt > read_n50.txt
    >>>

    output {
        Int num_reads = read_int("num_reads.txt")
        Int total_bases = read_int("total_bases.txt")
        Int read_n50 = read_int("read_n50.txt")
    }

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

        # Also default_attr's cpu_cores; a runtime_attr_override setting cpu_cores should match.
        Int num_threads = 4

        RuntimeAttr? runtime_attr_override
    }

    Int disk_size = 3 * ceil(size(fastq, "GB"))

    command <<<
        set -euxo pipefail

        # Subsample by stride, not a prefix: read quality drifts over a run.
        gzip -dcf ~{fastq} | awk -v stride=~{sample_every_nth} 'NR % 4 == 1 { keep = (++rec % stride == 0) } keep' > sample.fq

        minimap2 -x ava-ont -t ~{num_threads} sample.fq sample.fq > overlaps.paf

        # Divergence is minimap2's dv:f tag. Not 1 - matches/blocklen: without -c that counts seed bases.
        # match() rather than a for loop, whose semicolons confuse the tool-wheel command scanner.
        awk -v minlen=~{min_overlap_bp} '$1 != $6 { if ($11 >= minlen) if (match($0, /d[ve]:f:[0-9.]+/)) print substr($0, RSTART + 5, RLENGTH - 5) }' overlaps.paf > divergences.txt
        # LC_ALL=C: a comma-decimal locale breaks sort -n.
        LC_ALL=C sort -n divergences.txt > sorted_divergences.txt

        awk 'END { print NR + 0 }' sorted_divergences.txt > num_overlaps.txt
        awk 'END { print int((NR + 1) / 2) }' sorted_divergences.txt > median_index.txt

        # -1 means too few overlaps to trust a median.
        awk -v k="$(cat median_index.txt)" -v n="$(cat num_overlaps.txt)" -v need=~{min_overlaps} 'NR == k { if (n >= need) print }' sorted_divergences.txt > median_candidates.txt
        printf -- '-1\n' >> median_candidates.txt
        head -1 median_candidates.txt > divergence.txt
    >>>

    output {
        Float divergence = read_float("divergence.txt")
        Int num_overlaps = read_int("num_overlaps.txt")
    }

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
