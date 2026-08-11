#!/usr/bin/env bash
# Build and publish the template's demo data: region slices of chr20 ONT reads for the
# GIAB Ashkenazi trio, each paired with the matching reference slice.
#
#   s3://anyscale-public-materials/genomics/giab-trio-chr20/
#     quick/     chr20:1,000,000-3,000,000    (~2 Mbp;  CI, ~15 min per sample)
#     standard/  chr20:1,000,000-11,000,000   (~10 Mbp; the notebook default)
#     full/      chr20:1-64,444,167           (64 Mbp;  job.yaml, the production cohort)
#
# each containing, per sample:
#
#     HG002.reads.fastq.gz  HG003.reads.fastq.gz  HG004.reads.fastq.gz
#     reference.fa          MANIFEST.json
#
# HG002 is the GIAB Ashkenazi son, HG003 the father, HG004 the mother -- the most
# thoroughly characterised human samples there are. They are three independent
# single-sample assemblies here, not a pedigree-aware analysis; see the note in
# wdl/pipelines/ONT/Assembly/ONTAssembleCohort.wdl.
#
# ---------------------------------------------------------------------------------
# Source, and why it changed
#
# Everything is derived from ONT's public open-data bucket, which is anonymously
# readable, so this script is reproducible by anyone rather than only by whoever holds
# credentials to a private bucket:
#
#   reads      s3://ont-open-data/giab_2023.05/analysis/<sample>/sup/*.pass.cram
#   reference  s3://ont-open-data/giab_2023.05/analysis/benchmarking/
#                GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
#
# Those CRAMs are aligned and indexed, which changes the slicing method for the
# better. The previous version of this script started from an unaligned FASTQ and
# recovered placement with a whole-file `minimap2 -x map-ont` pass; now the placement
# is already in the file and samtools reads only the region's blocks over HTTPS. That
# is minutes instead of hours, no aligner in the loop, and no dependence on this
# script's minimap2 version matching anything.
#
# The reads it emits are still *reference-selected*, and that is a real limitation
# worth stating rather than hiding: a read is in the slice because it aligned to the
# region, so reads from a divergent haplotype that failed to align are absent by
# construction, and reads from paralogous sequence elsewhere in the genome are absent
# too. An assembly of them is therefore easier than a whole-genome de novo assembly
# and its contiguity and genome-fraction numbers are optimistic relative to one. It
# is a demo of the pipeline, not a benchmark of the assembler.
#
# Ultra-long reads overhang the slice edges by construction. That is fine: Flye never
# sees the reference, and QUAST reports the overhang as unaligned contig ends.
#
# ---------------------------------------------------------------------------------
# Needs: samtools (built with libcurl), aws, jq, awk, gzip -- all in the template
# image except jq. Reading the source needs no credentials; writing $DEST does. Run it
# inside the image:
#
#   podman run --rm -v ~/.aws:/home/ray/.aws:ro -v "$PWD":/work -w /work \
#     us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates/wdl-genomics-on-ray:2.56.0 \
#     bash tools/stage-demo-data.sh
#
# `--dry-run` derives and verifies everything locally without uploading.
set -euo pipefail

SRC_BUCKET="${SRC_BUCKET:-https://ont-open-data.s3.amazonaws.com}"
SRC_RELEASE="${SRC_RELEASE:-giab_2023.05}"
SRC_BASECALL="${SRC_BASECALL:-sup}"
SRC_REF_KEY="${SRC_REF_KEY:-analysis/benchmarking/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna}"

DEST="${DEST:-s3://anyscale-public-materials/genomics/giab-trio-chr20}"
SAMPLES="${SAMPLES:-HG002 HG003 HG004}"
CHROM="${CHROM:-chr20}"
THREADS="${THREADS:-$(nproc)}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"   # a caller-supplied WORK need not pre-exist
DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# The basecaller that produced these reads. Not cosmetic: it is what decides Flye's
# read mode (R10.4.1 -> --nano-hq) and which medaka model is correct, and the reason
# the manifest records it is that neither decision is recoverable from a FASTQ.
CHEMISTRY="${CHEMISTRY:-R10.4.1 (LSK114, E8.2, 400 bps)}"
BASECALLER="${BASECALLER:-dorado sup v4.1.0}"
MEDAKA_MODEL="${MEDAKA_MODEL:-r1041_e82_400bps_sup_v4.1.0}"

echo "work dir: $WORK"
cd "$WORK"

for tool in samtools aws jq; do
  command -v "$tool" >/dev/null || { echo "$tool not on PATH" >&2; exit 1; }
done
# samtools must be able to open an https:// URL, or every region read below fails with
# a bare "fail to open file". Checking the build once beats decoding that message.
samtools --version | grep -qi 'libcurl' \
  || echo "WARNING: samtools may lack libcurl; https:// CRAM access will fail" >&2

echo "== reference"
REF_URL="$SRC_BUCKET/$SRC_RELEASE/$SRC_REF_KEY"
[ -f ref.fa ] || curl -fsSL "$REF_URL" -o ref.fa
[ -f ref.fa.fai ] || curl -fsSL "$REF_URL.fai" -o ref.fa.fai
CHROM_LEN=$(awk -v c="$CHROM" '$1 == c { print $2 }' ref.fa.fai)
[ -n "$CHROM_LEN" ] || { echo "$CHROM not in the reference" >&2; exit 1; }
echo "   $CHROM is $CHROM_LEN bp"

# CRAM stores sequence as a diff against the reference, so decoding needs the *same*
# reference the CRAM was written against. It is the one in this bucket, which is why
# the script fetches it rather than accepting one as input.
export REF_PATH=
export REF_CACHE=

cram_urls() {
  # Every pass CRAM for one sample, at the requested basecall quality. Listing the
  # bucket rather than hardcoding filenames: the trio has 2, 3 and 6 flowcells and
  # ONT may add more.
  local sample="$1" lower
  lower=$(echo "$sample" | tr '[:upper:]' '[:lower:]')
  curl -fsS "$SRC_BUCKET/?list-type=2&prefix=$SRC_RELEASE/analysis/$lower/$SRC_BASECALL/&max-keys=1000" \
    | tr '<' '\n' | sed -n 's/^Key>//p' | grep '\.pass\.cram$' \
    | while read -r key; do echo "$SRC_BUCKET/$key"; done
}

slice_sample() {
  # Extract one sample's reads over one region, straight from the aligned CRAMs.
  local outdir="$1" sample="$2" region="$3" url
  local fastq="$outdir/$sample.reads.fastq.gz"
  [ -f "$fastq" ] && { echo "   $sample: already built"; return; }

  : > "$outdir/$sample.reads.fastq"
  while read -r url; do
    [ -n "$url" ] || continue
    echo "   $sample <- $(basename "$url") $region"
    # -F 0x900 drops secondary (0x100) and supplementary (0x800) records, so each
    # read contributes its primary alignment once. Without it a read crossing a
    # supplementary breakpoint is emitted twice under the same name, and the
    # assembler sees a duplicate.
    samtools view -@ "$THREADS" -T ref.fa -F 0x900 -u "$url" "$region" \
      | samtools fastq -@ "$THREADS" -n - >> "$outdir/$sample.reads.fastq"
  done < <(cram_urls "$sample")

  # gzip renames in place to <name>.gz, which is exactly $fastq.
  gzip -f "$outdir/$sample.reads.fastq"
}

build() {
  local name="$1" start="$2" end="$3" region="$CHROM:$2-$3"
  echo "== $name  ($region)"
  mkdir -p "$name"

  # samtools faidx regions are 1-based inclusive, which is also what `samtools view`
  # takes, so the read filter and the reference slice agree on their boundaries. The
  # region header (">chr20:1000000-3000000") is kept on purpose: coordinates in the
  # outputs then say what they are relative to.
  samtools faidx ref.fa "$region" > "$name/reference.fa"
  samtools faidx "$name/reference.fa"

  # Every scale carries every sample. `full` used to be HG002 only, on the reasoning that
  # three 14-hour assemblies is not a demo -- but the production job is the cohort, and a
  # cohort with one sample cannot show the thing the cohort exists to show. Set SAMPLES to
  # narrow it.
  local samples_for_scale="$SAMPLES"

  for sample in $samples_for_scale; do
    slice_sample "$name" "$sample" "$region"
  done

  manifest "$name" "$start" "$end" "$samples_for_scale"
}

manifest() {
  # Published alongside the data because a FASTQ carries none of it, and every one of
  # these is something a reader needs in order to judge or reproduce a run: what the
  # reads are, how they were selected, what chemistry they are (which decides Flye's
  # read mode and medaka's model), and a checksum to verify the download.
  local name="$1" start="$2" end="$3" samples="$4"
  local entries="[]"
  for sample in $samples; do
    local f="$name/$sample.reads.fastq.gz"
    local n bases n50 sha
    # "%.0f", not "%d": the image's mawk converts %d through a 32-bit int, so the full
    # chr20 set's 6.1 Gbp published as exactly 2147483647 and its coverage as 33.32x instead
    # of ~95x. Same fix, and same reason, as ReadStats.FastqStats.
    n=$(gzip -dc "$f" | awk 'END { printf "%.0f", NR / 4 }')
    bases=$(gzip -dc "$f" | awk 'NR % 4 == 2 { b += length($0) } END { printf "%.0f", b }')
    # No early `exit` in the reader: under `set -o pipefail` an awk that quits at the
    # halfway mark SIGPIPEs the still-writing sort (exit 141) once the read set outgrows
    # the pipe buffer. Read it all; FastqStats solves its N50 the same way.
    n50=$(gzip -dc "$f" | awk 'NR % 4 == 2 { print length($0) }' | sort -rn \
          | awk -v total="$bases" '{ acc += $1; if (!n50 && acc >= total / 2) n50 = $1 } END { print n50 + 0 }')
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    entries=$(echo "$entries" | jq \
      --arg s "$sample" --arg f "$(basename "$f")" --arg sha "$sha" \
      --argjson n "$n" --argjson b "$bases" --argjson n50 "${n50:-0}" \
      --argjson len "$((end - start + 1))" \
      '. + [{sample: $s, file: $f, reads: $n, bases: $b, read_n50: $n50,
             coverage: (($b / $len) * 100 | round / 100), sha256: $sha}]')
  done

  jq -n \
    --arg region "$CHROM:$start-$end" --arg chrom "$CHROM" \
    --argjson start "$start" --argjson end "$end" \
    --arg source "s3://ont-open-data/$SRC_RELEASE/analysis/<sample>/$SRC_BASECALL/*.pass.cram" \
    --arg reference "$SRC_BUCKET/$SRC_RELEASE/$SRC_REF_KEY" \
    --arg chemistry "$CHEMISTRY" --arg basecaller "$BASECALLER" \
    --arg medaka_model "$MEDAKA_MODEL" \
    --arg ref_sha "$(sha256sum "$name/reference.fa" | cut -d' ' -f1)" \
    --argjson samples "$entries" \
    '{region: $region, chrom: $chrom, start: $start, end: $end,
      source_reads: $source, source_reference: $reference,
      chemistry: $chemistry, basecaller: $basecaller,
      recommended_medaka_model: $medaka_model,
      selection: "reads whose primary alignment overlaps the region, extracted from the aligned CRAMs; secondary and supplementary records dropped (-F 0x900). Reference-selected, so reads from divergent haplotypes that failed to align are absent by construction.",
      reference_sha256: $ref_sha,
      samples: $samples}' > "$name/MANIFEST.json"

  echo "   manifest:"
  jq -r '.samples[] | "     \(.sample)  \(.reads) reads  \(.bases) bases  N50 \(.read_n50)  \(.coverage)x"' \
    "$name/MANIFEST.json"
}

# SCALES narrows what gets rebuilt and republished. All three by default; set it to one
# name to add or correct a single tier without re-uploading the others. Note that a scale
# is rebuilt whole, manifest included, so narrowing SAMPLES *and* SCALES together would
# publish a manifest listing only those samples.
SCALES="${SCALES:-quick standard full}"

case " $SCALES " in *" quick "*)    build quick    1000000  3000000     ;; esac
case " $SCALES " in *" standard "*) build standard 1000000 11000000     ;; esac
case " $SCALES " in *" full "*)     build full           1 "$CHROM_LEN" ;; esac

if [ -n "$DRY_RUN" ]; then
  echo "== dry run: not uploading. Output in $WORK"
  exit 0
fi

echo "== upload to $DEST"
for scale in $SCALES; do
  aws s3 cp --recursive "$scale/" "$DEST/$scale/" --exclude "*.fai"
done

echo "== verify anonymously (what the notebook and CI actually do)"
fail=0
for scale in $SCALES; do
  for object in MANIFEST.json reference.fa; do
    if aws s3 cp --no-sign-request "$DEST/$scale/$object" - >/dev/null 2>&1; then
      echo "   $scale/$object: anonymously readable"
    else
      echo "   $scale/$object: NOT anonymously readable -- check the bucket policy" >&2
      fail=1
    fi
  done
done

echo "done; work dir kept at $WORK"
exit "$fail"
