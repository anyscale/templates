#!/usr/bin/env bash
# Build and publish the demo data: chr20 region slices (quick, standard, full) of the GIAB
# trio's ONT reads from s3://ont-open-data, each with its reference slice and MANIFEST.json.
# Run inside the template image; --dry-run builds and verifies without uploading.
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

# Recorded in the manifest: it decides Flye's read mode and the medaka model, and no FASTQ has it.
CHEMISTRY="${CHEMISTRY:-R10.4.1 (LSK114, E8.2, 400 bps)}"
BASECALLER="${BASECALLER:-dorado sup v4.1.0}"
MEDAKA_MODEL="${MEDAKA_MODEL:-r1041_e82_400bps_sup_v4.1.0}"

echo "work dir: $WORK"
cd "$WORK"

for tool in samtools aws jq; do
  command -v "$tool" >/dev/null || { echo "$tool not on PATH" >&2; exit 1; }
done
# Without libcurl every region read fails with a bare "fail to open file".
samtools --version | grep -qi 'libcurl' \
  || echo "WARNING: samtools may lack libcurl; https:// CRAM access will fail" >&2

echo "== reference"
REF_URL="$SRC_BUCKET/$SRC_RELEASE/$SRC_REF_KEY"
[ -f ref.fa ] || curl -fsSL "$REF_URL" -o ref.fa
[ -f ref.fa.fai ] || curl -fsSL "$REF_URL.fai" -o ref.fa.fai
CHROM_LEN=$(awk -v c="$CHROM" '$1 == c { print $2 }' ref.fa.fai)
[ -n "$CHROM_LEN" ] || { echo "$CHROM not in the reference" >&2; exit 1; }
echo "   $CHROM is $CHROM_LEN bp"

# CRAM decodes against the exact reference it was written with: this bucket's, never a lookup.
export REF_PATH=
export REF_CACHE=

cram_urls() {
  # Listed, not hardcoded: the trio has 2, 3 and 6 flow cells.
  local sample="$1" lower
  lower=$(echo "$sample" | tr '[:upper:]' '[:lower:]')
  curl -fsS "$SRC_BUCKET/?list-type=2&prefix=$SRC_RELEASE/analysis/$lower/$SRC_BASECALL/&max-keys=1000" \
    | tr '<' '\n' | sed -n 's/^Key>//p' | grep '\.pass\.cram$' \
    | while read -r key; do echo "$SRC_BUCKET/$key"; done
}

slice_sample() {
  local outdir="$1" sample="$2" region="$3" url
  local fastq="$outdir/$sample.reads.fastq.gz"
  [ -f "$fastq" ] && { echo "   $sample: already built"; return; }

  : > "$outdir/$sample.reads.fastq"
  while read -r url; do
    [ -n "$url" ] || continue
    echo "   $sample <- $(basename "$url") $region"
    # -F 0x900 drops secondary and supplementary records, so each read appears once.
    samtools view -@ "$THREADS" -T ref.fa -F 0x900 -u "$url" "$region" \
      | samtools fastq -@ "$THREADS" -n - >> "$outdir/$sample.reads.fastq"
  done < <(cram_urls "$sample")

  gzip -f "$outdir/$sample.reads.fastq"
}

build() {
  local name="$1" start="$2" end="$3" region="$CHROM:$2-$3"
  echo "== $name  ($region)"
  mkdir -p "$name"

  # Same 1-based inclusive region as samtools view; the ">chr20:a-b" header is kept on purpose.
  samtools faidx ref.fa "$region" > "$name/reference.fa"
  samtools faidx "$name/reference.fa"

  local samples_for_scale="$SAMPLES"

  for sample in $samples_for_scale; do
    slice_sample "$name" "$sample" "$region"
  done

  manifest "$name" "$start" "$end" "$samples_for_scale"
}

manifest() {
  local name="$1" start="$2" end="$3" samples="$4"
  local entries="[]"
  for sample in $samples; do
    local f="$name/$sample.reads.fastq.gz"
    local n bases n50 sha
    # "%.0f", not "%d": mawk's %d goes through a 32-bit int and clamps.
    n=$(gzip -dc "$f" | awk 'END { printf "%.0f", NR / 4 }')
    bases=$(gzip -dc "$f" | awk 'NR % 4 == 2 { b += length($0) } END { printf "%.0f", b }')
    # No early exit: under pipefail it would SIGPIPE the sort once reads outgrow the pipe buffer.
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

# SCALES narrows the rebuild; each scale is rebuilt whole, manifest included.
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
