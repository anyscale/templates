#!/usr/bin/env bash
# Build and publish the template's demo data: chr20 Illumina reads for the GIAB
# Ashkenazi trio, with the reference, known-sites resources and each sample's own
# GIAB v4.2.1 truth set.
#
#   s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20/
#     quick/     chr20:1,000,000-3,000,000    (2 Mbp;  CI)
#     standard/  chr20:1,000,000-11,000,000   (10 Mbp; the notebook default)
#     full/      chr20                        (64 Mbp; job.yaml)
#
# each self-contained, so the notebook syncs one prefix and has everything:
#
#     MANIFEST.json                          what bin/make_samplesheet.py reads
#     HG00{2,3,4}_R{1,2}.fastq.gz            reads over the scale's region, ~30x
#     reference/chr20.fa (.fai, .dict)       all of chr20, every scale
#     known_sites/dbsnp138.chr20.vcf.gz (.tbi)                        BQSR
#     known_sites/hapmap_3.3.chr20.vcf.gz (.tbi)                      FilterVariantTranches,
#     known_sites/1000G_phase1.snps.high_confidence.chr20.vcf.gz (.tbi)   for the CNN arm
#     known_sites/Mills_and_1000G_gold_standard.indels.chr20.vcf.gz (.tbi)
#     truth/HG00{2,3,4}.vcf.gz (.tbi)        GIAB v4.2.1, chr20, one per sample
#     truth/HG00{2,3,4}.bed                  its high-confidence regions
#
# The region lines below are also what main.nf's scales() uses, and
# tests/nextflow-genomics-on-ray/test_config_agreement.py fails if the two drift:
# the pipeline would otherwise score a region the reads do not cover.
#
# ---------------------------------------------------------------------------------
# Source. All of it is public and anonymously readable, so anyone can rebuild this.
#
#   reads      GIAB's NHGRI Illumina 300x novoalign BAMs, GRCh38 (contig `chr20`,
#              LN 64,444,167 in the BAM header, the same as the reference below)
#   reference  chr20 of GCA_000001405.15_GRCh38_no_alt_analysis_set, the copy in
#              ONT's open-data bucket
#   known      the Broad hg38 bundle's dbSNP 138, sliced to chr20
#   tranche    the Broad hg38 bundle's HapMap 3.3, 1000G phase 1 high-confidence SNPs
#              and Mills/1000G gold-standard indels, sliced to chr20: the resources
#              GATK's CNN workflow (gatk-workflows/gatk4-cnn-variant-filter) gives
#              FilterVariantTranches
#   truth      GIAB NISTv4.2.1 GRCh38 benchmark VCF + _noinconsistent BED, per sample
#
# The Broad files are read by range request, chr20's blocks only, and each object's
# MD5 is pinned below and checked first, against the value GCS reports in
# `x-goog-hash`: a re-cut upstream file stops the build instead of quietly moving
# BQSR or every tranche cutoff. MANIFEST.json records the pinned MD5 beside the
# sha256 of each published slice.
#
# The BAMs are indexed, so samtools reads only the region's blocks over HTTPS. Reads
# are subsampled to ~30x by read name (`-s SEED.FRAC`, which keeps mates together),
# collated, and written as paired FASTQ with singletons dropped.
#
# The reads are *reference-selected*, and that bounds what the results mean: a pair
# is here because GIAB's whole-genome alignment put it in the region, and the
# pipeline realigns it against chr20 alone, so reads from elsewhere in the genome
# that a whole-genome run would misplace into chr20 are mostly absent, and
# precision reads high against a whole-genome run. pipeline/PIPELINE.md, "Bounds
# on the numbers", has the rest.
#
# ---------------------------------------------------------------------------------
# Needs samtools built with libcurl, bcftools, tabix, bgzip, python3, curl and (to
# publish) aws -- all in the template image. Reading needs no credentials; writing
# $DEST does. Run it inside the image, as a job:
#
#   bash tools/stage-demo-data.sh              # build every scale and upload
#   bash tools/stage-demo-data.sh --dry-run    # build and verify, upload nothing
#
# A scale whose reads are already in $WORK/<scale>/ keeps them, so adding a resource to
# the published data without re-deriving the reads is: sync each published scale into
# $WORK, then run this with that WORK. The FASTQ checksums in the new MANIFEST.json come
# out the same, and sync uploads the new files and the rewritten ones.
#
# A small local check of the whole derivation, a few MB of reads:
#
#   SCALES=quick REGION_QUICK=chr20:1000000-1020000 bash tools/stage-demo-data.sh --dry-run
set -euo pipefail

GIAB="${GIAB:-https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab}"
REF_URL="${REF_URL:-https://ont-open-data.s3.amazonaws.com/giab_2023.05/analysis/benchmarking/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna}"
BROAD="${BROAD:-https://storage.googleapis.com/gcp-public-data--broad-references/hg38/v0}"
DBSNP_URL="${DBSNP_URL:-$BROAD/Homo_sapiens_assembly38.dbsnp138.vcf.gz}"
DBSNP_MD5="${DBSNP_MD5:-110aaeb8130bf4edb544a72d2c7829f7}"

# FilterVariantTranches' resources: published name, object in the Broad bundle, and the
# object's MD5 (GCS metadata, 2019-12-06 uploads).
RESOURCES=(
  "hapmap_3.3 hapmap_3.3.hg38.vcf.gz d05ac6b9a247a21ce0030c7494194da9"
  "1000G_phase1.snps.high_confidence 1000G_phase1.snps.high_confidence.hg38.vcf.gz b2979b47800b59b41920bf5432c4b2a0"
  "Mills_and_1000G_gold_standard.indels Mills_and_1000G_gold_standard.indels.hg38.vcf.gz 2e02696032dcfe95ff0324f4a13508e3"
)

DEST="${DEST:-s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20}"
SCALES="${SCALES:-quick standard full}"
CHROM=chr20
REGION_QUICK="${REGION_QUICK:-chr20:1000000-3000000}"
REGION_STANDARD="${REGION_STANDARD:-chr20:1000000-11000000}"
REGION_FULL="${REGION_FULL:-chr20}"

# 300x down to ~30x. The seed is fixed so a rebuild draws the same reads.
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-42}"
SUBSAMPLE_FRACTION="${SUBSAMPLE_FRACTION:-0.1}"
THREADS="${THREADS:-$(nproc)}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"
DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

declare -A BAM=(
  [HG002]="$GIAB/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.GRCh38.300x.bam"
  [HG003]="$GIAB/data/AshkenazimTrio/HG003_NA24149_father/NIST_HiSeq_HG003_Homogeneity-12389378/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG003.GRCh38.300x.bam"
  [HG004]="$GIAB/data/AshkenazimTrio/HG004_NA24143_mother/NIST_HiSeq_HG004_Homogeneity-14572558/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG004.GRCh38.300x.bam"
)
declare -A TRUTH=(
  [HG002]="$GIAB/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark"
  [HG003]="$GIAB/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38/HG003_GRCh38_1_22_v4.2.1_benchmark"
  [HG004]="$GIAB/release/AshkenazimTrio/HG004_NA24143_mother/NISTv4.2.1/GRCh38/HG004_GRCh38_1_22_v4.2.1_benchmark"
)
SAMPLES=(HG002 HG003 HG004)

echo "work dir: $WORK"
cd "$WORK"

need=(samtools bcftools tabix bgzip python3 curl)
[ -z "$DRY_RUN" ] && need+=(aws)
for tool in "${need[@]}"; do
  command -v "$tool" > /dev/null || { echo "$tool not on PATH" >&2; exit 1; }
done
# Every remote read below fails with a bare "fail to open file" if htslib was built
# without libcurl. Say so once, up front.
samtools --version | grep -qi 'libcurl' \
  || { echo "samtools here is not built with libcurl; https:// reads will fail" >&2; exit 1; }

# --- shared inputs, built once -----------------------------------------------------

echo "== reference: $CHROM"
mkdir -p shared/reference shared/known_sites shared/truth
if [ ! -s shared/reference/$CHROM.fa ]; then
  # Remote faidx: htslib fetches the .fai and then only chr20's byte range.
  samtools faidx "$REF_URL" "$CHROM" > shared/reference/$CHROM.fa
fi
samtools faidx shared/reference/$CHROM.fa
samtools dict shared/reference/$CHROM.fa -o shared/reference/$CHROM.dict
echo "   $(cut -f2 shared/reference/$CHROM.fa.fai) bp"

# Stop unless the object at $1 is the one pinned by MD5 $2. GCS reports an object's MD5,
# base64-encoded, in an `x-goog-hash: md5=` header.
check_md5() {
  local got
  got="$(curl -fsSI "$1" | tr -d '\r' | sed -n 's/^[Xx]-[Gg]oog-[Hh]ash: md5=//p' \
    | python3 -c 'import base64, sys; print(base64.b64decode(sys.stdin.read().strip()).hex())')" \
    || got=""
  if [ "$got" != "$2" ]; then
    echo "$1: MD5 is '$got', pinned $2. The upstream file changed; review it and re-pin." >&2
    exit 1
  fi
}

# chr20 of a Broad VCF, by range request, after its MD5 check.
slice_broad() {  # url md5 out
  check_md5 "$1" "$2"
  tabix -h "$1" "$CHROM" | bgzip -@ "$THREADS" > "$3"
}

echo "== known sites: dbSNP 138, $CHROM"
if [ ! -s shared/known_sites/dbsnp138.$CHROM.vcf.gz ]; then
  slice_broad "$DBSNP_URL" "$DBSNP_MD5" shared/known_sites/dbsnp138.$CHROM.vcf.gz
fi
tabix -f -p vcf shared/known_sites/dbsnp138.$CHROM.vcf.gz

echo "== tranche resources, $CHROM"
for entry in "${RESOURCES[@]}"; do
  read -r name object md5 <<< "$entry"
  slice=shared/known_sites/$name.$CHROM.vcf.gz
  if [ ! -s "$slice" ]; then
    slice_broad "$BROAD/$object" "$md5" "$slice"
  fi
  tabix -f -p vcf "$slice"
  echo "   $name  $(bcftools view -H "$slice" | wc -l) records"
done

echo "== truth: GIAB v4.2.1, $CHROM"
for s in "${SAMPLES[@]}"; do
  if [ ! -s shared/truth/$s.vcf.gz ]; then
    # Renamed to the sample id: RTG_VCFEVAL passes `--sample <id>,<id>`, so the
    # truth VCF's sample column has to say HG002, whatever GIAB calls it.
    echo "$s" > "$s.name"
    tabix -h "${TRUTH[$s]}.vcf.gz" "$CHROM" | bcftools reheader -s "$s.name" - \
      | bgzip > shared/truth/$s.vcf.gz
    curl -fsSL "${TRUTH[$s]}_noinconsistent.bed" | awk -v c="$CHROM" '$1 == c' > shared/truth/$s.bed
  fi
  tabix -f -p vcf shared/truth/$s.vcf.gz
  echo "   $s  $(bcftools view -H shared/truth/$s.vcf.gz | wc -l) variants, $(wc -l < shared/truth/$s.bed) BED intervals"
done

# --- one scale -----------------------------------------------------------------------

build() {
  local scale="$1" region="$2" out="$WORK/$1" s
  echo "== $scale  ($region)"
  mkdir -p "$out"
  cp -R shared/reference shared/known_sites shared/truth "$out/"
  for s in "${SAMPLES[@]}"; do
    if [ -s "$out/${s}_R1.fastq.gz" ]; then
      echo "   $s: already built"
      continue
    fi
    echo "   $s <- $(basename "${BAM[$s]}") $region, -s $SUBSAMPLE_SEED$(echo "$SUBSAMPLE_FRACTION" | sed 's/^0//')"
    # -F 0x900: primary alignments only, so each read appears once. `collate` puts
    # mates next to each other; `fastq -s /dev/null` drops a pair's survivor when
    # its mate fell outside the region, which paired-end aligners cannot use.
    samtools view -@ "$THREADS" -u -F 0x900 \
        -s "$SUBSAMPLE_SEED$(echo "$SUBSAMPLE_FRACTION" | sed 's/^0//')" "${BAM[$s]}" "$region" \
      | samtools collate -@ "$THREADS" -O -u - "$WORK/collate.$s" \
      | samtools fastq -@ "$THREADS" -n -c 6 \
          -1 "$out/${s}_R1.fastq.gz" -2 "$out/${s}_R2.fastq.gz" -0 /dev/null -s /dev/null -
  done

  SCALE="$scale" REGION="$region" OUT="$out" SEED="$SUBSAMPLE_SEED" FRACTION="$SUBSAMPLE_FRACTION" \
  REF_URL="$REF_URL" DBSNP_URL="$DBSNP_URL" DBSNP_MD5="$DBSNP_MD5" BROAD="$BROAD" CHROM="$CHROM" \
  BAMS="$(for s in "${SAMPLES[@]}"; do printf '%s=%s\n' "$s" "${BAM[$s]}"; done)" \
  TRUTHS="$(for s in "${SAMPLES[@]}"; do printf '%s=%s\n' "$s" "${TRUTH[$s]}"; done)" \
  TRANCHE="$(printf '%s\n' "${RESOURCES[@]}")" \
  python3 - <<'PY'
import gzip, hashlib, json, os

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()

def pairs(path):
    with gzip.open(path, "rb") as handle:
        return sum(1 for _ in handle) // 4

out = os.environ["OUT"]
bams = dict(line.split("=", 1) for line in os.environ["BAMS"].splitlines())
truths = dict(line.split("=", 1) for line in os.environ["TRUTHS"].splitlines())
chrom = os.environ["CHROM"]
samples = []
for s in sorted(bams):
    r1, r2 = f"{s}_R1.fastq.gz", f"{s}_R2.fastq.gz"
    samples.append({
        "id": s,
        "fastq_1": r1, "fastq_2": r2,
        "sha256_1": sha256(os.path.join(out, r1)),
        "sha256_2": sha256(os.path.join(out, r2)),
        "read_pairs": pairs(os.path.join(out, r1)),
        "source_bam": bams[s],
        "truth_vcf": f"truth/{s}.vcf.gz",
        "truth_bed": f"truth/{s}.bed",
        "truth_source": truths[s] + ".vcf.gz",
    })
resources = []
for line in os.environ["TRANCHE"].splitlines():
    name, obj, md5 = line.split()
    vcf = f"known_sites/{name}.{chrom}.vcf.gz"
    resources.append({
        "name": name,
        "vcf": vcf,
        "sha256": sha256(os.path.join(out, vcf)),
        "source": f"{os.environ['BROAD']}/{obj} ({chrom})",
        "source_md5": md5,
    })
manifest = {
    "scale": os.environ["SCALE"],
    "region": os.environ["REGION"],
    "platform": "Illumina HiSeq, 2x148 (GIAB NHGRI 300x)",
    "subsample": {"seed": int(os.environ["SEED"]), "fraction": float(os.environ["FRACTION"])},
    "reference": {"fasta": f"reference/{chrom}.fa",
                  "sha256": sha256(os.path.join(out, "reference", f"{chrom}.fa")),
                  "source": os.environ["REF_URL"] + f" ({chrom})"},
    "known_sites": {"vcf": f"known_sites/dbsnp138.{chrom}.vcf.gz",
                    "source": os.environ["DBSNP_URL"] + f" ({chrom})",
                    "source_md5": os.environ["DBSNP_MD5"]},
    # FilterVariantTranches' resources, for the CNN arm (main.nf --tranche_resources).
    # make_samplesheet.py checks each against its sha256.
    "tranche_resources": resources,
    "truth_version": "GIAB NISTv4.2.1, GRCh38",
    "selection": ("Read pairs whose primary alignment in the GIAB novoalign GRCh38 BAM falls "
                  "in the region, subsampled by read name. Reference-selected: reads that "
                  "would mismap into the region from elsewhere are absent by construction."),
    "samples": samples,
}
with open(os.path.join(out, "MANIFEST.json"), "w") as handle:
    json.dump(manifest, handle, indent=2)
for s in samples:
    print(f"   {s['id']}  {s['read_pairs']:,} read pairs")
PY
}

for scale in $SCALES; do
  case "$scale" in
    quick)    build quick "$REGION_QUICK" ;;
    standard) build standard "$REGION_STANDARD" ;;
    full)     build full "$REGION_FULL" ;;
    *) echo "unknown scale '$scale'" >&2; exit 1 ;;
  esac
done

# --- publish ---------------------------------------------------------------------------

for scale in $SCALES; do
  if [ -n "$DRY_RUN" ]; then
    echo "== dry run: would upload $WORK/$scale/ -> $DEST/$scale/"
    (cd "$WORK/$scale" && find . -type f | sort | while read -r f; do
      printf '   %10s  %s\n' "$(wc -c < "$f" | tr -d ' ')" "$f"; done)
  else
    echo "== upload $scale -> $DEST/$scale/"
    aws s3 sync --only-show-errors "$WORK/$scale/" "$DEST/$scale/"
  fi
done
