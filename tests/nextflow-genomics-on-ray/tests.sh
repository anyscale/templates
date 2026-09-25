#!/usr/bin/env bash
set -euxo pipefail

# Three stages, cheapest first. Only the third needs staged demo data.
#
#   1. unit checks      offline, seconds
#   2. synthetic trio   main.nf end to end under -profile ray on this cluster, on data this
#                       script generates. The gate: it asserts a perfect score.
#   3. the notebook     at `quick`, over the GIAB reads staged at
#                       s3://anyscale-public-materials/genomics/giab-trio-illumina-chr20/quick/
#
# Run from the template directory. rayapp flattens templates/<name>/ and tests/<name>/ into one
# directory, so this script, synthetic_trio.py and the test_*.py files sit beside README.ipynb
# and pipeline/. In a repo checkout, `cd templates/nextflow-genomics-on-ray && bash
# ../../tests/nextflow-genomics-on-ray/tests.sh` is the same layout.
here="$(dirname "$0")"

# Pinned, so that a papermill release cannot turn this template's test red.
uv pip install -q --system 'papermill==2.7.0'

# The notebook's default is `standard`, three samples x 10 Mbp of chr20, which is what a
# reader gets and more than CI should spend. CI runs `quick`, 2 Mbp over 8 intervals: the same
# processes, tools and resource requests on fewer bases. The Buildkite step is capped at
# max(75, timeout_in_sec/60 + 30) minutes, 90 for this template.
export NF_DEMO_SCALE=quick

# The GPU leg stays in the template and out of CI: ANNOTATE_VARIANTS asks for an L4, and a run
# that waits on L4 capacity in the test cloud's zone is red for a reason the template cannot
# fix. Stage 2 passes it to main.nf; the notebook reads it and skips the process and its Ray
# Data step (Step 7).
export NF_ANNOTATE=false

# --- 1. unit checks: offline ---------------------------------------------------------------
#
# Plain scripts, not pytest (pytest collects none of them), each exiting non-zero on a failed
# check:
#
#   test_nf_ray            the #RAY header, directive-to-Ray mapping, exit codes, config
#   test_daemon            the daemon's task table, against a stand-in ray module
#   test_config_agreement  the node ceiling, scale regions and plugin version, which live in
#                          several files each (its compute-config checks skip here: rayapp
#                          does not ship configs/ to the cluster)
#   test_make_samplesheet  per-sample truth wiring, and a manifest that disagrees with disk
#   test_collect_vcfeval   the vcfeval readout, against summaries captured from rtg itself
#   test_score_variants    the reference and VCF plumbing around the scorer's model
#
# Each finds nf_ray/ and pipeline/ from the working directory.
for check in test_nf_ray test_daemon test_config_agreement test_make_samplesheet \
             test_collect_vcfeval test_score_variants; do
  python "$here/$check.py"
done

# --- 2. the synthetic trio: main.nf end to end, no staged data -----------------------------
#
# 200 kb, 657 planted variants, Mendelian by construction; synthetic_trio.py has the details.
# The truth set is what was planted, so the only passing score is a perfect one, and `--check`
# fails the build on anything else: precision, recall or F1 below 1.0000 for any sample and
# type, or true positives that do not add up to each sample's planted set.
#
# `-profile ray`, as the notebook runs it: the executor, the plugin, the shared work directory
# and the autoscaling workers, over the whole DAG from FASTQ to benchmark.tsv. --region and
# --intervals replace the scale preset's, which name chr20. A fresh directory per run, on
# shared storage because the workers read the inputs.
mkdir -p /mnt/cluster_storage/nf-genomics
synth="$(mktemp -d /mnt/cluster_storage/nf-genomics/synthetic.XXXXXX)"
python "$here/synthetic_trio.py" --out "$synth/data"
bgzip -f "$synth/data/known_sites.vcf"
tabix -f -p vcf "$synth/data/known_sites.vcf.gz"
for truth in "$synth"/data/truth/*.vcf; do
  bgzip -f "$truth"
  tabix -f -p vcf "$truth.gz"
done
python pipeline/bin/make_samplesheet.py --data-dir "$synth/data" \
  --output "$synth/data/samplesheet.csv"
region="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["region"])' \
  "$synth/data/MANIFEST.json")"

nextflow run pipeline/main.nf -profile ray -ansi-log false \
  --samplesheet "$synth/data/samplesheet.csv" \
  --reference "$synth/data/reference/chrS.fa" \
  --known_sites "$synth/data/known_sites.vcf.gz" \
  --region "$region" --intervals 4 \
  --annotate "$NF_ANNOTATE" \
  --outdir "$synth/results"
python "$here/synthetic_trio.py" --check "$synth/data" "$synth/results/benchmark/benchmark.tsv"

# --- 3. the notebook at `quick`: needs the staged demo data --------------------------------
#
# Step 4 syncs the `quick` prefix above, which tools/stage-demo-data.sh publishes. If it has
# not been published, the notebook fails in Step 4 with make_samplesheet.py's "MANIFEST.json
# not found", after the gate has already passed.
papermill README.ipynb /tmp/nextflow-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
