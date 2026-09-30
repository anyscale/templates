#!/usr/bin/env bash
set -euxo pipefail

# Run from the template dir: rayapp flattens templates/<name>/ and tests/<name>/ into one.
here="$(dirname "$0")"

uv pip install -q --system 'papermill==2.7.0'

# The default `standard` run on fewer bases: same processes, tools and requests.
export NF_DEMO_SCALE=quick

# The CNN arm needs an L4, and a run waiting on L4 capacity fails for a reason this can't fix.
export NF_CNN=false

# 1. Unit checks: plain scripts (pytest collects none), offline.
for check in test_nf_ray test_daemon test_config_agreement test_make_samplesheet \
             test_collect_vcfeval; do
  python "$here/$check.py"
done

# 2. The gate: main.nf under -profile ray on a planted trio must score a perfect 1.0000.
# On shared storage, because the workers read the inputs.
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
  --cnn "$NF_CNN" \
  --outdir "$synth/results"
python "$here/synthetic_trio.py" --check "$synth/data" "$synth/results/benchmark/benchmark.tsv"

# 3. The notebook, on the `quick` data tools/stage-demo-data.sh publishes.
papermill README.ipynb /tmp/nextflow-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
