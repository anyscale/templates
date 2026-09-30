#!/usr/bin/env bash
# Score two records with NVScoreVariants ([cpu|gpu]); fail unless both come back scored, since it
# exits 0 when its Python fails. Sets up the environment as cnn_filter.nf does: change both together.
set -euo pipefail

accelerator="${1:-cpu}"
work="$(mktemp -d)"
cd "$work"

# A SNP and a 1 bp deletion, each with the seven INFO annotations the 1D model reads.
python3 - <<'PY'
import random

random.seed(1)
seq = "".join(random.choice("ACGT") for _ in range(2000))
with open("ref.fa", "w") as fa:
    fa.write(">chrT\n")
    for i in range(0, len(seq), 60):
        fa.write(seq[i : i + 60] + "\n")

types = [("DP", "Integer"), ("FS", "Float"), ("MQ", "Float"), ("MQRankSum", "Float"),
         ("QD", "Float"), ("ReadPosRankSum", "Float"), ("SOR", "Float")]
info = "DP=30;FS=0.000;MQ=60.00;MQRankSum=0.000;QD=20.00;ReadPosRankSum=0.300;SOR=0.700"
header = [
    "##fileformat=VCFv4.2",
    f"##contig=<ID=chrT,length={len(seq)}>",
    *(f'##INFO=<ID={key},Number=1,Type={kind},Description="{key}">' for key, kind in types),
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSMOKE",
]
ref = seq[999]
records = [
    f"chrT\t1000\t.\t{ref}\t{'A' if ref != 'A' else 'C'}\t500\t.\t{info}\tGT\t0/1",
    f"chrT\t1200\t.\t{seq[1199:1201]}\t{seq[1199]}\t500\t.\t{info}\tGT\t0/1",
]
with open("smoke.vcf", "w") as vcf:
    vcf.write("\n".join(header + records) + "\n")
PY

py="${NF_GATK_PYTHON:-$(command -v python3)}"
mkdir -p gatk-python
for name in python python3; do
    printf '#!/bin/sh\nexec "%s" "$@"\n' "$py" > "gatk-python/$name"
    chmod +x "gatk-python/$name"
done
export PATH="$PWD/gatk-python:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

gatk --java-options "-Xmx2g -Djava.io.tmpdir=$PWD" NVScoreVariants \
    --variant smoke.vcf \
    --reference ref.fa \
    --tensor-type reference \
    --accelerator "$accelerator" \
    --output scored.vcf

scored=$(bcftools query -f '%INFO/CNN_1D\n' scored.vcf | awk '$1 != "." { n++ } END { print n + 0 }')
if [ "$scored" -ne 2 ]; then
    echo "NVScoreVariants scored $scored of 2 records (work dir $work)" >&2
    exit 1
fi
bcftools query -f '%POS %REF>%ALT CNN_1D=%INFO/CNN_1D\n' scored.vcf
echo "NVScoreVariants ok on $accelerator ($py)"
