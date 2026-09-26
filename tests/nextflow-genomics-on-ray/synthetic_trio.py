#!/usr/bin/env python3
"""Write a small synthetic trio that the whole pipeline can run on, with a known answer.

    python synthetic_trio.py --out /tmp/synth [--length 200000] [--coverage 30]
    python synthetic_trio.py --check /tmp/synth results/benchmark/benchmark.tsv

What it writes, in the layout tools/stage-demo-data.sh publishes, so the pipeline
and bin/make_samplesheet.py read it exactly as they read the real data:

    MANIFEST.json
    reference/chrS.fa  (+ .fai, .dict)
    known_sites.vcf               sites-only; bgzip + tabix it before use
    HG00{2,3,4}_R{1,2}.fastq.gz
    truth/HG00{2,3,4}.vcf         one per sample; bgzip + tabix before use
    truth/HG00{2,3,4}.bed

Why this exists. It is the CI gate: tests.sh runs main.nf on it under
`-profile ray` before the notebook, and it needs no staged data. It takes the
same code path as the real data (FASTQ in, a vcfeval table out) on a genome small
enough to call in minutes, and the truth set is the list of variants that were
planted, so a low recall here is a pipeline bug rather than a question about GIAB.

``--check`` is that gate's verdict on the run's benchmark.tsv; see check().

The trio is Mendelian by construction. Four parental haplotypes each carry a
random half of the planted variants; the father (HG003) gets haplotypes A and B,
the mother (HG004) C and D, and the child (HG002) A and C. Reads are drawn from a
sample's two haplotypes with equal probability, with a 0.1% substitution error
rate and a flat base quality, which makes this far easier than a real genome: it
is a plumbing test, and its F1 says nothing about GATK.

Standard library only, seeded, and deterministic for a given set of arguments.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import random

CONTIG = "chrS"
SAMPLES = {"HG002": ("A", "C"), "HG003": ("A", "B"), "HG004": ("C", "D")}
BASES = "ACGT"

#: The callset the gate scores: the hard-filtered joint callset, main.nf's `gatk_hard`.
#: The CNN arm's `gatk_cnn` is not gated. CI runs with it off, and these reads, with a
#: flat base quality on a random genome, are nothing like the data its model learned.
GATED_CALLER = "gatk_hard"


def plant_variants(seq: str, rng: random.Random, spacing: int, margin: int) -> list[dict]:
    """SNPs, insertions and deletions, one per ``spacing`` bases, never near an edge."""
    variants = []
    pos = margin
    while pos < len(seq) - margin:
        pos += rng.randint(spacing // 2, spacing)
        if pos >= len(seq) - margin:
            break
        ref_base = seq[pos - 1]  # VCF POS is 1-based
        kind = rng.random()
        if kind < 0.7:
            alt = rng.choice([b for b in BASES if b != ref_base])
            variants.append({"pos": pos, "ref": ref_base, "alt": alt})
        elif kind < 0.85:
            insert = "".join(rng.choice(BASES) for _ in range(rng.randint(1, 6)))
            variants.append({"pos": pos, "ref": ref_base, "alt": ref_base + insert})
        else:
            length = rng.randint(1, 6)
            ref = seq[pos - 1 : pos + length]
            variants.append({"pos": pos, "ref": ref, "alt": ref_base})
    return variants


def haplotype(seq: str, variants: list[dict], carried: set[int]) -> str:
    """Apply the carried variants to the reference, right to left so offsets hold."""
    out = seq
    for i in sorted(carried, key=lambda k: variants[k]["pos"], reverse=True):
        v = variants[i]
        start = v["pos"] - 1
        assert out[start : start + len(v["ref"])] == v["ref"]
        out = out[:start] + v["alt"] + out[start + len(v["ref"]) :]
    return out


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def simulate_reads(haps, n_pairs, read_len, insert_mean, insert_sd, error, rng, sample):
    """Yield (name, r1, r2) paired reads from a sample's two haplotypes."""
    for i in range(n_pairs):
        hap = haps[rng.randrange(2)]
        insert = max(read_len + 10, int(rng.gauss(insert_mean, insert_sd)))
        start = rng.randint(0, len(hap) - insert)
        fragment = hap[start : start + insert]
        r1, r2 = fragment[:read_len], revcomp(fragment[-read_len:])
        if rng.random() < 0.5:
            r1, r2 = r2, r1

        def noisy(read: str) -> str:
            return "".join(
                rng.choice([b for b in BASES if b != c]) if rng.random() < error else c
                for c in read
            )

        yield f"{sample}:{i}", noisy(r1), noisy(r2)


def write_vcf(path: str, header_sample: str | None, rows: list[str]) -> None:
    with open(path, "w") as out:
        out.write("##fileformat=VCFv4.2\n")
        out.write(f"##contig=<ID={CONTIG},length={LENGTH}>\n")
        if header_sample:
            out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            out.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{header_sample}\n")
        else:
            out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in rows:
            out.write(row + "\n")


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def planted(data_dir: str) -> dict[str, int]:
    """Variants planted per sample, from the MANIFEST.json the trio was written with."""
    with open(os.path.join(data_dir, "MANIFEST.json")) as handle:
        return {s["id"]: s["truth_variants"] for s in json.load(handle)["samples"]}


def check(data_dir: str, benchmark: str) -> list[str]:
    """What is wrong with ``benchmark`` as a score of the trio in ``data_dir``.

    Empty when the run is right. The truth set is what was planted, so the one
    right answer is a perfect one: for every sample, SNPs and indels of the
    hard-filtered callset each at precision, recall and F1 1.0000 with no false
    positive or negative, and the true positives of the two adding up to
    everything planted in that sample.

    The sum is not redundant. A truth variant that neither the SNP nor the indel
    split keeps is scored nowhere, and every rate still reads 1.0000 without it.
    """
    expected = planted(data_dir)
    with open(benchmark) as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["caller"] == GATED_CALLER]

    problems = []
    got = sorted((r["sample"], r["variant_type"]) for r in rows)
    want = sorted((sample, vtype) for sample in expected for vtype in ("indel", "snp"))
    if got != want:
        problems.append(
            f"expected one {GATED_CALLER} row per sample and type, {want}; got {got}"
        )

    true_pos = dict.fromkeys(expected, 0)
    for row in rows:
        label = f"{row['sample']}/{row['caller']}/{row['variant_type']}"
        for metric in ("precision", "recall", "f1"):
            if row[metric] != "1.0000":
                problems.append(f"{label}: {metric} {row[metric]}, expected 1.0000")
        for count in ("false_pos", "false_neg"):
            if row[count] != "0":
                problems.append(f"{label}: {count} {row[count]}, expected 0")
        if row["sample"] in true_pos:
            true_pos[row["sample"]] += int(row["true_pos_baseline"])
    for sample, n in expected.items():
        if true_pos[sample] != n:
            problems.append(f"{sample}: {true_pos[sample]} true positives, {n} planted")
    return problems


LENGTH = 0


def main(argv: list[str] | None = None) -> int:
    global LENGTH
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", help="write the trio here")
    mode.add_argument(
        "--check",
        nargs=2,
        metavar=("DATA_DIR", "BENCHMARK_TSV"),
        help="score a run of main.nf on the trio in DATA_DIR; exit 1 unless it is perfect",
    )
    parser.add_argument("--length", type=int, default=200_000)
    parser.add_argument("--coverage", type=float, default=30.0)
    parser.add_argument("--read-length", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args(argv)

    if args.check:
        problems = check(*args.check)
        for problem in problems:
            print(f"FAIL {problem}")
        if problems:
            return 1
        counts = ", ".join(f"{sample} {n}" for sample, n in planted(args.check[0]).items())
        print(
            f"synthetic trio scores perfect ({GATED_CALLER}): precision, recall and F1 "
            "1.0000 for every sample and type, and every planted variant a true "
            f"positive ({counts})"
        )
        return 0

    LENGTH = args.length
    rng = random.Random(args.seed)

    os.makedirs(os.path.join(args.out, "reference"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "truth"), exist_ok=True)

    seq = "".join(rng.choice(BASES) for _ in range(args.length))
    fasta = os.path.join(args.out, "reference", f"{CONTIG}.fa")
    with open(fasta, "w") as fh, open(fasta + ".fai", "w") as fai:
        fh.write(f">{CONTIG}\n")
        offset = fh.tell()
        for i in range(0, len(seq), 60):
            fh.write(seq[i : i + 60] + "\n")
        fai.write(f"{CONTIG}\t{len(seq)}\t{offset}\t60\t61\n")
    with open(os.path.join(args.out, "reference", f"{CONTIG}.dict"), "w") as out:
        md5 = hashlib.md5(seq.encode()).hexdigest()
        out.write("@HD\tVN:1.6\n")
        out.write(f"@SQ\tSN:{CONTIG}\tLN:{len(seq)}\tM5:{md5}\n")

    margin = 1_000
    variants = plant_variants(seq, rng, spacing=400, margin=margin)
    carriers = {h: {i for i in range(len(variants)) if rng.random() < 0.5} for h in "ABCD"}
    haps = {h: haplotype(seq, variants, carriers[h]) for h in "ABCD"}

    write_vcf(
        os.path.join(args.out, "known_sites.vcf"),
        None,
        [f"{CONTIG}\t{v['pos']}\t.\t{v['ref']}\t{v['alt']}\t.\t.\t." for v in variants],
    )

    n_pairs = int(args.length * args.coverage / (2 * args.read_length))
    manifest = {
        "scale": "synthetic",
        "region": f"{CONTIG}:1-{args.length}",
        "note": "synthetic trio from tests/nextflow-genomics-on-ray/synthetic_trio.py",
        "samples": [],
    }
    for sample, (h1, h2) in SAMPLES.items():
        rows = []
        for i, v in enumerate(variants):
            dose = (i in carriers[h1]) + (i in carriers[h2])
            if dose:
                gt = "1/1" if dose == 2 else "0/1"
                rows.append(f"{CONTIG}\t{v['pos']}\t.\t{v['ref']}\t{v['alt']}\t.\tPASS\t.\tGT\t{gt}")
        write_vcf(os.path.join(args.out, "truth", f"{sample}.vcf"), sample, rows)
        with open(os.path.join(args.out, "truth", f"{sample}.bed"), "w") as out:
            out.write(f"{CONTIG}\t{margin}\t{args.length - margin}\n")

        r1_path = os.path.join(args.out, f"{sample}_R1.fastq.gz")
        r2_path = os.path.join(args.out, f"{sample}_R2.fastq.gz")
        quality = "I" * args.read_length
        with gzip.open(r1_path, "wt", compresslevel=1) as o1, \
             gzip.open(r2_path, "wt", compresslevel=1) as o2:
            reads = simulate_reads(
                (haps[h1], haps[h2]), n_pairs, args.read_length, 350, 30, 0.001, rng, sample
            )
            for name, r1, r2 in reads:
                o1.write(f"@{name}/1\n{r1}\n+\n{quality}\n")
                o2.write(f"@{name}/2\n{r2}\n+\n{quality}\n")

        manifest["samples"].append({
            "id": sample,
            "fastq_1": os.path.basename(r1_path),
            "fastq_2": os.path.basename(r2_path),
            "sha256_1": sha256(r1_path),
            "sha256_2": sha256(r2_path),
            "read_pairs": n_pairs,
            "truth_vcf": f"truth/{sample}.vcf.gz",
            "truth_bed": f"truth/{sample}.bed",
            "truth_variants": len(rows),
        })

    with open(os.path.join(args.out, "MANIFEST.json"), "w") as out:
        json.dump(manifest, out, indent=2)
    print(
        f"synthetic trio: {args.length:,} bp {CONTIG}, {len(variants)} planted variants, "
        f"{n_pairs:,} read pairs per sample -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
