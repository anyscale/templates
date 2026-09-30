#!/usr/bin/env python3
"""Offline check that README.ipynb's readout cells agree with ONTAssembleCohort.wdl's outputs."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

#: rayapp flattens templates/<name>/ and tests/<name>/ into one dir; a checkout keeps it two up.
_HERE = Path(__file__).resolve().parent
TEMPLATE = next(
    (
        d
        for d in (_HERE, _HERE.parents[1] / "templates" / "wdl-genomics-on-ray")
        if (d / "README.ipynb").is_file()
    ),
    _HERE.parents[1] / "templates" / "wdl-genomics-on-ray",
)
SAMPLES = ["HG002", "HG003", "HG004"]

#: Found by content, not position, so an inserted cell cannot silently skip a check.
CELLS = [
    ("quast_key", "Step 6 QUAST readout"),
    ("def nx_points", "Nx / NGx curves"),
    ("parse_vcf_lines", "Step 6b Ray Data trio check"),
]

REPORT = textwrap.dedent("""\
    All statistics are based on contigs of size >= 500 bp, unless otherwise noted.

    Assembly                    {name}
    # contigs (>= 0 bp)         {contigs}
    # contigs                   {contigs}
    Largest contig              {largest}
    Total length                {total}
    GC (%)                      43.90
    N50                         {n50}
    NG50                        {n50}
    L50                         1
    # misassemblies             6
    Genome fraction (%)         {gf}
    # mismatches per 100 kbp    {mm}
    # indels per 100 kbp        {indels}
    NGA50                       {nga50}
    """)

SHAPES = [
    dict(contigs=4, largest=6_100_000, total=9_900_000, n50=6_100_000,
         gf="98.512", mm="112.30", indels="11.40", nga50=5_900_000),
    dict(contigs=7, largest=4_200_000, total=9_820_000, n50=3_900_000,
         gf="97.905", mm="118.77", indels="13.02", nga50=3_700_000),
    dict(contigs=5, largest=5_400_000, total=9_870_000, n50=5_100_000,
         gf="98.201", mm="115.44", indels="12.11", nga50=4_800_000),
]


def summarize_quast_report(report_txt: str) -> dict[str, str]:
    """Quast.wdl's SummarizeQuastReport command, reimplemented; keep the two in step by hand."""
    rows = []
    for line in report_txt.splitlines():
        if line.startswith("All statistics") or not line.strip():
            continue
        line = re.sub(r"_{2,}", "\t", line.replace(" ", "_"))  # GNU sed's s/__\\+/\\t/g
        rows.append(re.sub(r"\s+$", "", line).replace(">=", "gt").split("\t"))
    return {row[0]: row[1] for row in rows if len(row) > 1}


def write_fasta(path: Path, lengths: list[int], name: str = "contig") -> None:
    with path.open("w") as handle:
        for index, length in enumerate(lengths):
            handle.write(f">{name}_{index}\n" if len(lengths) > 1 or name == "contig" else f">{name}\n")
            for offset in range(0, length, 60):
                handle.write("A" * min(60, length - offset) + "\n")


def write_vcf(path: Path, sample: str, positions: list[int], ref_length: int) -> None:
    """The header paftools.js writes under ``call -f``; ``bcftools norm`` rejects a minimal one."""
    with path.open("w") as handle:
        handle.write("##fileformat=VCFv4.1\n")
        handle.write(f"##contig=<ID=chr20,length={ref_length}>\n")
        handle.write('##INFO=<ID=QNAME,Number=1,Type=String,Description="Query name">\n')
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
        for position in positions:
            handle.write(f"chr20\t{position}\t.\tA\tG\t60\t.\t.\tGT\t1/1\n")


def notebook_cells() -> dict[str, str]:
    notebook = json.loads((TEMPLATE / "README.ipynb").read_text())
    sources = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    found = {}
    for marker, label in CELLS:
        matches = [s for s in sources if marker in s]
        if len(matches) != 1:
            raise SystemExit(
                f"expected exactly one code cell containing {marker!r} ({label}retained?),"
                f" found {len(matches)}. The notebook changed shape; update CELLS."
            )
        found[marker] = matches[0]
    return found


def build_fixture(root: Path) -> tuple[Path, Path, dict[str, str]]:
    runs = root / "runs"
    run_dir = runs / "20260809_120000_ONTAssembleCohort"
    run_dir.mkdir(parents=True)

    # Named chr20 to match the VCFs: `bcftools norm -f` rejects a contig absent from the reference.
    ref_length = 10_000_001
    reference = root / "reference.fa"
    write_fasta(reference, [ref_length], name="chr20")

    assemblies, vcfs, summaries = [], [], []
    for sample, shape in zip(SAMPLES, SHAPES):
        assembly = root / f"{sample}.flye.consensus.fasta"
        remaining = shape["total"] - shape["largest"]
        write_fasta(
            assembly,
            [shape["largest"], *([remaining // (shape["contigs"] - 1)] * (shape["contigs"] - 1))],
        )
        assemblies.append(str(assembly))

        variants = root / f"{sample}.flye.paftools.vcf"
        shared = list(range(1_000, 1_000 + 40 * 100, 100))
        child_only = [500_000 + i for i in range(5)] if sample == "HG002" else []
        write_vcf(variants, sample, shared + child_only, ref_length)
        vcfs.append(str(variants))

        summaries.append(
            summarize_quast_report(REPORT.format(name=f"{sample}.flye.consensus", **shape))
        )

    outputs = {
        "ONTAssembleCohort.sample_names": SAMPLES,
        "ONTAssembleCohort.assemblies": assemblies,
        "ONTAssembleCohort.vcfs": vcfs,
        "ONTAssembleCohort.quast_summaries": summaries,
        "ONTAssembleCohort.flye_params": [
            {
                "read_mode": "--nano-hq",
                "extra_args": "--iterations 1 --asm-coverage 40 --genome-size 10000001",
                "asm_coverage": "40", "iterations": "1",
                "genome_size": "10000001", "imputed": "true",
            }
            for _ in SAMPLES
        ],
        "ONTAssembleCohort.read_stats": [
            {
                "num_reads": "61234", "total_bases": "546000000", "read_n50": "29100",
                "coverage": "54.6", "pairwise_divergence": "0.0612",
                "divergence_overlaps": "18422",
            }
            for _ in SAMPLES
        ],
    }
    # The bare mapping miniwdl writes to the run-root file; the envelope is CLI stdout only.
    (run_dir / "outputs.json").write_text(json.dumps(outputs))

    # Decoy: nested sub-workflow directories have their own, possibly newer, outputs.json.
    decoy = run_dir / "call-assemble-0" / "call-flye" / "outputs.json"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(json.dumps({"ONTAssembleWithFlye.asm_polished": "bare, no envelope"}))
    later = (run_dir / "outputs.json").stat().st_mtime + 60
    os.utime(decoy, (later, later))
    return runs, reference, summaries[0]


def main() -> int:
    cells = notebook_cells()

    # Step 6b's Ray Data read runs on a worker (the head has CPU: 0), which cannot see the
    # driver's local disk, so the fixture goes on shared storage when there is one.
    shared = Path("/mnt/cluster_storage")
    fixture_parent = str(shared) if shared.is_dir() and os.access(shared, os.W_OK) else None

    with tempfile.TemporaryDirectory(dir=fixture_parent, prefix="wdl-readout-") as tmp:
        root = Path(tmp)
        runs, reference, first_summary = build_fixture(root)

        # The named regression: QUAST display names must not be keys; the underscored forms must.
        for display in ("# contigs", "Genome fraction (%)", "# mismatches per 100 kbp"):
            assert display not in first_summary, (
                f"{display!r} is a key now -- SummarizeQuastReport stopped mangling names,"
                " so the notebook's quast_key() must stop mangling them too"
            )
        assert "Genome_fraction_(%)" in first_summary

        # WORK and run() come from the Step 1 cell, which is not replayed. Real definitions, not
        # stubs: 6b normalizes the VCFs with them.
        preamble = textwrap.dedent(f"""
            import json, pathlib, subprocess
            import matplotlib
            matplotlib.use("Agg")
            RUN_DIR = pathlib.Path({str(runs)!r})
            WORK = pathlib.Path({str(root)!r})
            SAMPLES = {SAMPLES!r}
            reference = pathlib.Path({str(reference)!r})

            def run(cmd, **kwargs):
                subprocess.run([str(c) for c in cmd], check=True, **kwargs)
        """)

        replayed = ""
        for marker, label in CELLS:
            if marker == "parse_vcf_lines":
                try:
                    import ray  # noqa: F401
                except ImportError:
                    print(f"SKIP  {label} (ray not installed)")
                    continue
                # On the template image, where CI runs. Skipping beats a stub that tests nothing.
                missing = [t for t in ("samtools", "bcftools") if shutil.which(t) is None]
                if missing:
                    print(f"SKIP  {label} ({', '.join(missing)} not on PATH)")
                    continue
            script = preamble + replayed + "\n" + cells[marker]
            result = subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"FAIL  {label}\n{result.stderr}", file=sys.stderr)
                return 1
            print(f"ok    {label}")
            replayed += "\n" + cells[marker]

    print("\nnotebook readout cells agree with ONTAssembleCohort.wdl's declared outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
