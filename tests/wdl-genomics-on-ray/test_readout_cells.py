#!/usr/bin/env python3
"""Offline check that the notebook's readout cells agree with the workflow's outputs.

Runs README.ipynb's Step 6 / 6b cells against a synthetic `outputs.json` in the shape
ONTAssembleCohort.wdl declares. No cluster, no data, no assembly -- seconds, not hours.

The bug this exists for shipped once and was invisible for months. `SummarizeQuastReport`
mangles QUAST metric names (`sed 's/ /_/g'`), so `quast_summary` is keyed
`Genome_fraction_(%)`, not `Genome fraction (%)`. The notebook looked up the display
names, which matched four keys out of nine and made its own guard unfalsifiable:

    assert "Genome fraction (%)" in summary   # never true, whatever QUAST produced

Nothing caught it, because CI had never reached that cell -- the demo data 404'd two
steps earlier. Any mismatch between a WDL output name and what the notebook reads is
the same class of failure, and this catches all of it before an assembly runs.

Executes the cells' real source, so it cannot drift from what ships.
"""

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

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "wdl-genomics-on-ray"
SAMPLES = ["HG002", "HG003", "HG004"]

#: Cells to exercise, and what each is. Indices are resolved by content, not position,
#: so inserting a cell does not silently skip a check.
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
    """Quast.wdl's SummarizeQuastReport command block, reimplemented faithfully.

    Kept in step with that task by hand. If the sed pipeline there changes, this is the
    other half of the contract and has to change with it -- which is the point: the
    mangling is load-bearing and deserves to be written down twice.
    """
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
    """The header paftools.js writes under `call -f`, not a minimal one.

    paftools.js:467-474 emits ##contig for every reference sequence plus ##FORMAT=GT.
    A fixture without them passes a naive line parser and is rejected by `bcftools norm`,
    which is how the first version of this fixture reported a bug in the notebook that
    was really a bug in the fixture.
    """
    with path.open("w") as handle:
        handle.write("##fileformat=VCFv4.1\n")
        handle.write(f"##contig=<ID=chr20,length={ref_length}>\n")
        handle.write('##INFO=<ID=QNAME,Number=1,Type=String,Description="Query name">\n')
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n")
        for position in positions:
            handle.write(f"chr20\t{position}\t.\tA\tG\t60\t.\t.\tGT\t1/1\n")


def notebook_cells() -> dict[str, str]:
    """Locate each cell by a marker in its source, so a reordered notebook still works."""
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

    # Named chr20 to match the VCFs below, because Step 6b now runs `bcftools norm -f`
    # over them and bcftools rejects a record whose contig is absent from the reference.
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
    # The bare mapping, because that is what miniwdl actually writes to the run-root file --
    # the {"dir", "outputs"} envelope is CLI stdout only. The first fixture used the envelope
    # (copied from the then-unexercised notebook, circularly) and hid exactly that mismatch.
    (run_dir / "outputs.json").write_text(json.dumps(outputs))

    # The decoy that broke the first real cohort run:
    # miniwdl writes an envelope-less outputs.json inside every nested sub-workflow
    # directory, and one of them carried a newer mtime than the run root's. A readout that
    # globs recursively and sorts by mtime picks this file and dies on the missing
    # "outputs" key -- so the fixture plants one, newer, exactly where miniwdl puts it.
    decoy = run_dir / "call-assemble-0" / "call-flye" / "outputs.json"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_text(json.dumps({"ONTAssembleWithFlye.asm_polished": "bare, no envelope"}))
    later = (run_dir / "outputs.json").stat().st_mtime + 60
    os.utime(decoy, (later, later))
    return runs, reference, summaries[0]


def main() -> int:
    cells = notebook_cells()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs, reference, first_summary = build_fixture(root)

        # The named regression: QUAST's display names must NOT be keys, and the
        # underscored forms must be. If this ever inverts, the notebook's lookups
        # need to invert with it.
        for display in ("# contigs", "Genome fraction (%)", "# mismatches per 100 kbp"):
            assert display not in first_summary, (
                f"{display!r} is a key now -- SummarizeQuastReport stopped mangling names,"
                " so the notebook's quast_key() must stop mangling them too"
            )
        assert "Genome_fraction_(%)" in first_summary

        # WORK and run() come from the notebook's Step 1 cell, which this harness does not
        # replay: it exercises the readout cells only. Step 6b uses both, to normalize the
        # VCFs before comparing them, so bind the same definitions here rather than stubbing
        # them -- a stubbed `run` would make the normalization untested precisely where it
        # matters.
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

        # Cells run cumulatively, as in a notebook: the curve cell uses names/outputs
        # bound by the QUAST cell.
        replayed = ""
        for marker, label in CELLS:
            if marker == "parse_vcf_lines":
                try:
                    import ray  # noqa: F401
                except ImportError:
                    print(f"SKIP  {label} (ray not installed)")
                    continue
                # Present on the template image, where CI runs this; absent on a bare
                # laptop. Skipping beats a stub, which would leave the normalization
                # unexercised on the one machine that can exercise it.
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
