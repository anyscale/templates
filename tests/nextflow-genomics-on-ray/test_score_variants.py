#!/usr/bin/env python3
"""Offline tests for bin/score_variants.py.

No GPU, no model download, no torch import: everything below is the plumbing
*around* the model, which is where the bugs that produce quietly wrong numbers
live.

Three things are worth this much attention.

`IndexedFasta` does byte arithmetic against a ``.fai`` to turn a base coordinate
into a file offset. Get the newline accounting wrong and it still returns
plausible-looking sequence -- just shifted -- and every downstream score is
computed on the wrong window with nothing to indicate it. So it is checked
exhaustively against ground truth at several line widths, including across line
boundaries and at both contig edges.

The window substitution has to put the ALT allele exactly where the REF was. Off
by one and the model scores a variant that is not the one in the VCF.

And a REF that disagrees with the reference FASTA has to be *rejected*. Scoring
it anyway would produce a number, and a number is much harder to notice than a
gap.
"""

from __future__ import annotations

import gzip
import importlib.util
import os
import random
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "templates", "nextflow-genomics-on-ray")
)
_SOURCE = None
for candidate in (os.getcwd(), _TEMPLATE):
    # pipeline/bin, not bin: Nextflow puts <projectDir>/bin on PATH for every
    # task, and projectDir is where main.nf lives.
    probe = os.path.join(candidate, "pipeline", "bin", "score_variants.py")
    if os.path.exists(probe):
        _SOURCE = probe
        break
if _SOURCE is None:
    raise SystemExit("could not find pipeline/bin/score_variants.py")

_spec = importlib.util.spec_from_file_location("score_variants", _SOURCE)
sv = importlib.util.module_from_spec(_spec)
sys.modules["score_variants"] = sv  # dataclasses resolves types through sys.modules
_spec.loader.exec_module(sv)

_FAILURES: list[str] = []


def check(name: str):
    def wrap(fn):
        try:
            fn()
        except Exception:
            _FAILURES.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
        return fn

    return wrap


def write_fasta(path: str, contigs: dict[str, str], width: int) -> None:
    """Write a FASTA and the ``.fai`` samtools would produce for it."""
    with open(path, "w") as fh, open(path + ".fai", "w") as fai:
        for name, seq in contigs.items():
            fh.write(f">{name}\n")
            offset = fh.tell()
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
            fai.write(f"{name}\t{len(seq)}\t{offset}\t{width}\t{width + 1}\n")


@check("fasta: fetch matches ground truth at every line width and boundary")
def _() -> None:
    random.seed(7)
    with tempfile.TemporaryDirectory() as tmp:
        # 60 is the samtools default; 70/80 are common in distributed references;
        # 13 is deliberately pathological, to catch arithmetic that only works
        # when the width is a round number.
        for width in (60, 70, 80, 13):
            contigs = {
                "chr20": "".join(random.choice("ACGT") for _ in range(1000)),
                "chr21": "".join(random.choice("ACGT") for _ in range(137)),
            }
            path = os.path.join(tmp, f"ref{width}.fa")
            write_fasta(path, contigs, width)
            fasta = sv.IndexedFasta(path)
            try:
                for name, seq in contigs.items():
                    assert fasta.length(name) == len(seq)
                    cases = [
                        (0, len(seq)),                 # whole contig
                        (0, 1),                        # first base
                        (len(seq) - 1, len(seq)),      # last base
                        (width - 1, width + 2),        # across a line boundary
                        (width, 2 * width),            # a whole interior line
                        (-50, 10),                     # clamped left
                        (len(seq) - 5, len(seq) + 50), # clamped right
                        (10, 10),                      # empty
                    ]
                    cases += [
                        (s, s + random.randint(1, 90))
                        for s in (random.randint(0, len(seq) - 1) for _ in range(60))
                    ]
                    for start, end in cases:
                        want = seq[max(0, start) : min(len(seq), end)]
                        got = fasta.fetch(name, start, end)
                        assert got == want, f"w={width} {name}[{start}:{end}]"
            finally:
                fasta.close()


@check("fasta: a missing .fai says how to make one")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bare.fa")
        with open(path, "w") as fh:
            fh.write(">chr1\nACGT\n")
        try:
            sv.IndexedFasta(path)
        except FileNotFoundError as exn:
            assert "samtools faidx" in str(exn), exn
        else:
            raise AssertionError("expected FileNotFoundError")


@check("fasta: an unknown contig names the mismatch")
def _() -> None:
    # chr20 vs 20 is the single most common way a VCF and a reference disagree.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.fa")
        write_fasta(path, {"chr20": "ACGT" * 30}, 60)
        fasta = sv.IndexedFasta(path)
        try:
            fasta.fetch("20", 0, 10)
        except KeyError as exn:
            assert "chr20" in str(exn)
        else:
            raise AssertionError("expected KeyError")
        finally:
            fasta.close()


@check("vcf: multi-allelic records split, no-alt records skipped, case normalised")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.vcf")
        with open(path, "w") as fh:
            fh.write(
                "##fileformat=VCFv4.2\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                "chr20\t100\t.\tA\tG\t50\tPASS\t.\n"
                "chr20\t200\t.\tAT\tA,ATT\t50\tPASS\t.\n"
                "chr20\t300\t.\tC\t.\t50\tPASS\t.\n"
                "chr20\t350\t.\tC\t*\t50\tPASS\t.\n"
                "chr20\t400\t.\tg\tt\t50\tPASS\t.\n"
            )
        want = [
            "chr20:100:A:G",
            "chr20:200:AT:A",
            "chr20:200:AT:ATT",
            "chr20:400:G:T",
        ]
        assert [v.key for v in sv.read_vcf(path)] == want

        gz = os.path.join(tmp, "t.vcf.gz")
        with gzip.open(gz, "wt") as fh, open(path) as src:
            fh.write(src.read())
        assert [v.key for v in sv.read_vcf(gz)] == want


class _WindowProbe:
    """Just enough of VariantScorer to exercise _windows without importing torch."""

    def __init__(self, fasta, context: int) -> None:
        self.fasta = fasta
        self.context = context

    _windows = sv.VariantScorer._windows


@check("windows: the ALT lands exactly where the REF was")
def _() -> None:
    random.seed(11)
    with tempfile.TemporaryDirectory() as tmp:
        seq = "".join(random.choice("ACGT") for _ in range(400))
        path = os.path.join(tmp, "w.fa")
        write_fasta(path, {"chr1": seq}, 60)
        fasta = sv.IndexedFasta(path)
        try:
            probe = _WindowProbe(fasta, context=10)
            pos = 101  # 1-based
            ref_base = seq[pos - 1]
            alt_base = "T" if ref_base != "T" else "A"
            ref_win, alt_win = probe._windows(sv.Variant("chr1", pos, ref_base, alt_base))

            assert ref_win == seq[pos - 1 - 10 : pos - 1 + 1 + 10]
            assert alt_win == ref_win[:10] + alt_base + ref_win[11:]
            assert len(alt_win) == len(ref_win)
            # And the substitution really changed something.
            assert alt_win != ref_win
        finally:
            fasta.close()


@check("windows: an insertion lengthens the alt window by exactly the insert")
def _() -> None:
    random.seed(13)
    with tempfile.TemporaryDirectory() as tmp:
        seq = "".join(random.choice("ACGT") for _ in range(400))
        path = os.path.join(tmp, "w.fa")
        write_fasta(path, {"chr1": seq}, 60)
        fasta = sv.IndexedFasta(path)
        try:
            probe = _WindowProbe(fasta, context=10)
            pos = 200
            ref = seq[pos - 1]
            ref_win, alt_win = probe._windows(sv.Variant("chr1", pos, ref, ref + "GGGG"))
            assert len(alt_win) == len(ref_win) + 4
        finally:
            fasta.close()


@check("windows: a REF that disagrees with the FASTA is rejected, not scored")
def _() -> None:
    # The failure this prevents is silent: score the variant anyway and you get a
    # number computed on a sequence the VCF never described.
    random.seed(17)
    with tempfile.TemporaryDirectory() as tmp:
        seq = "".join(random.choice("ACGT") for _ in range(400))
        path = os.path.join(tmp, "w.fa")
        write_fasta(path, {"chr1": seq}, 60)
        fasta = sv.IndexedFasta(path)
        try:
            probe = _WindowProbe(fasta, context=10)
            wrong = "T" if seq[199] != "T" else "A"
            _, alt_win = probe._windows(sv.Variant("chr1", 200, wrong, "C"))
            assert alt_win == "", "a mismatched REF must produce no alt window"
        finally:
            fasta.close()


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall score_variants unit checks passed")
