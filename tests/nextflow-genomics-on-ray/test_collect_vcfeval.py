#!/usr/bin/env python3
"""Offline tests for pipeline/bin/collect_vcfeval.py.

The header this parser expects is not a guess. It is transcribed from rtg-tools'
own source, ``src/main/java/com/rtg/vcf/eval/RocContainer.java``::

    table.addRow("Threshold", "True-pos-baseline", "True-pos-call",
                 "False-pos", "False-neg", "Precision", "Sensitivity", "F-measure");

That matters more than it looks. The sibling WDL template shipped a readout bug
for months because its test fixture encoded the same wrong assumption as the code
it was testing, so the test could only ever agree with itself. A fixture has to
come from the tool, not from the parser.

The same source also settles two behaviours worth pinning:

* the threshold column reads ``None`` when the score is NaN (``RocContainer:318``),
  which is the row to report -- a threshold chosen to maximise F-measure against
  the very truth set being scored is fitted to the answer;
* when the baseline has no variants, rtg writes a one-line message and **no
  header** (``RocContainer:420``), so "nothing matched" and "the format changed"
  are different failures and must not report as the same one.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "templates", "nextflow-genomics-on-ray")
)
_SOURCE = None
for candidate in (os.getcwd(), _TEMPLATE):
    probe = os.path.join(candidate, "pipeline", "bin", "collect_vcfeval.py")
    if os.path.exists(probe):
        _SOURCE = probe
        break
if _SOURCE is None:
    raise SystemExit("could not find pipeline/bin/collect_vcfeval.py")

_spec = importlib.util.spec_from_file_location("collect_vcfeval", _SOURCE)
cv = importlib.util.module_from_spec(_spec)
sys.modules["collect_vcfeval"] = cv
_spec.loader.exec_module(cv)

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


#: Column layout per RocContainer.java. Whitespace-aligned the way rtg's TextTable
#: renders it -- the parser splits on whitespace, so the exact padding does not
#: matter, but ragged columns are what the real file looks like.
SUMMARY = """\
Threshold  True-pos-baseline  True-pos-call  False-pos  False-neg  Precision  Sensitivity  F-measure
----------------------------------------------------------------------------------------------------------
   16.500              44523          44530        612        988     0.9864       0.9783     0.9823
    None               44980          44991       1204        531     0.9739       0.9882     0.9810
"""

NO_VARIANTS = "0 total baseline variants, no summary statistics available\n"


def write_summary(root: str, directory: str, text: str = SUMMARY) -> str:
    path = os.path.join(root, directory)
    os.makedirs(path, exist_ok=True)
    summary = os.path.join(path, "summary.txt")
    with open(summary, "w") as handle:
        handle.write(text)
    return summary


@check("header matches rtg-tools' RocContainer.writeSummary exactly")
def _() -> None:
    assert cv.EXPECTED_COLUMNS == [
        "Threshold",
        "True-pos-baseline",
        "True-pos-call",
        "False-pos",
        "False-neg",
        "Precision",
        "Sensitivity",
        "F-measure",
    ]
    assert SUMMARY.split("\n")[0].split() == cv.EXPECTED_COLUMNS


@check("the unthresholded row is the one reported")
def _() -> None:
    # Not the best-F-measure row: that threshold was chosen against the same truth
    # set being scored, so reporting it would flatter the result.
    with tempfile.TemporaryDirectory() as tmp:
        row = cv.parse_summary(write_summary(tmp, "HG002.gatk.snp"))
        assert row["Threshold"] == "None", row
        assert row["F-measure"] == "0.9810"
        assert row["Precision"] == "0.9739"
        assert row["Sensitivity"] == "0.9882"


@check("labels are recovered from the directory, right to left")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = write_summary(tmp, "HG002.gatk.snp")
        assert cv.label_from_path(p) == ("HG002", "gatk", "snp")
        # A sample id containing a dot is ordinary; caller and type never do, so
        # parsing from the right is what keeps this correct.
        p2 = write_summary(tmp, "HG002.hiseq.deepvariant.indel")
        assert cv.label_from_path(p2) == ("HG002.hiseq", "deepvariant", "indel")


@check("a changed header is loud, not a silent column shift")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = write_summary(tmp, "HG002.gatk.snp", "Threshold  TP  FP\n---\n None 1 2\n")
        try:
            cv.parse_summary(p)
        except cv.SummaryFormatError as exn:
            assert "unexpected header" in str(exn)
            assert "re-capture the test fixture" in str(exn)
        else:
            raise AssertionError("expected SummaryFormatError")


@check("'nothing matched' is a distinct failure from 'format changed'")
def _() -> None:
    # rtg writes this with no header at all. Reporting it as a format error would
    # send a reader to the parser when the actual problem is almost always contig
    # naming or a region that does not intersect the truth BED.
    with tempfile.TemporaryDirectory() as tmp:
        p = write_summary(tmp, "HG002.gatk.snp", NO_VARIANTS)
        try:
            cv.parse_summary(p)
        except cv.NoBaselineVariants as exn:
            assert "chr20 vs 20" in str(exn)
        else:
            raise AssertionError("expected NoBaselineVariants")
        # and it is still a SummaryFormatError, so a caller catching the base
        # class does not suddenly stop catching this
        assert issubclass(cv.NoBaselineVariants, cv.SummaryFormatError)


@check("the collected table is sorted and complete")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = [
            write_summary(tmp, "HG004.gatk.indel"),
            write_summary(tmp, "HG002.deepvariant.snp"),
            write_summary(tmp, "HG002.gatk.snp"),
        ]
        out = os.path.join(tmp, "benchmark.tsv")
        assert cv.main(["--output", out] + paths) == 0
        with open(out) as handle:
            lines = [line.rstrip("\n").split("\t") for line in handle]

        assert lines[0] == cv.OUTPUT_COLUMNS
        got = [(r[0], r[1], r[2]) for r in lines[1:]]
        # Sorted, so two runs of the same pipeline produce a byte-identical table.
        assert got == sorted(got), got
        assert len(got) == 3
        assert ("HG002", "deepvariant", "snp") in got
        # f1 column carries the unthresholded value
        assert lines[1][cv.OUTPUT_COLUMNS.index("f1")] == "0.9810"


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall collect_vcfeval unit checks passed")
