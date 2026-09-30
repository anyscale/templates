#!/usr/bin/env python3
"""Offline tests for pipeline/bin/collect_vcfeval.py, against summaries rtg itself wrote."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "templates", "nextflow-genomics-on-ray")
)
_SOURCE = None
for candidate in (os.getcwd(), _HERE, _TEMPLATE):
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

# rtg-tools 3.13's own bytes: refresh them by re-running rtg, never by editing to fit the parser.
FIXTURES = os.path.join(_HERE, "fixtures", "vcfeval")

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


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name, "summary.txt")


def staged(task_dir: str, work: str, labels: list[tuple[str, str]]) -> list[str]:
    """Symlink each output from its own hash-named work dir into ``task_dir``, as Nextflow does."""
    names = []
    for i, (name, source) in enumerate(labels):
        real = os.path.join(work, f"{i:02x}", f"{i:030x}", name)
        os.makedirs(real)
        shutil.copy(fixture(source), os.path.join(real, "summary.txt"))
        os.symlink(real, os.path.join(task_dir, name))
        names.append(name)
    return names


@check("header: rtg's own output matches RocContainer.writeSummary exactly")
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
    for name in ("HG002.gatk.snp", "HG003.gatk.indel", "crossed-HG003-calls-vs-HG002-truth"):
        with open(fixture(name)) as handle:
            assert handle.readline().split() == cv.EXPECTED_COLUMNS, name


@check("the unthresholded row is the one reported, not the best-threshold one")
def _() -> None:
    # The best-threshold row, which rtg writes first, is fitted to the truth being scored.
    row = cv.parse_summary(fixture("HG002.gatk.snp"))
    assert row["Threshold"] == "None", row
    assert (row["True-pos-baseline"], row["False-pos"], row["False-neg"]) == ("344", "0", "0")
    crossed = cv.parse_summary(fixture("crossed-HG003-calls-vs-HG002-truth"))
    assert crossed["Threshold"] == "None"
    assert (crossed["Precision"], crossed["Sensitivity"], crossed["F-measure"]) == (
        "0.4586",
        "0.3374",
        "0.3888",
    ), crossed


@check("'nothing matched' is a distinct failure from 'format changed'")
def _() -> None:
    # rtg writes one headerless line and exits 0; the cause is contig naming or region, not format.
    with open(fixture("no-baseline")) as handle:
        assert cv.NO_VARIANTS_MARKER in handle.read()
    try:
        cv.parse_summary(fixture("no-baseline"))
    except cv.NoBaselineVariants as exn:
        assert "chr20 vs 20" in str(exn)
    else:
        raise AssertionError("expected NoBaselineVariants")
    assert issubclass(cv.NoBaselineVariants, cv.SummaryFormatError)


@check("a changed header is loud, not a silent column shift")
def _() -> None:
    # Hand-written on purpose: this is the output rtg does *not* write.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "summary.txt")
        with open(path, "w") as out:
            out.write("Threshold  TP  FP\n---\n None 1 2\n")
        try:
            cv.parse_summary(path)
        except cv.SummaryFormatError as exn:
            assert "unexpected header" in str(exn)
            assert "re-capture the test fixture" in str(exn)
        else:
            raise AssertionError("expected SummaryFormatError")


@check("labels come from the directory, right to left")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for name, want in (
            ("HG002.gatk.snp", ("HG002", "gatk", "snp")),
            # A sample id containing a dot is ordinary; caller and type never do.
            ("HG002.hiseq.gatk.indel", ("HG002.hiseq", "gatk", "indel")),
            ("HG002.gatk_hard.snp", ("HG002", "gatk_hard", "snp")),
            ("HG002.hiseq.gatk_cnn.indel", ("HG002.hiseq", "gatk_cnn", "indel")),
        ):
            os.makedirs(os.path.join(tmp, name))
            assert cv.label_from_path(os.path.join(tmp, name)) == want
            assert cv.label_from_path(os.path.join(tmp, name, "summary.txt")) == want


@check("staged directories: the shape COLLECT_BENCHMARK actually gets")
def _() -> None:
    with tempfile.TemporaryDirectory() as root:
        work, task = os.path.join(root, "work"), os.path.join(root, "task")
        os.makedirs(task)
        names = staged(task, work, [
            ("HG004.gatk.indel", "HG003.gatk.indel"),
            ("HG002.gatk.snp", "HG002.gatk.snp"),
            ("HG002.gatk.indel", "HG003.gatk.indel"),
        ])
        cwd = os.getcwd()
        os.chdir(task)
        try:
            assert cv.main(["--output", "benchmark.tsv", *names]) == 0
            with open("benchmark.tsv") as handle:
                lines = [line.rstrip("\n").split("\t") for line in handle]
        finally:
            os.chdir(cwd)

        assert lines[0] == cv.OUTPUT_COLUMNS
        got = [tuple(r[:3]) for r in lines[1:]]
        # Sorted, so two runs of the same pipeline produce a byte-identical table.
        assert got == [("HG002", "gatk", "indel"), ("HG002", "gatk", "snp"),
                       ("HG004", "gatk", "indel")], got
        snp = lines[1 + got.index(("HG002", "gatk", "snp"))]
        assert snp[cv.OUTPUT_COLUMNS.index("threshold")] == "None"
        assert snp[cv.OUTPUT_COLUMNS.index("true_pos_baseline")] == "344"


@check("staged summary.txt files fail loudly: their parent is a hash directory")
def _() -> None:
    # What collecting summary.txt files would stage, had Nextflow not refused the name clash.
    with tempfile.TemporaryDirectory() as root:
        work, task = os.path.join(root, "work"), os.path.join(root, "task")
        os.makedirs(task)
        staged(task, work, [("HG002.gatk.snp", "HG002.gatk.snp")])
        real = os.path.realpath(os.path.join(task, "HG002.gatk.snp", "summary.txt"))
        hashed = os.path.join(root, "hashdir")
        os.makedirs(hashed)
        shutil.copy(real, os.path.join(hashed, "summary.txt"))
        try:
            cv.label_from_path(os.path.join(hashed, "summary.txt"))
        except cv.SummaryFormatError as exn:
            assert "hash directory" in str(exn), exn
        else:
            raise AssertionError("expected SummaryFormatError")


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall collect_vcfeval unit checks passed")
