#!/usr/bin/env python3
"""Turn a pile of ``rtg vcfeval`` summaries into one tidy table.

vcfeval writes a fixed-width ``summary.txt`` per comparison and says nothing in it
about what was compared -- the sample, the caller and the variant type live only
in the directory name. This collects both halves into ``benchmark.tsv``, which is
the single artifact the notebook reads and the one a reader can diff between runs.

    collect_vcfeval.py --output benchmark.tsv HG002.gatk.snp HG002.gatk.indel ...

Give it the vcfeval output *directories*, as COLLECT_BENCHMARK stages them. A
staged directory keeps the name RTG_VCFEVAL gave it, which is where the labels
are. A staged ``summary.txt`` does not help: its parent is a Nextflow hash
directory, and several of them in one task are an input file name collision.

**On the parsing.** The format below was read off the tool's output, and the
tests use summaries captured from real ``rtg vcfeval`` runs (see
tests/nextflow-genomics-on-ray/test_collect_vcfeval.py for which run) rather than
text written to match this code, so a test cannot pass by sharing the parser's
assumptions. If you change this parser, re-capture the fixtures from a real run;
do not edit them to match.

vcfeval emits two rows: one at the best-F-measure score threshold, and one labelled
``None`` for "no threshold applied". The ``None`` row is the one to report, because
a threshold chosen to maximise F-measure *on the truth set being scored against*
is fitted to the answer.
"""

from __future__ import annotations

import argparse
import os
import sys

#: vcfeval's column order, verbatim from rtg-tools'
#: ``RocContainer.writeSummary`` (``src/main/java/com/rtg/vcf/eval/RocContainer.java``):
#:
#:     table.addRow("Threshold", "True-pos-baseline", "True-pos-call",
#:                  "False-pos", "False-neg", "Precision", "Sensitivity", "F-measure");
#:
#: Checked against the header at runtime rather than trusted, so a format change
#: is a loud failure instead of a silent column shift.
EXPECTED_COLUMNS = [
    "Threshold",
    "True-pos-baseline",
    "True-pos-call",
    "False-pos",
    "False-neg",
    "Precision",
    "Sensitivity",
    "F-measure",
]

OUTPUT_COLUMNS = [
    "sample",
    "caller",
    "variant_type",
    "threshold",
    "true_pos_baseline",
    "true_pos_call",
    "false_pos",
    "false_neg",
    "precision",
    "recall",
    "f1",
]


#: What rtg writes instead of a table when the baseline had nothing to match --
#: `RocContainer.writeSummary` emits this string with no header at all. Worth
#: recognising by name: it means the truth set and the calls did not overlap
#: (usually a contig-naming or region mismatch), which is a different problem from
#: a format change and wants a different fix.
NO_VARIANTS_MARKER = "0 total baseline variants"


class SummaryFormatError(RuntimeError):
    """vcfeval's summary.txt did not look the way this parser expects."""


class NoBaselineVariants(SummaryFormatError):
    """vcfeval scored nothing: the baseline and the calls did not overlap."""


def parse_summary(path: str) -> dict[str, str]:
    """Return the unthresholded row of one ``summary.txt``."""
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]

    if not lines:
        raise SummaryFormatError(f"{path} is empty")

    if any(NO_VARIANTS_MARKER in line for line in lines):
        raise NoBaselineVariants(
            f"{path}: {NO_VARIANTS_MARKER}.\n"
            "vcfeval found no truth variants in the evaluated region, so there was\n"
            "nothing to score against. Usual causes, in order of likelihood:\n"
            "  - contig naming differs between the calls and the truth VCF "
            "(chr20 vs 20)\n"
            "  - --region and the high-confidence BED do not intersect\n"
            "  - the sample name given to --sample is not in one of the VCFs"
        )

    header = lines[0].split()
    if header != EXPECTED_COLUMNS:
        raise SummaryFormatError(
            f"{path}: unexpected header.\n"
            f"  expected {EXPECTED_COLUMNS}\n"
            f"  got      {header}\n"
            "rtg vcfeval's output format changed; update EXPECTED_COLUMNS and "
            "re-capture the test fixture from a real run."
        )

    rows = [line.split() for line in lines[1:] if not line.startswith("---")]
    rows = [row for row in rows if len(row) == len(EXPECTED_COLUMNS)]
    if not rows:
        raise SummaryFormatError(f"{path}: header present but no data rows")

    # Prefer the row whose threshold is literally "None" -- see the module
    # docstring. Fall back to the last row, which is where vcfeval puts it.
    for row in rows:
        if row[0] == "None":
            return dict(zip(EXPECTED_COLUMNS, row, strict=True))
    return dict(zip(EXPECTED_COLUMNS, rows[-1], strict=True))


def resolve(path: str) -> tuple[str, str]:
    """``(summary.txt path, directory whose name carries the labels)``.

    Accepts a vcfeval output directory, which is what COLLECT_BENCHMARK passes,
    or a ``summary.txt`` inside one. ``abspath``, never ``realpath``: a staged
    input is a symlink named ``HG002.gatk.snp`` pointing into a hash directory,
    and resolving it would throw the name away.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return os.path.join(path, "summary.txt"), os.path.basename(path)
    return path, os.path.basename(os.path.dirname(path))


def label_from_path(path: str) -> tuple[str, str, str]:
    """Recover (sample, caller, variant_type) from the vcfeval output directory.

    RTG_VCFEVAL names its output directory ``<sample>.<caller>.<vtype>``. Parsed
    from the right, because a sample id may itself contain a dot (HG002.hiseq is a
    perfectly ordinary thing to call a sample) while the caller and type never do.
    """
    _summary, directory = resolve(path)
    parts = directory.rsplit(".", 2)
    if len(parts) != 3:
        raise SummaryFormatError(
            f"cannot read sample/caller/type from directory {directory!r} "
            f"(expected <sample>.<caller>.<vtype>). Pass the vcfeval output "
            "directories themselves: a staged summary.txt's parent is a hash directory."
        )
    return parts[0], parts[1], parts[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "eval_dirs", nargs="+", help="vcfeval output directories (or summary.txt inside them)"
    )
    args = parser.parse_args(argv)

    rows = []
    for path in args.eval_dirs:
        sample, caller, vtype = label_from_path(path)
        parsed = parse_summary(resolve(path)[0])
        rows.append(
            {
                "sample": sample,
                "caller": caller,
                "variant_type": vtype,
                "threshold": parsed["Threshold"],
                "true_pos_baseline": parsed["True-pos-baseline"],
                "true_pos_call": parsed["True-pos-call"],
                "false_pos": parsed["False-pos"],
                "false_neg": parsed["False-neg"],
                "precision": parsed["Precision"],
                "recall": parsed["Sensitivity"],
                "f1": parsed["F-measure"],
            }
        )

    rows.sort(key=lambda r: (r["sample"], r["caller"], r["variant_type"]))
    with open(args.output, "w") as out:
        out.write("\t".join(OUTPUT_COLUMNS) + "\n")
        for row in rows:
            out.write("\t".join(row[col] for col in OUTPUT_COLUMNS) + "\n")

    print(f"collect_vcfeval: {len(rows)} comparison(s) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
