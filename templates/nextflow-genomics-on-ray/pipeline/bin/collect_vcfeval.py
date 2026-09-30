#!/usr/bin/env python3
"""Turn a pile of ``rtg vcfeval`` summaries into one tidy table."""

from __future__ import annotations

import argparse
import os
import sys

# rtg-tools' RocContainer.writeSummary columns; checked against the header, not trusted.
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


# Written instead of a table when truth and calls do not overlap (usually contig naming).
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

    # The unthresholded row: the best-F threshold is fitted to the truth set being scored.
    for row in rows:
        if row[0] == "None":
            return dict(zip(EXPECTED_COLUMNS, row, strict=True))
    return dict(zip(EXPECTED_COLUMNS, rows[-1], strict=True))


def resolve(path: str) -> tuple[str, str]:
    """``(summary.txt path, directory whose name carries the labels)``."""
    # abspath, not realpath: the staged symlink's name carries the labels.
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return os.path.join(path, "summary.txt"), os.path.basename(path)
    return path, os.path.basename(os.path.dirname(path))


def label_from_path(path: str) -> tuple[str, str, str]:
    """Recover (sample, caller, variant_type) from the vcfeval output directory."""
    _summary, directory = resolve(path)
    # From the right: a sample id may contain a dot.
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
