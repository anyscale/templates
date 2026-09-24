#!/usr/bin/env python3
"""Offline tests for pipeline/bin/make_samplesheet.py.

The samplesheet is where main.nf learns which truth set belongs to which sample,
and a mistake there is silent: calls scored against the wrong genome still
produce a table, just a wrong one. So what is pinned here is the per-sample
wiring, and that a manifest that disagrees with the files on disk stops the run
before it starts rather than several processes in.

The manifests below are in the shape tools/stage-demo-data.sh writes and
tests/nextflow-genomics-on-ray/synthetic_trio.py writes; the second is also run
end to end, through the real pipeline, in the local check described in that
script's docstring.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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
    probe = os.path.join(candidate, "pipeline", "bin", "make_samplesheet.py")
    if os.path.exists(probe):
        _SOURCE = probe
        break
if _SOURCE is None:
    raise SystemExit("could not find pipeline/bin/make_samplesheet.py")

_spec = importlib.util.spec_from_file_location("make_samplesheet", _SOURCE)
ms = importlib.util.module_from_spec(_spec)
sys.modules["make_samplesheet"] = ms
_spec.loader.exec_module(ms)

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


def stage(root: str, samples=("HG002", "HG003", "HG004"), truth=True, corrupt=None) -> None:
    """Write a manifest and the files it names, the way a finished download looks."""
    os.makedirs(os.path.join(root, "truth"), exist_ok=True)
    entries = []
    for s in samples:
        entry = {"id": s, "fastq_1": f"{s}_R1.fastq.gz", "fastq_2": f"{s}_R2.fastq.gz"}
        for key, suffix in (("fastq_1", "1"), ("fastq_2", "2")):
            body = f"@{s}:{key}\nACGT\n+\nIIII\n".encode()
            with open(os.path.join(root, entry[key]), "wb") as out:
                out.write(body)
            entry[f"sha256_{suffix}"] = hashlib.sha256(body).hexdigest()
        if truth:
            for rel in (f"truth/{s}.vcf.gz", f"truth/{s}.vcf.gz.tbi", f"truth/{s}.bed"):
                open(os.path.join(root, rel), "w").close()
            entry["truth_vcf"] = f"truth/{s}.vcf.gz"
            entry["truth_bed"] = f"truth/{s}.bed"
        entries.append(entry)
    if corrupt:
        with open(os.path.join(root, corrupt), "ab") as out:
            out.write(b"truncated?")
    with open(os.path.join(root, "MANIFEST.json"), "w") as out:
        json.dump({"scale": "quick", "region": "chr20:1000000-3000000", "samples": entries}, out)


def read_sheet(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@check("each sample gets its own truth set, not the first one's")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        out = os.path.join(tmp, "samplesheet.csv")
        assert ms.main(["--data-dir", tmp, "--output", out]) == 0
        rows = read_sheet(out)
        assert [r["sample"] for r in rows] == ["HG002", "HG003", "HG004"]
        for row in rows:
            s = row["sample"]
            assert row["truth_vcf"] == os.path.join(tmp, "truth", f"{s}.vcf.gz"), row
            assert row["truth_bed"] == os.path.join(tmp, "truth", f"{s}.bed"), row
            assert os.path.isabs(row["fastq_1"]) and row["fastq_1"].endswith(f"{s}_R1.fastq.gz")


@check("the header is the one main.nf reads")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        out = os.path.join(tmp, "samplesheet.csv")
        ms.main(["--data-dir", tmp, "--output", out])
        with open(out) as handle:
            header = handle.readline().strip().split(",")
        # main.nf reads row.sample/fastq_1/fastq_2 and, when present,
        # row.truth_vcf/truth_bed. A renamed column is a silent unscored run.
        assert header == ["sample", "fastq_1", "fastq_2", "truth_vcf", "truth_bed"], header
        main_nf = os.path.join(os.path.dirname(os.path.dirname(_SOURCE)), "main.nf")
        with open(main_nf) as handle:
            text = handle.read()
        for column in header:
            assert f"row.{column}" in text, f"main.nf does not read row.{column}"


@check("no truth set is fine: the columns are empty and the sample goes unscored")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp, truth=False)
        out = os.path.join(tmp, "samplesheet.csv")
        assert ms.main(["--data-dir", tmp, "--output", out]) == 0
        assert all(r["truth_vcf"] == "" and r["truth_bed"] == "" for r in read_sheet(out))


@check("a truth VCF without its .tbi stops here, not in vcfeval")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        os.unlink(os.path.join(tmp, "truth", "HG003.vcf.gz.tbi"))
        try:
            ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv")])
        except ms.ManifestError as exn:
            assert "HG003" in str(exn) and ".tbi" in str(exn), exn
        else:
            raise AssertionError("expected ManifestError")


@check("half a truth set is an error, not a guess")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        path = os.path.join(tmp, "MANIFEST.json")
        with open(path) as handle:
            manifest = json.load(handle)
        del manifest["samples"][1]["truth_bed"]
        with open(path, "w") as out:
            json.dump(manifest, out)
        try:
            ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv")])
        except ms.ManifestError as exn:
            assert "HG003" in str(exn) and "not both" in str(exn), exn
        else:
            raise AssertionError("expected ManifestError")


@check("a FASTQ that does not match its checksum is caught")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp, corrupt="HG004_R2.fastq.gz")
        try:
            ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv")])
        except ms.ManifestError as exn:
            assert "HG004" in str(exn) and "checksum" in str(exn), exn
        else:
            raise AssertionError("expected ManifestError")
        # --no-verify is the documented way past it, for a re-run you trust.
        out = os.path.join(tmp, "s.csv")
        assert ms.main(["--data-dir", tmp, "--output", out, "--no-verify"]) == 0


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall make_samplesheet unit checks passed")
