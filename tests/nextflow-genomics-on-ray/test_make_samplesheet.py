#!/usr/bin/env python3
"""Offline tests for pipeline/bin/make_samplesheet.py."""

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


# As tools/stage-demo-data.sh publishes them, in order.
RESOURCES = (
    "hapmap_3.3",
    "1000G_phase1.snps.high_confidence",
    "Mills_and_1000G_gold_standard.indels",
)


def stage(root: str, samples=("HG002", "HG003", "HG004"), truth=True, corrupt=None,
          resources=RESOURCES) -> None:
    """Write a manifest and the files it names, the way a finished download looks."""
    os.makedirs(os.path.join(root, "truth"), exist_ok=True)
    os.makedirs(os.path.join(root, "known_sites"), exist_ok=True)
    tranche = []
    for name in resources:
        rel = f"known_sites/{name}.chr20.vcf.gz"
        body = f"{name} sites\n".encode()
        with open(os.path.join(root, rel), "wb") as out:
            out.write(body)
        open(os.path.join(root, f"{rel}.tbi"), "w").close()
        tranche.append({"name": name, "vcf": rel, "sha256": hashlib.sha256(body).hexdigest()})
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
    manifest = {"scale": "quick", "region": "chr20:1000000-3000000", "samples": entries}
    if tranche:
        manifest["tranche_resources"] = tranche
    with open(os.path.join(root, "MANIFEST.json"), "w") as out:
        json.dump(manifest, out)


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
        # A renamed column is a silently unscored run.
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
        out = os.path.join(tmp, "s.csv")
        assert ms.main(["--data-dir", tmp, "--output", out, "--no-verify"]) == 0


@check("tranche resources come out comma-separated, absolute, in manifest order")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        res = os.path.join(tmp, "tranche_resources.txt")
        assert ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv"),
                        "--resources-output", res]) == 0
        with open(res) as handle:
            got = handle.read().strip().split(",")
        want = [os.path.join(tmp, "known_sites", f"{name}.chr20.vcf.gz") for name in RESOURCES]
        # main.nf splits --tranche_resources on commas and adds .tbi to each.
        assert got == want, got


@check("a manifest with no tranche resources writes an empty list, not an error")
def _() -> None:
    # The synthetic trio's manifest has none.
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp, resources=())
        res = os.path.join(tmp, "tranche_resources.txt")
        assert ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv"),
                        "--resources-output", res]) == 0
        with open(res) as handle:
            assert handle.read().strip() == ""


@check("a tranche resource without its .tbi stops here, not in FilterVariantTranches")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp)
        os.unlink(os.path.join(tmp, "known_sites", "hapmap_3.3.chr20.vcf.gz.tbi"))
        try:
            ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv")])
        except ms.ManifestError as exn:
            assert "hapmap_3.3" in str(exn) and ".tbi" in str(exn), exn
        else:
            raise AssertionError("expected ManifestError")


@check("a tranche resource that does not match its checksum is caught")
def _() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage(tmp, corrupt="known_sites/Mills_and_1000G_gold_standard.indels.chr20.vcf.gz")
        try:
            ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv")])
        except ms.ManifestError as exn:
            assert "Mills" in str(exn) and "checksum" in str(exn), exn
        else:
            raise AssertionError("expected ManifestError")
        assert ms.main(["--data-dir", tmp, "--output", os.path.join(tmp, "s.csv"),
                        "--no-verify"]) == 0


if _FAILURES:
    print(f"\n{len(_FAILURES)} failing: {', '.join(_FAILURES)}", file=sys.stderr)
    sys.exit(1)
print("\nall make_samplesheet unit checks passed")
